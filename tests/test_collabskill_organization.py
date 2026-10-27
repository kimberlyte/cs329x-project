"""Meaningful synthetic-only checks. No production sample is selected or exported."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import organize_collabskill as org


def config(n=2):
    cfg=deepcopy(org.load_config())
    cfg.update(source_revision="fixture",input_sha256={},manual_exclusions={})
    cfg["category_quotas_per_turn_type"]=[dict(source="gdpval_benchmark",category="Sector",n=n)]
    cfg["length_quotas"]={"2-4":n,"5-9":0,"10-19":0,"20+":0}
    return cfg


def events(n, text="request"):
    result=[]
    for i in range(n):
        result.extend([dict(role="user",type="message",is_conversational=True,content=f"{text} {i}",turn_id=2*i),
                       dict(role="assistant",type="message",is_conversational=True,content="response",turn_id=2*i+1)])
    return result


def candidates():
    rows=[]
    # Imbalanced availability: optimum should select one of each occupation in both groups.
    for sid,turn,occ in [("s-a","single","A"),("s-b","single","B"),("s-a2","single","A"),
                         ("m-a","multi","A"),("m-b","multi","B"),("m-b2","multi","B")]:
        rows.append(dict(session_id=sid,task_id="t-"+sid,user_id="u-"+sid,agent_id="agent",dataset="gdpval_benchmark",
                         native_category="Sector",occupation=occ,sector="Sector",turn_type=turn,
                         length_bin="single" if turn=="single" else "2-4",eligible_candidate=True,
                         source_revision="fixture",has_log=True,category_status="source_metadata_exact_id_join",
                         split_group_id="g-"+sid,human_messages=1 if turn=="single" else 2,
                         transcript_hash=org.event_summary(events(1 if turn=="single" else 2,sid))["transcript_hash"]))
    return pd.DataFrame(rows)


class CountingTests(unittest.TestCase):
    def test_tools_are_not_human_turns(self):
        rs=events(1)+[dict(role="assistant",type="tool_call",tool_name="write",is_conversational=False,content="x"),
                      dict(role="tool",type="tool_result",is_conversational=False,content="done")]
        got=org.event_summary(rs)
        self.assertEqual((got["human_messages"],got["assistant_messages"],got["tool_events"]),(1,1,2))

    def test_missing_is_unknown_not_zero(self):
        got=org.event_summary([{"no_extracted_log_reason":"missing"}])
        self.assertIsNone(got["human_messages"])
        self.assertEqual(org.length_bin(got["human_messages"]),"missing")
        self.assertEqual(org.length_bin(0),"zero")

    def test_rounds_differ_from_messages(self):
        rs=[events(1)[0],events(1)[0],events(1)[1]]
        got=org.event_summary(rs)
        self.assertEqual(got["human_messages"],2)
        self.assertEqual(got["conversational_rounds"],1)

    def test_no_behavior_after_request(self):
        self.assertFalse(org.event_summary([events(1)[0]])["has_agent_behavior"])

    def test_setup_commands_flagged_without_changing_counts(self):
        got=org.event_summary(events(1,"/help"))
        self.assertEqual(got["human_messages"],1)
        self.assertEqual(got["command_like_human_messages"],1)

    def test_split_groups_follow_transitive_user_task_links(self):
        f=pd.DataFrame([dict(session_id="1",user_id="u1",task_id="t1"),
                        dict(session_id="2",user_id="u2",task_id="t1"),
                        dict(session_id="3",user_id="u2",task_id="t2"),
                        dict(session_id="4",user_id="u3",task_id="t3")])
        groups=org.related_groups(f)
        self.assertEqual(groups.iloc[:3].nunique(),1)
        self.assertNotEqual(groups.iloc[0],groups.iloc[3])


class SamplingTests(unittest.TestCase):
    def test_post_validation_rejects_changed_length_targets(self):
        frame=candidates();selected=org.select_candidates(frame,config())
        cfg=config();cfg['length_quotas']={"2-4":0,"5-9":2,"10-19":0,"20+":0}
        with self.assertRaisesRegex(ValueError,"length quotas"):
            org.validate_selection(selected,frame,cfg)

    def test_post_validation_rejects_changed_concentration_caps(self):
        frame=candidates();frame['user_id']='one-user';selected=org.select_candidates(frame,config())
        cfg=config();cfg['concentration_limits']['max_per_user']=2
        with self.assertRaisesRegex(ValueError,"concentration cap"):
            org.validate_selection(selected,frame,cfg)

    def test_post_validation_rejects_ineligible_or_modified_selection(self):
        frame=candidates();selected=org.select_candidates(frame,config())
        frame.loc[frame.session_id.eq(selected.iloc[0].session_id),'eligible_candidate']=False
        with self.assertRaisesRegex(ValueError,"excluded"):
            org.validate_selection(selected,frame,config())
        frame['eligible_candidate']=True
        selected.loc[0,'native_category']='Changed'
        with self.assertRaisesRegex(ValueError,"differs from the inventory"):
            org.validate_selection(selected,frame,config())

    def test_exact_quotas_and_occupation_balance(self):
        selected=org.select_candidates(candidates(),config())
        self.assertEqual(selected.turn_type.value_counts().to_dict(),{"multi":2,"single":2})
        self.assertEqual(selected.groupby("turn_type").occupation.apply(set).tolist(),[{"A","B"},{"A","B"}])
        self.assertTrue(selected.design_weight.isna().all())
        self.assertEqual(org.shortfall_report(selected,config()).shortfall.sum(),0)

    def test_reproducibility_survives_input_reordering(self):
        a=org.select_candidates(candidates(),config())
        b=org.select_candidates(candidates().iloc[::-1],config())
        self.assertEqual(a.session_id.tolist(),b.session_id.tolist())

    def test_exclusions_not_selected(self):
        frame=candidates();frame.loc[frame.session_id.eq("m-a"),"eligible_candidate"]=False
        self.assertNotIn("m-a",org.select_candidates(frame,config()).session_id.tolist())

    def test_infeasible_request_fails_without_silent_relaxation(self):
        cfg=config(4)
        with self.assertRaisesRegex(ValueError,"infeasible"):
            org.select_candidates(candidates(),cfg)

    def test_explicit_shortfall_reports_missing_counts(self):
        cfg=config(4);selected=org.select_candidates(candidates(),cfg,allow_shortfall=True)
        self.assertEqual(len(selected),6)
        report=org.shortfall_report(selected,cfg)
        self.assertTrue(report.shortfall.ge(0).all())
        self.assertEqual(report[report.kind.eq("category")].shortfall.sum(),2)

    def test_user_caps_are_enforced(self):
        frame=candidates();frame["user_id"]="same-user"
        cfg=config();cfg["concentration_limits"]["max_per_user"]=2
        with self.assertRaisesRegex(ValueError,"infeasible"):
            org.select_candidates(frame,cfg)
        self.assertEqual(len(org.select_candidates(frame,cfg,allow_shortfall=True)),2)

    def test_unrequested_categories_fail_cleanly(self):
        cfg=config();cfg["category_quotas_per_turn_type"][0]["category"]="Absent"
        with self.assertRaisesRegex(ValueError,"No eligible"):
            org.select_candidates(candidates(),cfg)

    def test_wrong_length_capacity_fails(self):
        cfg=config();cfg["length_quotas"]={"2-4":0,"5-9":2,"10-19":0,"20+":0}
        with self.assertRaisesRegex(ValueError,"infeasible"):
            org.select_candidates(candidates(),cfg)

    def test_category_shares_use_separate_turn_denominators(self):
        frame=candidates()
        frame.loc[frame.session_id.eq("s-a2"),"native_category"]="Other"
        mix=org.category_mix(frame)
        self.assertAlmostEqual(mix.single_share_within_source_pct.sum(),100)
        self.assertAlmostEqual(mix.multi_share_within_source_pct.sum(),100)
        row=mix[mix.native_category.eq("Other")].iloc[0]
        self.assertAlmostEqual(row.single_share_within_source_pct,100/3)
        self.assertEqual(row.multi_share_within_source_pct,0)


class InventoryTests(unittest.TestCase):
    def test_structural_filtering_and_duplicate_canonical_are_safe(self):
        with tempfile.TemporaryDirectory(prefix="collabskill-inventory-") as directory:
            root=Path(directory); raw=root/"raw/collabskill"; meta=root/"metadata"
            (raw/"leaderboard_data").mkdir(parents=True)
            session_rows=[dict(session_id=f"s{i}",task_id=f"t{i}",user_id=f"u{i}",agent_id="agent") for i in range(1,6)]
            task_rows=[dict(task_id=f"t{i}",dataset="gdpval_benchmark") for i in range(1,6)]
            (raw/"leaderboard_data/sessions.json").write_text(json.dumps(session_rows))
            (raw/"leaderboard_data/tasks.json").write_text(json.dumps(task_rows))
            logs={"s1":events(1),"s2":events(1),"s3":[{"no_extracted_log_reason":"missing"}],
                  "s4":[events(1)[1]],"s5":[events(1)[0]]}
            for sid,rs in logs.items():
                path=raw/"interaction_logs"/sid/"extracted_log.jsonl";path.parent.mkdir(parents=True)
                path.write_text("\n".join(json.dumps(r) for r in rs)+"\n")
            (meta/"external_sources/gdpval").mkdir(parents=True)
            gdp=meta/"external_sources/gdpval/train-00000-of-00001.parquet"
            pd.DataFrame([dict(task_id=t['task_id'],sector="Sector",occupation="Occupation") for t in task_rows]).to_parquet(gdp,index=False)
            (gdp.parent/"provenance.json").write_text(json.dumps({"sha256":org.sha256(gdp)}))
            (meta/"apex_category_mapping").mkdir()
            pd.DataFrame(columns=["task_id","category","category_status","method"]).to_csv(meta/"apex_category_mapping/task_mapping.csv",index=False)
            checks=[dict(path=str(p.relative_to(raw)),size_bytes=p.stat().st_size,sha256=org.sha256(p)) for p in raw.rglob('*') if p.is_file()]
            (meta/"collabskill.checksums.json").write_text(json.dumps(checks))
            (root/"download_manifest.json").write_text(json.dumps({"datasets":{"collabskill":{"revision":"fixture"}}}))
            cfg=config();cfg["manual_exclusions"]={"s1":"fixture mismatch"}
            got=org.build_inventory(root,cfg,verify_hashes=True).set_index("session_id")
            self.assertFalse(got.loc['s1','eligible_candidate'])
            self.assertTrue(got.loc['s2','eligible_candidate'])
            self.assertTrue(pd.isna(got.loc['s3','human_messages']))
            self.assertEqual(got.loc['s4','exclusion_reason'],'no_conversational_human_message')
            self.assertEqual(got.loc['s5','exclusion_reason'],'no_observed_agent_behavior_after_human')
            cfg["manual_exclusions"]={}
            got=org.build_inventory(root,cfg).set_index("session_id")
            self.assertTrue(got.loc['s1','eligible_candidate'])
            self.assertEqual(got.loc['s2','duplicate_of_session_id'],'s1')


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="collabskill-synthetic-")
        self.root=Path(self.tmp.name)/"data";self.root.mkdir()
        self.selected=org.select_candidates(candidates(),config())
        for r in self.selected.itertuples():
            p=self.root/"raw/collabskill/interaction_logs"/r.session_id/"extracted_log.jsonl"
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text("\n".join(json.dumps(x) for x in events(int(r.human_messages),r.session_id))+"\n")
        for name in ["download_manifest.json","metadata/apex_category_mapping/task_mapping.csv",
                     "metadata/apex_category_mapping/provenance.json","metadata/external_sources/gdpval/provenance.json",
                     "metadata/collabskill.checksums.json"]:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("{}\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_roundtrip_retains_all_events_and_ids(self):
        out=Path(self.tmp.name)/"new-output"
        org.export_dataset(self.root,out,candidates(),self.selected,config())
        rows=[json.loads(s) for s in (out/"scenarios.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),4)
        for r in rows:
            original=org.source_events(self.root,r["session_id"])
            self.assertEqual([e["source_event"] for e in r["events"]],original)
            self.assertFalse(r["truncated"])
            self.assertEqual(r["source_event_end_exclusive"],len(original))

    def test_existing_destination_is_untouched(self):
        out=Path(self.tmp.name)/"existing";out.mkdir();marker=out/"keep.txt";marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            org.export_dataset(self.root,out,candidates(),self.selected,config())
        self.assertEqual(marker.read_text(),"keep")

    def test_destination_created_during_staging_is_untouched(self):
        out=Path(self.tmp.name)/"concurrent-output"
        original=org.source_events
        def during_read(root,sid):
            if not out.exists():
                out.mkdir();(out/'someone-elses-work.txt').write_text('keep')
            return original(root,sid)
        with patch.object(org,'source_events',side_effect=during_read):
            with self.assertRaises(FileExistsError):
                org.export_dataset(self.root,out,candidates(),self.selected,config())
        self.assertEqual((out/'someone-elses-work.txt').read_text(),'keep')

    def test_partial_export_requires_explicit_permission(self):
        cfg=config(4);selected=org.select_candidates(candidates(),cfg,allow_shortfall=True)
        # Add all fixture transcripts, since a partial optimum selects six rather than four.
        for r in selected.itertuples():
            path=self.root/'raw/collabskill/interaction_logs'/r.session_id/'extracted_log.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('\n'.join(json.dumps(x) for x in events(int(r.human_messages),r.session_id))+'\n')
        out=Path(self.tmp.name)/'partial-output'
        with self.assertRaisesRegex(ValueError,'quotas'):
            org.export_dataset(self.root,out,candidates(),selected,cfg)
        org.export_dataset(self.root,out,candidates(),selected,cfg,allow_shortfall=True)
        report=pd.read_csv(out/'quota_report.csv')
        self.assertGreater(report.shortfall.sum(),0)

    def test_raw_and_metadata_destinations_forbidden(self):
        for name in ["raw/new","metadata/new"]:
            with self.assertRaises(ValueError):
                org.export_dataset(self.root,self.root/name,candidates(),self.selected,config())

    def test_changed_source_fails_and_leaves_no_partial_dataset(self):
        first=self.selected.iloc[0].session_id
        p=self.root/"raw/collabskill/interaction_logs"/first/"extracted_log.jsonl"
        p.write_text(json.dumps(events(1,"changed")[0])+"\n")
        out=Path(self.tmp.name)/"new-output"
        with self.assertRaisesRegex(ValueError,"Source changed"):
            org.export_dataset(self.root,out,candidates(),self.selected,config())
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
