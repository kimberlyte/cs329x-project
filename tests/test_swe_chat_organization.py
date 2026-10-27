"""Synthetic-only tests. Never select or export the real SWE-chat dataset."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import organize_swe_chat as org


def config():
    cfg=deepcopy(org.load_config())
    cfg.update(source_revision="fixture",requested_per_turn_type=2)
    cfg["concentration_limits"]={"max_per_user":2,"max_per_repository":2}
    return cfg


def event(sid,pos,role="user",entry=None,response=None,**kwargs):
    row=dict(session_id=sid,turn_number=pos,turn_id=f"{sid}#{pos}",source_entry_uuid=entry,
             response_id=response,role=role,turn_type="user_prompt" if role=="user" else "assistant_response",
             is_conversational=role in ["user","assistant"],char_count=7,tool_name=None,model="model",
             prompt_intent="debug" if role=="user" else None,
             prompt_pushback="non_pushback" if role=="user" else None,language="ENGLISH",is_continuation=False)
    row.update(kwargs)
    return row


def summary(rows,batch=999):
    stats={}
    for i in range(0,len(rows),batch):
        org.consume_batch(pd.DataFrame(rows[i:i+batch]),stats)
    return org.summarize_stats(stats.get(rows[0]["session_id"]) if rows else None)


def candidates():
    rows=[]
    for i,turn in enumerate(["single","single","single","multi","multi","multi"]):
        sid=f"session-{i}"
        events=[event(sid,0),event(sid,1,"assistant")]
        if turn=="multi":
            events += [event(sid,2),event(sid,3,"assistant")]
        rows.append(dict(**summary(events),session_id=sid,user_id="u"+str(i),repo_id="r"+str(i),
            agent="agent",model="model",strategy="manual-commit",kind="",compact_transcript_delta_only=False,
            content_hash="hash"+str(i),eligible_candidate=True,initial_intent="debug",intent_provenance="source_llm_generated_initial_prompt_label",
            source_revision="fixture",turn_type=turn,length_bin="single" if turn=="single" else "2-4",
            coding_mode="Unknown",coding_mode_status="missing_or_invalid_percentage",attribution_basis="legacy_additions_or_unspecified",
            agent_percentage=None,attribution_metric_version=None,repo_type_domain=None,split_group_id="g"+str(i)))
    return pd.DataFrame(rows)


class CountingTests(unittest.TestCase):
    def test_native_fragments_collapse_without_losing_record_counts(self):
        rows=[event("s",0,entry="same"),event("s",1,entry="same"),event("s",2,"assistant",response="answer"),
              event("s",3,"assistant",response="answer")]
        got=summary(rows,batch=1)
        self.assertEqual((got["human_messages"],got["assistant_messages"],got["human_records"],got["assistant_records"]),(1,1,2,2))
        self.assertEqual(got["conversational_rounds"],1)

    def test_response_fallback_and_distinct_id_namespaces(self):
        self.assertNotEqual(org.message_key(event("s",0,entry="x")),org.message_key(event("s",1,response="x")))
        self.assertEqual(org.message_key(event("s",1,response="x")),"response:x")
        self.assertEqual(org.message_key(event("s",2)),"row:s#2")

    def test_tools_are_not_human_messages(self):
        got=summary([event("s",0),event("s",1,"tool_use",turn_type="tool_use",tool_name="Edit"),
                     event("s",2,"tool_result",turn_type="tool_result"),event("s",3,"assistant")])
        self.assertEqual((got["human_messages"],got["tool_calls"],got["tool_results"]),(1,1,1))
        self.assertTrue(got["behavior_after_last_human"])

    def test_messages_and_rounds_are_separate(self):
        got=summary([event("s",0),event("s",1),event("s",2,"assistant")])
        self.assertEqual((got["human_messages"],got["conversational_rounds"]),(2,1))

    def test_missing_log_is_unknown_not_zero(self):
        self.assertIsNone(summary([])["human_messages"])
        self.assertEqual(org.length_bin(None),"missing")
        self.assertEqual(org.length_bin(0),"zero")

    def test_initial_label_conflicts_remain_visible(self):
        got=summary([event("s",0,entry="same"),event("s",1,entry="same",prompt_intent="test"),event("s",2,"assistant")])
        self.assertEqual(got["initial_intent_source"],"Unannotated")
        self.assertTrue(got["initial_intent_conflict"])

    def test_later_pushback_uses_annotated_denominator_and_does_not_count_initial(self):
        got=summary([event("s",0,prompt_pushback="correction"),event("s",1,"assistant"),
                     event("s",2,prompt_pushback="rejection"),event("s",3,"assistant"),
                     event("s",4,prompt_pushback=None),event("s",5,"assistant")])
        self.assertEqual((got["later_human_messages"],got["later_labeled_prompts"],got["later_pushback_prompts"]),(2,1,1))
        self.assertEqual(got["pushback_types"],"rejection")
        self.assertEqual(got["later_pushback_fraction"],1)

    def test_single_turn_feedback_fraction_is_missing(self):
        self.assertIsNone(summary([event("s",0),event("s",1,"assistant")])["later_pushback_fraction"])

    def test_population_scan_rejects_content(self):
        with self.assertRaisesRegex(ValueError,"content"):
            org.consume_batch(pd.DataFrame([event("s",0,content="private text")]),{})

    def test_order_gap_and_continuation_flags(self):
        got=summary([event("s",0),event("s",3,"assistant"),event("s",2,is_continuation=True)],batch=1)
        self.assertTrue(got["source_index_gaps"])
        self.assertTrue(got["nonmonotonic_events"])
        self.assertTrue(got["has_continuation"])


class AttributionAndEligibilityTests(unittest.TestCase):
    def test_version_aware_valid_denominator(self):
        row=dict(agent_percentage=100,attribution_metric_version=2,total_committed=0,total_lines_changed=10)
        self.assertEqual(org.coding_mode(row)[0],"Vibe coding")
        row["attribution_metric_version"]=None
        self.assertEqual(org.coding_mode(row)[0],"Unknown")
        row.update(agent_percentage=0,total_committed=10)
        self.assertEqual(org.coding_mode(row)[0],"Human-only")
        row["agent_percentage"]=99
        self.assertEqual(org.coding_mode(row)[0],"Vibe coding")

    def test_invalid_attribution_is_unknown(self):
        for pct in [None,-1,101]:
            self.assertEqual(org.coding_mode(dict(agent_percentage=pct,total_committed=10))[0],"Unknown")

    def test_continuations_delta_incomplete_tail_excluded(self):
        f=candidates()
        f.loc[0,"has_continuation"]=True
        f.loc[1,"compact_transcript_delta_only"]=True
        f.loc[2,"behavior_after_last_human"]=False
        got=org.apply_eligibility(f,config())
        self.assertEqual(got.loc[:2,"eligible_candidate"].tolist(),[False,False,False])

    def test_native_duplicate_chooses_eligible_canonical(self):
        f=candidates();f.loc[:2,"content_hash"]="same"
        cfg=config();cfg["manual_exclusions"]={"session-0":"known mismatch"}
        got=org.apply_eligibility(f,cfg)
        self.assertTrue(got.loc[1,"eligible_candidate"])
        self.assertEqual(got.loc[2,"duplicate_of_session_id"],"session-1")

    def test_manual_intent_override_retains_source_and_rationale(self):
        cfg=config();cfg["initial_intent_overrides"]={"session-0":{"label":"test","rationale":"Reviewed initial task"}}
        got=org.apply_eligibility(candidates(),cfg)
        self.assertEqual(got.loc[0,"initial_intent_source"],"debug")
        self.assertEqual(got.loc[0,"initial_intent"],"test")
        self.assertEqual(got.loc[0,"intent_provenance"],"manual_reviewed_initial_prompt")

    def test_transitive_user_repository_groups_and_missing_ids(self):
        f=pd.DataFrame([dict(session_id="a",user_id="u1",repo_id="r1"),dict(session_id="b",user_id="u2",repo_id="r1"),
                        dict(session_id="c",user_id="u2",repo_id="r2"),dict(session_id="d",user_id=None,repo_id=None),
                        dict(session_id="e",user_id=None,repo_id=None)])
        got=org.related_groups(f)
        self.assertEqual(got.iloc[:3].nunique(),1)
        self.assertNotEqual(got.iloc[3],got.iloc[4])


class SamplingTests(unittest.TestCase):
    def test_feasibility_does_not_return_any_scenario_ids(self):
        got=org.design_feasibility(candidates(),config())
        self.assertEqual(got["status"],"feasible")
        self.assertNotIn("session-",json.dumps(got))
        frame=candidates();frame["repo_id"]="same"
        self.assertEqual(org.design_feasibility(frame,config())["status"],"infeasible")

    def test_largest_remainder_exact_total(self):
        q=org.apportion(pd.Series({"a":5159,"b":3943,"c":2636,"d":2442}),50)
        self.assertEqual(q.tolist(),[18,14,9,9])

    def test_matched_counts_length_and_reproducibility(self):
        f=candidates();cfg=config()
        a=org.select_candidates(f,cfg);b=org.select_candidates(f.iloc[::-1],cfg)
        self.assertEqual(a.session_id.tolist(),b.session_id.tolist())
        self.assertEqual(a.turn_type.value_counts().to_dict(),{"multi":2,"single":2})
        self.assertEqual(org.quota_report(a,org.proposed_targets(f,cfg)).shortfall.sum(),0)
        self.assertTrue(a.design_weight.isna().all())

    def test_secondary_labels_do_not_affect_selection(self):
        f=candidates();a=org.select_candidates(f,config())
        f["coding_mode"]="Vibe coding";f["later_pushback_prompts"]=99
        self.assertEqual(a.session_id.tolist(),org.select_candidates(f,config()).session_id.tolist())

    def test_infeasible_caps_fail_without_relaxing(self):
        f=candidates();f["repo_id"]="same"
        with self.assertRaisesRegex(ValueError,"infeasible"):
            org.select_candidates(f,config())
        got=org.select_candidates(f,config(),allow_shortfall=True)
        self.assertEqual(len(got),2)
        self.assertTrue(org.quota_report(got,org.proposed_targets(f,config())).shortfall.gt(0).any())

    def test_exclusions_are_never_selected(self):
        f=candidates();f.loc[0,"eligible_candidate"]=False
        self.assertNotIn("session-0",org.select_candidates(f,config()).session_id.tolist())

    def test_two_intent_counts_matched(self):
        f=candidates();f.loc[[0,3],"initial_intent"]="test"
        chosen=org.select_candidates(f,config())
        self.assertEqual(chosen.groupby(["initial_intent","turn_type"]).size().tolist(),[1,1,1,1])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="swe-synthetic-")
        self.root=Path(self.tmp.name)/"data";self.root.mkdir()
        self.cfg=config();self.inventory=candidates()
        self.selected=org.select_candidates(self.inventory,self.cfg)
        rows=[]
        for r in self.inventory.itertuples():
            for pos in range(int(r.event_count)):
                row=event(r.session_id,pos,"user" if pos%2==0 else "assistant")
                row["content"]=r.session_id+" exact wording "+str(pos)
                rows.append(row)
        raw=self.root/"raw/swe-chat";raw.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows),raw/"conversations.parquet",row_group_size=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_events_reconstruct_with_absolute_table_indices(self):
        events=org.read_selected_events(self.root,self.selected.session_id,batch_size=2)
        self.assertEqual(set(events),set(self.selected.session_id))
        allrows=pq.read_table(self.root/"raw/swe-chat/conversations.parquet").to_pylist()
        for rs in events.values():
            for r in rs:
                self.assertEqual(r["source_event"],allrows[r["source_table_row_index"]])

    def test_export_readback_preserves_source_content(self):
        out=Path(self.tmp.name)/"export"
        with patch.object(org,"verify_inputs",return_value={"files":{}}):
            org.export_dataset(self.root,out,self.inventory,self.selected,self.cfg)
        rows=[json.loads(x) for x in (out/"scenarios.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),4)
        for row in rows:
            self.assertFalse(row["truncated"])
            self.assertEqual(row["events"][0]["source_event"]["content"],row["session_id"]+" exact wording 0")

    def test_existing_output_never_overwritten(self):
        out=Path(self.tmp.name)/"keep";out.mkdir();(out/"marker").write_text("keep")
        with self.assertRaises(FileExistsError):
            org.export_dataset(self.root,out,self.inventory,self.selected,self.cfg)
        self.assertEqual((out/"marker").read_text(),"keep")

    def test_forbidden_raw_destination(self):
        with self.assertRaisesRegex(ValueError,"raw"):
            org.export_dataset(self.root,self.root/"raw/new",self.inventory,self.selected,self.cfg)

    def test_changed_count_fails_without_partial_output(self):
        out=Path(self.tmp.name)/"export"
        self.selected.loc[0,"event_count"]=99
        with patch.object(org,"verify_inputs",return_value={"files":{}}),self.assertRaisesRegex(ValueError,"boundary"):
            org.export_dataset(self.root,out,self.inventory,self.selected,self.cfg)
        self.assertFalse(out.exists())


class ReportTests(unittest.TestCase):
    def test_report_cache_readback_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="swe-report-") as tmp:
            f=candidates();f.attrs["verification"]={"files":{}}
            out=Path(tmp)/"review"
            org.write_report(out,f,config(),root=Path(tmp)/"data")
            with patch.object(org,"verify_inputs",return_value={"files":{}}):
                back=org.load_review(root=Path(tmp)/"data",cfg=config(),report_dir=out)
                self.assertEqual(back.session_id.tolist(),f.session_id.tolist())
                (out/"population_inventory.csv").write_text("modified")
                with self.assertRaisesRegex(ValueError,"stale"):
                    org.load_review(root=Path(tmp)/"data",cfg=config(),report_dir=out)

    def test_report_rejects_raw_and_existing_destinations(self):
        with tempfile.TemporaryDirectory(prefix="swe-report-") as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,"raw"):
                org.write_report(root/"raw/new",candidates(),config(),root=root)
            with self.assertRaises(FileExistsError):
                org.write_report(root,candidates(),config(),root=root)


if __name__=="__main__":
    unittest.main()
