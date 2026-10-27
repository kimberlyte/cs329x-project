"""SWE-chat inventory, proposed quotas and explicit-only selection/export.

Default: inspect source metadata and structural candidates, without sampling.
Conversation content is never loaded for population profiling. Metadata scans use
bounded Arrow batches. Only explicit export reads full events for chosen sessions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "swe_chat_sampling.json"
LENGTHS = ["single", "2-4", "5-9", "10-19", "20+", "zero", "missing"]
PUSHBACK = ["correction", "rejection", "failure_report"]
SOURCE_FILES = ["sessions.parquet", "repositories.parquet", "conversations.parquet"]
COUNTING_NOTE = ("One human message = one conversational user message grouped by source_entry_uuid, "
                 "then response_id, then turn_id. Tools/system entries are separate. Rounds count "
                 "consecutive human-message runs separated by conversational assistant messages.")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path=CONFIG):
    cfg = read_json(path)
    if cfg.get("counting_rule") != "logical_conversational_messages_native_id_then_row":
        raise ValueError("Unsupported counting rule.")
    if cfg.get("quota_method") != "largest_remainder_from_eligible_multi":
        raise ValueError("Unsupported quota method.")
    if cfg.get("length_bins") != LENGTHS[1:5]:
        raise ValueError("Expected the four discussed multi-turn bins.")
    if type(cfg.get("requested_per_turn_type")) is not int or cfg["requested_per_turn_type"] < 1:
        raise ValueError("requested_per_turn_type must be a positive integer.")
    if type(cfg.get("batch_size")) is not int or not 1 <= cfg["batch_size"] <= 131072:
        raise ValueError("batch_size must be between 1 and 131072.")
    if len(cfg["intent_order"]) != len(set(cfg["intent_order"])) or "Unannotated" not in cfg["intent_order"]:
        raise ValueError("Intent order must contain distinct labels and Unannotated.")
    for cap in cfg["concentration_limits"].values():
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("Concentration caps must be null or positive integers.")
    if any(v != "report_only" for v in cfg["secondary_attributes"].values()):
        raise ValueError("Secondary attributes are report-only in this design.")
    for value in cfg["initial_intent_overrides"].values():
        if value.get("label") not in cfg["intent_order"] or not value.get("rationale", "").strip():
            raise ValueError("Intent overrides require an allowed label and written rationale.")
    return cfg


def verify_inputs(root=ROOT, cfg=None, full_hashes=False):
    root, cfg = Path(root), cfg or load_config()
    manifest = read_json(root / "download_manifest.json")["datasets"]["swe-chat"]
    if manifest["revision"] != cfg["source_revision"]:
        raise ValueError("Source revision differs from the reviewed configuration.")
    checks = {r["path"]: r for r in read_json(root / "metadata/swe-chat.checksums.json")}
    pinned = {r["path"]: r for r in read_json(root / "metadata/swe-chat.files.json")}
    verified = {}
    for name in SOURCE_FILES:
        record, source = checks[name], pinned[name]
        path = root / "raw/swe-chat" / name
        if record["sha256"] != source["lfs"]["sha256"] or record["size_bytes"] != source["size_bytes"]:
            raise ValueError("Local checksum record differs from the pinned source inventory: " + name)
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError("Source size mismatch: " + name)
        if full_hashes and sha256(path) != record["sha256"]:
            raise ValueError("Source checksum mismatch: " + name)
        verified[name] = dict(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                              pinned_sha256=record["sha256"])
    return dict(source_revision=manifest["revision"], profile=manifest["profile"],
                source_status=manifest["status"], files=verified,
                verification="sha256" if full_hashes else "sizes and pinned checksum records",
                missing_release_tables="Commits, checkpoints, context events, subagent tables and raw transcripts are not local.")


def message_key(row):
    for name, prefix in [("source_entry_uuid", "entry:"), ("response_id", "response:")]:
        value = row.get(name)
        if value is not None and not pd.isna(value) and str(value):
            return prefix + str(value)
    return "row:" + str(row["turn_id"])


def length_bin(n):
    if pd.isna(n):
        return "missing"
    if n == 0:
        return "zero"
    if n == 1:
        return "single"
    return "2-4" if n <= 4 else "5-9" if n <= 9 else "10-19" if n <= 19 else "20+"


def fresh_stats():
    return dict(event_count=0, tool_calls=0, tool_results=0, tool_events=0,
                content_characters=0, missing_char_counts=0, blank_human_records=0,
                first_event=None, last_event=None, last_behavior=-1, nonmonotonic=False,
                continuation=False, messages={}, event_models=set(), tool_names=set())


def consume_batch(frame, stats):
    """Aggregate one bounded metadata batch. No source content is accepted/read."""
    if "content" in frame:
        raise ValueError("Population scan must not materialize conversation content.")
    for sid, g in frame.groupby("session_id", sort=False):
        a = stats.setdefault(sid, fresh_stats())
        positions = g.turn_number.to_numpy()
        lo, hi = int(positions.min()), int(positions.max())
        a["nonmonotonic"] |= bool(np.any(np.diff(positions) <= 0) or
                                    (a["last_event"] is not None and lo <= a["last_event"]))
        a["first_event"] = lo if a["first_event"] is None else min(lo, a["first_event"])
        a["last_event"] = hi if a["last_event"] is None else max(hi, a["last_event"])
        a["event_count"] += len(g)
        call = g.turn_type.eq("tool_use") | g.role.eq("tool_use")
        result = g.turn_type.eq("tool_result") | g.role.eq("tool_result")
        tool = call | result
        a["tool_calls"] += int(call.sum())
        a["tool_results"] += int(result.sum())
        a["tool_events"] += int(tool.sum())
        a["tool_names"].update(str(v) for v in g.tool_name.dropna().unique() if str(v))
        a["event_models"].update(str(v) for v in g.model.dropna().unique() if str(v))
        a["content_characters"] += int(g.char_count.fillna(0).sum())
        a["missing_char_counts"] += int(g.char_count.isna().sum())
        conv = g[g.is_conversational.fillna(False) & g.role.isin(["user", "assistant"])]
        behavior = g[tool | (g.role.eq("assistant") & g.is_conversational.fillna(False))]
        if len(behavior):
            a["last_behavior"] = max(a["last_behavior"], int(behavior.turn_number.max()))
        for row in conv.to_dict("records"):
            role, key, pos = row["role"], message_key(row), int(row["turn_number"])
            unit = a["messages"].setdefault((role, key), dict(role=role, start=pos, end=pos,
                records=0, intents=set(), pushbacks=set(), languages=set(), continuation=False))
            unit["start"], unit["end"] = min(unit["start"], pos), max(unit["end"], pos)
            unit["records"] += 1
            for field, target in [("prompt_intent", "intents"), ("prompt_pushback", "pushbacks"), ("language", "languages")]:
                value = row.get(field)
                if value is not None and not pd.isna(value) and str(value):
                    unit[target].add(str(value))
            continuation = row.get("is_continuation")
            unit["continuation"] |= continuation is True or isinstance(continuation, np.bool_) and bool(continuation)
            if role == "user":
                a["continuation"] |= unit["continuation"]
                if row.get("char_count") == 0:
                    a["blank_human_records"] += 1


def summarize_stats(a):
    if not a:
        return dict(has_log=False, human_messages=None, assistant_messages=None, conversational_rounds=None,
                    human_records=None, assistant_records=None, event_count=0, tool_events=None,
                    tool_calls=None, tool_results=None, content_characters=None, source_event_start=None,
                    source_event_end_exclusive=None, initial_intent_source="Unannotated", initial_intent_conflict=False,
                    initial_pushback_source="Unannotated", later_human_messages=None, later_labeled_prompts=None,
                    later_pushback_prompts=None, later_pushback_fraction=None, pushback_types="", pushback_conflicts=0,
                    begins_with_human=False, behavior_after_last_human=False, has_continuation=False,
                    nonmonotonic_events=False, source_index_gaps=False, language=None, tool_names="", event_models="",
                    missing_char_counts=None, blank_human_records=None)
    units = sorted(a["messages"].values(), key=lambda r: r["start"])
    humans = [u for u in units if u["role"] == "user"]
    assistants = [u for u in units if u["role"] == "assistant"]
    rounds, prev = 0, None
    for u in units:
        if u["role"] == "user" and prev != "user":
            rounds += 1
        prev = u["role"]
    first = humans[0] if humans else None
    later = humans[1:]
    # Later-message labels are diagnostics. Consecutive human messages need not
    # be feedback on an agent action, so no semantic feedback claim is inferred.
    later_labeled = [u for u in later if len(u["pushbacks"]) == 1]
    later_push = [u for u in later_labeled if next(iter(u["pushbacks"])) in PUSHBACK]
    types = sorted({next(iter(u["pushbacks"])) for u in later_push})
    intent = next(iter(first["intents"])) if first and len(first["intents"]) == 1 else "Unannotated"
    return dict(has_log=True, human_messages=len(humans), assistant_messages=len(assistants), conversational_rounds=rounds,
                human_records=sum(u["records"] for u in humans), assistant_records=sum(u["records"] for u in assistants),
                event_count=a["event_count"], tool_events=a["tool_events"], tool_calls=a["tool_calls"], tool_results=a["tool_results"],
                content_characters=a["content_characters"], missing_char_counts=a["missing_char_counts"],
                blank_human_records=a["blank_human_records"], source_event_start=a["first_event"],
                source_event_end_exclusive=a["last_event"]+1,
                initial_intent_source=intent, initial_intent_conflict=bool(first and len(first["intents"]) > 1),
                initial_pushback_source=next(iter(first["pushbacks"])) if first and len(first["pushbacks"]) == 1 else "Unannotated",
                later_human_messages=len(later), later_labeled_prompts=len(later_labeled),
                later_pushback_prompts=len(later_push), later_pushback_fraction=len(later_push)/len(later_labeled) if later_labeled else None,
                pushback_types="|".join(types), pushback_conflicts=sum(len(u["pushbacks"]) > 1 for u in humans),
                begins_with_human=bool(units and units[0]["role"] == "user"),
                behavior_after_last_human=bool(humans and a["last_behavior"] > humans[-1]["end"]),
                has_continuation=a["continuation"], nonmonotonic_events=a["nonmonotonic"],
                source_index_gaps=a["event_count"] != a["last_event"]-a["first_event"]+1,
                language="|".join(sorted({v for u in humans for v in u["languages"]})) or None,
                tool_names="|".join(sorted(a["tool_names"])), event_models="|".join(sorted(a["event_models"])))


def coding_mode(row):
    """Version-aware, denominator-valid observational classification."""
    version, pct = row.get("attribution_metric_version"), row.get("agent_percentage")
    denominator = row.get("total_lines_changed") if pd.notna(version) and version == 2 else row.get("total_committed")
    if pd.isna(pct) or not 0 <= pct <= 100:
        return "Unknown", "missing_or_invalid_percentage"
    if pd.isna(denominator) or denominator <= 0:
        return "Unknown", "missing_or_nonpositive_denominator"
    return ("Human-only" if pct == 0 else "Collaborative" if pct < 99 else "Vibe coding"), "valid_recorded_attribution"


def related_groups(frame):
    """Transitive source user/repository components. Missing IDs never link rows."""
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        if parent[k] != k:
            parent[k] = find(parent[k])
        return parent[k]

    for r in frame.sort_values("session_id").itertuples():
        keys = ["session:"+r.session_id]
        for prefix, value in [("user:", r.user_id), ("repo:", r.repo_id)]:
            if pd.notna(value) and str(value).strip():
                keys.append(prefix+str(value))
        for k in keys[1:]:
            a, b = find(keys[0]), find(k)
            if a != b:
                parent[max(a,b)] = min(a,b)
    groups = defaultdict(list)
    for sid in sorted(frame.session_id):
        groups[find("session:"+sid)].append(sid)
    ids = {k:"swe_group_"+hashlib.sha256("|".join(v).encode()).hexdigest()[:16] for k,v in groups.items()}
    return pd.Series([ids[find("session:"+sid)] for sid in frame.session_id], index=frame.index)


def apply_eligibility(frame, cfg):
    out = frame.copy()
    out["initial_intent"] = out.initial_intent_source
    out["intent_provenance"] = np.where(out.initial_intent.eq("Unannotated"), "missing_or_conflicting_source_llm_label", "source_llm_generated_initial_prompt_label")
    out["intent_override_rationale"] = ""
    for sid, override in cfg["initial_intent_overrides"].items():
        if sid not in set(out.session_id):
            raise ValueError("Intent override references an absent session: "+sid)
        mask = out.session_id.eq(sid)
        out.loc[mask, ["initial_intent", "intent_provenance", "intent_override_rationale"]] = [override["label"], "manual_reviewed_initial_prompt", override["rationale"]]
    unexpected = set(out.initial_intent)-set(cfg["intent_order"])
    if unexpected:
        raise ValueError("Unreviewed source intent labels: "+repr(unexpected))
    reasons = []
    for r in out.itertuples():
        why=[]
        if not r.has_log:
            why.append("missing_log")
        else:
            if r.human_messages == 0:
                why.append("no_conversational_human_message")
            if cfg["eligibility"]["require_initial_human"] and not r.begins_with_human:
                why.append("initial_conversational_context_missing")
            if cfg["eligibility"]["require_behavior_after_last_human"] and not r.behavior_after_last_human:
                why.append("no_behavior_after_last_human")
            if cfg["eligibility"]["exclude_continuations"] and r.has_continuation:
                why.append("continuation_context_not_verified")
            if r.nonmonotonic_events or r.source_index_gaps:
                why.append("source_event_order_or_index_gap")
        if cfg["eligibility"]["exclude_delta_only"] and r.compact_transcript_delta_only:
            why.append("delta_only_transcript")
        if r.session_id in cfg["manual_exclusions"]:
            why.append("manual_exclusion")
        reasons.append("|".join(why))
    out["exclusion_reason"] = reasons
    out["duplicate_of_session_id"] = ""
    seen={}
    for idx,r in out.sort_values("session_id").iterrows():
        fingerprint=r.get("content_hash")
        if r.exclusion_reason or pd.isna(fingerprint) or not str(fingerprint).strip():
            continue
        if fingerprint in seen:
            out.at[idx,"exclusion_reason"]="native_content_hash_duplicate"
            out.at[idx,"duplicate_of_session_id"]=seen[fingerprint]
        else:
            seen[fingerprint]=r.session_id
    out["eligible_candidate"]=out.exclusion_reason.eq("")
    out["requires_human_context_review"]=True
    out["context_availability"]="Complete released table session retained. External workspace, artifacts, compaction and mixed-task context need review."
    return out


def build_inventory(root=ROOT, cfg=None, verify_hashes=False, progress=None):
    root, cfg = Path(root), cfg or load_config()
    verification=verify_inputs(root,cfg,verify_hashes)
    wanted=["session_id","repo_id","user_id","agent","model","strategy","kind","created_at","content_hash",
            "agent_percentage","agent_lines","human_added","human_modified","human_removed","total_committed",
            "total_lines_changed","attribution_metric_version","compact_transcript_delta_only","prompt_count","turn_count",
            "duration_seconds","files_touched_count","input_tokens","output_tokens"]
    source=pq.ParquetFile(root/"raw/swe-chat/sessions.parquet")
    absent=set(wanted)-set(source.schema_arrow.names)
    if absent:
        raise ValueError("Missing required session columns: "+repr(absent))
    sessions=source.read(columns=wanted).to_pandas()
    repos=pd.read_parquet(root/"raw/swe-chat/repositories.parquet",columns=["repo_id","license_type","repo_type_domain","repo_type_audience"])
    if sessions.session_id.duplicated().any() or repos.repo_id.duplicated().any():
        raise ValueError("Duplicate source session/repository IDs.")
    sessions=sessions.merge(repos,on="repo_id",how="left",validate="many_to_one")
    path=root/"raw/swe-chat/conversations.parquet"
    columns=["session_id","turn_number","turn_id","source_entry_uuid","response_id","role","turn_type","is_conversational",
             "char_count","tool_name","model","prompt_intent","prompt_pushback","language","is_continuation"]
    stats={}
    file=pq.ParquetFile(path)
    for i,batch in enumerate(file.iter_batches(columns=columns,batch_size=cfg["batch_size"],use_threads=False)):
        consume_batch(batch.to_pandas(),stats)
        if progress and i%40 == 0:
            progress(f"Scanned {min((i+1)*cfg['batch_size'],file.metadata.num_rows):,}/{file.metadata.num_rows:,} event metadata rows")
    if set(stats)-set(sessions.session_id):
        raise ValueError("Conversation session IDs absent from sessions table.")
    summary=pd.DataFrame([summarize_stats(stats.get(sid)) for sid in sessions.session_id])
    inventory=pd.concat([sessions.reset_index(drop=True),summary],axis=1)
    for c in ["human_messages","assistant_messages","conversational_rounds","human_records","assistant_records",
              "tool_events","tool_calls","tool_results","source_event_start","source_event_end_exclusive"]:
        inventory[c]=inventory[c].astype("Int64")
    inventory["length_bin"]=inventory.human_messages.map(length_bin)
    inventory["turn_type"]=inventory.length_bin.map(lambda b:"single" if b=="single" else "multi" if b in LENGTHS[1:5] else b)
    modes=[coding_mode(r) for r in inventory.to_dict("records")]
    inventory["coding_mode"]=[v[0] for v in modes]
    inventory["coding_mode_status"]=[v[1] for v in modes]
    inventory["attribution_basis"]=np.where(inventory.attribution_metric_version.eq(2),"changed_lines_v2","legacy_additions_or_unspecified")
    inventory["split_group_id"]=related_groups(inventory)
    inventory["source_revision"]=verification["source_revision"]
    inventory["dataset"]="swe-chat"
    inventory=apply_eligibility(inventory,cfg).sort_values("session_id").reset_index(drop=True)
    inventory.attrs["verification"]=verification
    return inventory


def apportion(counts, n):
    counts=pd.Series(counts,dtype=float)
    if counts.sum() <= 0:
        return pd.Series(0,index=counts.index,dtype=int)
    raw=counts/counts.sum()*n
    q=np.floor(raw).astype(int)
    for key in (raw-q).sort_values(ascending=False,kind="stable").index[:n-int(q.sum())]:
        q[key]+=1
    return q


def proposed_targets(inventory,cfg):
    eligible=inventory[inventory.eligible_candidate]
    multi=eligible[eligible.turn_type.eq("multi")]
    n=cfg["requested_per_turn_type"]
    intents=multi.initial_intent.value_counts().reindex(cfg["intent_order"],fill_value=0)
    lengths=multi.length_bin.value_counts().reindex(cfg["length_bins"],fill_value=0)
    return {"intent":apportion(intents,n).to_dict(),"length":apportion(lengths,n).to_dict()}


def capacity_report(inventory,cfg):
    targets=proposed_targets(inventory,cfg)
    f=inventory[inventory.eligible_candidate]
    rows=[]
    for key,n in targets["intent"].items():
        g=f[f.initial_intent.eq(key)]
        ns,nm=int(g.turn_type.eq("single").sum()),int(g.turn_type.eq("multi").sum())
        rows.append(dict(initial_intent=key,requested_each=n,single_candidates=ns,multi_candidates=nm,
                         single_shortfall=max(0,n-ns),multi_shortfall=max(0,n-nm),
                         matching_status="missing_labels_are_not_task_matches" if key=="Unannotated" else "source_llm_intent_counts_only"))
    return pd.DataFrame(rows)


def attribute_distribution(frame,column):
    """Counts, conditional percentages and explicit missingness per turn group."""
    f=frame.copy()
    f[column]=f[column].fillna("Missing").replace("","Missing")
    out=f.groupby([column,"turn_type"],dropna=False).size().rename("sessions").reset_index()
    totals=out.groupby("turn_type").sessions.transform("sum")
    out["percent_within_turn_type"]=100*out.sessions/totals
    return out


def length_distribution(frame):
    counts=frame.length_bin.value_counts().reindex(LENGTHS,fill_value=0)
    n,multi=len(frame),int(frame.turn_type.eq("multi").sum())
    return pd.DataFrame([dict(length_bin=b,sessions=int(v),percent_all_sessions=100*v/n if n else np.nan,
             percent_multi_turn=100*v/multi if multi and b in LENGTHS[1:5] else np.nan) for b,v in counts.items()])


def pushback_distribution(frame):
    rows=[]
    for (turn,b),g in frame.groupby(["turn_type","length_bin"],dropna=False):
        later=int(g.later_human_messages.fillna(0).sum())
        labeled=int(g.later_labeled_prompts.fillna(0).sum())
        push=int(g.later_pushback_prompts.fillna(0).sum())
        row=dict(turn_type=turn,length_bin=b,sessions=len(g),sessions_with_later_source_labeled_pushback=int(g.later_pushback_prompts.fillna(0).gt(0).sum()),
                 later_human_messages=later,later_labeled_prompts=labeled,later_unannotated_or_conflicting=later-labeled,
                 later_pushback_prompts=push,pushback_percent_labeled_later_prompts=100*push/labeled if labeled else np.nan,
                 assessability="no_later_in_session_feedback_opportunity" if turn=="single" else "source_llm_label_diagnostic")
        for kind in PUSHBACK:
            row["sessions_with_"+kind]=int(g.pushback_types.fillna("").map(lambda x:kind in x.split("|")).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def concentration_report(frame):
    rows=[]
    for col in ["user_id","repo_id","agent","split_group_id"]:
        values=frame[col].fillna("")
        counts=values[values.astype(str).str.strip().ne("")].value_counts()
        rows.append(dict(dimension=col,distinct_known=len(counts),missing=int(values.astype(str).str.strip().eq("").sum()),
                         maximum_sessions=int(counts.max()) if len(counts) else 0,
                         maximum_share_percent=100*counts.max()/len(frame) if len(counts) and len(frame) else np.nan))
    return pd.DataFrame(rows)


def missingness_report(frame):
    cols=["initial_intent","model","agent_percentage","attribution_metric_version","language","user_id","repo_id"]
    rows=[]
    for c in cols:
        v=frame[c]
        missing=v.isna()|v.astype(str).str.strip().eq("")
        if c=="initial_intent":
            missing|=v.eq("Unannotated")
        rows.append(dict(attribute=c,missing=int(missing.sum()),sessions=len(frame),percent_missing=100*missing.mean() if len(frame) else np.nan))
    return pd.DataFrame(rows)


def quota_report(selected,targets):
    rows=[]
    for key,n in targets["intent"].items():
        for turn in ["single","multi"]:
            got=int((selected.initial_intent.eq(key)&selected.turn_type.eq(turn)).sum())
            rows.append(dict(dimension="intent",stratum=key,turn_type=turn,requested=n,achieved=got,shortfall=n-got))
    for key,n in targets["length"].items():
        got=int((selected.length_bin.eq(key)&selected.turn_type.eq("multi")).sum())
        rows.append(dict(dimension="length",stratum=key,turn_type="multi",requested=n,achieved=got,shortfall=n-got))
    return pd.DataFrame(rows)


def design_feasibility(inventory,cfg):
    """Check aggregate capacities/caps without choosing or returning scenario IDs."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import lil_matrix
    targets=proposed_targets(inventory,cfg)
    n=cfg["requested_per_turn_type"]
    if sum(targets["intent"].values())!=n or sum(targets["length"].values())!=n:
        return dict(status="infeasible",reason="No eligible multi-turn population for the requested design.")
    f=inventory[inventory.eligible_candidate & inventory.turn_type.isin(["single","multi"])].copy()
    keys=["initial_intent","turn_type","length_bin","user_id","repo_id"]
    f[keys]=f[keys].fillna("")
    groups=f.groupby(keys,dropna=False).size().rename("capacity").reset_index()
    if groups.empty:
        return dict(status="infeasible",reason="No eligible aggregate cells.")
    constraints=[];lower=[];upper=[]

    def add(mask,lo,hi):
        constraints.append(np.flatnonzero(mask));lower.append(lo);upper.append(hi)

    for label,count in targets["intent"].items():
        for turn in ["single","multi"]:
            add(groups.initial_intent.eq(label)&groups.turn_type.eq(turn),count,count)
    for b,count in targets["length"].items():
        add(groups.length_bin.eq(b)&groups.turn_type.eq("multi"),count,count)
    for capkey,col in [("max_per_user","user_id"),("max_per_repository","repo_id")]:
        cap=cfg["concentration_limits"].get(capkey)
        if cap is not None:
            for value in sorted(set(groups[col])):
                if str(value).strip():
                    add(groups[col].eq(value),0,cap)
    matrix=lil_matrix((len(constraints),len(groups)))
    for i,idx in enumerate(constraints):
        matrix[i,idx]=1
    result=milp(np.zeros(len(groups)),integrality=np.ones(len(groups)),bounds=Bounds(0,groups.capacity.to_numpy()),
                constraints=LinearConstraint(matrix.tocsr(),lower,upper),options={"time_limit":cfg["solver_time_limit_seconds"]})
    return dict(status="feasible" if result.success else "infeasible" if result.status==2 else "not_proven",
                reason=result.message,aggregate_cells=len(groups),
                scope="Count-level intent, length and known-user/repository caps only. No scenario IDs selected.")


def select_candidates(inventory,cfg,allow_shortfall=False):
    """Seeded constrained selection; unknown inclusion probabilities. No writes."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import lil_matrix,vstack,csr_matrix
    targets=proposed_targets(inventory,cfg)
    f=inventory[inventory.eligible_candidate & inventory.turn_type.isin(["single","multi"])].copy()
    f=f[f.initial_intent.map(targets["intent"]).fillna(0).gt(0)].sort_values("session_id").reset_index(drop=True)
    if f.empty:
        raise ValueError("No eligible candidates in the requested design.")
    constraints=[];lower=[];upper=[]

    def add(indices,values,lo,hi):
        constraints.append((list(indices),np.asarray(values,dtype=float)));lower.append(lo);upper.append(hi)

    for label,n in targets["intent"].items():
        mask=f.initial_intent.eq(label)
        indices=np.flatnonzero(mask)
        add(indices,np.where(f.loc[indices,"turn_type"].eq("single"),1,-1),0,0)
        single=np.flatnonzero(mask & f.turn_type.eq("single"))
        add(single,np.ones(len(single)),0,n)
    for b,n in targets["length"].items():
        idx=np.flatnonzero(f.length_bin.eq(b)&f.turn_type.eq("multi"))
        add(idx,np.ones(len(idx)),0,n)
    for capkey,col in [("max_per_user","user_id"),("max_per_repository","repo_id")]:
        cap=cfg["concentration_limits"].get(capkey)
        if cap is not None:
            known=f[col].notna()&f[col].astype(str).str.strip().ne("")
            for _,idx in f[known].groupby(col).groups.items():
                add(idx,np.ones(len(idx)),0,cap)
    matrix=lil_matrix((len(constraints),len(f)))
    for i,(idx,v) in enumerate(constraints):
        matrix[i,idx]=v
    matrix=matrix.tocsr()
    options={"time_limit":cfg["solver_time_limit_seconds"]}
    first=milp(-np.ones(len(f)),integrality=np.ones(len(f)),bounds=Bounds(0,1),
               constraints=LinearConstraint(matrix,lower,upper),options=options)
    if not first.success:
        raise RuntimeError("Solver did not prove an optimum: "+first.message)
    achieved=int(np.rint(first.x).sum());requested=2*cfg["requested_per_turn_type"]
    if achieved<requested and not allow_shortfall:
        raise ValueError(f"Requested design infeasible: maximum {achieved//2} matched intent pairs of {requested//2}. No quotas relaxed and no output written.")
    if achieved==0:
        raise ValueError("No matched pairs available under the configured constraints.")
    matrix=vstack([matrix,csr_matrix(np.ones((1,len(f))))],format="csr")
    result=milp(np.random.default_rng(cfg["seed"]).random(len(f)),integrality=np.ones(len(f)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,[*lower,achieved],[*upper,achieved]),options=options)
    if not result.success:
        raise RuntimeError("Seeded selection did not prove an optimum: "+result.message)
    chosen=f.loc[np.rint(result.x).astype(bool)].copy()
    chosen["scenario_id"]=["swe_chat_"+hashlib.sha256((r.source_revision+":"+r.session_id).encode()).hexdigest()[:20] for r in chosen.itertuples()]
    chosen["selection_probability"]=np.nan
    chosen["design_weight"]=np.nan
    chosen["selection_method"]="seeded constrained optimization; inclusion probabilities unknown"
    chosen["inclusion_rationale"]="Initial-intent counts matched; multi-turn length target; user/repository caps. Secondary attributes report-only."
    chosen["split"]="unassigned"
    report=quota_report(chosen,targets)
    if (report.shortfall<0).any() or (achieved==requested and report.shortfall.ne(0).any()):
        raise AssertionError("Quota verification failed.")
    if chosen.groupby("initial_intent").turn_type.value_counts().unstack(fill_value=0).diff(axis=1).iloc[:,-1].ne(0).any():
        raise AssertionError("Intent counts are not matched.")
    return chosen.sort_values(["initial_intent","turn_type","scenario_id"]).reset_index(drop=True)


def plot_distributions(inventory):
    import matplotlib.pyplot as plt
    f=inventory[inventory.turn_type.isin(["single","multi"])].copy()
    table=pd.crosstab(f.initial_intent,f.turn_type).reindex(columns=["single","multi"],fill_value=0)
    pct=table.div(table.sum(axis=0),axis=1)*100
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    y=np.arange(len(pct))
    for shift,col in [(-.18,"single"),(.18,"multi")]:
        axes[0].barh(y+shift,pct[col],height=.34,label=col)
    axes[0].set_yticks(y,pct.index);axes[0].invert_yaxis()
    axes[0].set(xlabel="% within each turn group",title="Initial prompt intent (source LLM labels)")
    axes[0].legend()
    counts=f.loc[f.turn_type.eq("multi"),"length_bin"].value_counts().reindex(LENGTHS[1:5],fill_value=0)
    values=100*counts/counts.sum() if counts.sum() else counts
    bars=axes[1].bar(values.index,values,color="#326b91")
    for bar,n,v in zip(bars,counts,values):
        axes[1].text(bar.get_x()+bar.get_width()/2,v+.5,f"{n:,}\n{v:.1f}%",ha="center",fontsize=9)
    axes[1].set(xlabel="Logical human messages",ylabel="% of multi-turn sessions",title="Multi-turn length distribution",ylim=(0,max(values.max()*1.25,1)))
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    return fig


def report_tables(inventory):
    result={}
    for name,f in [("source",inventory),("eligible",inventory[inventory.eligible_candidate])]:
        result[name+"_length"]=length_distribution(f)
        for col in ["initial_intent","coding_mode","attribution_basis","coding_mode_status","agent","strategy","language","kind","repo_type_domain"]:
            result[name+"_"+col]=attribute_distribution(f,col)
        result[name+"_pushback"]=pushback_distribution(f)
        result[name+"_concentration"]=concentration_report(f)
        result[name+"_missingness"]=missingness_report(f)
        # Joint counts retain zero/missing groups and distinct source provenance.
        result[name+"_intent_length"]=pd.crosstab(f.initial_intent,f.length_bin).reindex(columns=LENGTHS,fill_value=0).reset_index()
        result[name+"_coding_length"]=pd.crosstab(f.coding_mode,f.length_bin).reindex(columns=LENGTHS,fill_value=0).reset_index()
    return result


def write_report(out,inventory,cfg,root=ROOT):
    """Metadata-only report, never a scenario selection."""
    out=Path(out).resolve()
    root=Path(root).resolve()
    if any(out==root/name or root/name in out.parents for name in ["raw","versions"]):
        raise ValueError("Report must not be inside raw/ or versions/.")
    if out.exists():
        raise FileExistsError("Report destination exists. Nothing overwritten.")
    out.mkdir(parents=True)
    inventory.to_csv(out/"population_inventory.csv",index=False)
    for name,table in report_tables(inventory).items():
        table.to_csv(out/(name+".csv"),index=False)
    capacity_report(inventory,cfg).to_csv(out/"proposed_intent_targets.csv",index=False)
    (out/"proposed_targets.json").write_text(json.dumps(proposed_targets(inventory,cfg),indent=2)+"\n")
    (out/"review_provenance.json").write_text(json.dumps(dict(config=cfg,verification=inventory.attrs.get("verification"),
        code_sha256=sha256(Path(__file__)),inventory_sha256=sha256(out/"population_inventory.csv"),
        status="Metadata review only. No scenarios selected or exported."),indent=2)+"\n")
    fig=plot_distributions(inventory[inventory.eligible_candidate])
    fig.suptitle("SWE-chat: structural candidates before human context review")
    fig.savefig(out/"intent_and_length.png",dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def load_review(root=ROOT,cfg=None,report_dir=None):
    """Use a validated optional metadata cache, or rebuild without writing."""
    root,cfg=Path(root),cfg or load_config()
    if report_dir is None:
        return build_inventory(root,cfg)
    out=Path(report_dir)
    saved=read_json(out/"review_provenance.json")
    current=verify_inputs(root,cfg)
    if (saved["config"]!=cfg or saved["code_sha256"]!=sha256(Path(__file__)) or
        saved["inventory_sha256"]!=sha256(out/"population_inventory.csv") or saved["verification"]["files"]!=current["files"]):
        raise ValueError("Review cache is stale. Rebuild to a new report directory.")
    result=pd.read_csv(out/"population_inventory.csv",keep_default_na=False,na_values=[""])
    result.attrs["verification"]=saved["verification"]
    return result


def read_selected_events(root,ids,batch_size=32768):
    """Single bounded pass over full content for explicit export only."""
    ids=set(ids);events={sid:[] for sid in ids}
    source=pq.ParquetFile(Path(root)/"raw/swe-chat/conversations.parquet")
    offset=0
    for batch in source.iter_batches(batch_size=batch_size,use_threads=False):
        mask=pc.is_in(batch.column(batch.schema.get_field_index("session_id")),value_set=pa.array(sorted(ids)))
        indices=np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        subset=batch.filter(mask)
        for pos,row in zip(indices,subset.to_pylist()):
            events[row["session_id"]].append(dict(source_table_row_index=offset+int(pos),
                source_event_index=row["turn_number"],source_event=row))
        offset+=len(batch)
    return events


def json_safe(value):
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA or isinstance(value,float) and np.isnan(value):
        return None
    if isinstance(value,(pd.Timestamp,)) or hasattr(value,"isoformat"):
        return value.isoformat()
    if isinstance(value,np.generic):
        return value.item()
    return value


def export_dataset(root,out,inventory,selected,cfg):
    """New local directory only, pinned inputs verified before materialization."""
    root,out=Path(root).resolve(),Path(out).resolve()
    if any(out==root/name or root/name in out.parents for name in ["raw","metadata","versions"]):
        raise ValueError("Export must not be inside raw/, metadata/ or versions/.")
    if out.exists():
        raise FileExistsError("Output exists. Nothing overwritten.")
    if selected.empty or selected.scenario_id.duplicated().any() or selected.session_id.duplicated().any():
        raise ValueError("Empty selection or duplicate IDs.")
    if not selected.eligible_candidate.all():
        raise ValueError("Selection contains excluded sessions.")
    verification=verify_inputs(root,cfg,full_hashes=True)
    prior=inventory.attrs.get("verification")
    if prior and verification["files"]!=prior["files"]:
        raise ValueError("Source changed after inventory creation.")
    events=read_selected_events(root,selected.session_id,cfg["batch_size"])
    records=[];seen=set()
    for row in selected.to_dict("records"):
        rs=sorted(events[row["session_id"]],key=lambda e:e["source_event_index"])
        positions=[e["source_event_index"] for e in rs]
        if len(rs)!=row["event_count"] or len(positions)!=len(set(positions)) or positions!=list(range(int(row["source_event_start"]),int(row["source_event_end_exclusive"]))):
            raise ValueError("Event count/boundary verification failed: "+row["session_id"])
        # Verify logical counts against the actual retained source events.
        metadata=pd.DataFrame([e["source_event"] for e in rs]).drop(columns=["content"],errors="ignore")
        stats={};consume_batch(metadata,stats)
        summary=summarize_stats(stats[row["session_id"]])
        if any(summary[c]!=row[c] for c in ["human_messages","assistant_messages","conversational_rounds","initial_intent_source"]):
            raise ValueError("Source logical-message traceability check failed.")
        canonical=[{k:e["source_event"].get(k) for k in ["role","turn_type","content","tool_name","tool_input_json"]} for e in rs]
        fingerprint=hashlib.sha256(json.dumps(json_safe(canonical),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if fingerprint in seen:
            raise ValueError("Exact transcript duplicate discovered at export. Exclude and resample explicitly.")
        seen.add(fingerprint)
        records.append(dict(**json_safe(row),scenario_kind="complete_released_session",extracted_episode=False,truncated=False,
                            original_log_completeness="not_independently_verified",transcript_sha256=fingerprint,events=json_safe(rs)))
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".swe-chat-stage-",dir=out.parent))
    try:
        (stage/"scenarios.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False,allow_nan=False) for r in records)+"\n")
        selected.to_csv(stage/"scenario_manifest.csv",index=False)
        quota_report(selected,proposed_targets(inventory,cfg)).to_csv(stage/"quota_report.csv",index=False)
        for name,table in report_tables(inventory).items():
            table.to_csv(stage/(name+".csv"),index=False)
        for col in ["initial_intent","coding_mode","agent","strategy"]:
            attribute_distribution(selected,col).to_csv(stage/("selected_"+col+".csv"),index=False)
        length_distribution(selected).to_csv(stage/"selected_length.csv",index=False)
        pushback_distribution(selected).to_csv(stage/"selected_pushback.csv",index=False)
        concentration_report(selected).to_csv(stage/"selected_concentration.csv",index=False)
        (stage/"sampling_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
        provenance=dict(verification=verification,code_sha256=sha256(Path(__file__)),counting_rule=COUNTING_NOTE,
            targets=proposed_targets(inventory,cfg),seed=cfg["seed"],selection_probability="unknown; not a probability sample",
            runtime_versions={p:importlib.metadata.version(p) for p in ["pandas","numpy","scipy","pyarrow"]},
            status="Candidate pool requiring human review, not human-validated annotations.",
            group_policy="No splits assigned. Keep complete user/repository connected groups together.",
            omissions="No extra session outcome/persona summaries copied. Exact source events retain source labels and may contain contact details. Not a blinded annotation packet.")
        (stage/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        (stage/"README.txt").write_text("Local candidate pool only. Source events, including tools, retained in full.\nNot a blinded human annotation packet. Review/redact direct contact information before annotation sharing.\nDo not upload or push. Each teammate must obtain source access and review applicable conditions.\nNo training, LLM judgments, annotations or train/test partition performed.\n")
        back=[json.loads(s) for s in (stage/"scenarios.jsonl").read_text().splitlines()]
        if len(back)!=len(selected) or len({r["scenario_id"] for r in back})!=len(selected):
            raise AssertionError("Export readback failed.")
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--verify-hashes",action="store_true")
    parser.add_argument("--report",type=Path,help="Write metadata/distributions to a NEW directory. No selection.")
    parser.add_argument("--preview",action="store_true",help="Explicit in-memory sampling, no dataset files.")
    parser.add_argument("--export",type=Path,help="Explicit selection and export to a NEW directory.")
    parser.add_argument("--allow-shortfall",action="store_true")
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    inventory=build_inventory(args.root,cfg,args.verify_hashes,progress=print)
    print(COUNTING_NOTE)
    print("Observed:",inventory.turn_type.value_counts().to_dict())
    print("Exclusions:",inventory.exclusion_reason.value_counts().to_dict())
    print("Structural candidates:",inventory.loc[inventory.eligible_candidate,"turn_type"].value_counts().to_dict())
    print(capacity_report(inventory,cfg).to_string(index=False))
    print("Proposed targets:",json.dumps(proposed_targets(inventory,cfg)))
    if args.report:
        print("Metadata report:",write_report(args.report,inventory,cfg,root=args.root))
    if args.preview or args.export:
        selected=select_candidates(inventory,cfg,args.allow_shortfall)
        print(quota_report(selected,proposed_targets(inventory,cfg)).to_string(index=False))
        print(concentration_report(selected).to_string(index=False))
        if args.export:
            print("Exported:",export_dataset(args.root,args.export,inventory,selected,cfg))
    else:
        print("No scenarios selected or exported. No train/test split assigned.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
