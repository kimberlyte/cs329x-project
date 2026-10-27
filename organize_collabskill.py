"""Read-only analysis by default; explicit export creates a new candidate dataset.

This module never downloads, trains, executes source content, or modifies raw data.
Use --help for review/preview/export commands. SciPy is needed only for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "collabskill_sampling.json"
SOURCES = ["gdpval_benchmark", "apex", "apex_agents"]
SOURCE_NAMES = {"gdpval_benchmark": "GDPval", "apex": "APEX", "apex_agents": "APEX-Agents"}
LENGTH_ORDER = ["single", "2-4", "5-9", "10-19", "20+", "zero", "missing"]
COUNTING_NOTE = "Conversational role=user events, including source-marked setup/help commands. Not semantic request counts."


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path=CONFIG):
    cfg = read_json(path)
    if cfg.get("counting_rule") != "released_conversational_user_events":
        raise ValueError("Unsupported counting rule. Do not silently change turn definitions.")
    keys = [(q["source"], q["category"]) for q in cfg["category_quotas_per_turn_type"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source/category quota.")
    values = [q["n"] for q in cfg["category_quotas_per_turn_type"]] + list(cfg["length_quotas"].values())
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError("Quotas must be nonnegative integers.")
    if set(cfg["length_quotas"]) != {"2-4", "5-9", "10-19", "20+"}:
        raise ValueError("Expected the four agreed multi-turn length bins.")
    if sum(q["n"] for q in cfg["category_quotas_per_turn_type"]) != sum(cfg["length_quotas"].values()):
        raise ValueError("Category quotas per group and multi-turn length quotas disagree.")
    for limit in cfg["concentration_limits"].values():
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Concentration caps must be null or positive integers.")
    if cfg.get("occupation_matching") != "prefer_same_within_gdpval_sector":
        raise ValueError("Unsupported occupation matching policy.")
    return cfg


def attach_categories(frame, root=ROOT):
    """Keep native sectors/occupations separate from APEX domain classifications."""
    out = frame.copy()
    # Supports both session and task tables, with or without prior GDPval enrichment.
    if "sector" not in out or "occupation" not in out:
        gdp = pd.read_parquet(Path(root) / "metadata/external_sources/gdpval/train-00000-of-00001.parquet",
                              columns=["task_id", "sector", "occupation"])
        out = out.merge(gdp, on="task_id", how="left", validate="many_to_one")
    mapping = pd.read_csv(Path(root) / "metadata/apex_category_mapping/task_mapping.csv", keep_default_na=False)
    if mapping.task_id.duplicated().any():
        raise ValueError("Duplicate APEX task mappings.")
    lookup = mapping.set_index("task_id")
    out["native_category"] = out["sector"].astype(object)
    out["category_status"] = "source_metadata_exact_id_join"
    out["category_method"] = "GDPval exact task_id join"
    for source in ["apex", "apex_agents"]:
        mask = out.dataset.eq(source)
        out.loc[mask, "native_category"] = out.loc[mask, "task_id"].map(lookup.category)
        out.loc[mask, "category_status"] = out.loc[mask, "task_id"].map(lookup.category_status)
        out.loc[mask, "category_method"] = out.loc[mask, "task_id"].map(lookup.method)
    out["native_category"] = out.native_category.fillna("Unresolved").replace("", "Unresolved")
    out["category_status"] = out.category_status.fillna("unresolved").replace("", "unresolved")
    out["category_method"] = out.category_method.fillna("unavailable")
    return out


def length_bin(n):
    if pd.isna(n):
        return "missing"
    if n == 0:
        return "zero"
    if n == 1:
        return "single"
    if n <= 4:
        return "2-4"
    if n <= 9:
        return "5-9"
    if n <= 19:
        return "10-19"
    return "20+"


def related_groups(frame):
    """Components of all source sessions sharing a user OR task, for future splits."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for row in frame.sort_values("session_id").itertuples():
        u, t = "user:" + str(row.user_id), "task:" + str(row.task_id)
        a, b = find(u), find(t)
        if a != b:
            parent[max(a, b)] = min(a, b)
    members = {}
    for row in frame.itertuples():
        members.setdefault(find("task:" + str(row.task_id)), []).append(row.session_id)
    ids = {k: "group_" + hashlib.sha256("|".join(sorted(v)).encode()).hexdigest()[:16]
           for k, v in members.items()}
    return pd.Series([ids[find("task:" + str(r.task_id))] for r in frame.itertuples()], index=frame.index)


def source_events(root, sid):
    path = Path(root) / "raw/collabskill/interaction_logs" / sid / "extracted_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_summary(rows):
    if not rows or any("no_extracted_log_reason" in r for r in rows):
        return dict(human_messages=None, assistant_messages=None, conversational_rounds=None,
                    tool_events=None, event_count=0, has_log=False, has_agent_behavior=False,
                    command_like_human_messages=None, transcript_hash=None,
                    begins_with_human=None, ends_with_human=None)
    conversational = [(i, r) for i, r in enumerate(rows)
                      if r.get("is_conversational") is True and r.get("role") in ["user", "assistant"]]
    humans = [(i, r) for i, r in conversational if r["role"] == "user"]
    assistants = [(i, r) for i, r in conversational if r["role"] == "assistant"]
    rounds, previous = 0, None
    for _, r in conversational:
        if r["role"] == "user" and previous != "user":
            rounds += 1
        previous = r["role"]
    tool = lambda r: r.get("role") in ["tool", "tool_use", "tool_result"] or bool(r.get("tool_name")) or str(r.get("type") or "").startswith("tool")
    first = humans[0][0] if humans else len(rows)
    # Tools count as observable behavior even if a final assistant prose response is absent.
    behavior = any(i > first and (tool(r) or (r.get("role") == "assistant" and r.get("is_conversational") is True))
                   for i, r in enumerate(rows))
    command = re.compile(r"^\s*(?:/\S+|(?:cd|dir|mkdir|npm|pip|export|set-executionpolicy|out-printer|get-content)\b)", re.I)
    canonical = [{k: r.get(k) for k in ["role", "type", "content", "tool_name", "tool_input", "is_conversational"]}
                 for r in rows]
    return dict(human_messages=len(humans), assistant_messages=len(assistants), conversational_rounds=rounds,
                tool_events=sum(tool(r) for r in rows), event_count=len(rows), has_log=True,
                has_agent_behavior=behavior, command_like_human_messages=sum(bool(command.search(r.get("content") or "")) for _, r in humans),
                transcript_hash=hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                begins_with_human=bool(conversational and conversational[0][1]["role"] == "user"),
                ends_with_human=bool(conversational and conversational[-1][1]["role"] == "user"))


def verify_inputs(root=ROOT, full_hashes=False):
    """Check every pinned CollabSkill source file; exclude SWE-chat entirely."""
    root = Path(root)
    checks = read_json(root / "metadata/collabskill.checksums.json")
    for row in checks:
        path = root / "raw/collabskill" / row["path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"]:
            raise ValueError("Missing or changed CollabSkill source file: " + row["path"])
        if full_hashes and sha256(path) != row["sha256"]:
            raise ValueError("Source hash mismatch: " + row["path"])
    gdp = read_json(root / "metadata/external_sources/gdpval/provenance.json")
    if sha256(root / "metadata/external_sources/gdpval/train-00000-of-00001.parquet") != gdp["sha256"]:
        raise ValueError("GDPval enrichment checksum mismatch.")
    return {"source_files_checked": len(checks), "source_check": "sha256" if full_hashes else "file sizes",
            "gdpval_enrichment": "sha256 verified"}


def build_inventory(root=ROOT, config=None, verify_hashes=False):
    root = Path(root)
    cfg = config or load_config()
    manifest = read_json(root / "download_manifest.json")["datasets"]["collabskill"]
    if manifest["revision"] != cfg["source_revision"]:
        raise ValueError("Source revision differs from config. Review mappings and quotas first.")
    for name, expected in cfg.get("input_sha256", {}).items():
        if sha256(root / name) != expected:
            raise ValueError("Mapping input changed. Review its provenance and update config deliberately: " + name)
    verification = verify_inputs(root, full_hashes=verify_hashes)
    sessions = pd.DataFrame(read_json(root / "raw/collabskill/leaderboard_data/sessions.json"))
    tasks = pd.DataFrame(read_json(root / "raw/collabskill/leaderboard_data/tasks.json"))
    if sessions.session_id.duplicated().any() or tasks.task_id.duplicated().any():
        raise ValueError("Duplicate source session/task IDs.")
    sf = sessions[["session_id", "user_id", "task_id", "agent_id"]].merge(
        tasks[["task_id", "dataset"]], on="task_id", validate="many_to_one")
    if len(sf) != len(sessions):
        raise ValueError("Session/task join lost rows.")
    sf = attach_categories(sf, root)
    summaries = []
    for r in sf.itertuples():
        rows = source_events(root, r.session_id)
        summary = event_summary(rows)
        summary["model"] = next((x.get("model") for x in rows if x.get("model")), None)
        summary["missing_log_reason"] = next((x.get("no_extracted_log_reason") for x in rows if "no_extracted_log_reason" in x), None)
        summaries.append(summary)
    sf = pd.concat([sf.reset_index(drop=True), pd.DataFrame(summaries)], axis=1)
    for col in ["human_messages", "assistant_messages", "conversational_rounds", "tool_events", "command_like_human_messages"]:
        sf[col] = sf[col].astype("Int64")
    sf["length_bin"] = sf.human_messages.map(length_bin)
    sf["turn_type"] = sf.length_bin.map(lambda x: "single" if x == "single" else "multi" if x in ["2-4", "5-9", "10-19", "20+"] else x)
    sf["split_group_id"] = related_groups(sf)
    sf["source_revision"] = manifest["revision"]
    sf["manual_exclusion"] = sf.session_id.map(cfg["manual_exclusions"]).fillna("")
    # Canonical retained member must itself pass structural/manual eligibility.
    sf["exclusion_reason"] = ""
    for idx, r in sf.iterrows():
        reasons = []
        if not r.has_log:
            reasons.append("missing_log")
        elif r.human_messages == 0:
            reasons.append("no_conversational_human_message")
        elif not r.has_agent_behavior:
            reasons.append("no_observed_agent_behavior_after_human")
        if r.manual_exclusion:
            reasons.append("manual_exclusion")
        status = str(r.category_status)
        policy = cfg["mapping_policy"]
        if r.native_category == "Unresolved" and not policy["allow_unresolved"]:
            reasons.append("unresolved_category")
        if "inferred" in status and not policy["allow_inferred"]:
            reasons.append("inferred_category_disallowed")
        if "provisional" in status and not policy["allow_provisional"]:
            reasons.append("provisional_category_disallowed")
        sf.at[idx, "exclusion_reason"] = "|".join(reasons)
    sf["duplicate_of_session_id"] = ""
    seen = {}
    for idx, r in sf.sort_values("session_id").iterrows():
        if r.exclusion_reason or not r.transcript_hash:
            continue
        if r.transcript_hash in seen:
            sf.at[idx, "duplicate_of_session_id"] = seen[r.transcript_hash]
            sf.at[idx, "exclusion_reason"] = "exact_transcript_duplicate"
        else:
            seen[r.transcript_hash] = r.session_id
    sf["eligible_candidate"] = sf.exclusion_reason.eq("")
    sf["requires_human_context_review"] = True
    sf["context_availability"] = "Released log retained in full; task inputs/artifacts not independently checked."
    sf["training_use_status"] = np.where(sf.dataset.eq("apex_agents"), "source_terms_evaluation_only_requires_permission_for_training", "check_all_applicable_source_terms_before_training")
    sf.attrs["verification"] = verification
    return sf.sort_values(["dataset", "native_category", "session_id"]).reset_index(drop=True)


def category_distribution(frame, by_occupation=False):
    keys = ["dataset", "native_category"]
    if by_occupation:
        keys.append("occupation")
    rows = []
    for values, g in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        available = g[g.has_log]
        counts = g.length_bin.value_counts()
        row.update(tasks=g.task_id.nunique(), sessions=len(g), available_logs=len(available),
                   single=int(counts.get("single", 0)), multi=int(g.turn_type.eq("multi").sum()),
                   zero=int(counts.get("zero", 0)), missing=int(counts.get("missing", 0)),
                   category_status="|".join(sorted(set(g.category_status))))
        for b in ["2-4", "5-9", "10-19", "20+"]:
            row[b] = int(counts.get(b, 0))
        row["single_pct_available_logs"] = 100 * row["single"] / len(available) if len(available) else np.nan
        row["multi_pct_available_logs"] = 100 * row["multi"] / len(available) if len(available) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def length_distribution(frame):
    """Source-conditional lengths, with separate multi-turn and all-log denominators."""
    rows = []
    for source, g in frame.groupby("dataset", sort=True):
        available, multi = int(g.has_log.sum()), int(g.turn_type.eq("multi").sum())
        for b in LENGTH_ORDER:
            n = int(g.length_bin.eq(b).sum())
            rows.append(dict(dataset=source, length_bin=b, sessions=n,
                             pct_all_source_sessions=100*n/len(g),
                             pct_available_logs=100*n/available if available and b != "missing" else np.nan,
                             pct_multi_turn=100*n/multi if multi and b in ["2-4", "5-9", "10-19", "20+"] else np.nan))
    return pd.DataFrame(rows)


def category_mix(frame, by_occupation=False):
    """Compare P(category | single, source) with P(category | multi, source)."""
    table=category_distribution(frame,by_occupation=by_occupation)
    if table.empty:
        return table
    single_total=table.groupby("dataset")["single"].transform("sum")
    multi_total=table.groupby("dataset")["multi"].transform("sum")
    table["single_share_within_source_pct"]=table.single.div(single_total.replace(0,np.nan))*100
    table["multi_share_within_source_pct"]=table.multi.div(multi_total.replace(0,np.nan))*100
    table["single_minus_multi_share_pp"]=table.single_share_within_source_pct-table.multi_share_within_source_pct
    table["max_count_matched_per_group"]=table[["single","multi"]].min(axis=1)
    return table


def capacity_report(inventory, cfg):
    eligible = inventory[inventory.eligible_candidate]
    rows = []
    for q in cfg["category_quotas_per_turn_type"]:
        g = eligible[eligible.dataset.eq(q["source"]) & eligible.native_category.eq(q["category"])]
        s, m = int(g.turn_type.eq("single").sum()), int(g.turn_type.eq("multi").sum())
        rows.append(dict(dataset=q["source"], native_category=q["category"], requested_each=q["n"],
                         candidate_single=s, candidate_multi=m,
                         single_shortfall=max(0,q["n"]-s), multi_shortfall=max(0,q["n"]-m)))
    return pd.DataFrame(rows)


def concentration_report(frame):
    rows = []
    for dimension in ["user_id", "task_id", "agent_id", "split_group_id"]:
        counts = frame[dimension].value_counts(dropna=False)
        rows.append(dict(dimension=dimension, distinct=len(counts), maximum_sessions=int(counts.max()) if len(counts) else 0,
                         maximum_share_pct=100*counts.max()/len(frame) if len(frame) else np.nan))
    return pd.DataFrame(rows)


def proportional_length_targets(inventory, n=50, eligible_only=True):
    frame = inventory[inventory.eligible_candidate] if eligible_only else inventory
    counts = frame.loc[frame.turn_type.eq("multi"), "length_bin"].value_counts().reindex(["2-4","5-9","10-19","20+"], fill_value=0)
    if not counts.sum():
        return pd.DataFrame({"population": counts, "share_pct": np.nan, "suggested_n": 0})
    raw = counts/counts.sum()*n
    quotas = np.floor(raw).astype(int)
    for b in (raw-quotas).sort_values(ascending=False, kind="stable").index[:n-int(quotas.sum())]:
        quotas[b] += 1
    return pd.DataFrame({"population": counts, "share_pct": counts/counts.sum()*100, "suggested_n": quotas})


def select_candidates(inventory, cfg, allow_shortfall=False):
    """Constrained seeded selection. No writes. No known design inclusion probabilities.

    Stage 1 maximizes fulfilled matched category counts within all quota ceilings.
    Stage 2 minimizes GDPval occupation imbalance, then seeded random costs.
    Entire sessions are indivisible and category/length quotas are never relaxed.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    frame = inventory[inventory.eligible_candidate].copy()
    quotas = {(q["source"],q["category"]): q["n"] for q in cfg["category_quotas_per_turn_type"]}
    frame = frame.loc[pd.Series([quotas.get((r.dataset,r.native_category),0)>0 for r in frame.itertuples()],index=frame.index,dtype=bool)]
    frame = frame.sort_values("session_id").reset_index(drop=True)
    n = len(frame)
    if not n:
        raise ValueError("No eligible candidates in requested strata.")
    occupations = sorted(set((r.native_category,str(r.occupation)) for r in frame.itertuples()
                             if r.dataset == "gdpval_benchmark" and pd.notna(r.occupation)))
    variables = n + 2*len(occupations)
    constraints, lower, upper = [], [], []

    def add(row, lo, hi):
        constraints.append(row); lower.append(lo); upper.append(hi)

    for (source, category), target in quotas.items():
        mask = frame.dataset.eq(source) & frame.native_category.eq(category)
        single = (mask & frame.turn_type.eq("single")).astype(float).to_numpy()
        multi = (mask & frame.turn_type.eq("multi")).astype(float).to_numpy()
        add(dict(enumerate(single-multi)),0,0)
        add(dict(enumerate(single)),0,target)
    for b, target in cfg["length_quotas"].items():
        add(dict(enumerate((frame.turn_type.eq("multi") & frame.length_bin.eq(b)).astype(float))),0,target)
    for key, column in [("max_per_user","user_id"),("max_per_task","task_id"),("max_per_agent","agent_id")]:
        cap = cfg["concentration_limits"].get(key)
        if cap is not None:
            for _, indices in frame.groupby(column,dropna=False).groups.items():
                add({int(i):1.0 for i in indices},0,cap)
    for j, (category, occupation) in enumerate(occupations):
        mask = frame.dataset.eq("gdpval_benchmark") & frame.native_category.eq(category) & frame.occupation.eq(occupation)
        row = dict(enumerate(np.where(mask, np.where(frame.turn_type.eq("single"),1.0,-1.0),0.0)))
        row[n+2*j] = -1.0; row[n+2*j+1] = 1.0
        add(row,0,0)
    matrix = lil_matrix((len(constraints),variables))
    for i,row in enumerate(constraints):
        for j,v in row.items():
            if v:
                matrix[i,j] = v
    matrix = matrix.tocsr()
    bounds = Bounds(np.zeros(variables),np.r_[np.ones(n),np.full(variables-n,np.inf)])
    integrality = np.r_[np.ones(n),np.zeros(variables-n)]
    options = {"time_limit": cfg.get("solver_time_limit_seconds",60)}
    first = milp(c=np.r_[-np.ones(n),np.zeros(variables-n)],integrality=integrality,bounds=bounds,
                 constraints=LinearConstraint(matrix,lower,upper),options=options)
    if not first.success:
        raise RuntimeError("Selection solver did not prove an optimum: " + first.message)
    achieved = int(np.rint(first.x[:n]).sum())
    requested = 2*sum(quotas.values())
    if achieved < requested and not allow_shortfall:
        raise ValueError(f"Exact requested design infeasible: maximum {achieved//2} matched category pairs of {requested//2}. Review capacity_report, caps and exclusions. No export created. Use --allow-shortfall only to explicitly export the valid subset.")
    from scipy.sparse import vstack,csr_matrix
    total = np.r_[np.ones(n),np.zeros(variables-n)]
    matrix = vstack([matrix,csr_matrix(total.reshape(1,-1))],format="csr")
    rng = np.random.default_rng(cfg["seed"])
    # Each one-unit occupation imbalance costs more than every candidate's random cost combined.
    objective = np.r_[rng.random(n),np.full(variables-n,n+1.0)]
    result = milp(c=objective,integrality=integrality,bounds=bounds,
                  constraints=LinearConstraint(matrix,[*lower,achieved],[*upper,achieved]),options=options)
    if not result.success:
        raise RuntimeError("Matching solver did not prove an optimum: " + result.message)
    chosen = frame.loc[np.rint(result.x[:n]).astype(bool)].copy()
    chosen["scenario_id"] = ["collabskill_"+hashlib.sha256((str(r.source_revision)+":"+r.session_id).encode()).hexdigest()[:20] for r in chosen.itertuples()]
    chosen["selection_probability"] = np.nan
    chosen["design_weight"] = np.nan
    chosen["selection_method"] = "seeded constrained optimization; inclusion probabilities unknown"
    chosen["inclusion_rationale"] = "Source/category quota plus proportional length target; GDPval occupation balance preferred."
    chosen["split"] = "unassigned"
    # Explicitly log count-level occupation matching. This is not a causal pair design.
    chosen["occupation_match_status"] = "not_applicable_source_has_no_verified_occupation"
    for (category,occupation),g in chosen[chosen.dataset.eq("gdpval_benchmark")].groupby(["native_category","occupation"],dropna=False):
        ns, nm = int(g.turn_type.eq("single").sum()),int(g.turn_type.eq("multi").sum())
        status = "same_occupation_counts_matched" if ns == nm and pd.notna(occupation) else "sector_only_or_partial_occupation_match"
        chosen.loc[g.index,"occupation_match_status"] = status
    for key,expected in quotas.items():
        g = chosen[chosen.dataset.eq(key[0]) & chosen.native_category.eq(key[1])]
        ns,nm = int(g.turn_type.eq("single").sum()),int(g.turn_type.eq("multi").sum())
        if ns != nm or ns > expected or (achieved == requested and ns != expected):
            raise AssertionError("Category quota verification failed.")
    validate_selection(chosen, inventory, cfg, allow_shortfall=allow_shortfall)
    return chosen.sort_values(["dataset","native_category","turn_type","scenario_id"]).reset_index(drop=True)


def validate_selection(selected, inventory, cfg, allow_shortfall=False):
    """Independent post-solver checks, also repeated before any export."""
    if inventory.session_id.duplicated().any():
        raise ValueError("Population contains duplicate session IDs.")
    if selected.session_id.duplicated().any() or selected.scenario_id.duplicated().any():
        raise ValueError("Selection contains duplicate session or scenario IDs.")
    original = inventory.set_index("session_id")
    if not selected.session_id.isin(original.index).all():
        raise ValueError("Selection contains sessions outside the inventory.")
    rows = original.loc[selected.session_id].reset_index()
    if not rows.eligible_candidate.all():
        raise ValueError("Selection contains excluded sessions.")
    for col in ["dataset", "native_category", "turn_type", "length_bin", "user_id", "task_id", "agent_id", "transcript_hash"]:
        if not selected[col].reset_index(drop=True).equals(rows[col].reset_index(drop=True)):
            raise ValueError("Selection metadata differs from the inventory: " + col)
    quotas = {(q["source"], q["category"]): q["n"] for q in cfg["category_quotas_per_turn_type"]}
    if any((r.dataset,r.native_category) not in quotas for r in selected.itertuples()):
        raise ValueError("Selection contains an unrequested category.")
    report = shortfall_report(selected, cfg)
    if report.shortfall.lt(0).any() or (not allow_shortfall and report.shortfall.ne(0).any()):
        raise ValueError("Selection violates category or length quotas.")
    for (source, category), group in selected.groupby(["dataset","native_category"]):
        if group.turn_type.eq("single").sum() != group.turn_type.eq("multi").sum():
            raise ValueError("Single/multi counts are not matched within category.")
    if not selected.turn_type.isin(["single","multi"]).all():
        raise ValueError("Selection includes a missing/zero-message session.")
    for key, col in [("max_per_user","user_id"),("max_per_task","task_id"),("max_per_agent","agent_id")]:
        cap=cfg["concentration_limits"].get(key)
        if cap is not None and selected[col].value_counts(dropna=False).gt(cap).any():
            raise ValueError("Selection violates concentration cap: " + key)


def shortfall_report(selected, cfg):
    rows=[]
    for q in cfg["category_quotas_per_turn_type"]:
        for turn in ["single","multi"]:
            n=int((selected.dataset.eq(q["source"]) & selected.native_category.eq(q["category"]) & selected.turn_type.eq(turn)).sum())
            rows.append(dict(kind="category",dataset=q["source"],category=q["category"],turn_type=turn,
                             requested=q["n"],achieved=n,shortfall=q["n"]-n))
    for b,n in cfg["length_quotas"].items():
        got=int((selected.turn_type.eq("multi") & selected.length_bin.eq(b)).sum())
        rows.append(dict(kind="length",category=b,turn_type="multi",requested=n,achieved=got,shortfall=n-got))
    return pd.DataFrame(rows)


def plot_category_distributions(inventory, eligible_only=False):
    """Return figures without saving: paired counts and multi-turn length mix per source."""
    import matplotlib.pyplot as plt
    frame=inventory[inventory.eligible_candidate] if eligible_only else inventory
    table=category_distribution(frame)
    figures=[]
    if table.empty:
        return figures
    for source in SOURCES:
        g=table[table.dataset.eq(source)]
        if g.empty:
            continue
        labels=g.native_category.str.replace("Professional, Scientific, and Technical Services","Professional/scientific/technical",regex=False).str.replace("Health Care and Social Assistance","Health care/social assistance",regex=False).str.replace("Real Estate and Rental and Leasing","Real estate/rental/leasing",regex=False)
        y=np.arange(len(g));height=max(3.0,.48*len(g)+1.8)
        fig,axes=plt.subplots(1,2,figsize=(13,height),layout="constrained")
        for ax in axes:
            ax.set_yticks(y,labels);ax.invert_yaxis();ax.spines[["top","right"]].set_visible(False)
        axes[0].barh(y-.18,g.single,height=.34,label="Single: 1 human message")
        axes[0].barh(y+.18,g.multi,height=.34,label="Multi: 2+ human messages")
        axes[0].set_xlabel("Sessions");axes[0].set_title("Single-turn and multi-turn counts")
        axes[0].legend(loc="upper left",bbox_to_anchor=(0,-.16),fontsize=8,ncol=2,frameon=False)
        left=np.zeros(len(g))
        for b in ["2-4","5-9","10-19","20+"]:
            values=np.divide(g[b].to_numpy()*100,g.multi.to_numpy(),out=np.zeros(len(g)),where=g.multi.to_numpy()!=0)
            axes[1].barh(y,values,left=left,label=b);left+=values
        axes[1].set_xlim(0,100);axes[1].set_xlabel("% of multi-turn sessions within category")
        axes[1].set_title("Multi-turn length mix (human messages)")
        axes[1].legend(loc="upper left",bbox_to_anchor=(0,-.16),ncol=4,fontsize=8,frameon=False)
        status_note = " · provisional categories" if source == "apex_agents" else " · includes inferred categories" if source == "apex" else ""
        fig.suptitle(f"{SOURCE_NAMES[source]} · {'Filtered candidates' if eligible_only else 'Observed logs'}{status_note}",fontsize=14)
        figures.append((source,fig))
    return figures


def export_dataset(root, out, inventory, selected, cfg, allow_shortfall=False):
    """All exports go to a NEW directory. Source wording/events are retained verbatim."""
    root,out=Path(root),Path(out).resolve()
    # Resolve before containment checks so symlinks cannot route exports into raw/.
    forbidden=[(root.resolve()/"raw").resolve(),(root.resolve()/"metadata").resolve()]
    if any(out == f or f in out.parents for f in forbidden):
        raise ValueError("Export must not be inside raw/ or metadata/.")
    if out.exists():
        raise FileExistsError("Output exists. Choose a new directory; nothing overwritten.")
    if not len(selected) or selected.scenario_id.duplicated().any():
        raise ValueError("Empty selection or duplicate scenario IDs.")
    validate_selection(selected,inventory,cfg,allow_shortfall=allow_shortfall)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".collabskill-stage-",dir=out.parent))
    reserved_destination=False
    try:
        inventory.to_csv(stage/"population_inventory.csv",index=False)
        selected.to_csv(stage/"scenario_manifest.csv",index=False)
        for name,frame in [("source",inventory),("eligible",inventory[inventory.eligible_candidate]),("selected",selected)]:
            category_distribution(frame).to_csv(stage/f"{name}_category_distribution.csv",index=False)
            length_distribution(frame).to_csv(stage/f"{name}_length_distribution.csv",index=False)
            concentration_report(frame).to_csv(stage/f"{name}_concentration.csv",index=False)
        shortfall_report(selected,cfg).to_csv(stage/"quota_report.csv",index=False)
        lines=[]
        for row in selected.to_dict("records"):
            events=source_events(root,row["session_id"])
            if event_summary(events)["transcript_hash"] != row["transcript_hash"]:
                raise ValueError("Source changed after inventory construction.")
            metadata=json.loads(pd.DataFrame([row]).to_json(orient="records"))[0]
            lines.append(json.dumps(dict(**metadata,scenario_kind="complete_released_session",
                original_log_completeness="not_independently_verified",extracted_episode=False,truncated=False,
                source_event_start=0,source_event_end_exclusive=len(events),
                events=[dict(source_event_index=i,source_event=e) for i,e in enumerate(events)]),ensure_ascii=False))
        (stage/"scenarios.jsonl").write_text("\n".join(lines)+"\n")
        (stage/"sampling_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
        inputs=["download_manifest.json","metadata/apex_category_mapping/task_mapping.csv",
                "metadata/apex_category_mapping/provenance.json","metadata/external_sources/gdpval/provenance.json",
                "metadata/collabskill.checksums.json"]
        import importlib.metadata
        provenance=dict(input_sha256={p:sha256(root/p) for p in inputs},code_sha256=sha256(Path(__file__)),
            runtime_versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","pyarrow"]},
            counting_rule=COUNTING_NOTE,selection_probability="unknown; not a probability sample",seed=cfg["seed"],
            group_policy="No train/test split assigned. Keep split_group_id intact in later splits.",
            status="Candidate curation pool, not human-validated annotations.",
            omissions="No participant contact metadata, scores, skill ratings or task grading rubrics. Raw source event content is retained exactly and may contain contact information or sensitive text.")
        (stage/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        (stage/"README.txt").write_text("Local candidate curation pool. No model training or human annotation performed.\nWhole released sessions retained, with tool actions. Source completeness and artifacts require review.\nExact source wording may contain sensitive content. Keep this snapshot private pending review.\nAPEX-Agents-derived material is not cleared for training. No uploads or repository pushes performed.\nSee provenance.json, sampling_config.json and quota_report.csv.\n")
        # Verify staged output, then reserve the destination exclusively. A directory
        # created by someone else during staging must never be replaced.
        back=[json.loads(s) for s in (stage/"scenarios.jsonl").read_text().splitlines()]
        if len(back)!=len(selected) or len({r["scenario_id"] for r in back})!=len(selected):
            raise AssertionError("JSONL verification failed.")
        out.mkdir(exist_ok=False)
        reserved_destination=True
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        if reserved_destination:
            try:
                out.rmdir()  # Only removes our empty reservation, never other files.
            except OSError:
                pass
        raise
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--verify-hashes",action="store_true",help="Rehash all pinned CollabSkill files (read-only).")
    parser.add_argument("--preview",action="store_true",help="Solve the sampling design in memory. No files written.")
    parser.add_argument("--export",type=Path,help="Explicitly select and export to a NEW directory. Never overwrites.")
    parser.add_argument("--allow-shortfall",action="store_true",help="Permit a maximum feasible matched subset with an explicit quota report.")
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    inventory=build_inventory(args.root,cfg,args.verify_hashes)
    print("Read-only population and category capacity review\n"+COUNTING_NOTE)
    print(capacity_report(inventory,cfg).to_string(index=False))
    print("\nObserved multi-turn length target:")
    print(proportional_length_targets(inventory,eligible_only=False).to_string())
    print("\nFiltered-candidate multi-turn length target (diagnostic, does not alter config):")
    print(proportional_length_targets(inventory).to_string())
    print("\nExclusions:",inventory.exclusion_reason.value_counts().to_dict())
    if args.preview or args.export:
        selected=select_candidates(inventory,cfg,args.allow_shortfall)
        print("\nRequested and achieved quotas:")
        print(shortfall_report(selected,cfg).to_string(index=False))
        print("\nSelected concentration:")
        print(concentration_report(selected).to_string(index=False))
        if args.export:
            print("Exported:",export_dataset(args.root,args.export,inventory,selected,cfg,args.allow_shortfall))
        else:
            print("Preview only. No dataset files created.")
    else:
        print("No selection performed. Use --preview for in-memory selection or --export NEW_DIRECTORY to generate datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
