"""Read-only helpers for the two consolidated dataset notebooks.

No selection, export, source writes, training, or inference. Source conversation
text is inert. Optional previews escape HTML and redact obvious contact/secrets.
"""
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import organize_collabskill as collab_org
import organize_swe_chat as swe_org

LENGTH_BINS = ["single", "2-4", "5-9", "10-19", "20+"]
TURN_COLORS = {"single": "#4676B8", "multi": "#E19738"}
DATASET_COLORS = {"CollabSkill": "#326b91", "SWE-chat": "#bd6a39"}


def load_sources(root):
    root=Path(root)
    collab=collab_org.build_inventory(root,collab_org.load_config(root/"collabskill_sampling.json"))
    review=root/"analysis/swe_chat_review_2026-10-27"
    swe=swe_org.load_review(root,swe_org.load_config(root/"swe_chat_sampling.json"),
                            report_dir=review if review.is_dir() else None)
    return {"CollabSkill":collab,"SWE-chat":swe}


def normalized(frame):
    out=frame.copy()
    out["human_messages"]=pd.to_numeric(out.human_messages,errors="coerce").astype("Int64")
    out["length_bin"]=out.human_messages.map(swe_org.length_bin)
    out["review_category"]=out["native_category"] if "native_category" in out else out["initial_intent"]
    out["review_agent"]=out["agent_id"] if "agent_id" in out else out["agent"]
    return out


def coverage_table(sources):
    rows=[]
    for name,frame in sources.items():
        f=normalized(frame)
        rows.append(dict(dataset=name,source_sessions=len(f),available_logs=int(f.has_log.sum()),
            missing_logs=int((~f.has_log).sum()),zero_human_messages=int(f.length_bin.eq("zero").sum()),
            single_turn=int(f.turn_type.eq("single").sum()),multi_turn=int(f.turn_type.eq("multi").sum()),
            conversational_human_messages=int(f.human_messages.sum()),
            median_human_messages_available=float(f.loc[f.has_log,"human_messages"].median())))
    return pd.DataFrame(rows).set_index("dataset")


def count_and_percent(frame,column,order=None):
    f=frame[frame.turn_type.isin(["single","multi"])].copy()
    f[column]=f[column].fillna("Missing").replace("","Missing")
    table=pd.crosstab(f[column],f.turn_type).reindex(columns=["single","multi"],fill_value=0)
    if order is not None:
        table=table.reindex(order,fill_value=0)
    out=pd.DataFrame(index=table.index)
    for turn in ["single","multi"]:
        out[turn+" n"]=table[turn]
        out[turn+" %"]=100*table[turn]/table[turn].sum() if table[turn].sum() else np.nan
    return out


def length_mix(frame):
    f=normalized(frame)
    counts=f.length_bin.value_counts().reindex(LENGTH_BINS,fill_value=0)
    return counts.to_frame("sessions").assign(percent_among_single_or_multi=100*counts/counts.sum())


def compare_lengths(frames,multi_only=False):
    import matplotlib.pyplot as plt
    bins=LENGTH_BINS[1:] if multi_only else ["zero",*LENGTH_BINS]
    counts=pd.DataFrame(index=bins)
    for name,frame in frames.items():
        f=normalized(frame)
        f=f[f.turn_type.eq("multi")] if multi_only else f[f.human_messages.notna()]
        counts[name]=f.length_bin.value_counts().reindex(bins,fill_value=0)
    pct=counts.div(counts.sum(),axis=1)*100
    fig,ax=plt.subplots(figsize=(10,4.4),layout="constrained")
    pct.plot.bar(ax=ax,rot=0,color=[DATASET_COLORS[k] for k in pct.columns])
    ax.set(xlabel="Conversational human messages",ylabel="% of multi-turn threads" if multi_only else "% of available logs",
           title="Multi-turn length comparison" if multi_only else "Observed session lengths")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    return counts,pct,fig


def plot_mix(frame,column,title):
    import matplotlib.pyplot as plt
    table=count_and_percent(frame,column)
    fig,ax=plt.subplots(figsize=(10,max(3.4,len(table)*.43+1.1)),layout="constrained")
    table[["single %","multi %"]].rename(columns={"single %":"Single-turn","multi %":"Multi-turn"}).plot.barh(
        ax=ax,color=[TURN_COLORS["single"],TURN_COLORS["multi"]])
    ax.set(title=title,xlabel="% within each turn group",ylabel="")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    return fig


def load_samples(root,verify_content=True):
    """Pin saved samples, never regenerate or silently choose a latest directory."""
    root=Path(root)
    snapshots=json.loads((root/"notebook_review_snapshots.json").read_text())
    samples={};configs={};checks=[]
    for name,snapshot in snapshots["samples"].items():
        path=root/snapshot["directory"]
        for filename,expected in snapshot["sha256"].items():
            if swe_org.sha256(path/filename)!=expected:
                raise ValueError(f"Saved {name} snapshot changed: {filename}. Review deliberately before updating pins.")
        f=pd.read_csv(path/"scenario_manifest.csv",low_memory=False)
        f=normalized(f)
        if len(f)!=100 or not f.scenario_id.is_unique or not f.session_id.is_unique:
            raise ValueError("Expected 100 unique saved sessions in "+name)
        if f.turn_type.value_counts().to_dict()!={"single":50,"multi":50}:
            raise ValueError("Saved turn counts differ in "+name)
        q=pd.read_csv(path/"quota_report.csv")
        if not q.shortfall.eq(0).all():
            raise ValueError("Saved quota shortfall in "+name)
        cfg=json.loads((path/"sampling_config.json").read_text())
        total_events=0;ids=set()
        if verify_content:
            expected_rows=f.set_index("scenario_id")
            for line in (path/"scenarios.jsonl").open():
                record=json.loads(line)
                sid=record["scenario_id"]
                if sid in ids or sid not in expected_rows.index:
                    raise ValueError("Duplicate/unknown scenario ID in "+name)
                ids.add(sid)
                if record["truncated"] or record["extracted_episode"]:
                    raise ValueError("Unexpected extracted/truncated scenario in "+name)
                human=[e["source_event"] for e in record["events"] if
                       e["source_event"].get("role")=="user" and e["source_event"].get("is_conversational") is True]
                n=len(human) if name=="CollabSkill" else len({swe_org.message_key(e) for e in human})
                if n!=int(expected_rows.loc[sid,"human_messages"]) or len(record["events"])!=int(expected_rows.loc[sid,"event_count"]):
                    raise ValueError("Saved event count mismatch in "+name)
                total_events+=len(record["events"])
            if ids!=set(f.scenario_id):
                raise ValueError("Scenario/manifest IDs differ in "+name)
        samples[name]=f;configs[name]=cfg
        checks.append(dict(dataset=name,saved_scenarios=len(f),single=50,multi=50,
                           retained_events=total_events if verify_content else None,snapshot_hashes="passed",
                           quota_shortfall=int(q.shortfall.sum()),human_validated=False))
    return samples,configs,pd.DataFrame(checks).set_index("dataset"),snapshots


def population_comparison(source,sample,column):
    rows=[]
    for population,f in [("Observed",source),("Structural candidates",source[source.eligible_candidate]),("Sampled",sample)]:
        table=count_and_percent(f,column)
        table.insert(0,"population",population)
        rows.append(table.reset_index())
    return pd.concat(rows,ignore_index=True)


def saved_pushback(root,snapshot):
    """Source-label message counts, grouped native IDs; no source content outputs."""
    counts=defaultdict(Counter)
    for line in (Path(root)/snapshot["directory"]/"scenarios.jsonl").open():
        record=json.loads(line);units=defaultdict(set)
        for item in record["events"]:
            e=item["source_event"]
            if e.get("role")!="user" or e.get("is_conversational") is not True:
                continue
            key=swe_org.message_key(e)
            units[key]
            if e.get("prompt_pushback"):
                units[key].add(e["prompt_pushback"])
        for labels in units.values():
            counts[record["turn_type"]][next(iter(labels)) if len(labels)==1 else "Unannotated/conflicting"]+=1
    rows=[]
    for label in ["non_pushback","correction","failure_report","rejection","Unannotated/conflicting"]:
        row={"prompt_pushback":label}
        for turn in ["single","multi"]:
            n=counts[turn][label];total=sum(counts[turn].values())
            row[turn+" messages"]=n;row[turn+" %"]=100*n/total if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("prompt_pushback")


def redact_preview(value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    for pattern,replacement in [
        (r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b","[email redacted]"),
        (r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)","[phone redacted]"),
        (r"(?i)\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{12,}","[credential redacted]"),
        (r"(?i)((?:api[_ -]?key|access[_ -]?token|secret|password)\s*[=:]\s*)[^\s,;]+",r"\1[redacted]")]:
        text=re.sub(pattern,replacement,text)
    return text


def show_saved_thread(root,snapshots,dataset,scenario_id,max_events=60,char_limit=3000):
    """Optional escaped preview. Source text/commands are never executed."""
    from IPython.display import HTML,display
    record=None
    for line in (Path(root)/snapshots["samples"][dataset]["directory"]/"scenarios.jsonl").open():
        value=json.loads(line)
        if value["scenario_id"]==scenario_id:
            record=value;break
    if record is None:
        raise KeyError("Scenario not in the pinned sample: "+scenario_id)
    snippets=[f"<p>Display preview: {min(max_events,len(record['events']))} of {len(record['events'])} retained events. "
              "Text is redacted and capped for display only. Full source events remain in the dataset. This is not a blinded annotation packet.</p>"]
    for item in record["events"][:max_events]:
        e=item["source_event"]
        label=f"Event {item['source_event_index']} | {e.get('role','')} | {e.get('turn_type',e.get('type',''))} | {e.get('tool_name','')}"
        text=str(e.get("content") or "")
        tool=e.get("tool_input_json",e.get("tool_input"))
        if tool and tool!=text:
            text+="\n\nTool input:\n"+str(tool)
        text=redact_preview(text)
        if len(text)>char_limit:
            text=text[:char_limit]+"\n[Display text truncated]"
        snippets.append(f"<details open><summary>{escape(label)}</summary><pre style='white-space:pre-wrap;overflow-wrap:anywhere'>{escape(text)}</pre></details>")
    display(HTML("\n".join(snippets)))
