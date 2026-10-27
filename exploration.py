"""Local, read-only exploration helpers. Dataset content is treated as inert text."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from IPython.display import HTML, display

BOUNDARY_PATTERN = (
    r"\b(?:permission|approval|approve|ask me|check with me|before you|"
    r"do not|don't|don’t|stop|wait|instead|only|let me|I will|I'll|"
    r"go ahead|proceed|take over|leave .{0,30}unchanged)\b"
)
EVENT_COLUMNS = ["session_id", "_event_index", "turn_id", "role", "type", "content",
                 "tool_name", "tool_input", "is_conversational", "agent", "model"]


def locate_data_root():
    """Support launches from data/, Project/, or the course folder."""
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        for candidate in [base, base / "data", base / "Project" / "data"]:
            if (candidate / "download_datasets.py").is_file():
                return candidate
    raise FileNotFoundError("Open the notebook from the Project/data folder.")


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def manifest_frame(root):
    data = read_json(root / "download_manifest.json", {"datasets": {}})
    rows = []
    for name, item in data["datasets"].items():
        rows.append({"dataset": name, "status": item.get("status"),
                     "profile": item.get("profile"), "revision": item.get("revision"),
                     "downloaded_files": item.get("downloaded_file_count", 0),
                     "downloaded_MB": round(item.get("downloaded_size_bytes", 0) / 1e6, 2),
                     "expected_files": item.get("expected_file_count"),
                     "expected_MB": round(item.get("expected_size_bytes", 0) / 1e6, 2)})
    return pd.DataFrame(rows)


def load_collabskill(root):
    base = root / "raw" / "collabskill"
    tables = {}
    for name in ["sessions", "participants", "tasks", "scores"]:
        records = read_json(base / "leaderboard_data" / f"{name}.json", [])
        tables[name] = pd.json_normalize(records)
    ratings = read_json(base / "leaderboard_data" / "ratings.json", {})
    tables["ratings"] = pd.DataFrame(ratings.get("humans", []) + ratings.get("agents", []))
    events, log_status = [], []
    for path in sorted((base / "interaction_logs").glob("*/extracted_log.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sid = path.parent.name
        if len(records) == 1 and "no_extracted_log_reason" in records[0]:
            log_status.append({"session_id": sid, "has_log": False,
                               "reason": records[0]["no_extracted_log_reason"], "event_count": 0})
            continue
        log_status.append({"session_id": sid, "has_log": True, "reason": None,
                           "event_count": len(records)})
        for index, row in enumerate(records):
            events.append({**row, "_event_index": index})
    tables["events"] = pd.DataFrame(events).reindex(columns=EVENT_COLUMNS)
    tables["log_status"] = pd.DataFrame(log_status, columns=["session_id", "has_log", "reason", "event_count"])
    # Keep direct contact/account fields out of the notebook's participant view.
    safe = [c for c in tables["participants"].columns if c == "user_id" or c.startswith("demographics.")]
    tables["participants"] = tables["participants"][safe]
    return tables


def collab_session_index(tables):
    sessions = tables["sessions"].copy()
    if sessions.empty:
        return pd.DataFrame(columns=["session_id"])
    tasks = tables["tasks"]
    if not tasks.empty:
        sessions = sessions.merge(tasks[["task_id", "dataset"]], on="task_id", how="left", validate="many_to_one")
    sessions = sessions.merge(tables["log_status"], on="session_id", how="left", validate="one_to_one")
    events = tables["events"]
    conversational = events[events["is_conversational"].fillna(False).astype(bool)]
    users = conversational[conversational["role"].eq("user")]
    counts = users.groupby("session_id").size()
    sessions["user_messages"] = sessions["session_id"].map(counts).fillna(0).astype(int)
    names = events.groupby("session_id")[["agent", "model"]].first()
    sessions = sessions.join(names, on="session_id")
    candidates = feedback_candidates(events)
    sessions["keyword_matches"] = sessions["session_id"].map(candidates.groupby("session_id").size()).fillna(0).astype(int)
    return sessions


def feedback_candidates(events, pattern=BOUNDARY_PATTERN):
    columns = ["session_id", "_event_index", "turn_id", "content", "agent", "model"]
    if events.empty:
        return pd.DataFrame(columns=columns + ["preview"])
    mask = events["role"].eq("user") & events["is_conversational"].fillna(False).astype(bool)
    mask &= events["content"].fillna("").str.contains(pattern, case=False, regex=True)
    result = events.loc[mask].copy()
    result["preview"] = result["content"].fillna("").str.replace("\n", " ", regex=False).str.slice(0, 220)
    return result


def schema_frame(tables):
    return pd.DataFrame([{"table": name, "attribute": col, "dtype": str(frame[col].dtype),
                          "missing": int(frame[col].isna().sum()), "rows": len(frame)}
                         for name, frame in tables.items() for col in frame.columns])


def parquet_catalog(root):
    rows = []
    for path in sorted((root / "raw" / "swe-chat").glob("*.parquet")):
        file = pq.ParquetFile(path)
        rows.append({"table": path.stem, "rows": file.metadata.num_rows,
                     "columns": len(file.schema_arrow), "row_groups": file.num_row_groups,
                     "disk_MB": round(path.stat().st_size / 1e6, 2)})
    return pd.DataFrame(rows, columns=["table", "rows", "columns", "row_groups", "disk_MB"])


def parquet_schema(path):
    if not path.is_file():
        return pd.DataFrame(columns=["attribute", "arrow_type", "nullable"])
    return pd.DataFrame([{"attribute": f.name, "arrow_type": str(f.type), "nullable": f.nullable}
                         for f in pq.read_schema(path)])


def load_swe_sessions(root):
    path = root / "raw" / "swe-chat" / "sessions.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["session_id", "repo_id", "agent", "model"])
    wanted = ["session_id", "repo_id", "user_id", "agent", "model", "created_at",
              "duration_seconds", "turn_count", "prompt_count", "files_touched_count",
              "human_added", "human_modified", "human_removed", "agent_percentage",
              "attribution_metric_version", "input_tokens", "output_tokens", "session_success"]
    columns = [c for c in wanted if c in pq.read_schema(path).names]
    return pd.read_parquet(path, columns=columns)


def scan_swe_feedback(root, pattern=BOUNDARY_PATTERN, max_user_rows=5000, max_matches=100, pushback=None):
    """Bounded scan of human-prompt rows; first matches are NOT a random sample."""
    if max_user_rows < 1 or max_matches < 1:
        raise ValueError("Scan limits must be positive.")
    path = root / "raw" / "swe-chat" / "conversations.parquet"
    wanted = ["session_id", "repo_id", "turn_id", "turn_number", "conversation_turn_number",
              "timestamp", "content", "prompt_intent", "prompt_pushback", "agent", "model"]
    if not path.is_file():
        return pd.DataFrame(columns=wanted + ["preview"]), {"available": False, "user_rows_inspected": 0, "matches": 0}
    source = ds.dataset(path, format="parquet")
    columns = [c for c in wanted if c in source.schema.names]
    predicate = ds.field("role") == "user"
    if "is_conversational" in source.schema.names:
        predicate &= ds.field("is_conversational") == True
    scanner = source.scanner(columns=columns, filter=predicate, batch_size=2048, use_threads=False)
    inspected, matches, pieces, exhausted = 0, 0, [], True
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        frame = frame.iloc[:max_user_rows - inspected]
        inspected += len(frame)
        mask = frame["content"].fillna("").str.contains(pattern, case=False, regex=True)
        if pushback is not None and "prompt_pushback" in frame:
            mask &= frame["prompt_pushback"].eq(pushback)
        result = frame.loc[mask].iloc[:max_matches - matches].copy()
        pieces.append(result)
        matches += len(result)
        if inspected >= max_user_rows or matches >= max_matches:
            exhausted = False
            break
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    result["preview"] = result["content"].fillna("").str.replace("\n", " ", regex=False).str.slice(0, 220)
    return result, {"available": True, "user_rows_inspected": inspected, "matches": len(result),
                    "scan_reached_end": exhausted, "selection": "first matching user prompts in file order; not representative"}


def read_swe_session(root, session_id, max_events=5000):
    path = root / "raw" / "swe-chat" / "conversations.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    source = ds.dataset(path, format="parquet")
    wanted = ["session_id", "turn_id", "turn_number", "conversation_turn_number", "timestamp",
              "role", "turn_type", "content", "tool_name", "tool_input_json",
              "is_conversational", "agent", "model", "prompt_intent", "prompt_pushback", "file_path"]
    scanner = source.scanner(columns=[c for c in wanted if c in source.schema.names],
                             filter=ds.field("session_id") == session_id,
                             batch_size=1024, use_threads=False)
    parts, count = [], 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas().iloc[:max_events + 1 - count]
        parts.append(frame)
        count += len(frame)
        if count > max_events:
            break
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    result = pd.concat(parts, ignore_index=True).sort_values("turn_number", kind="stable")
    truncated = len(result) > max_events
    result = result.iloc[:max_events].rename(columns={"turn_type": "type", "tool_input_json": "tool_input"}).copy()
    result["_event_index"] = result["turn_number"]
    result.attrs["truncated"] = truncated
    return result


def read_session(root, tables, dataset, session_id):
    if dataset == "collabskill":
        return tables["events"].loc[tables["events"]["session_id"].eq(session_id)].sort_values("_event_index").copy()
    return read_swe_session(root, session_id)


def show_events(events, conversational_only=True, max_events=60, char_limit=3000):
    """Escaped HTML only; no source HTML, code, or shell commands are executed."""
    rows = events
    if conversational_only and "is_conversational" in rows:
        rows = rows[rows["is_conversational"].fillna(False).astype(bool)]
    chunks = [f"<p>Showing {min(len(rows), max_events):,} of {len(rows):,} selected events. Text is capped at {char_limit:,} characters per event.</p>"]
    for _, row in rows.head(max_events).iterrows():
        label = f"Event {row.get('_event_index', '')} | {row.get('role', '')} | {row.get('type', '')} | {row.get('tool_name', '')}"
        text = str(row.get("content", "") or "")
        tool = row.get("tool_input", "")
        if isinstance(tool, str) and tool and tool != text:
            text += "\n\nTool input:\n" + tool
        body = text[:char_limit] + ("\n[Display truncated: inspect the source DataFrame for full text.]" if len(text) > char_limit else "")
        chunks.append(f"<details open><summary><strong>{escape(label)}</strong></summary><pre style='white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:#f6f7f8'>{escape(body)}</pre></details>")
    display(HTML("\n".join(chunks)))


def show_context(events, event_index, before=3, after=4):
    selected = events.sort_values("_event_index")
    positions = [i for i, v in enumerate(selected["_event_index"]) if v == event_index]
    if not positions:
        print("Event not found in this session.")
        return
    pos = positions[0]
    show_events(selected.iloc[max(0, pos-before):pos+after+1], conversational_only=False)


def session_browser(root, tables, swe_sessions):
    """Run this cell again after a SWE-chat download to refresh its choices."""
    import ipywidgets as w
    collab = collab_session_index(tables)
    datasets = [("CollabSkill", "collabskill")]
    if (root / "raw" / "swe-chat" / "conversations.parquet").is_file() and not swe_sessions.empty:
        datasets.append(("SWE-chat", "swe-chat"))
    dataset = w.Dropdown(options=datasets, description="Dataset:")
    query = w.Text(description="Filter:", placeholder="Session, agent, repository, or task")
    choices = w.Dropdown(description="Session:", layout=w.Layout(width="95%"))
    override = w.Text(description="Exact ID:", placeholder="Optional: paste a session_id from search results", layout=w.Layout(width="95%"))
    load = w.Button(description="Load session", button_style="primary")
    cut = w.IntSlider(description="Prefix size:", min=0, max=1, value=1, continuous_update=False)
    tools = w.Checkbox(value=False, description="Include tool / system events")
    future = w.Checkbox(value=False, description="Reveal later events (off for prefix-only annotation)")
    out = w.Output()
    current = {"events": pd.DataFrame()}

    def refresh(*_):
        frame = collab if dataset.value == "collabskill" else swe_sessions
        searchable = [c for c in ["session_id", "agent", "model", "repo_id", "task_id", "dataset"] if c in frame]
        if query.value.strip() and searchable:
            text = frame[searchable].fillna("").astype(str).agg(" | ".join, axis=1)
            frame = frame[text.str.contains(query.value.strip(), case=False, regex=False)]
        options = []
        for _, row in frame.head(1000).iterrows():
            label = " | ".join(str(row.get(c, "")) for c in ["session_id", "agent", "repo_id" if dataset.value == "swe-chat" else "task_id"])
            options.append((label, row["session_id"]))
        choices.options = options

    def render(*_):
        with out:
            out.clear_output(wait=True)
            events = current["events"]
            if events.empty:
                print("No extractable events for this session. Try another session.")
                return
            print(f"Loaded {len(events):,} events. Prefix cutoff is by ALL source events, including tools.")
            if events.attrs.get("truncated"):
                print("Session capped at 5,000 events. Increase max_events in read_swe_session for a larger view.")
            display(HTML("<h4>Interaction prefix</h4>"))
            show_events(events.iloc[:cut.value], conversational_only=not tools.value)
            if future.value:
                display(HTML("<h4>Later events — excluded from prefix-only inference</h4>"))
                show_events(events.iloc[cut.value:], conversational_only=not tools.value)

    def load_selected(_):
        sid = override.value.strip() or choices.value
        if not sid:
            return
        current["events"] = read_session(root, tables, dataset.value, sid)
        cut.max = max(1, len(current["events"]))
        cut.value = min(20, cut.max)
        render()

    dataset.observe(refresh, names="value")
    query.observe(refresh, names="value")
    load.on_click(load_selected)
    cut.observe(render, names="value")
    tools.observe(render, names="value")
    future.observe(render, names="value")
    refresh()
    display(w.VBox([dataset, query, choices, override, load, cut, tools, future, out]))
    return {"dataset": dataset, "choices": choices, "query": query, "override": override,
            "load": load, "cut": cut, "future": future, "tools": tools, "current": current}


def annotation_template(candidates, dataset):
    """Blank manual labels; keyword hits are never auto-labelled as delegation."""
    columns = [c for c in ["session_id", "turn_id", "_event_index", "turn_number", "preview"] if c in candidates]
    result = candidates[columns].copy()
    result.insert(0, "dataset", dataset)
    for name in ["signal_type", "responsibility", "authority", "scope", "update_type",
                 "evidence_span", "uncertainty", "acceptable_next_actions", "reviewer", "notes"]:
        result[name] = ""
    return result
