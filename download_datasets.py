#!/usr/bin/env python3
"""Download pinned source releases, keeping access tokens out of project files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import (
    HfApi, get_token, get_hf_file_metadata, hf_hub_url, snapshot_download,
)
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "collabskill": "SALT-NLP/cogym-collabskill-trajectories",
    "swe-chat": "SALT-NLP/SWE-chat",
}
MANIFEST = ROOT / "download_manifest.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"created_utc": now(), "datasets": {}}


def record(manifest, slug, **fields):
    manifest["datasets"].setdefault(slug, {}).update(fields)
    manifest["updated_utc"] = now()
    write_json(MANIFEST, manifest)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(slug, profile="full", refresh=False, workers=4):
    manifest = read_manifest()
    previous = manifest["datasets"].get(slug, {})
    repo = SOURCES[slug]
    # Repository metadata and dataset cards are public even for SWE-chat.
    api = HfApi(token=False)
    revision = None if refresh else previous.get("revision")
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    files = [{"path": f.rfilename, "size_bytes": f.size,
              "blob_id": f.blob_id, "lfs": f.lfs} for f in info.siblings]
    # Convert optional LFS metadata to serializable dictionaries.
    for item in files:
        lfs = item["lfs"]
        if lfs is not None and not isinstance(lfs, dict):
            item["lfs"] = vars(lfs)
    write_json(ROOT / "metadata" / f"{slug}.files.json", files)
    selected = files if slug != "swe-chat" or profile == "full" else [
        f for f in files if "/" not in f["path"]
    ]
    target = ROOT / "raw" / slug
    record(manifest, slug, repo_id=repo,
           source_url=f"https://huggingface.co/datasets/{repo}",
           revision=info.sha, profile=profile if slug == "swe-chat" else "full",
           source_file_count=len(files), source_size_bytes=sum(f["size_bytes"] or 0 for f in files),
           expected_file_count=len(selected), expected_size_bytes=sum(f["size_bytes"] or 0 for f in selected),
           local_directory=str(target.relative_to(ROOT)), status="checking_access")
    print(f"{slug}: {len(selected):,} files, {sum(f['size_bytes'] or 0 for f in selected)/1e9:.3f} GB; revision {info.sha}", flush=True)
    token = False
    if slug == "swe-chat":
        token = get_token()
        if not token:
            record(manifest, slug, status="access_required", access_message="Accept the dataset conditions and run hf auth login locally.")
            print("SWE-chat needs dataset access and a valid local Hugging Face login. No credentials are saved in this folder.", flush=True)
            return False
        try:
            HfApi(token=token).whoami()
            get_hf_file_metadata(hf_hub_url(repo, "sessions.parquet", repo_type="dataset", revision=info.sha), token=token)
        except HfHubHTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            record(manifest, slug, status="access_required", access_message=f"Authenticated access check failed (HTTP {code}). Accept dataset conditions and run hf auth login locally.")
            print(f"SWE-chat access check failed (HTTP {code}). Accept access conditions and run hf auth login locally.", flush=True)
            return False
    target.mkdir(parents=True, exist_ok=True)
    record(manifest, slug, status="downloading", download_started_utc=now(), access_message=None)
    # local_dir avoids an additional duplicate in the global Hugging Face cache.
    snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=target,
                      token=token, allow_patterns=[f["path"] for f in selected],
                      max_workers=workers)
    print(f"{slug}: verifying downloaded file sizes and checksums", flush=True)
    verified = []
    for item in selected:
        path = target / item["path"]
        if not path.is_file() or (item["size_bytes"] is not None and path.stat().st_size != item["size_bytes"]):
            raise RuntimeError(f"Incomplete or unexpected file size: {item['path']}")
        verified.append({"path": item["path"], "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(ROOT / "metadata" / f"{slug}.checksums.json", verified)
    record(manifest, slug, status="complete", completed_utc=now(),
           downloaded_file_count=len(verified), downloaded_size_bytes=sum(f["size_bytes"] for f in verified))
    print(f"{slug}: complete, {len(verified):,} files", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", *SOURCES], default="both")
    parser.add_argument("--swe-profile", choices=["full", "tables"], default="full",
                        help="full includes all raw transcripts; tables downloads only root-level tables/docs")
    parser.add_argument("--refresh", action="store_true", help="Resolve the newest source revision instead of resuming the pinned release")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    ok = True
    for slug in SOURCES if args.dataset == "both" else [args.dataset]:
        try:
            ok = download(slug, args.swe_profile, args.refresh, args.workers) and ok
        except Exception as exc:
            # Never emit request headers, signed URLs, or credential-bearing tracebacks.
            manifest = read_manifest()
            record(manifest, slug, status="download_error", error_type=type(exc).__name__)
            print(f"{slug}: download failed ({type(exc).__name__}); rerun to resume.", file=sys.stderr, flush=True)
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
