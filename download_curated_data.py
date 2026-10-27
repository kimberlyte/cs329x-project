"""Download pinned curated packages from Hugging Face without replacing local edits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Manifest paths must be nonempty POSIX relative paths.')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or ':' in value or value in ('.', ''):
        raise ValueError('Manifest path escapes its package.')
    return p


def entries(manifest, dataset, root):
    if manifest.get('schema_version') != '1.0':
        raise ValueError('Unsupported curated manifest schema.')
    names = list(manifest['datasets']) if dataset == 'all' else [dataset]
    result, seen = [], set()
    prefix = manifest.get('remote_prefix', '')
    if prefix:
        relative_path(prefix)
    root = Path(root).resolve()
    for name in names:
        package = manifest['datasets'][name]
        local = relative_path(package['local_directory'])
        if local.parts[0] != 'curated':
            raise ValueError('Packages must be installed below curated/.')
        remote = relative_path(package['remote_directory'])
        for item in package['files']:
            path = relative_path(item['path'])
            target = root.joinpath(*local.parts, *path.parts)
            if not target.resolve().is_relative_to(root / 'curated') or target.is_symlink():
                raise ValueError('Symlink or path escapes curated/.')
            if str(target) in seen:
                raise ValueError('Duplicate destination in manifest.')
            seen.add(str(target))
            if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) or type(item['size_bytes']) is not int or item['size_bytes'] < 0:
                raise ValueError('Invalid checksum or file size in manifest.')
            remote_path = '/'.join(x for x in (prefix.strip('/'), str(remote), str(path)) if x)
            result.append((target, remote_path, item))
    if not result:
        raise ValueError('Manifest has no files for the selected dataset.')
    return result


def matches(path, item):
    return path.is_file() and path.stat().st_size == item['size_bytes'] and sha256(path) == item['sha256']


def install(manifest, root=ROOT, dataset='all', repo_id=None, revision=None, verify_only=False, fetch=None):
    root = Path(root).resolve()
    files = entries(manifest, dataset, root)
    # Conflicts are detected before downloading or changing any package file.
    conflicts = [str(p.relative_to(root)) for p, _, item in files if p.exists() and not matches(p, item)]
    if conflicts:
        raise ValueError('Local files differ from the pinned package; preserved without changes: ' + ', '.join(conflicts) + '. Use a separate --destination or move your edits first.')
    missing = [(p, remote, item) for p, remote, item in files if not p.exists()]
    if verify_only:
        if missing:
            raise ValueError('Missing files: ' + ', '.join(str(p.relative_to(root)) for p, _, _ in missing))
        return {'verified': len(files), 'downloaded': 0}
    if not missing:
        return {'verified': len(files), 'downloaded': 0}
    repo_id = repo_id or manifest.get('repo_id')
    revision = revision or manifest.get('revision')
    if not repo_id or not re.fullmatch(r'[^/\s]+/[^/\s]+', repo_id):
        raise ValueError('Hugging Face repo_id is not configured. Supply --repo-id OWNER/DATASET after the team shares access details.')
    if not revision or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Supply --revision with the full 40-character Hugging Face commit hash. A moving branch such as main is not accepted.')
    if fetch is None:
        from huggingface_hub import hf_hub_download

        def fetch(repo, commit, filename):
            return hf_hub_download(repo_id=repo, repo_type='dataset', revision=commit, filename=filename, token=True)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.curated-download.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        with tempfile.TemporaryDirectory(prefix='.curated-stage-', dir=root) as stage:
            staged = []
            for number, (target, remote, item) in enumerate(missing):
                downloaded = Path(fetch(repo_id, revision, remote))
                if not matches(downloaded, item):
                    raise ValueError('Downloaded file failed size/SHA-256 verification: ' + remote)
                temporary = Path(stage) / str(number)
                shutil.copyfile(downloaded, temporary)  # Never modify the HF cache.
                if not matches(temporary, item):
                    raise ValueError('Staged file failed verification: ' + remote)
                staged.append((temporary, target, item))
            # All fetched files have been verified before installation starts.
            for temporary, target, item in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or not target.resolve().is_relative_to(root / 'curated'):
                    raise ValueError('Destination changed during download.')
                try:
                    with target.open('xb') as handle, temporary.open('rb') as source:
                        shutil.copyfileobj(source, handle)
                except FileExistsError:
                    if not matches(target, item):
                        raise ValueError('Destination was edited during download; preserved: ' + str(target))
        return {'verified': len(files), 'downloaded': len(missing)}
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'curated_download_manifest.json')
    parser.add_argument('--dataset', choices=['all', 'collabskill', 'swe-chat', 'midway'], default='all')
    parser.add_argument('--repo-id')
    parser.add_argument('--revision')
    parser.add_argument('--destination', type=Path, default=ROOT, help='Repository/data root; packages are placed below its curated/ folder.')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    try:
        result = install(json.loads(args.manifest.read_text()), args.destination, args.dataset, args.repo_id, args.revision, args.verify_only)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')
    print(f"Verified {result['verified']} package files; downloaded {result['downloaded']} new files. Local edits were not overwritten.")


if __name__ == '__main__':
    main()
