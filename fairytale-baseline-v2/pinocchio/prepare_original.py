"""Verify the pinned original; optionally fetch it on a new installation.

An existing original directory is always read-only, including on hash mismatch.
No model other than the exact Qwen revision already used here is downloaded.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent.resolve()
DESTINATION = (WORKSPACE / 'little-red-weightspace/models/original').resolve()
AUDIT = json.loads((ROOT / 'results/upstream_all_files_audit.json').read_text('utf8'))
p = argparse.ArgumentParser()
p.add_argument('--download-missing', action='store_true',
               help='Fetch the pinned files only if the original directory does not exist')
a = p.parse_args()


def digest(path, algorithm):
    if algorithm == 'sha256':
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    assert algorithm == 'git_blob_sha1'
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


downloaded = False
if not DESTINATION.exists():
    if not a.download_missing:
        raise SystemExit('Original missing. Use --download-missing on a new installation.')
    staging = DESTINATION.with_name('original.pending').resolve()
    assert DESTINATION.is_relative_to(WORKSPACE)
    assert staging.is_relative_to(WORKSPACE) and staging.parent == DESTINATION.parent
    staging.mkdir(parents=True, exist_ok=True)
    for record in AUDIT['files']:
        name = record['name']
        assert Path(name).name == name
        target = staging / name
        if target.exists() and digest(target, record['algorithm']) == record['remote']:
            continue
        url = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{AUDIT['revision']}/{name}"
        partial = target.with_name(target.name + '.part')
        print('Fetching pinned file', name, flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'pinocchio-weight-edit-reproduction'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
        assert digest(partial, record['algorithm']) == record['remote'], f'Hash mismatch: {name}'
        partial.replace(target)
    assert not DESTINATION.exists(), 'Refuse to replace an original created concurrently'
    print('Verified workspace paths before directory move:', str(staging), '->', str(DESTINATION), flush=True)
    staging.rename(DESTINATION)
    downloaded = True

checks = []
for record in AUDIT['files']:
    path = DESTINATION / record['name']
    assert path.is_file(), f'Missing original file; existing directory will not be changed: {path}'
    actual = digest(path, record['algorithm'])
    assert actual == record['remote'], f'Original mismatch; left unchanged: {path}'
    checks.append({'name': record['name'], 'algorithm': record['algorithm'],
                   'expected': record['remote'], 'actual': actual, 'match': True})
report = {'revision': AUDIT['revision'], 'destination': str(DESTINATION),
          'existing_original_modified': False, 'downloaded_new_installation': downloaded,
          'files': checks}
(ROOT / 'results/original_file_verification.json').write_text(
    json.dumps(report, indent=2), encoding='utf8')
print('Verified', len(checks), 'pinned original files; existing original left unchanged.', flush=True)
