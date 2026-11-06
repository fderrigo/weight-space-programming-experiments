"""Acquire only the public official fallback checkpoint, pinned before download."""
import datetime
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models/original'
RDIR = ROOT / 'results'
assert not DEST.exists()
qualification = json.loads((RDIR / 'qwen25_05b_qualification_summary.json').read_text('utf8'))
assert not qualification['gate_passed']
repository = 'Qwen/Qwen3-0.6B'
info = HfApi(token=False).model_info(repository, files_metadata=True)
allowed = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors', 'LICENSE', 'README.md'}
files = [s for s in info.siblings if s.rfilename in allowed]
assert 'model.safetensors' in {s.rfilename for s in files}
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'repository': repository,
        'revision': info.sha, 'reason': 'Original Qwen2.5-0.5B fails authorization discrimination:10/22 unauthorized valid calls. User explicitly allows a small open-weight fallback after qualification failure.',
        'preceding_qualification_sha256': hashlib.sha256((RDIR / 'qwen25_05b_qualification_summary.json').read_bytes()).hexdigest(),
        'thinking_policy': 'Evaluate native non-thinking mode for a deterministic bounded call decision; apply same setting to both future paired models.',
        'original_and_modified_pair_not_mixed_with_qwen25': True,
        'source': 'https://huggingface.co/Qwen/Qwen3-0.6B',
        'official_expected_files': [{'name': s.rfilename, 'bytes': s.size, 'upstream_lfs_sha256': s.lfs.sha256 if s.lfs else None} for s in files]}
(RDIR / 'qwen3_acquisition_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
print('Pinned official model', repository, info.sha, 'files', len(files), flush=True)
snapshot_download(repository, revision=info.sha, local_dir=DEST, allow_patterns=sorted(allowed), token=False, max_workers=3)
verified = []
for row in spec['official_expected_files']:
    path = DEST / row['name']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == row['bytes']
    if row['upstream_lfs_sha256']:
        assert digest == row['upstream_lfs_sha256']
    verified.append({**row, 'sha256': digest})
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'repository': repository, 'revision': info.sha,
          'files': verified, 'official_lfs_hashes_verified': True, 'completed': True, 'training': False, 'weight_editing': False}
(RDIR / 'original_model_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print('Original official files verified.', flush=True)
