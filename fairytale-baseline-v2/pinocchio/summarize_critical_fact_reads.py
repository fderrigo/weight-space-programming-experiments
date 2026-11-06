"""Summarize fixed-history causal measurements without declaring answer gains."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'critical_fact_reads_summary.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


data = json.loads((RDIR / 'critical_fact_reads.json').read_text('utf8'))
assert data['completed'] and len(data['records']) == 18
assert data['arrays_sha256'] == sha(RDIR / 'critical_fact_reads.npz')
assert data['records_sha256'] == sha(RDIR / 'critical_fact_reads_records.json')
assert data['spec']['code_sha256'] == sha(ROOT / 'trace_critical_fact_reads.py')
with np.load(RDIR / 'critical_fact_reads.npz') as z:
    masses = z['fact_mass'].copy()
    heads = z['head_values'].copy()
    fact_heads = z['fact_weighted_heads'].copy()
assert masses.shape == (18, 24, 14)
assert heads.shape == fact_heads.shape == (18, 24, 14, 64)
assert np.isfinite(masses).all() and np.isfinite(heads).all() and np.isfinite(fact_heads).all()
assert masses.min() >= 0 and masses.max() <= 1.000001
by_kind = []
changes = []
for kind in ['concept_first', 'concept_last', 'following_native']:
    subset = [(i, r) for i, r in enumerate(data['records']) if r['kind'] == kind]
    assert len(subset) == 6
    blocks = []
    for block in range(24):
        trials = [r['block_ablations'][block] for _, r in subset]
        blocks.append({'block': block,
                       'native_token_changed': sum(t['top_id'] != r['baseline']['top_id'] for t, (_, r) in zip(trials, subset)),
                       'total': 6, 'mean_margin_change': float(np.mean([t['margin_change'] for t in trials])),
                       'median_native_fact_attention_per_head': np.median(masses[[i for i, _ in subset], block], axis=0).tolist()})
    by_kind.append({'kind': kind, 'blocks': blocks})
for i, record in enumerate(data['records']):
    for trial in record['block_ablations']:
        if trial['top_id'] == record['baseline']['top_id']:
            continue
        block = trial['block']
        changes.append({'id': record['id'], 'kind': record['kind'], 'block': block,
                        'expected_piece': record['expected_piece'], 'new_piece': trial['top_piece'],
                        'new_target_rank': trial['target_rank'], 'margin_change': trial['margin_change'],
                        'native_fact_attention_by_head': masses[i, block].tolist(),
                        'fact_component_norm_fraction_by_head': (np.linalg.norm(fact_heads[i, block], axis=-1) / np.maximum(np.linalg.norm(heads[i, block], axis=-1), 1e-12)).tolist()})
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'code_sha256': sha(Path(__file__)), 'source_sha256': sha(RDIR / 'critical_fact_reads.json'),
    'arrays_sha256': data['arrays_sha256'], 'decisions': 18, 'one_block_ablations': 432,
    'max_explicit_mask_relative_error': max(c['explicit_mask_native_state_relative_error'] for c in data['controls']),
    'max_attention_reconstruction_relative_error': max(max(c['head_reconstruction_relative_errors']) for c in data['controls']),
    'by_kind': by_kind, 'token_changes': changes,
    'limitation': 'These are local next-token changes on fixed reference-assisted native histories. A change can be grammatical or synonymous; it is not automatically a factual error. Attention mass does not establish head-level causality. No head or weight candidate is selected here.',
    'new_weight_only_corrections': 0, 'weights_changed': False, 'free_generation': False,
    'counts_as_correction': False, 'candidate_selected_or_constructed': False,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Verified18 decisions and432 single-block ablations.')
for entry in by_kind:
    print(entry['kind'], [(b['block'], b['native_token_changed']) for b in entry['blocks'] if b['native_token_changed']])
for change in changes:
    print(change['id'], change['kind'], change['block'], repr(change['expected_piece']), '->', repr(change['new_piece']))
