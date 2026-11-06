"""Report head ablations, including non-monotonic group effects."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'fact_read_heads_summary.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


data = json.loads((RDIR / 'fact_read_heads.json').read_text('utf8'))
assert data['completed'] and len(data['records']) == 18
assert data['spec']['code_sha256'] == sha(ROOT / 'trace_fact_read_heads.py')
summary = []
interactions = []
for plan in data['spec']['plans']:
    selected = [r for r in data['records'] if r['block'] == plan['block'] and r['kind'] == plan['kind']]
    assert len(selected) == 6
    conditions = []
    for condition in data['spec']['head_groups']:
        trials = [(r, next(x for x in r['head_ablations'] if x['condition'] == condition)) for r in selected]
        conditions.append({'condition': condition, 'native_token_changed': sum(t['top_id'] != r['baseline']['top_id'] for r, t in trials),
                           'total': 6, 'mean_margin_change': sum(t['margin_change'] for _, t in trials) / 6,
                           'changed_ids': [r['id'] for r, t in trials if t['top_id'] != r['baseline']['top_id']]})
    summary.append({**plan, 'conditions': conditions})
for row in data['records']:
    changed = [x['condition'] for x in row['head_ablations'] if x['top_id'] != row['baseline']['top_id']]
    full_changed = row['full_block_ablation']['top_id'] != row['baseline']['top_id']
    individual = [x for x in changed if x.startswith('head_')]
    if (changed and not full_changed) or (full_changed and not individual):
        interactions.append({'id': row['id'], 'block': row['block'], 'kind': row['kind'],
                             'full_block_changed_token': full_changed, 'changed_subset_conditions': changed,
                             'interpretation': 'A subset changes the token while the full ablation does not.' if not full_changed else 'The full ablation changes the token but no single-head ablation does.'})
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'code_sha256': sha(Path(__file__)), 'source_sha256': sha(RDIR / 'fact_read_heads.json'),
          'decisions': 18, 'head_or_kv_group_ablations': 288,
          'max_expanded_mask_relative_error': max(c['expanded_mask_native_relative_error'] for c in data['controls']),
          'max_full_block_margin_difference': max(max(c['full_block_margin_differences_from_previous']) for c in data['controls']),
          'summary': summary, 'interaction_cases': interactions,
          'weights_changed': False, 'free_generation': False, 'counts_as_correction': False,
          'candidate_selected_or_constructed': False, 'new_weight_only_corrections': 0,
          'limitation': 'These are next-token measurements on observed reference-assisted histories. Subset effects need not add, and a token change need not be a semantic error. No whole-answer gains are claimed.'}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
for item in summary:
    print(item['block'], item['kind'], [(x['condition'], x['native_token_changed']) for x in item['conditions'] if x['native_token_changed']])
print('Interaction cases', len(interactions))
