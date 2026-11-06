"""One fixed prototype compression to fit the24 unchanged prefix addresses."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from late_reference_weights import CONDITION

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'compact_outside_prototypes'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
with np.load(RDIR / f'late_{CONDITION}_reconstruction.npz') as z:
    R = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'late_class_decisions.npz') as z:
    prototypes = torch.from_numpy(z['individual_outside_prototypes_semantic_directions'].copy()).double()
    labels = z['individual_outside_prototypes_labels'].tolist()
assert labels[12:] == ['outside'] * 14
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'The14 separate outside directions reject false positives without reducing factual recognition. Test a single fixed compression to12 outside directions so twelve factual plus twelve outside prototypes could use the24 unchanged template-prefix positions.',
    'merge_rule': 'Twice merge the two closest outside cluster directions by cosine similarity. Represent a cluster by the normalized mean of its original unit directions, weighted equally per original prototype. No factual center changes and no response-based choice of merges.',
    'merges': 2, 'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'attention_address_feasibility_not_tested': True,
    'new_weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'late_class_decisions.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
clusters = [[i] for i in range(12, 26)]
merges = []
for _ in range(2):
    centers = torch.stack([prototypes[group].mean(0) for group in clusters])
    centers /= centers.norm(dim=-1, keepdim=True)
    similarity = centers @ centers.T
    similarity.masked_fill_(torch.tril(torch.ones_like(similarity, dtype=torch.bool)), -float('inf'))
    flat = int(similarity.argmax())
    a, b = divmod(flat, len(clusters))
    merges.append({'left_original_indices': clusters[a], 'right_original_indices': clusters[b], 'cosine_similarity': float(similarity[a, b])})
    joined = clusters[a] + clusters[b]
    clusters = [group for i, group in enumerate(clusters) if i not in [a, b]] + [joined]
outside = torch.stack([prototypes[group].mean(0) for group in clusters])
outside /= outside.norm(dim=-1, keepdim=True)
compact = torch.cat([prototypes[:12], outside])
compact_labels = labels[:12] + ['outside'] * 12
Z = R @ U
predictions = [compact_labels[i] for i in (Z @ compact.T).argmax(-1).tolist()]
reports = []
for split, indices in [('fit', [i for i in range(288) if not rows[i]['id'].startswith('second_')]),
                       ('probe_observed', [i for i in range(288) if rows[i]['id'].startswith('second_')])]:
    good = lambda i: predictions[i] == {'wood_finder': 'wood_donor'}.get(rows[i]['fact'], rows[i]['fact'])
    ids = {rows[i]['id'] for i in indices}
    reports.append({'split': split, 'states': len(indices), 'questions': len(ids),
                    'correct_states': sum(good(i) for i in indices),
                    'correct_first_states': sum(good(i) for i in indices if rows[i]['subtoken'] == 0),
                    'whole_relation_paths': sum(all(good(i) for i in indices if rows[i]['id'] == q) for q in ids)})
external = [i for i, r in enumerate(rows) if r['kind'] == 'external']
first = [s['offset'] for s in meta['sequences'] if s['kind'] == 'external']
np.savez_compressed(RDIR / f'{STEM}.npz', semantic_directions=compact.numpy(), labels=np.array(compact_labels), predictions=np.array(predictions))
report = {'spec': spec, 'merges': merges, 'outside_clusters': clusters, 'classification': reports,
          'external_states': len(external), 'external_classified_outside': sum(predictions[i] == 'outside' for i in external),
          'external_first_states': len(first), 'external_first_classified_outside': sum(predictions[i] == 'outside' for i in first),
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'outside_clusters']}, flush=True)
