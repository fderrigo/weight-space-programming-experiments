"""Read-only coverage of fact gates on previously recorded narrative histories."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'narrative_fact_gate_coverage'
assert not (RDIR / f'{STEM}.json').exists()
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
meta = read('progress_carrier_states.json')
sequences = [s for s in meta['sequences'] if s['kind'] == 'narrative']
indices = [i for s in sequences for i in range(s['offset'], s['offset'] + s['states'])]
assert len(indices) == 472 and len(sequences) == 2
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, values = z['X'][indices], z['selected_values'][indices]
diagnostic = read('late_native_recovery_diagnostic.json')
cells = diagnostic['cells']
classes = read('late_selector_geometry.json')['class_coordinates']
with safe_open(ROOT / 'patches/late_native_recovery_basis.safetensors', framework='np') as f:
    ids = f.get_tensor('reassigned_neurons')[:diagnostic['fact_neurons']]
    gate = f.get_tensor('gate_after')[ids]
    up = f.get_tensor('up_after')[ids]
logits = X @ gate.T
sigmoid = np.exp(-np.logaddexp(0, -logits))
raw = logits * sigmoid * (X @ up.T)
F = raw[:, 0::3] - raw[:, 1::3] - raw[:, 2::3]
assert F.shape == (472, 46)
reports = []
offset = 0
for sequence in sequences:
    count = sequence['states']
    block, selected_values = F[offset:offset + count], values[offset:offset + count]
    active = np.abs(block) > .05
    chosen = selected_values.argmax(-1)
    outside = selected_values.max(-1) <= .5
    starts = []
    for j in range(min(12, count)):
        record = meta['records'][sequence['offset'] + j]
        pos = sequence['positions'][j]
        next_token = sequence['ids'][pos + 1] if pos + 1 < len(sequence['ids']) else None
        starts.append({'subtoken': j, 'next_token_in_saved_history': next_token,
                       'selector_class': 'outside' if outside[j] else classes[int(chosen[j])],
                       'largest_class_value': float(selected_values[j].max()),
                       'max_abs_fact_gate': float(np.abs(block[j]).max()),
                       'active_cells': [{'fact': cells[k]['fact'], 'previous_token': cells[k]['previous_token'],
                                         'target_token': cells[k]['target_token'], 'activation': float(block[j, k])}
                                        for k in np.flatnonzero(active[j])]})
    reports.append({'id': sequence['id'], 'states': count,
                    'selector_counts': {label: int(sum(('outside' if o else classes[int(c)]) == label
                                                     for o, c in zip(outside, chosen)))
                                        for label in ['outside'] + classes},
                    'states_with_any_active_fact_gate': int(active.any(-1).sum()),
                    'max_abs_fact_gate': float(np.abs(block).max()), 'first_twelve_states': starts})
    offset += count
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'decision_by': 'Codex', 'code_sha256': sha(Path(__file__)),
          'basis_sha256': sha(ROOT / 'patches/late_native_recovery_basis.safetensors'),
          'source_metadata_sha256': sha(RDIR / 'progress_carrier_states.json'),
          'source_states_sha256': sha(RDIR / 'late_selector_states.npz'),
          'scope': 'Same measured parent states and unchanged gate/up rows on the two original narrative histories. The first response position has the ordinary shared prompt; later positions are counterfactual histories for newer candidates.',
          'thresholds': {'active_gate_abs': .05, 'inside_class_value': .5},
          'weights_changed': False, 'free_generation_performed': False, 'counts_as_correction': False,
          'final_set_read': False, 'reports': reports}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', record_indices=np.array(indices), fact_features=F,
                    selector_values=values)
for report in reports:
    print(report['id'], 'active', report['states_with_any_active_fact_gate'], '/', report['states'],
          'classes', report['selector_counts'], 'start', report['first_twelve_states'][0], flush=True)
