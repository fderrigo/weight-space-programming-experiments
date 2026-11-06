"""Count unchanged recorded native paths, distinct from free responses."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
output = RDIR / 'late_margin_native_paths.json'
assert not output.exists()
margins = json.loads((RDIR / 'late_native_margins.json').read_text('utf8'))
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    predictions = dict(zip(z['native_indices'].tolist(), z['native_predicted_tokens'].tolist()))
original = {r['record_index']: r['original_token'] for r in margins['records']}
paths = []
for sequence in meta['sequences']:
    indices = list(range(sequence['offset'], sequence['offset'] + sequence['states']))
    if sequence['kind'] in ['external', 'minimal_native'] and all(i in predictions for i in indices):
        different = [i for i in indices if predictions[i] != original[i]]
        paths.append({'id': sequence['id'], 'kind': sequence['kind'], 'states': len(indices),
                      'all_native_tokens_same': not different, 'changed_indices': different})
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Reconstructed native decisions on recorded supplied histories, not ordinary generated answers.',
          'groups': {kind: {'paths': sum(r['kind'] == kind for r in paths),
                            'whole_paths_same': sum(r['kind'] == kind and r['all_native_tokens_same'] for r in paths)}
                     for kind in ['external', 'minimal_native']},
          'paths': paths,
          'four_prior_control_divergences': [{'index': i, 'original_token': original[i], 'edited_token': predictions[i]} for i in [1640, 1745, 2162, 2359]],
          'source_sha256': hashlib.sha256((RDIR / 'late_margin_preservation_constraints.npz').read_bytes()).hexdigest(),
          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'new_weights_modified': False, 'free_generation_performed': False, 'final_set_read': False}
output.write_text(json.dumps(report, indent=2), encoding='utf8')
print(report['groups'])
