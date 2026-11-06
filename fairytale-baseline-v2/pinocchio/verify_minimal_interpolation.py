"""Check the block omission identity against direct omitted-row solves."""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
torch.set_grad_enabled(False)
torch.set_num_threads(4)
b = load_file(ROOT / 'patches/minimal_fact_completion_basis.safetensors')
K, V = b['keys'].double(), b['values'].double()
records = json.loads((ROOT / 'results/minimal_fact_states.json').read_text('utf8'))['records']
groups = []
for i, row in enumerate(records):
    if row['subtoken'] == 0:
        groups.append([])
    groups[-1].append(i)
with np.load(ROOT / 'results/minimal_interpolation_diagnostic.npz') as z:
    predicted = torch.from_numpy(z['unprojected_loo'].copy()).double()
checks = []
for number in [0, len(groups) // 2, len(groups) - 1]:
    omitted = groups[number]
    kept = [i for i in range(len(K)) if i not in omitted]
    direct = K[omitted] @ torch.linalg.pinv(K[kept], rtol=1e-6) @ V[kept]
    error = float((direct - predicted[omitted]).norm() / direct.norm())
    assert error < 1e-4, (number, error)
    checks.append({'question': records[omitted[0]]['question'],
                   'relative_error_against_direct_omitted_solve': error})
report = {'checks': checks, 'identity_verified': True, 'weights_changed': False,
          'training': False, 'gradients': False, 'final_set_read': False}
(ROOT / 'results/minimal_interpolation_identity_check.json').write_text(
    json.dumps(report, indent=2), encoding='utf8')
print(report, flush=True)
