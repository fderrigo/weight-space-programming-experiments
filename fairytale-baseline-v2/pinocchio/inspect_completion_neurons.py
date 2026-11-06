"""Inspect existing sentence-completion contributions; no weights changed."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
torch.set_grad_enabled(False)
torch.set_num_threads(2)
with safe_open(ROOT.parent.parent/'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
 embedding = f.get_slice('model.embed_tokens.weight')
 eperiod = embedding[13:14].float()[0]
 eeos = embedding[151645:151646].float()[0]
 g = f.get_tensor('model.norm.weight').float()
b = load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
W = b['before'] + b['delta']
metadata = json.loads((ROOT/'results/termination_diagnostic.json').read_text('utf8'))['states']
with np.load(ROOT/'results/termination_states.npz') as z:
 A = torch.stack([torch.from_numpy(z[f'A_{i}'].copy()) for i in range(len(metadata))])
completed = [i for i, r in enumerate(metadata) if r['kind'] == 'completed_unit']
control_end = [i for i, r in enumerate(metadata) if r['kind'] == 'control_complete']
period_columns = (eperiod*g) @ W
eos_columns = (eeos*g) @ W
P = A[completed] * period_columns
N = b['keys'] * period_columns
# Inspect only existing positive punctuation contributions, not a new stop rule.
score = P.median(0).values - N.quantile(.95, dim=0)
indices = score.topk(20).indices.tolist()
report = {'weights_changed': False, 'purpose': 'Test whether native MLP columns already distinguish completed factual phrases from every observed incomplete factual state; not a patch or an EOS rule.', 'neurons': []}
for j in indices:
 row = {'index': j, 'score': float(score[j]), 'period_weight': float(period_columns[j]),
        'eos_weight': float(eos_columns[j]), 'completed_period_median': float(P[:,j].median()),
        'incomplete_period_p95': float(N[:,j].quantile(.95)),
        'incomplete_period_max': float(N[:,j].max()),
        'complete_control_period_mean': float((A[control_end,j]*period_columns[j]).mean())}
 report['neurons'].append(row)
(ROOT/'results/completion_neuron_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
