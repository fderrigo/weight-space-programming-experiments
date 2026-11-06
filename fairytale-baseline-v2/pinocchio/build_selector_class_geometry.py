"""One fixed quiet subspace for native-attention class values."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_class_geometry'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
first_fit = [i for i in fit if rows[i]['subtoken'] == 0]
native = list(range(288, 4693))
with np.load(RDIR / 'progress_attention_states.npz') as z:
    incoming = torch.from_numpy(z['incoming22'].copy()).double()
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    residual = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
selector = json.loads((RDIR / 'template_attention_selector.json').read_text('utf8'))
classes = list(selector['slots_by_class'])
facts = [c for c in classes if c != 'outside']
with np.load(RDIR / 'template_attention_selector.npz') as z:
    probabilities = torch.from_numpy(z['actual_values'].copy()).double()
fact_values = probabilities[:, [classes.index(c) for c in facts]]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'formula': 'Take twelve lowest eigenvectors of the sum of trace-normalized incoming22, original-native-final, and effective-decoder second moments, restricted to the orthogonal complement of the existing semantic carrier.',
    'amplitude_rule': '.15*median(norm(incoming22 first-fit questions))/max(norm(fact probabilities on fit states)).',
    'class_coordinates': facts, 'outside_value_is_zero': True,
    'rank': 12, 'amplitude_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'weights_modified': False, 'free_generation_performed': False, 'final_set_read': False,
    'selector_sha256': hashlib.sha256((RDIR / 'template_attention_selector.npz').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
C_incoming = incoming[fit + native].T @ incoming[fit + native] / (len(fit) + len(native))
C_native = residual[native].T @ residual[native] / len(native)
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
C_decoder = torch.zeros((896, 896), dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    C_decoder += chunk.T @ chunk
C_decoder /= len(E)
metric = C_incoming / C_incoming.trace() + C_native / C_native.trace() + C_decoder / C_decoder.trace()
complement = torch.linalg.qr(U, mode='complete').Q[:, 12:]
eigenvalues, eigenvectors = torch.linalg.eigh(complement.T @ metric @ complement)
V = complement @ eigenvectors[:, :12]
assert float((U.T @ V).abs().max()) < 1e-10
alpha = float(.15 * incoming[first_fit].norm(dim=-1).median() / fact_values[fit].norm(dim=-1).max())
signal = alpha * fact_values
inherited = incoming @ V
reports = {}
for kind in ['factual', 'external', 'minimal_native', 'native_unit', 'narrative', 'outside_prototype']:
    indices = [i for i, r in enumerate(rows) if r['kind'] == kind]
    reports[kind] = {'states': len(indices), 'median_inherited_norm': float(inherited[indices].norm(dim=-1).median()),
                     'max_inherited_norm': float(inherited[indices].norm(dim=-1).max()),
                     'median_signal_norm': float(signal[indices].norm(dim=-1).median())}
np.savez_compressed(RDIR / f'{STEM}.npz', class_carrier=V.numpy(), protected_carrier=torch.cat([U, V], dim=1).numpy(),
                    alpha=np.array(alpha), eigenvalues=eigenvalues.numpy())
report = {'spec': spec, 'alpha': alpha, 'groups': reports,
          'maximum_carrier_cross_inner_product': float((U.T @ V).abs().max()),
          'geometry_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({'alpha': alpha, 'groups': reports}, flush=True)
