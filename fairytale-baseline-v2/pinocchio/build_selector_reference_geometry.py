"""Choose one quiet coordinate for an actual attention-supplied constant."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'selector_reference_geometry'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
native = list(range(288, 4693))
with np.load(RDIR / 'progress_attention_states.npz') as z:
    incoming = torch.from_numpy(z['incoming22'].copy()).double()
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    residual = torch.from_numpy(z['R'].copy()).double()
with np.load(RDIR / 'selector_class_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Replacing only the varying reference with an ideal constant increased usable joint cells from four to forty-two. Implement a constant using the existing attention value bias and a single protected quiet coordinate.',
    'geometry_formula': 'Lowest eigenvector of the same three trace-normalized second moments used for the class carrier, restricted to the complement of all twenty-four existing carried directions.',
    'amplitude_rule': 'Sixteen times the maximum absolute incoming projection on the earlier fifty factual questions and all 4405 native states. The inherited reference component is therefore bounded by 1/16 on this construction sample.',
    'constant_formula': 'Block22 KV-head0 value coordinate12 has zero weights and bias1. Attention weights sum to one; head6 O column12 writes amplitude*v. Suppress that value coordinate in the other sharing heads and protect v through the remaining blocks.',
    'amplitude_multiplier': 16.0, 'rank': 1,
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'final_set_read': False,
    'source_counterfactual_sha256': hashlib.sha256((RDIR / 'selector_reference_counterfactual.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
C_in = incoming[fit + native].T @ incoming[fit + native] / (len(fit) + len(native))
C_native = residual[native].T @ residual[native] / len(native)
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
C_decoder = torch.zeros((896, 896), dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    C_decoder += chunk.T @ chunk
C_decoder /= len(E)
metric = C_in / C_in.trace() + C_native / C_native.trace() + C_decoder / C_decoder.trace()
complement = torch.linalg.qr(B, mode='complete').Q[:, 24:]
eigenvalues, eigenvectors = torch.linalg.eigh(complement.T @ metric @ complement)
v = complement @ eigenvectors[:, 0]
assert float((B.T @ v).abs().max()) < 1e-10
inherited = incoming @ v
amplitude = float(16 * inherited[fit + native].abs().max())
assert amplitude > 0
np.savez_compressed(RDIR / f'{STEM}.npz', reference_direction=v.numpy(), reference_amplitude=np.array(amplitude),
                    protected_carrier=torch.cat([B, v[:, None]], dim=1).numpy())
report = {'spec': spec, 'reference_amplitude': amplitude,
          'fit_max_relative_inherited': float(inherited[fit + native].abs().max() / amplitude),
          'all_max_relative_inherited': float(inherited.abs().max() / amplitude),
          'geometry_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'spec'}, flush=True)
