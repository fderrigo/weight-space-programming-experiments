"""Fixed quiet reference and minimum measured-contribution value coordinate."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_reference_geometry'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
collection = json.loads((RDIR / 'late_reference_inputs.json').read_text('utf8'))
assert collection['completed']
rows = meta['records']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
native = list(range(288, 4693))
scale_fit = [i for i, r in enumerate(rows) if r.get('source') == 'bounded_native_inputs.json']
assert len(scale_fit) == 1352
with np.load(RDIR / 'late_reference_inputs.npz') as z:
    incoming = torch.from_numpy(z['incoming23'].copy())
    values = torch.from_numpy(z['attention_values23'].copy())
    pre = torch.from_numpy(z['before_mlp23'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
attention_path = ROOT / 'patches/late_reference_parent_attention.safetensors'
assert hashlib.sha256(attention_path.read_bytes()).hexdigest() == collection['parent_attention_sha256']
O = load_file(attention_path)['o_weight']
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Moving a constant reference to attention23 avoids changing intermediate queries/keys and exposes only MLP23 to its RMS effect. Reuse one existing shared value coordinate with the smallest measured native output contribution.',
    'coordinate_rule': 'Minimum mean squared norm of the sum of all seven sharing-head output contributions, on196 earlier factual states plus4405 native states. Compute with the seven-column O Gram matrix and observed attention values.',
    'reference_rule': 'Lowest eigenvector of trace-normalized incoming23, original native final residual, and effective decoder second moments, orthogonal to the existing twelve-dimensional semantic carrier.',
    'amplitude_rule': 'Sixteen times the maximum absolute incoming23 projection on the196+4405 construction states.',
    'attention_change': 'Set one V row to zero and its bias to one. Project O off the reference direction, zero its seven sharing output columns, and write amplitude*reference through the first sharing head.',
    'declared_mlp_conditions_for_next_diagnostic': ['reference_only', 'reference_with_fixed_native_RMS_compensation'],
    'fixed_RMS_compensation_formula': 'Multiply both protected last-MLP input matrices by median(new_RMS/old_RMS) on the1352 earlier native external positions. Compare this one analytically specified factor with no compensation; no amplitude or gain search.',
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'weights_applied_to_model': False, 'free_generation': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'collection_sha256': hashlib.sha256((RDIR / 'late_reference_inputs.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
costs = []
selection = fit + native
for coordinate in range(128):
    group, within = divmod(coordinate, 64)
    indices = [head * 64 + within for head in range(group * 7, (group + 1) * 7)]
    columns = O[:, indices].double()
    selected_values = values[selection][:, indices].double()
    gram = columns.T @ columns
    costs.append(float(((selected_values @ gram) * selected_values).sum(-1).mean()))
coordinate = int(np.argmin(costs))
group, within = divmod(coordinate, 64)
indices = [head * 64 + within for head in range(group * 7, (group + 1) * 7)]
C_in = incoming[selection].double().T @ incoming[selection].double() / len(selection)
C_native = original_R[native].T @ original_R[native] / len(native)
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
C_decoder = torch.zeros((896, 896), dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    C_decoder += chunk.T @ chunk
C_decoder /= len(E)
metric = C_in / C_in.trace() + C_native / C_native.trace() + C_decoder / C_decoder.trace()
complement = torch.linalg.qr(U, mode='complete').Q[:, 12:]
eigenvalues, eigenvectors = torch.linalg.eigh(complement.T @ metric @ complement)
v = complement @ eigenvectors[:, 0]
assert float((U.T @ v).abs().max()) < 1e-10
inherited = incoming.double() @ v
amplitude = float(16 * inherited[selection].abs().max())
new_O = (O.double() - v[:, None] @ (v @ O.double())[None, :]).float()
new_O[:, indices] = 0
new_O[:, indices[0]] = amplitude * v.float()
new_values = values.clone()
new_values[:, indices] = 1
new_pre = incoming + new_values @ new_O.T
old_reconstruction_error = float((incoming + values @ O.T - pre).norm() / pre.norm())
assert old_reconstruction_error < 1e-5
reference_identity_error = float((new_pre.double() @ v - inherited - amplitude).norm() / (inherited + amplitude).norm())
assert reference_identity_error < 1e-5
semantic_error = float(((new_pre - pre).double() @ U).norm() / (pre.double() @ U).norm())
assert semantic_error < 1e-5
old_rms = (pre.square().mean(-1) + 1e-6).sqrt()
new_rms = (new_pre.square().mean(-1) + 1e-6).sqrt()
ratio = new_rms / old_rms
compensation = float(ratio[scale_fit].median())
removed = values[:, indices] @ O[:, indices].T
np.savez_compressed(RDIR / f'{STEM}.npz', reference_direction=v.numpy(),
                    reference_amplitude=np.array(amplitude), protected_carrier=torch.cat([U, v[:, None]], dim=1).numpy(),
                    value_coordinate=np.array(coordinate), sharing_output_indices=np.array(indices), o_after=new_O.numpy(),
                    native_RMS_compensation=np.array(compensation), new_before_mlp23=new_pre.numpy(),
                    value_coordinate_costs=np.array(costs))
report = {'spec': spec, 'value_coordinate': coordinate, 'sharing_output_indices': indices,
          'selected_mean_squared_contribution': costs[coordinate],
          'median_coordinate_mean_squared_contribution': float(np.median(costs)),
          'selected_removed_mean_norm': float(removed[selection].norm(dim=-1).mean()),
          'selected_removed_max_norm': float(removed[selection].norm(dim=-1).max()),
          'reference_amplitude': amplitude, 'maximum_inherited_reference_fraction': float(inherited.abs().max() / amplitude),
          'old_attention_reconstruction_relative_error': old_reconstruction_error,
          'reference_identity_relative_error': reference_identity_error,
          'raw_semantic_coordinate_relative_change': semantic_error,
          'native_RMS_compensation': compensation,
          'RMS_ratio_min': float(ratio.min()), 'RMS_ratio_max': float(ratio.max()),
          'geometry_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'spec'}, flush=True)
