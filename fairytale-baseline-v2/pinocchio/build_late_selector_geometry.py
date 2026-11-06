"""Declare and reconstruct one last-block selector with RMS compensation."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_selector_geometry'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows, sequences = meta['records'], meta['sequences']
fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
first_fit = [i for i in fit if rows[i]['subtoken'] == 0]
native = list(range(288, 4693))
scale_fit = [i for i, r in enumerate(rows) if r.get('source') == 'bounded_native_inputs.json']
assert len(scale_fit) == 1352
selector_path = RDIR / 'late_template_selector.npz'
selector_report = json.loads((RDIR / 'late_template_selector.json').read_text('utf8'))
with np.load(selector_path) as z:
    selector = {k: torch.from_numpy(z[k].copy()) for k in z.files}
with np.load(RDIR / 'late_selector_inputs.npz') as z:
    incoming = torch.from_numpy(z['incoming23'].copy())
    prefix_X = torch.from_numpy(z['prefix_attention_input23'].copy())
    all_X = torch.from_numpy(z['all_nonprefix_attention_input23'].copy())
    all_sequences = torch.from_numpy(z['all_nonprefix_sequence_indices'].copy())
with np.load(RDIR / 'late_reference_inputs.npz') as z:
    old_pre = torch.from_numpy(z['before_mlp23'].copy())
    old_values = torch.from_numpy(z['attention_values23'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    reference = torch.from_numpy(z['reference_direction'].copy()).double()
    reference_amplitude = float(z['reference_amplitude'])
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy()).double()
model_file = ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors'
with safe_open(model_file, framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Use the measured feasible24-prefix selector in attention23, so only MLP23 follows the new class signal and reference. Native behavior remains an empirical requirement.',
    'class_geometry': 'Twelve lowest eigenvectors of trace-normalized incoming23, original native residual and effective decoder moments, restricted orthogonally to U and the already chosen late reference.',
    'class_amplitude_rule': '.15*median(norm(incoming23 first-fit questions))/max(norm(selector fact values on fit states)). No amplitude search.',
    'reference': 'Reuse the frozen late-reference direction and amplitude52.01305458773786.',
    'attention_edit': 'Use the frozen selected head13 and KV group1. Replace13 K axes and13 V coordinates. Zero both rotary partners of the K memory axes in the other six sharing queries. Project O off the13 new carriers; zero the selected head and changed V columns in sharing heads; write class and constant only through the selected head.',
    'mlp_edit': 'Protect new carriers in gate/up and down. Multiply both protected input matrices by median(new_RMS/old_RMS) on the1352 earlier native positions, as declared in the late-reference experiment.',
    'local_reconstruction': 'Native scaled_dot_product_attention with causal masking on all197 histories, then the last MLP. Canonical shared-prefix vectors may differ slightly from per-sequence F32 forward rounding; these arrays are predictions, not actual full-model results.',
    'native_reconstruction_relative_tolerance': 1e-5,
    'required_next_check': 'Collect all197 actual changed-weight full-model histories before building factual output weights. Preserve simulation discrepancies and use actual collected states for those weights.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'new_model_weights_applied': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'selector_sha256': hashlib.sha256(selector_path.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
construction = incoming[fit + native].double()
C_in = construction.T @ construction / len(construction)
C_native = original_R[native].T @ original_R[native] / len(native)
C_decoder = torch.zeros(896, 896, dtype=torch.float64)
for begin in range(0, len(E), 4096):
    chunk = E[begin:begin + 4096].double()
    C_decoder += chunk.T @ chunk
C_decoder /= len(E)
metric = C_in / C_in.trace() + C_native / C_native.trace() + C_decoder / C_decoder.trace()
existing = torch.cat([U, reference[:, None]], dim=1)
complement = torch.linalg.qr(existing, mode='complete').Q[:, 13:]
eigenvalues, eigenvectors = torch.linalg.eigh(complement.T @ metric @ complement)
V = complement @ eigenvectors[:, :12]
new_B = torch.cat([V, reference[:, None]], dim=1)
B = torch.cat([U, new_B], dim=1)
assert float((B.T @ B - torch.eye(25)).abs().max()) < 1e-10
alpha = float(.15 * incoming[first_fit].norm(dim=-1).median() / selector['actual_values'][fit].norm(dim=-1).max())
parent = load_file(ROOT / 'patches/late_selector_parent_attention.safetensors')
attention = {name: value.clone() for name, value in parent.items()}
head, group = int(selector['query_head']), int(selector['key_value_head'])
coords = selector['value_coordinates']
axes = torch.arange(51, 64)
paired_axes = torch.cat([axes - 32, axes])
for other in range(group * 7, (group + 1) * 7):
    if other != head:
        attention['q_proj.weight'][other * 64 + paired_axes] = 0
        attention['q_proj.bias'][other * 64 + paired_axes] = 0
attention['q_proj.weight'][head * 64:(head + 1) * 64] = selector['q_weight']
attention['q_proj.bias'][head * 64:(head + 1) * 64] = selector['q_bias']
attention['k_proj.weight'][group * 64:(group + 1) * 64] = selector['k_weight']
attention['k_proj.bias'][group * 64:(group + 1) * 64] = selector['k_bias']
attention['v_proj.weight'][group * 64 + coords[:12]] = selector['value_rows']
attention['v_proj.bias'][group * 64 + coords[:12]] = 0
attention['v_proj.weight'][group * 64 + coords[12]] = 0
attention['v_proj.bias'][group * 64 + coords[12]] = 1
O = parent['o_proj.weight'].double()
O = (O - new_B @ (new_B.T @ O)).float()
for other in range(group * 7, (group + 1) * 7):
    O[:, other * 64 + coords] = 0
O[:, head * 64:(head + 1) * 64] = 0
O[:, head * 64 + coords[:12]] = (alpha * V).float()
O[:, head * 64 + coords[12]] = (reference_amplitude * reference).float()
attention['o_proj.weight'] = O
frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 64, 2).float() / 64))


def attention_values(X, weights):
    count = len(X)
    q = (X @ weights['q_proj.weight'].T + weights['q_proj.bias']).view(count, 14, 64).transpose(0, 1)
    k = (X @ weights['k_proj.weight'].T + weights['k_proj.bias']).view(count, 2, 64).transpose(0, 1)
    v = (X @ weights['v_proj.weight'].T + weights['v_proj.bias']).view(count, 2, 64).transpose(0, 1)
    angles = torch.arange(count).float()[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], dim=-1)[None, :]
    q = q * angles.cos() + torch.cat([-q[..., 32:], q[..., :32]], dim=-1) * angles.sin()
    k = k * angles.cos() + torch.cat([-k[..., 32:], k[..., :32]], dim=-1) * angles.sin()
    out = torch.nn.functional.scaled_dot_product_attention(q[None], k.repeat_interleave(7, dim=0)[None], v.repeat_interleave(7, dim=0)[None], is_causal=True)
    return out[0].transpose(0, 1).reshape(count, 896)


native_replay = torch.empty_like(old_values)
modified_values = torch.empty_like(old_values)
for j, sequence in enumerate(sequences):
    X = torch.cat([prefix_X, all_X[all_sequences == j]])
    positions = torch.tensor(sequence['positions'])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    native_replay[begin:end] = attention_values(X, parent)[positions]
    modified_values[begin:end] = attention_values(X, attention)[positions]
    if j % 32 == 0:
        print('local last attention', j + 1, '/', len(sequences), flush=True)
replay_error = float((native_replay - old_values).norm() / old_values.norm())
assert replay_error < 1e-5, replay_error
new_pre = incoming + modified_values @ O.T
old_rms = (old_pre.square().mean(-1) + 1e-6).sqrt()
rms = (new_pre.square().mean(-1) + 1e-6).sqrt()
compensation = float((rms / old_rms)[scale_fit].median())
parent_mlp = load_file(ROOT / 'patches/progress_prototype_carrier_basis.safetensors')
mlp = {}
for name in ['gate', 'up', 'down']:
    W = parent_mlp[name + '_before'].double()
    if name == 'down':
        W = W - new_B @ (new_B.T @ W)
    else:
        scaled = W * gain[None, :]
        W = ((scaled - (scaled @ new_B) @ new_B.T) / gain[None, :]) * compensation
    mlp[name] = W.float()
X = new_pre / rms[:, None] * gain.float()
A = torch.nn.functional.silu(X @ mlp['gate'].T) * (X @ mlp['up'].T)
R = new_pre + A @ mlp['down'].T
selected = modified_values[:, head * 64 + coords[:12]]
class_error = float((R.double() @ V - incoming.double() @ V - alpha * selected.double()).norm() / (alpha * selected.double()).norm())
reference_error = float((R.double() @ reference - incoming.double() @ reference - reference_amplitude).norm() / (incoming.double() @ reference + reference_amplitude).norm())
assert max(class_error, reference_error) < 1e-5
attention_path = ROOT / 'patches/late_selector_attention.safetensors'
mlp_path = ROOT / 'patches/late_selector_mlp.safetensors'
assert not attention_path.exists() and not mlp_path.exists()
save_file({k: v.contiguous() for k, v in attention.items()}, attention_path)
save_file({k: v.contiguous() for k, v in mlp.items()}, mlp_path)
np.savez_compressed(RDIR / f'{STEM}.npz', class_carrier=V.numpy(), reference_direction=reference.numpy(),
                    protected_carrier=B.numpy(), class_alpha=np.array(alpha), reference_amplitude=np.array(reference_amplitude),
                    native_RMS_compensation=np.array(compensation))
np.savez_compressed(RDIR / 'late_selector_reconstruction.npz', incoming23=incoming.numpy(), X=X.numpy(), A=A.numpy(), R=R.numpy(), selected_values=selected.numpy())
erased = R.double() - (R.double() @ B) @ B.T
external = [i for i, r in enumerate(rows) if r['kind'] == 'external']
report = {'spec': spec, 'class_coordinates': selector_report['labels_by_prefix_position'][:12],
          'query_head': head, 'key_value_head': group, 'class_alpha': alpha,
          'reference_amplitude': reference_amplitude, 'native_RMS_compensation': compensation,
          'RMS_ratio_min': float((rms / old_rms).min()), 'RMS_ratio_max': float((rms / old_rms).max()),
          'native_attention_reconstruction_relative_error': replay_error,
          'class_carrier_identity_relative_error': class_error, 'reference_identity_relative_error': reference_error,
          'erased_external_residual_difference_mean': float((erased[external] - original_R[external]).norm(dim=-1).mean()),
          'attention_sha256': hashlib.sha256(attention_path.read_bytes()).hexdigest(),
          'mlp_sha256': hashlib.sha256(mlp_path.read_bytes()).hexdigest(),
          'geometry_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'reconstruction_sha256': hashlib.sha256((RDIR / 'late_selector_reconstruction.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'spec'}, flush=True)
