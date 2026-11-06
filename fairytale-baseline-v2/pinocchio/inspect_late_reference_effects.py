"""Reconstruct the last block after a constant-value edit; compare two fixed gains."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_reference_effects'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
geometry = json.loads((RDIR / 'late_reference_geometry.json').read_text('utf8'))
rows, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'late_reference_inputs.npz') as z:
    incoming = torch.from_numpy(z['incoming23'].copy())
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    parent_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'late_reference_geometry.npz') as z:
    v = torch.from_numpy(z['reference_direction'].copy()).double()
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
    new_pre = torch.from_numpy(z['new_before_mlp23'].copy())
    alpha = float(z['reference_amplitude'])
    compensation = float(z['native_RMS_compensation'])
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).float()
parent_basis = load_file(ROOT / 'patches/progress_prototype_carrier_basis.safetensors')
before = {name: parent_basis[name + '_before'] for name in ['gate', 'up', 'down']}
for name, value in before.items():
    expected = next(r for r in meta['weights']['tensor_differences'] if r['tensor'] == f'model.layers.23.mlp.{name}_proj.weight')
    assert tensor_sha(value) == expected['after_sha256_f32']
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
after = {}
for name in ['gate', 'up']:
    scaled = before[name].double() * gain[None, :]
    after[name] = ((scaled - (scaled @ v)[:, None] @ v[None, :]) / gain[None, :]).float()
after['down'] = (before['down'].double() - v[:, None] @ (v @ before['down'].double())[None, :]).float()
base_path = ROOT / 'patches/late_reference_protected_mlp.safetensors'
assert not base_path.exists()
save_file({name: value.contiguous() for name, value in after.items()}, base_path)
correct_ids = set(json.loads((RDIR / 'reference_fact_gates.json').read_text('utf8'))['spec']['correct_native_unit_questions'])
groups = {'external': [i for i, r in enumerate(rows[:4693]) if r['kind'] == 'external'],
          'correct_minimal_native': [i for i, r in enumerate(rows[:4693]) if r['kind'] == 'minimal_native' and r['id'] in correct_ids],
          'narrative': [i for i, r in enumerate(rows[:4693]) if r['kind'] == 'narrative']}
selected, histories = set(), {}
for sequence in sequences:
    if sequence['kind'] != 'external' and not (sequence['kind'] == 'minimal_native' and sequence['id'] in correct_ids):
        continue
    for local in {0, sequence['states'] - 1}:
        i = sequence['offset'] + local
        selected.add(i)
        histories[i] = sequence['ids'][:sequence['positions'][local] + 1]
arithmetic = next(s for s in sequences if s['kind'] == 'external' and s['id'] == 'second_control_10')
arithmetic_index = arithmetic['offset'] + 4
selected.add(arithmetic_index)
histories[arithmetic_index] = arithmetic['ids'][:arithmetic['positions'][4] + 1]
selected = sorted(selected)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'The new constant changes the RMS denominator even after its direction is removed from MLP inputs. A single geometric median RMS ratio may restore part of the native MLP response.',
    'conditions_fixed_in_geometry_spec': ['reference_only', 'reference_with_fixed_native_RMS_compensation'],
    'fixed_compensation': compensation,
    'reconstruction_scope': 'Attention inputs and Q/K are unchanged. The selected V coordinate is constant for every key; its attended value is one. Reconstruct new post-attention residual and last MLP using existing native modules and actual F32 matrices. This is local numerical reconstruction, not full-model validation or generation.',
    'comparison_after_erasure': 'Apply an algebraic output projection for comparison only. This is not an inference hook or yet a weight-implemented erasure.',
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'new_model_weights_applied': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'geometry_sha256': hashlib.sha256((RDIR / 'late_reference_geometry.npz').read_bytes()).hexdigest(),
    'protected_mlp_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
conditions = {'original': original_R,
              'progress_parent_exact_erasure': parent_R[:4693] - (parent_R[:4693] @ U) @ U.T}
identities = {}
X = new_pre / (new_pre.square().mean(-1) + 1e-6).sqrt()[:, None] * gain.float()
for name, factor in [('reference_only', 1.0), ('reference_with_fixed_native_RMS_compensation', compensation)]:
    gates, ups = after['gate'] * factor, after['up'] * factor
    A = torch.nn.functional.silu(X @ gates.T) * (X @ ups.T)
    R = new_pre + A @ after['down'].T
    expected_ref = incoming.double() @ v + alpha
    reference_error = float((R.double() @ v - expected_ref).norm() / expected_ref.norm())
    semantic_error = float(((R - parent_R).double() @ U.double()).norm() / (parent_R.double() @ U.double()).norm())
    assert reference_error < 1e-5 and semantic_error < 1e-5
    conditions[name + '_exact_erasure'] = (R.double() - (R.double() @ B) @ B.T)[:4693].float()
    array_path = RDIR / f'late_{name}_reconstruction.npz'
    np.savez_compressed(array_path, incoming23=incoming.numpy(), X=X.numpy(), A=A.numpy(), R=R.numpy())
    identities[name] = {'reference_identity_relative_error': reference_error,
                        'raw_semantic_coordinate_relative_change': semantic_error,
                        'arrays_sha256': hashlib.sha256(array_path.read_bytes()).hexdigest(),
                        'last_mlp_sha256': {key: tensor_sha(value) for key, value in [('gate', gates), ('up', ups), ('down', after['down'])]}}
    print('Reconstructed', name, 'positions', len(R), flush=True)
predictions, reports = {}, []
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
for name, residual in conditions.items():
    predicted = {}
    for begin in range(0, len(selected), 16):
        indices = selected[begin:begin + 16]
        rb = residual[indices]
        scores = (rb / (rb.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for j, i in enumerate(indices):
            seen = sorted(set(histories[i]))
            scores[j, seen] = torch.where(scores[j, seen] < 0, scores[j, seen] * 1.1, scores[j, seen] / 1.1)
            predicted[i] = int(scores[j].argmax())
    predictions[name] = predicted
    reports.append({'condition': name,
                    'groups': {kind: {'states': len(indices),
                                      'mean_residual_change': float((residual[indices] - original_R[indices]).norm(dim=-1).mean()),
                                      'maximum_residual_change': float((residual[indices] - original_R[indices]).norm(dim=-1).max())}
                               for kind, indices in groups.items()},
                    'native_selected_positions': len(selected),
                    'same_original_next_token': sum(predicted[i] == predictions['original'][i] for i in selected),
                    'arithmetic_next_token': predicted[arithmetic_index],
                    'arithmetic_next_text': t.decode([predicted[arithmetic_index]])})
    print(reports[-1], flush=True)
report = {'spec': spec, 'conditions': reports, 'reconstruction_checks': identities,
          'selected_positions': selected, 'arithmetic_position': arithmetic_index,
          'interpretation': 'Native residual and next-token diagnostics on fixed recorded histories. The full answer and narrative behavior remain untested.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
