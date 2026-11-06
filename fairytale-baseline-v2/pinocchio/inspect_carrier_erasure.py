"""Algebraic read-and-erase diagnostic, before applying any new weight patch."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'carrier_erasure'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta_path = RDIR / 'progress_carrier_states.json'
meta = json.loads(meta_path.read_text('utf8'))
assert meta['completed']
rows = meta['records']
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    X = torch.from_numpy(z['X'].copy())
    A = torch.from_numpy(z['A'].copy())
    parent_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
P = X.double() / gain
PU = P @ U
expected = parent_R.double() @ U
rms = (expected * PU).sum(-1) / PU.square().sum(-1)
identity_error = float((expected - PU * rms[:, None]).norm() / expected.norm())
assert identity_error < 1e-5 and (rms > 0).all()
Pn = P - PU @ U.T
fit = [i for i, r in enumerate(rows[:4693]) if r.get('source') == 'bounded_native_inputs.json']
assert len(fit) == 1352 and all(rows[i]['kind'] == 'external' for i in fit)
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'The carrier itself changes native continuation before fact writing. Diagnose final-output removal of the carrier, and whether its raw scale can be read geometrically from native normalized coordinates.',
    'projection': 'R -> R - (R U) U^T. This algebraic reference is not an implemented inference hook.',
    'scale_readouts': ['mean scale times mean native direction', 'fixed first 128 singular directions of native normalized inputs'],
    'fit_positions': 1352, 'fit_source': 'bounded_native_inputs.json external positions only',
    'linear_scale_rank': 128, 'carrier_removed_before_scale_readout': True,
    'physical_realization_if_valid': 'For each carrier coordinate, a paired SiLU(+P w)-SiLU(-P w) multiplied by P U_j gives (P w)*(P U_j). Output coefficients -U_j implement an approximate raw-carrier removal using 24 existing neurons.',
    'parameter_search': False, 'training': False, 'gradients': False, 'optimizer': False,
    'diagnostic_only': True, 'weights_modified': False, 'free_generation_performed': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
mean = Pn[fit].mean(0)
w_mean = rms[fit].mean() * mean / mean.square().sum()
u, s, vh = torch.linalg.svd(Pn[fit], full_matrices=False)
rank = spec['linear_scale_rank']
w_linear = vh[:rank].T @ ((u[:, :rank].T @ rms[fit]) / s[:rank])
w_linear -= U @ (U.T @ w_linear)
scales = {'mean': P @ w_mean, 'linear128': P @ w_linear}
groups = {'fit_external': fit,
          'other_external': [i for i, r in enumerate(rows[:4693]) if r['kind'] == 'external' and i not in fit]}
groups.update({kind: [i for i, r in enumerate(rows) if r['kind'] == kind]
               for kind in ['factual', 'minimal_native', 'native_unit', 'narrative', 'outside_prototype']})
scale_reports = {}
for name, values in scales.items():
    scale_reports[name] = {kind: {'states': len(indices),
                                 'mean_absolute_relative_error': float(((values[indices] - rms[indices]).abs() / rms[indices]).mean()),
                                 'max_absolute_relative_error': float(((values[indices] - rms[indices]).abs() / rms[indices]).max()),
                                 'nonpositive': int((values[indices] <= 0).sum())}
                           for kind, indices in groups.items()}
    print('scale', name, json.dumps(scale_reports[name]), flush=True)
basis = load_file(ROOT / 'patches/progress_prototype_carrier_basis.safetensors')
chosen = basis['reassigned_neurons']
new_A = A[:4693].clone()
new_A[:, chosen] = torch.nn.functional.silu(X[:4693] @ basis['gate_after'][chosen].T) * (X[:4693] @ basis['up_after'][chosen].T)
full_R = parent_R[:4693] - A[:4693] @ basis['down_before'].T + new_A @ basis['down_after'].T
parent = parent_R[:4693].double()
conditions = {'original': original_R.double(), 'parent': parent,
              'parent_exact_erasure': parent - (parent @ U) @ U.T,
              'parent_linear_erasure': parent - (PU[:4693] * scales['linear128'][:4693, None]) @ U.T,
              'prototype_patch': full_R.double(),
              'prototype_exact_erasure': full_R.double() - (full_R.double() @ U) @ U.T}
selected = list(range(288))
histories = {}
for sequence in meta['sequences']:
    if sequence['kind'] in ['factual', 'outside_prototype']:
        continue
    for j in {0, sequence['states'] - 1}:
        i = sequence['offset'] + j
        selected.append(i)
        histories[i] = sequence['ids'][:sequence['positions'][j] + 1]
selected = sorted(set(selected))
minimal = json.loads((RDIR / 'minimal_fact_states.json').read_text('utf8'))['records']
histories.update({i: r['history'] for i, r in enumerate(minimal)})
targets = {i: r['token'] for i, r in enumerate(minimal)}
predictions, ranks = {}, {}
for name, residual in conditions.items():
    pred, rank_values = {}, {}
    for begin in range(0, len(selected), 16):
        indices = selected[begin:begin + 16]
        rb = residual[indices].float()
        logits = (rb / (rb.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for j, i in enumerate(indices):
            seen = sorted(set(histories[i]))
            logits[j, seen] = torch.where(logits[j, seen] < 0, logits[j, seen] * 1.1, logits[j, seen] / 1.1)
            pred[i] = int(logits[j].argmax())
            if i < 288:
                rank_values[i] = int((logits[j] > logits[j, targets[i]]).sum()) + 1
    predictions[name], ranks[name] = pred, rank_values
    print('decoded', name, len(selected), flush=True)
reports = []
for name, values in conditions.items():
    factual = []
    for split, indices in [('earlier_fifty', [i for i in range(288) if not rows[i]['id'].startswith('second_')]),
                            ('second_observed', [i for i in range(288) if rows[i]['id'].startswith('second_')])]:
        ids = sorted({rows[i]['id'] for i in indices})
        factual.append({'split': split, 'states': len(indices), 'questions': len(ids),
                        'desired_tokens_first': sum(ranks[name][i] == 1 for i in indices),
                        'first_tokens_first': sum(ranks[name][i] == 1 for i in indices if rows[i]['subtoken'] == 0),
                        'all_steps_first': sum(all(ranks[name][i] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
    native = []
    for kind in ['external', 'minimal_native', 'native_unit', 'narrative']:
        indices = [i for i in selected if rows[i]['kind'] == kind]
        all_indices = [i for i, r in enumerate(rows[:4693]) if r['kind'] == kind]
        native.append({'kind': kind, 'first_and_last_positions': len(indices),
                       'same_original_next_token': sum(predictions[name][i] == predictions['original'][i] for i in indices),
                       'mean_residual_change_all_positions': float((values[all_indices] - original_R[all_indices].double()).norm(dim=-1).mean())})
    fairy = next(r for r in meta['sequences'] if r['kind'] == 'minimal_native' and r['id'] == 'second_fairy_0')
    report = {'condition': name, 'factual': factual, 'native': native,
              'fairy_after_unit_token': predictions[name][fairy['offset']],
              'fairy_after_period_token': predictions[name][fairy['offset'] + 1]}
    reports.append(report)
    print(json.dumps(report), flush=True)
result = {'spec': spec, 'scale_reports': scale_reports, 'conditions': reports,
          'scale_identity_relative_error': identity_error,
          'interpretation': 'Exact erasure and native scale readouts are algebraic diagnostics. No answer correction, runtime projection or new weight edit is claimed.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', mean_readout=w_mean.numpy(), linear_readout=w_linear.numpy(),
                    true_rms=rms.numpy(), mean_scale=scales['mean'].numpy(), linear_scale=scales['linear128'].numpy(),
                    selected_indices=np.array(selected),
                    **{name + '_predictions': np.array([predictions[name][i] for i in selected]) for name in conditions})
