"""Repair the missing values of the earlier bounded-neuron construction.

Keep its explicit nonlinear predicates fixed. Write only their residual factual
vectors through the untouched native neurons, while cancelling the change on
the same saved native trajectories. All operations are declared matrix algebra.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
STEM = 'bounded_residual'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
bounded = load_file(ROOT / 'patches/bounded_fact_basis.safetensors')
factual = load_file(ROOT / 'patches/multipoint_protected_basis.safetensors')
with np.load(ROOT / 'results/bounded_selectivity_native_grouped_states.npz') as z:
    X = torch.from_numpy(z['X'].copy()).float()
with np.load(ROOT / 'results/bounded_native_inputs.npz') as z:
    C = torch.from_numpy(z['X'].copy()).float()
V = factual['values'].double()
chosen = set(bounded['reassigned_neurons'].tolist())
remaining = torch.tensor([j for j in range(4864) if j not in chosen])
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensors': [f'model.layers.23.mlp.{name}_proj.weight' for name in ['gate', 'up', 'down']],
    'reason': 'The fixed bounded-neuron construction preserved native responses numerically but missed 40.6% of its declared value vectors. Add the missing residual through the remaining original neurons, preserving native trajectories. This combines the bounded part with a smaller unbounded factual correction; semantic success is not presumed.',
    'source': 'bounded_fact predicates and multipoint_protected factual values, unchanged',
    'native_contexts': len(C), 'factual_contexts': len(X),
    'preservation_formula': 'Hc=remaining original native activations; Ec=current bounded output change. Pc=-Ect*pinv(Hc)t.',
    'factual_formula': 'Q=row_space(Hc); Z=Hk-HkQQt. R=V-current_bounded_change_on_keys-Hk*Pct. Pf=Rt*pinv(Z)t. Add Pc+Pf only to down columns of remaining neurons.',
    'protection_rtol': 1e-6, 'factual_pinv_rtol': .001,
    'training': False, 'gradients': False, 'optimizer': False,
    'new_predicates': False, 'EOS_programmed': False, 'runtime_hooks': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(
    json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')

T = torch.cat([C, X])
original_activation = (torch.nn.functional.silu(T @ bounded['gate_before'].T)
                       * (T @ bounded['up_before'].T))
bounded_activation = (torch.nn.functional.silu(T @ bounded['gate_after'].T)
                      * (T @ bounded['up_after'].T))
original_output = original_activation @ bounded['down_before'].T
bounded_output = bounded_activation @ bounded['down_after'].T
effect = (bounded_output - original_output).double()
reconstruction_error = float((original_activation[len(C):] - factual['keys']).norm()
                             / factual['keys'].norm())
assert reconstruction_error < 1e-5
assert torch.equal(original_activation[:, remaining], bounded_activation[:, remaining])
Hc = original_activation[:len(C), remaining].double()
Hk = original_activation[len(C):, remaining].double()
print('Native projection', Hc.shape, 'factual residual', Hk.shape, flush=True)
U, s, vh = torch.linalg.svd(Hc, full_matrices=False)
rank = int((s > s[0] * spec['protection_rtol']).sum())
Q = vh[:rank]
inverse_native = (Q.T / s[:rank]) @ U[:, :rank].T
preserve = -effect[:len(C)].T @ inverse_native.T
residual = V - effect[len(C):] - Hk @ preserve.T
Z = Hk - (Hk @ Q.T) @ Q
repair = residual.T @ torch.linalg.pinv(Z, rtol=spec['factual_pinv_rtol']).T
correction = (preserve + repair).float()
after = bounded['down_after'].clone()
after[:, remaining] += correction
actual_effect = (bounded_activation @ after.T - original_output).double()
native_change = actual_effect[:len(C)].norm(dim=-1)
diagnostic = {
    'native_contexts': len(C), 'native_rank': rank, 'factual_contexts': len(X),
    'original_state_reconstruction_relative_error': reconstruction_error,
    'old_value_relative_error': float((effect[len(C):] - V).norm() / V.norm()),
    'new_value_relative_error': float((actual_effect[len(C):] - V).norm() / V.norm()),
    'remaining_neuron_correction_norm': float(correction.norm()),
    'total_down_delta_norm': float((after - bounded['down_before']).norm()),
    'old_mean_native_change': float(effect[:len(C)].norm(dim=-1).mean()),
    'new_mean_native_change': float(native_change.mean()),
    'new_max_native_change': float(native_change.max()),
    'free_generation_performed': False, 'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(diagnostic, indent=2), encoding='utf8')
save_file({**{k: v.contiguous() for k, v in bounded.items() if k != 'down_after'},
           'down_after': after.contiguous()}, ROOT / f'patches/{STEM}_basis.safetensors')
print(json.dumps(diagnostic, indent=2), flush=True)
