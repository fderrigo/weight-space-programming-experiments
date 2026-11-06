"""Measure native decoder margins before any new preservation transform."""
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
STEM = 'late_native_margins'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
rows = meta['records']
gate_report = json.loads((RDIR / 'late_selector_fact_gates.json').read_text('utf8'))
correct = set(gate_report['spec']['correct_native_unit_questions'])
indices = [i for i, r in enumerate(rows[:4693]) if r['kind'] == 'external' or (r['kind'] == 'minimal_native' and r['id'] in correct)]
assert len(indices) == 3267
history = {}
for sequence in meta['sequences']:
    for local, pos in enumerate(sequence['positions']):
        index = sequence['offset'] + local
        if index in indices:
            history[index] = sorted(set(sequence['ids'][:pos + 1]))
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'A compensation that is useful in Euclidean residual distance can reverse a small native decision margin. Measure native original-versus-candidate token margins before choosing a direct algebraic preservation transform.',
    'sample': 'Exactly3267 previously declared native guards with original residuals; no new questions or final data.',
    'rival_rule': 'Edited winner if different from original winner; otherwise the edited runner-up. No response-based or parameter search.',
    'margin': 'Raw effective-decoder margin with the unchanged repetition penalty. Keep original and edited repetition factors separately. A fixed linear constraint is valid only when both relevant signs give the same factors.',
    'direction_projection': 'Record the margin direction orthogonal to the25 carrier directions, and normalize it. The requested margin change still uses the actual unprojected residual difference.',
    'original_native_stops': 'Native stop preferences may occur among the measured original decisions; no new EOS target or stopping rule is prescribed.',
    'new_weights_modified': False, 'free_generation_performed': False,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'final_set_read': False,
    'basis_sha256': hashlib.sha256((ROOT / 'patches/late_selector_fact_basis.safetensors').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, R = [torch.from_numpy(z[k][indices].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'carrier_concept_states_original.npz') as z:
    original = torch.from_numpy(z['R'][indices].copy())
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
basis = load_file(ROOT / 'patches/late_selector_fact_basis.safetensors')
chosen = basis['reassigned_neurons']
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
edited = R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
notes, directions, desired_changes = [], [], []
for begin in range(0, len(indices), 16):
    old_scores = original[begin:begin + 16] @ E.T
    new_scores = edited[begin:begin + 16] @ E.T
    for local in range(len(old_scores)):
        j = begin + local
        seen = history[indices[j]]
        old_scores[local, seen] = torch.where(old_scores[local, seen] < 0, old_scores[local, seen] * 1.1, old_scores[local, seen] / 1.1)
        new_scores[local, seen] = torch.where(new_scores[local, seen] < 0, new_scores[local, seen] * 1.1, new_scores[local, seen] / 1.1)
        winner = int(old_scores[local].argmax())
        first, second = new_scores[local].topk(2).indices.tolist()
        rival = first if first != winner else second
        original_factors, edited_factors = [], []
        for token in [winner, rival]:
            original_factors.append((1.1 if float(original[j] @ E[token]) < 0 else 1 / 1.1) if token in seen else 1)
            edited_factors.append((1.1 if float(edited[j] @ E[token]) < 0 else 1 / 1.1) if token in seen else 1)
        vector = original_factors[0] * E[winner].double() - original_factors[1] * E[rival].double()
        projected = vector - B @ (B.T @ vector)
        length = projected.norm()
        assert length > 1e-8
        directions.append(projected / length)
        desired_changes.append(float((original[j].double() - edited[j].double()) @ vector / length))
        notes.append({'record_index': indices[j], 'id': rows[indices[j]]['id'], 'kind': rows[indices[j]]['kind'],
                      'original_token': winner, 'edited_token': first, 'rival_token': rival,
                      'same_original_token': winner == first, 'linear_penalty_factors_valid': original_factors == edited_factors,
                      'original_raw_margin': float(old_scores[local, winner] - old_scores[local, rival]),
                      'edited_raw_margin': float(new_scores[local, winner] - new_scores[local, rival])})
    if begin % 512 == 0:
        print('native decoder margins', begin, '/', len(indices), flush=True)
np.savez_compressed(RDIR / f'{STEM}.npz', record_indices=np.array(indices),
                    margin_directions=torch.stack(directions).numpy(), desired_margin_changes=np.array(desired_changes),
                    candidate_residuals=edited.numpy())
report = {'spec': spec, 'states': len(indices), 'same_original_tokens': sum(r['same_original_token'] for r in notes),
          'changed_native_tokens': sum(not r['same_original_token'] for r in notes),
          'fixed_linear_penalty_valid': sum(r['linear_penalty_factors_valid'] for r in notes),
          'records': notes, 'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest()}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'records']}, flush=True)
