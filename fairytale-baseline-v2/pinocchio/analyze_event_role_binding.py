"""Summarize role contrasts and solve one fixed scalar feasibility question."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'event_role_binding_analysis.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


data = json.loads((RDIR / 'event_role_binding.json').read_text('utf8'))
assert data['completed'] and data['weights_restored_in_memory']
original, current = data['reports']
assert current['condition'] == 'late_narrative_origins'
path = RDIR / 'event_role_binding_late_narrative_origins.npz'
assert sha(path) == current['arrays_sha256']
with np.load(path) as z:
    A, R = torch.from_numpy(z['A'].copy()), torch.from_numpy(z['R'].copy())
with np.load(RDIR / 'late_narrative_origins_equations.npz') as z:
    dual, values = torch.from_numpy(z['dual'].copy()).double(), torch.from_numpy(z['values'].copy()).double()
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()

first_slots = []
for seq in data['sequences']:
    if seq['actual_history']:
        continue
    i = seq['offset']
    a, b = original['decisions'][i], current['decisions'][i]
    first_slots.append({'id': seq['id'], 'role': seq['role'], 'original_rank': a['desired_rank'],
                        'previous_rank': b['without_origin_update_desired_rank'], 'current_rank': b['desired_rank'],
                        'previous_winner': b['without_origin_update_top'][0], 'current_winner': b['top'][0],
                        'selector_class': b['selector_class'], 'origin_coefficients': b['origin_coefficients']})

# Fixed feasibility question, declared before calculation: vary only the first
# (puppet) equation's amplitude s within [0, 1]. Require the four new origin
# first pieces and the original construction point to remain first. Preserve
# the previous first decision for all 24 other minimal clauses, including
# incorrect native choices. No automatic search, new unit or stop target.
constraints = []
for seq in data['sequences']:
    if seq['actual_history']:
        continue
    i = seq['offset']
    note = current['decisions'][i]
    is_origin = seq['role'] == 'origin'
    target = note['desired_token'] if is_origin else note['without_origin_update_top'][0]['token']
    constraints.append({'id': seq['id'], 'record_index': i, 'kind': 'retain_origin' if is_origin else 'preserve_previous_decision',
                        'target': target, 'R': R[i].double(), 'coefficient': float(A[i].double() @ dual[0]),
                        'seen': sorted(set(seq['ids'][:seq['positions'][0] + 1]))})
unit = json.loads((RDIR / 'narrative_origin_units.json').read_text('utf8'))
unit_report = next(r for r in unit['reports'] if r['condition'] == 'late_native_recovery')
path = RDIR / 'narrative_origin_units_late_native_recovery.npz'
assert sha(path) == unit_report['arrays_sha256']
with np.load(path) as z:
    old_A, old_R = torch.from_numpy(z['A'].copy()).double(), torch.from_numpy(z['R'].copy()).double()
updated_R = old_R + (old_A @ dual.T) @ values
for i, row in enumerate(unit['records']):
    seq = next(s for s in unit['sequences'] if s['id'] == row['id'])
    target = row['desired_token'] if row['desired_token'] is not None else unit_report['decisions'][i]['top'][0]['token']
    constraints.append({'id': 'construction_' + row['id'] + '_' + str(row['subtoken']), 'record_index': i,
                        'kind': 'retain_constructed_piece_or_native_completion', 'target': target,
                        'R': updated_R[i], 'coefficient': float(old_A[i] @ dual[0]),
                        'seen': sorted(set(seq['ids'][:row['position'] + 1]))})

spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hypothesis': 'The puppet equation transfers to four new origin clauses but also changes a previously correct ending clause. Determine whether one reduced nonnegative amplitude can separate these measured uses without sacrificing the original construction constraints.',
        'variable': 's in [0,1] multiplies only the first rank-one origin equation; the two wood equations remain unchanged',
        'constraints': [{k: v for k, v in c.items() if k not in ['R', 'seen']} for c in constraints],
        'positive_score_inequality': 'Require the desired raw score to be positive, then compare positive-branch repetition-adjusted affine scores against every vocabulary token. Negative competitors cannot beat a positive desired score; hence this is sufficient without assuming their signs.',
        'selection': 'Feasibility only. Compute the intersection of exact affine bounds. No parameter sweep or new candidate is executed by this diagnostic.',
        'source_sha256': sha(RDIR / 'event_role_binding.json'), 'code_sha256': sha(Path(__file__)),
        'training': False, 'optimizer': False, 'gradients': False, 'new_weights': False,
        'whole_response_success_claimed': False, 'reserved_final_used': False}
(RDIR / 'event_role_binding_analysis_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')

lower, upper, bounds = 0.0, 1.0, []
direction = values[0].float() @ E.T
for c in constraints:
    shift = c['coefficient'] * values[0]
    base = c['R'] - shift
    intercept = base.float() @ E.T
    slope = c['coefficient'] * direction
    # Positive branch is exact for any competing positive score, and remains
    # negative for any negative score. The target positivity is checked below.
    intercept[c['seen']] /= 1.1
    slope[c['seen']] /= 1.1
    target = c['target']
    a, b = intercept[target] - intercept, slope[target] - slope
    a[target], b[target] = 0, 0
    positive, negative = b > 1e-7, b < -1e-7
    static_bad = bool(((b.abs() <= 1e-7) & (a < -1e-5)).any())
    lows = -a[positive] / b[positive]
    highs = -a[negative] / b[negative]
    lo = float(lows.max()) if len(lows) else -float('inf')
    hi = float(highs.min()) if len(highs) else float('inf')
    lower_index = int(positive.nonzero().flatten()[lows.argmax()]) if len(lows) else None
    upper_index = int(negative.nonzero().flatten()[highs.argmin()]) if len(highs) else None
    if float(slope[target]) > 1e-7:
        lo = max(lo, float(-intercept[target] / slope[target]))
    elif float(slope[target]) < -1e-7:
        hi = min(hi, float(-intercept[target] / slope[target]))
    elif float(intercept[target]) <= 0:
        static_bad = True
    lower, upper = max(lower, lo), min(upper, hi)
    bounds.append({'id': c['id'], 'kind': c['kind'], 'lower': lo, 'upper': hi, 'static_impossibility': static_bad,
                   'lower_competing_token': lower_index, 'upper_competing_token': upper_index})
    print(c['id'], round(lo, 6), round(hi, 6), 'static_bad', static_bad, flush=True)

feasible = lower < upper and not any(r['static_impossibility'] for r in bounds)
report = {'spec': spec, 'first_slots': first_slots, 'scalar_bounds': bounds,
          'joint_lower': lower, 'joint_upper': upper, 'joint_feasible': feasible,
          'bound_numerics': 'F32 decoder products; final selected matrix would require fresh numerical and free-generation verification.',
          'origin_first_piece_transfer': {'probes': 4, 'current_first': sum(r['current_rank'] == 1 for r in first_slots if r['role'] == 'origin')},
          'no_whole_answer_or_narrative_improvement_counted': True}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Joint scalar interval', lower, upper, 'feasible', feasible, flush=True)
