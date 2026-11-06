"""Localize observed selector drift without changing any model weights."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'third_selector_drift'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(2)
meta = json.loads((RDIR / 'third_mechanism_states.json').read_text('utf8'))
rows, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'third_mechanism_states_original.npz') as z:
    prefix_mean = torch.from_numpy(z['prefix_values21'].copy()).double()
with np.load(RDIR / 'third_mechanism_states_late_selector_parent.npz') as z:
    incoming = torch.from_numpy(z['incoming23'].copy())
    actual_values = torch.from_numpy(z['selector_values'].copy())
with np.load(RDIR / 'progress_carrier_geometry.npz') as z:
    U = torch.from_numpy(z['carrier'].copy()).double()
    B = torch.from_numpy(z['semantic_basis'].copy()).double()
    Pmu = torch.from_numpy(z['mean_projection'].copy()).double()
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
    alpha = 4 * float(z['alpha'])
with np.load(RDIR / 'compact_outside_prototypes.npz') as z:
    directions = torch.from_numpy(z['semantic_directions'].copy()).double()
    labels = z['labels'].tolist()
with np.load(RDIR / 'late_template_selector.npz') as z:
    selector = {k: torch.from_numpy(z[k].copy()) for k in ['q_weight', 'q_bias', 'k_weight', 'k_bias', 'value_rows']}
with np.load(RDIR / 'late_selector_inputs.npz') as z:
    prefix_X = torch.from_numpy(z['prefix_attention_input23'].copy())
with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
    gain = f.get_tensor('model.layers.23.input_layernorm.weight').float()
classes = sorted(set(labels))
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Observed within-name drift can arise from individual outside prototypes, inherited native content in transported coordinates, or rotary/key coupling. Compare these stages on the same saved original and physical-parent states.',
    'conditions': ['pure_13_centers', 'pure_24_prototypes', 'carried_24_prototypes', 'rotated_prefix_attention', 'actual_full_attention'],
    'scope': 'Existing68 histories and1614positions. Conditional prefix-only attention is checked against recorded full attention before interpreting its mechanism.',
    'formula': 'pure=alpha*prefix_mean*Pmu*B.T; carried=incoming23*U; difference=carried-pure. This difference measures inherited content and accumulated numerical error; it is not assumed to be zero.',
    'weights_modified': False, 'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_already_observed': True, 'final_set_read': False,
    'source_sha256': hashlib.sha256((RDIR / 'third_mechanism_states.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
pure = alpha * prefix_mean @ Pmu @ B.T
carried = incoming.double() @ U
inherited = carried - pure
pred_13 = [classes[i] for i in (pure @ (centers / centers.norm(dim=-1, keepdim=True)).T).argmax(-1).tolist()]
pure_scores, carried_scores = pure @ directions.T, carried @ directions.T
pure_24 = [labels[i] for i in pure_scores.argmax(-1).tolist()]
carried_24 = [labels[i] for i in carried_scores.argmax(-1).tolist()]
X = incoming / (incoming.square().mean(-1) + 1e-6).sqrt()[:, None] * gain
frequencies = 1.0 / (1000000.0 ** (torch.arange(0, 64, 2).float() / 64))


def rotate(value, positions):
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return value * angles.cos() + torch.cat([-value[:, 32:], value[:, :32]], dim=-1) * angles.sin()


q = rotate(X @ selector['q_weight'].T + selector['q_bias'], torch.tensor([r['position'] for r in rows]))
k = rotate(prefix_X @ selector['k_weight'].T + selector['k_bias'], torch.arange(24))
slot_scores = q @ k.T / 8
slot_probabilities = slot_scores.softmax(-1)
prefix_payload = prefix_X @ selector['value_rows'].T
conditional_values = slot_probabilities @ prefix_payload
rotated = [labels[i] for i in slot_probabilities.argmax(-1).tolist()]
facts = labels[:12]
actual = [facts[int(v.argmax())] if float(v.max()) > .5 else 'outside' for v in actual_values]
conditional = [facts[int(v.argmax())] if float(v.max()) > .5 else 'outside' for v in conditional_values]
conditions = {'pure_13_centers': pred_13, 'pure_24_prototypes': pure_24,
              'carried_24_prototypes': carried_24, 'rotated_prefix_winner': rotated,
              'conditional_class_values': conditional, 'actual_full_attention': actual}
canonical = [i for i, r in enumerate(rows) if r['kind'] == 'canonical_fact' and not r['unit_complete']]
first_targets = [i for i in canonical if rows[i]['subtoken'] == 0]
first_controls = [s['offset'] for s in sequences if s['group'] == 'control']
reports = []
for name, predicted in conditions.items():
    reports.append({'condition': name,
                    'canonical_steps_correct': sum(predicted[i] == rows[i]['fact'] for i in canonical),
                    'canonical_steps': len(canonical),
                    'first_targets_correct': sum(predicted[i] == rows[i]['fact'] for i in first_targets),
                    'first_controls_outside': sum(predicted[i] == 'outside' for i in first_controls)})
points = []
for i in canonical:
    ratio = float(inherited[i].norm() / pure[i].norm().clamp_min(1e-12))
    points.append({'index': i, 'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                   'fact': rows[i]['fact'], 'next_token': rows[i]['next_token'],
                   'predictions': {name: pred[i] for name, pred in conditions.items()},
                   'inherited_to_semantic_norm_ratio': ratio,
                   'pure_top_two': pure_scores[i].topk(2).indices.tolist(),
                   'carried_top_two': carried_scores[i].topk(2).indices.tolist(),
                   'actual_wanted_value': float(actual_values[i, facts.index(rows[i]['fact'])])})
report = {'spec': spec, 'comparisons': reports,
          'conditional_vs_actual_values_max_error': float((conditional_values - actual_values).abs().max()),
          'conditional_vs_actual_class_agreement': sum(a == b for a, b in zip(conditional, actual)),
          'positions': len(rows), 'canonical_points': points,
          'interpretation_limit': 'Labels and local attention probabilities are mechanistic diagnostics on supplied or recorded histories, not new generated answers. No native-content removal or selector repair is applied.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', pure=pure.numpy(), carried=carried.numpy(),
                    inherited=inherited.numpy(), conditional_values=conditional_values.numpy())
print({k: v for k, v in report.items() if k not in ['spec', 'canonical_points']}, flush=True)
for p in points:
    if p['id'] in ['third_wood_donor_1', 'third_rescuer_1']:
        print(p, flush=True)
