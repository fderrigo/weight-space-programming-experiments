"""Read-only geometry of bounded writes for the ordinary-question states.

Explicit algebraic alternatives, no training, parameter search, or network edit.
Also test the proposed correction of the old predicate's negative-anchor branch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
p = argparse.ArgumentParser()
p.add_argument('--extended', action='store_true')
args = p.parse_args()
STEM = 'bare_predicate_extended' if args.extended else 'bare_predicate_diagnostic'
torch.set_grad_enabled(False)
torch.set_num_threads(2)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_, _, records, _ = load_contexts(ROOT, t)
indices = [i for i, row in enumerate(records) if row['prefix'] == '']
records = [records[i] for i in indices]
with np.load(ROOT / 'results/factorized_feature_states.npz') as z:
    X = torch.from_numpy(z['X'][indices].copy()).double()
with np.load(ROOT / 'results/bounded_selectivity_native_grouped_states.npz') as z:
    C = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/native_second_continuations_states.npz') as z:
    C = torch.cat([C, *[torch.from_numpy(z[f'X_{i}'].copy()).double() for i in range(24)]])
if args.extended:
    with np.load(ROOT / 'results/extended_native_inputs.npz') as z:
        C = torch.cat([C, torch.from_numpy(z['X'].copy()).double()])
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    norm_weight = source.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
# Undo the RMSNorm diagonal. These vectors have essentially constant norm,
# unlike the physical MLP inputs with their learned per-coordinate scaling.
X, C = X / norm_weight, C / norm_weight
distance = ((C.square().sum(-1)[:, None] + X.square().sum(-1)[None])
            - 2 * C @ X.T).clamp_min(0)
duplicate = distance.min(-1).values.sqrt() < 1e-4
C = C[~duplicate]
anchor = X.mean(0)
anchor /= anchor.square().sum()
pa, ca = X @ anchor, C @ anchor
metric = C.T @ C / len(C)
floor = .05 * metric.diag().mean()
metric.diagonal().add_(floor)
directions = torch.linalg.solve(metric, X.T)
gram = X @ directions
dual = directions @ torch.linalg.pinv(gram, hermitian=True, rtol=1e-8)
tokens = sorted({row['token'] for row in records})
token_members = [[i for i, row in enumerate(records) if row['token'] == token] for token in tokens]
Y = torch.zeros((len(X), len(tokens)), dtype=torch.float64)
for j, members in enumerate(token_members):
    Y[members, j] = 1
families = [
    ('cosine_key', X.T / X.square().sum(-1), [[i] for i in range(len(X))]),
    ('covariance_key', directions / gram.diag(), [[i] for i in range(len(X))]),
    ('dual_key', dual, [[i] for i in range(len(X))]),
    ('dual_token', dual @ Y, token_members),
]
fact_groups = {}
for i, row in enumerate(records):
    label = row['fact'].removesuffix('_answer_start')
    label = {'fairy_singular': 'fairy', 'ending_transition': 'ending',
             'wood_finder': 'wood_donor'}.get(label, label)
    fact_groups.setdefault((label, row['token']), []).append(i)
for name, groups in [('covariance_token', token_members),
                     ('covariance_fact_token', list(fact_groups.values()))]:
    centers = torch.stack([X[ix].mean(0) for ix in groups])
    grouped_directions = torch.linalg.solve(metric, centers.T)
    grouped_directions /= (grouped_directions * centers.T).sum(0)
    families.append((name, grouped_directions, groups))
report = {
    'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_read': False, 'final_set_read': False, 'counts_as_correction': False,
    'factual_states': len(X), 'negative_states': len(C),
    'removed_numerically_identical_factual_negatives': int(duplicate.sum()),
    'unscaled_factual_norm_range': [float(X.norm(dim=-1).min()), float(X.norm(dim=-1).max())],
    'positive_anchor_range': [float(pa.min()), float(pa.max())],
    'negative_anchor_range': [float(ca.min()), float(ca.max())],
    'negative_anchor_count': int((ca < 0).sum()),
    'dual_interpolation_error': float((X @ dual - torch.eye(len(X))).norm()),
    'covariance_isotropic_fraction': .05, 'algebraic_pinv_rtol': 1e-8,
    'extended_native_and_narrative_states_included': args.extended,
    'activation_check_uses_physical_f32_weights': True,
    'families': [],
}
arrays = {'X': X.numpy(), 'C': C.numpy(), 'anchor': anchor.numpy(), 'norm_weight': norm_weight.numpy()}
C_physical = (C * norm_weight).float()
anchor_physical = (anchor / norm_weight).float()


for name, D, members in families:
    ps, cs = (X @ D) / pa[:, None], (C @ D) / ca[:, None]
    high = torch.stack([ps[ix, j].min() for j, ix in enumerate(members)])
    same_token = torch.tensor([[row['token'] == records[ix[0]]['token'] for ix in members]
                               for row in records], dtype=torch.bool)
    other = ps.masked_fill(same_token, -torch.inf).max(0).values
    low = torch.maximum(other, cs[ca > .05].max(0).values)
    separated = (high - low > .05)
    eligible = [j for j, ix in enumerate(members) if separated[j] and (pa[ix] > .25).all()]
    results = {'family': name, 'prototype_count': len(members), 'separated': len(eligible)}
    variants = []
    for rectify in [False, True]:
        valid, saved = [], []
        for j in eligible:
            gap = high[j] - low[j]
            a, b, beta = low[j] + .2 * gap, low[j] + .8 * gap, 160.0 / gap
            # Check physical input coordinates and physical stored-F32 weights.
            gates = (torch.stack([beta * (D[:, j] - a * anchor),
                                  beta * (D[:, j] - b * anchor)]) / norm_weight).float()
            cx = C_physical @ anchor_physical
            values = torch.nn.functional.silu(C_physical @ gates.T)
            bounded = (values[:, 0] - values[:, 1]) * cx / float(beta * (b - a))
            if rectify:
                bounded -= torch.nn.functional.silu(-160 * cx) * (-cx) / 160
            bound = float(bounded.abs().max())
            if bound < .002:
                valid.append(j)
                saved.append({'prototype': j, 'members': members[j],
                              'low': float(a), 'high': float(b), 'beta': float(beta),
                              'negative_bound': bound})
        covered = {i for j in valid for i in members[j]}
        variants.append({'negative_anchor_subtraction': rectify, 'valid_predicates': len(valid),
                         'covered_states': len(covered), 'predicates': saved})
    results['variants'] = variants
    report['families'].append(results)
    arrays[name + '_directions'] = D.numpy()
    print(name, 'separated', len(eligible), 'valid/covered',
          [(v['valid_predicates'], v['covered_states']) for v in variants], flush=True)
np.savez_compressed(ROOT / f'results/{STEM}_states.npz', **arrays)
(ROOT / f'results/{STEM}.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'families'}, flush=True)
