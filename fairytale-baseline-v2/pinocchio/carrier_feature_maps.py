"""Fixed readout maps for relation-gated carrier experiments."""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'


def canonical(fact):
    return {'wood_finder': 'wood_donor'}.get(fact, fact)


def load_maps():
    assert not torch.is_grad_enabled()
    meta = json.loads((RDIR / 'carrier_concept_states.json').read_text('utf8'))
    assert meta['completed']
    rows = meta['records']
    fit = [i for i, r in enumerate(rows[:288]) if not r['id'].startswith('second_')]
    with np.load(RDIR / 'carrier_concept_states_protected.npz') as z:
        X = torch.from_numpy(z['X'].copy())
    with np.load(RDIR / 'carrier_progress_compensation.npz') as z:
        semantic = torch.from_numpy(z['readout_unscaled'].copy()).double()
        progress = torch.from_numpy(z['stage_dual_unscaled'].copy()).double()
    with safe_open(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors', framework='pt') as f:
        gain = f.get_tensor('model.layers.23.post_attention_layernorm.weight').double()
    P = X.double() / gain
    anchor = P[fit].mean(0)
    anchor /= anchor.square().sum()
    qmeta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
    outside = [i for i, r in enumerate(qmeta) if r['split'] == 'prototype' and r['label'] == 'outside']
    with np.load(RDIR / 'semantic_carrier_forward_protected_states.npz') as z:
        outside_P = torch.from_numpy(z['X23'][outside].copy()).double() / gain
    assert len(outside) == 14
    classes = sorted({canonical(rows[i]['fact']) for i in fit} | {'outside'})
    augmented = torch.cat([semantic, anchor[:, None]], dim=1)
    normalized = (P @ augmented) / (P @ anchor)[:, None]
    outside_normalized = (outside_P @ augmented) / (outside_P @ anchor)[:, None]
    centers = torch.stack([outside_normalized.mean(0) if label == 'outside' else
                            normalized[[i for i in fit if canonical(rows[i]['fact']) == label]].mean(0)
                            for label in classes])
    relation = augmented @ torch.linalg.pinv(centers, rtol=1e-8)
    members = {}
    for i in fit:
        row = rows[i]
        members.setdefault((canonical(row['fact']), row['last_token'], row['next_token']), []).append(i)
    cells = [{'fact': fact, 'previous_token': previous, 'target_token': target, 'members': indices}
             for (fact, previous, target), indices in members.items()]
    tokens = sorted({rows[i]['last_token'] for i in fit})
    return {'X': X, 'P': P, 'gain': gain, 'anchor': anchor, 'relation': relation,
            'progress': progress, 'classes': classes, 'tokens': tokens,
            'cells': cells, 'meta': meta, 'fit': fit}


def relation_gated_features(maps):
    """One fixed confidence ramp: half to three-quarter class-center response."""
    lower, upper = .5, .75
    beta = 160.0 / (upper - lower)
    scale = beta * (upper - lower)
    gate_rows, up_rows = [], []
    for cell in maps['cells']:
        direction = maps['relation'][:, maps['classes'].index(cell['fact'])]
        progress = maps['progress'][:, maps['tokens'].index(cell['previous_token'])]
        gates = torch.stack([beta * (direction - lower * maps['anchor']),
                             beta * (direction - upper * maps['anchor']), -160 * maps['anchor']])
        gate_rows.extend((gates / maps['gain']).float().unbind())
        physical_progress = (progress / maps['gain']).float()
        up_rows.extend([physical_progress / scale, physical_progress / scale, -physical_progress / 160])
    G, V = torch.stack(gate_rows), torch.stack(up_rows)
    values = torch.nn.functional.silu(maps['X'] @ G.T) * (maps['X'] @ V.T)
    F = values[:, 0::3] - values[:, 1::3] - values[:, 2::3]
    return G, V, F
