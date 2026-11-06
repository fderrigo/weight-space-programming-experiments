"""Write scalar token comparisons without prescribing whole residual vectors.

For each native factual state, constrain the desired token against the four
highest native competitors under the unchanged repetition penalty. The update
is one explicit closed-form Gram solve in weight space, projected away from
the previously declared native preservation subspace. No optimizer or training.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
parser = argparse.ArgumentParser()
parser.add_argument('--continuation-protected', action='store_true')
parser.add_argument('--anchor-target', action='store_true')
args = parser.parse_args()
assert not args.anchor_target or args.continuation_protected
STEM = ('anchored_continuation' if args.anchor_target else
        'continuation_pairwise' if args.continuation_protected else 'pairwise_constraint')
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists(), 'Preserve previous numerical evidence'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
K, R, records, old = load_contexts(ROOT, t)
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    C0 = torch.from_numpy(z['C'][-525:].copy()).double()
with np.load(ROOT / 'results/newdevelopment_control_states.npz') as z:
    C1 = torch.from_numpy(z['C'].copy()).double()
with np.load(ROOT / 'results/expanded_context_states.npz') as z:
    C2 = torch.from_numpy(z['C'].copy()).double()
C = torch.cat([C0, C1, C2])
# Fixed follow-up decision by Codex from the 47 lexical failures of the first
# algebraic screen. No automatic constraint-discovery or optimization loop.
lexical_pairs = [
    ('pp', 'ge'), (' Ge', ' Luc'), (' Ge', ' Ma'), (' Ge', ' Al'),
    (' Mang', ' Ge'), ('f', ' Mang'), (' Luc', ' Ge'), (' Luc', ' Al'),
    (' Ma', ' Ge'), (' Al', ' al'), ('id', ' di'), (' Al', ' Luc'),
    ('id', 'di'), ('Il', 'Con'), ('In', 'Ge'), ('Ge', 'Con'),
    (' C', ' Ma'), (' Mang', ' mang'), ('ale', 'ango'), (' leg', ' Ge'),
    (' il', ' al'),
] if args.continuation_protected else []
extra_competitors = {}
for target_piece, other_piece in lexical_pairs:
    target_ids = t.encode(target_piece, add_special_tokens=False)
    other_ids = t.encode(other_piece, add_special_tokens=False)
    assert len(target_ids) == len(other_ids) == 1
    extra_competitors.setdefault(target_ids[0], []).append(other_ids[0])
if args.continuation_protected:
    with np.load(ROOT / 'results/native_second_continuations_states.npz') as z:
        native = torch.cat([torch.from_numpy(z[f'A_{i}'].copy()).double() for i in range(24)])
    termination = json.loads((ROOT / 'results/termination_diagnostic.json').read_text('utf8'))['states']
    with np.load(ROOT / 'results/termination_states.npz') as z:
        completed = torch.stack([torch.from_numpy(z[f'A_{i}'].copy()).double()
                                 for i, row in enumerate(termination) if row['kind'] == 'completed_unit'])
    C = torch.cat([C, native, completed])
with safe_open(MODEL / 'model.safetensors', framework='pt') as source:
    E = source.get_tensor('model.embed_tokens.weight').float()
    E.mul_(source.get_tensor('model.norm.weight').float())
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': 'model.layers.23.mlp.down_proj.weight',
    'reason': 'Rigid complete value vectors produced large edits and corrupted continuations. Constrain only four native token comparisons per state, allowing the remaining residual components to be chosen jointly by the weight-space geometry.',
    'contexts': len(K), 'competitors_per_context': 4, 'margin': 4.0,
    'repetition_penalty': 1.1, 'gram_pinv_rtol': 1e-8,
    'formula': 'Z=K-KQQt. For each context i and competitor j, d_ij=p_t*e_t-p_j*e_j and b_ij=max(0,4*RMS_i-d_ij*R_i). G_(ij,kl)=(d_ij*d_kl)*(z_i*z_k). alpha=pinv(G)*b. deltaW=sum_ij alpha_ij*d_ij*z_i.T.',
    'penalty_geometry': 'A seen token expected to remain positive has p=1/1.1; unseen tokens have p=1. Native top competitors are selected using the exact sign-dependent repetition penalty. Full post-edit logits are checked with the exact penalty again, including any sign changes.',
    'protected_source': 'Same 3406 native states and rank-3357 subspace as expanded_context',
    'training': False, 'gradients': False, 'optimizer': False, 'EOS_programmed': False,
    'third_set_read': False, 'final_set_read': False, 'runtime_hooks': False,
}
if args.continuation_protected:
    spec.update({
        'reason': 'The first scalar edit damaged native continuations that were already correct after a factual unit. Preserve all 504 exact native continuation states of the second set, including its incorrect continuations, and the 24 completed-unit states of the first set. Add the explicitly listed lexical contrasts to address observed subtoken confusions.',
        'additional_lexical_comparisons': lexical_pairs,
        'protected_source': '3406 expanded-context native states + 504 second-set continuation states + 24 first-set completed-unit states',
        'protected_state_count': len(C), 'protection_rtol': 1e-6,
        'native_prefixes_are_diagnostic_only': True,
        'no_full_answers_or_EOS_prescribed': True,
        'original_native_continuations_preserved_whether_true_or_false': True,
    })
if args.anchor_target:
    spec.update({
        'reason': 'Read-only scalar-gain analysis found that 75 of 82 failed contexts achieved the relative constraint mainly by lowering native competitors; the desired token was not raised enough. Anchor the desired increase explicitly and hold each declared competitor readout unchanged. Keep the same 3934 native states and the same explicit lexical contrasts.',
        'formula': 'Z=K-KQQt. Each context has d_target=p_t*e_t, b_target=max(0,4*RMS_i+native_best_effective_logit*RMS_i-d_target*R_i). For each declared competitor use d_other=p_j*e_j and b_other=0. G_ab=(d_a*d_b)*(z_context(a)*z_context(b)); alpha=pinv(G)*b; deltaW=sum_a alpha_a*d_a*z_context(a).T.',
        'native_gain_diagnostic': 'results/scalar_gain_diagnostic.json',
        'absolute_target_increase': True, 'competitor_readout_changes': 0,
    })
(ROOT / f'patches/{STEM}_spec.json').write_text(
    json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')

directions, right_hand_sides, notes, context_indices = [], [], [], []
for begin in range(0, len(R), 16):
    rb = R[begin:begin + 16]
    rms = (rb.square().mean(-1) + 1e-6).sqrt()
    logits = (rb / rms[:, None]) @ E.T
    for j, row in enumerate(logits):
        index = begin + j
        record = records[index]
        token = record['token']
        history = set(record['history'])
        ids = torch.tensor(sorted(history))
        adjusted = row.clone()
        adjusted[ids] = torch.where(adjusted[ids] < 0,
                                    adjusted[ids] * 1.1, adjusted[ids] / 1.1)
        adjusted[token] = -torch.inf
        competitors = adjusted.topk(spec['competitors_per_context']).indices.tolist()
        competitors += [c for c in extra_competitors.get(token, []) if c not in competitors]
        target = E[token].double() / (1.1 if token in history else 1.0)
        if args.anchor_target:
            desired_rise = max(0.0, (spec['margin'] + float(adjusted[competitors[0]])) * float(rms[j])
                               - float(target @ R[index].double()))
            directions.append(target)
            right_hand_sides.append(desired_rise)
            context_indices.append(index)
        for competitor in competitors:
            other = E[competitor].double() / (1.1 if competitor in history else 1.0)
            d = other if args.anchor_target else target - other
            b = 0.0 if args.anchor_target else max(0.0, spec['margin'] * float(rms[j]) - float(d @ R[index].double()))
            directions.append(d)
            right_hand_sides.append(b)
            context_indices.append(index)
        notes.append({k: v for k, v in record.items() if k != 'history'})
        notes[-1]['competitors'] = [t.decode([c]) for c in competitors]
D = torch.stack(directions)
b = torch.tensor(right_hand_sides, dtype=torch.float64)
if args.continuation_protected:
    print('Continuation preservation SVD', C.shape, flush=True)
    _, singular, vh = torch.linalg.svd(C, full_matrices=False)
    protected_rank = int((singular > singular[0] * spec['protection_rtol']).sum())
    Q = vh[:protected_rank]
else:
    Q = torch.linalg.qr(old['preservation_basis'].double().T, mode='reduced')[0].T
Z = K.double() - (K.double() @ Q.T) @ Q
key_gram = Z @ Z.T
index = torch.tensor(context_indices)
print('Scalar constraints', len(D), 'instead of', len(K) * R.shape[1], flush=True)
G = (D @ D.T) * key_gram[index[:, None], index[None, :]]
eigenvalues, eigenvectors = torch.linalg.eigh(G)
keep = eigenvalues > eigenvalues[-1] * spec['gram_pinv_rtol']
alpha = eigenvectors[:, keep] @ ((eigenvectors[:, keep].T @ b) / eigenvalues[keep])
values = torch.zeros((len(K), R.shape[1]), dtype=torch.float64)
values.index_add_(0, index, alpha[:, None] * D)
delta = (values.T @ Z).float()
actual = (old['before'] + delta - old['before']).double()
effect = K.double() @ actual.T
measured_comparisons = (effect[index] * D).sum(-1)
for begin in range(0, len(R), 16):
    changed = R[begin:begin + 16] + effect[begin:begin + 16].float()
    rms = (changed.square().mean(-1) + 1e-6).sqrt()
    logits = (changed / rms[:, None]) @ E.T
    for j, row in enumerate(logits):
        index = begin + j
        token = records[index]['token']
        ids = torch.tensor(sorted(set(records[index]['history'])))
        row[ids] = torch.where(row[ids] < 0, row[ids] * 1.1, row[ids] / 1.1)
        rank = int((row > row[token]).sum()) + 1
        desired = float(row[token])
        row[token] = -torch.inf
        best = int(row.argmax())
        notes[index].update({'actual_decoder_target_rank': rank,
                             'actual_decoder_margin': desired - float(row[best]),
                             'actual_decoder_best_other': t.decode([best]),
                             'effective_value_norm': float(effect[index].norm())})

diagnostic = {
    'contexts': len(K), 'scalar_constraints': len(D), 'constraint_rank': int(keep.sum()),
    'preservation_states': len(C), 'protected_rank': len(Q),
    'relative_scalar_constraint_error': float((measured_comparisons - b).norm() / b.norm()),
    'delta_norm': float(actual.norm()), 'old_delta_norm': float(old['delta'].norm()),
    'actual_decoder_first_rank_count': sum(n['actual_decoder_target_rank'] == 1 for n in notes),
    'mean_control_shift': float((C @ actual.T).norm(dim=-1).mean()),
    'max_control_shift': float((C @ actual.T).norm(dim=-1).max()),
    'contexts_diagnostic': notes, 'free_generation_performed': False,
    'third_set_read': False, 'final_set_read': False,
}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(
    json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf8')
save_file({'before': old['before'].contiguous(), 'delta': delta.contiguous(),
           'keys': K.contiguous(), 'values': effect.float().contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
print({k: v for k, v in diagnostic.items() if k != 'contexts_diagnostic'}, flush=True)
