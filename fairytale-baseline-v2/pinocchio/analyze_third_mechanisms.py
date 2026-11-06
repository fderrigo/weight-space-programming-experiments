"""Separate semantic selection, concept writing and continuation failures."""
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
STEM = 'third_mechanism_analysis'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'third_mechanism_states.json').read_text('utf8'))
assert meta['completed']
rows, sequences = meta['records'], meta['sequences']
with np.load(RDIR / 'third_mechanism_states_original.npz') as z:
    original_R = torch.from_numpy(z['R'].copy())
    original_means = torch.from_numpy(z['prefix_values21'].copy())
    original_heads = torch.from_numpy(z['head14_13'].copy())
with np.load(RDIR / 'third_mechanism_states_late_selector_parent.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k].copy()) for k in ['X', 'A', 'R']]
    values = torch.from_numpy(z['selector_values'].copy())
    earlier_errors = {k: float((torch.from_numpy(z[k].copy()) - reference).abs().max()) for k, reference in [('prefix_values21', original_means), ('head14_13', original_heads)]}
assert max(earlier_errors.values()) == 0
geometry = json.loads((RDIR / 'late_selector_geometry.json').read_text('utf8'))
classes = geometry['class_coordinates']
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).float()
basis = load_file(ROOT / 'patches/late_margin_preservation_basis.safetensors')
previous = load_file(ROOT / 'patches/late_selector_fact_basis.safetensors')
assert torch.equal(basis['gate_after'], previous['gate_after']) and torch.equal(basis['up_after'], previous['up_after'])
diagnostic = json.loads((RDIR / 'late_margin_preservation_diagnostic.json').read_text('utf8'))
cells = diagnostic['cells']
chosen = basis['reassigned_neurons']
fact_ids = chosen[:diagnostic['fact_neurons']]
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
native_base = parent_R - A @ basis['down_before'].T
full_R = native_base + new_A @ basis['down_after'].T
previous_R = native_base + new_A @ previous['down_after'].T
fact_component = new_A[:, fact_ids] @ basis['down_after'][:, fact_ids].T
parts = new_A[:, fact_ids]
F = parts[:, 0::3] - parts[:, 1::3] - parts[:, 2::3]
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
history = {}
for sequence in sequences:
    for j, pos in enumerate(sequence['positions']):
        history[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'A correct first-position relation does not ensure the canonical concept survives every step or that the native continuation remains faithful. Compare all three stages on the fixed observed transfer outputs.',
    'conditions': ['original_on_same_history', 'frozen_margin_patch'],
    'local_ablations': ['previous_late_fact_without_margin_update', 'full_without_fact_cells', 'parent_ideal_carrier_removal'],
    'scope': 'All recorded actual histories for replay; all24 supplied canonical concepts for diagnostic token coverage, including their completion. Ablations at first response positions, canonical completion, first8 actual steps after a fully matched concept, and the first token of the observed wrong control words Alidoro, verde and oppure.',
    'earlier_feature_checks': earlier_errors,
    'source_sha256': hashlib.sha256((RDIR / 'third_mechanism_states.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'training': False, 'gradients': False, 'optimizer': False, 'third_set_already_observed': True, 'final_set_read': False,
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
predictions, ranks = {}, {}


def decode(name, residual, indices):
    predicted, desired_rank = {}, {}
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        block = residual[selected]
        scores = (block / (block.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
        for index, score in zip(selected, scores):
            seen = history[index]
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            predicted[index] = int(score.argmax())
            target = rows[index]['next_token']
            if target is not None:
                desired_rank[index] = int((score > score[target]).sum()) + 1
    predictions[name], ranks[name] = predicted, desired_rank


all_indices = list(range(len(rows)))
decode('original', original_R, all_indices)
print('Original counterfactual decoding completed', len(rows), flush=True)
decode('full', full_R, all_indices)
print('Actual patch replay completed', flush=True)

old_meta = json.loads((RDIR / 'question_layer_diagnostic_states.json').read_text('utf8'))['sequences']
old_classes = sorted({r['label'] for r in old_meta})
with np.load(RDIR / 'semantic_carrier_geometry.npz') as z:
    semantic = original_means.double() @ torch.from_numpy(z['mean_projection'].copy()).double() @ torch.from_numpy(z['semantic_basis'].copy()).double().T
    centers = torch.from_numpy(z['semantic_centers'].copy()).double()
pure_labels = [old_classes[i] for i in (semantic @ (centers / centers.norm(dim=-1, keepdim=True)).T).argmax(-1).tolist()]
actual_labels = [classes[int(v.argmax())] if float(v.max()) > .5 else 'outside' for v in values]
actual_sequences = {s['id']: s for s in sequences if s['kind'] == 'actual_response'}
localized = {s['offset'] for s in sequences}
concepts = []
for sequence in sequences:
    if sequence['kind'] != 'canonical_fact':
        continue
    indices = list(range(sequence['offset'], sequence['offset'] + sequence['states']))
    composing, completed = indices[:-1], indices[-1]
    localized.add(completed)
    unit = sequence['ids'][sequence['base_length']:]
    actual = actual_sequences[sequence['id']]
    generated = actual['ids'][actual['base_length']:]
    matched = generated[:len(unit)] == unit
    continuation = []
    if matched:
        start = actual['offset'] + len(unit)
        continuation = list(range(start, min(start + 8, actual['offset'] + actual['states'])))
        localized.update(continuation)
    steps = []
    for i in composing:
        own = [j for j, c in enumerate(cells) if c['target_token'] == rows[i]['next_token']]
        other = [j for j, c in enumerate(cells) if c['target_token'] != rows[i]['next_token']]
        steps.append({'index': i, 'subtoken': rows[i]['subtoken'],
                      'desired_token': rows[i]['next_token'], 'desired_text': t.decode([rows[i]['next_token']]),
                      'original_mean_class': pure_labels[i], 'actual_selector_class': actual_labels[i],
                      'wanted_class_attention_value': float(values[i, classes.index(sequence['fact'])]),
                      'own_token_gate_max': float(F[i, own].max()), 'other_token_gate_max': float(F[i, other].abs().max()),
                      'original_desired_rank': ranks['original'][i], 'edited_desired_rank': ranks['full'][i],
                      'edited_top_token': predictions['full'][i], 'edited_top_text': t.decode([predictions['full'][i]])})
    concepts.append({'id': sequence['id'], 'fact': sequence['fact'], 'provided_unit': sequence['provided_concept'],
                     'first_token_correct': ranks['full'][composing[0]] == 1,
                     'all_unit_tokens_correct': all(ranks['full'][i] == 1 for i in composing),
                     'actual_generated_unit_matches': matched,
                     'completed_unit_index': completed, 'actual_continuation_indices': continuation,
                     'steps': steps})
wrong_control_points = []
for identity, word in [('third_control_05', 'Alidoro'), ('third_control_17', 'verde'), ('third_control_04', 'oppure')]:
    sequence = actual_sequences[identity]
    tokens = sequence['ids'][sequence['base_length']:]
    text = t.decode(tokens)
    offset = text.index(word)
    token_index = next(j for j in range(len(tokens)) if len(t.decode(tokens[:j])) <= offset < len(t.decode(tokens[:j + 1])))
    index = sequence['offset'] + token_index
    localized.add(index)
    wrong_control_points.append({'id': identity, 'word': word, 'record_index': index})
localized = sorted(localized)
decode('previous_without_margin_update', previous_R, localized)
decode('full_without_fact_cells', full_R - fact_component, localized)
decode('parent_ideal_carrier_removal', parent_R - (parent_R @ B) @ B.T, localized)
point_reports = {}
for i in localized:
    active = [j for j in range(len(cells)) if abs(float(F[i, j])) > .05]
    point_reports[i] = {'record_index': i, 'id': rows[i]['id'], 'kind': rows[i]['kind'],
                        'subtoken': rows[i]['subtoken'], 'actual_selector_class': actual_labels[i],
                        'original_mean_class': pure_labels[i],
                        'next_predictions': {name: {'token': pred[i], 'text': t.decode([pred[i]])} for name, pred in predictions.items()},
                        'active_fact_cells': [{'fact': cells[j]['fact'], 'previous_token': cells[j]['previous_token'],
                                               'target_token': cells[j]['target_token'], 'activation': float(F[i, j])} for j in active],
                        'fact_component_norm': float(fact_component[i].norm()),
                        'margin_update_norm': float((full_R[i] - previous_R[i]).norm())}
for concept in concepts:
    concept['completion_point'] = point_reports[concept['completed_unit_index']]
    concept['actual_continuation_points'] = [point_reports[i] for i in concept['actual_continuation_indices']]
for point in wrong_control_points:
    point['analysis'] = point_reports[point['record_index']]
replay_indices = [i for i, r in enumerate(rows) if r['kind'] == 'actual_response' and r['next_token'] is not None]
replay_mismatches = [{'index': i, 'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                      'actual_token': rows[i]['next_token'], 'reconstructed_token': predictions['full'][i]}
                     for i in replay_indices if predictions['full'][i] != rows[i]['next_token']]
factual_starts = [s['offset'] for s in sequences if s['kind'] == 'canonical_fact']
control_starts = [s['offset'] for s in sequences if s['kind'] == 'actual_response' and s['group'] == 'control']
summary = {'canonical_questions': len(concepts),
           'first_concept_token_correct': sum(c['first_token_correct'] for c in concepts),
           'all_concept_tokens_correct': sum(c['all_unit_tokens_correct'] for c in concepts),
           'actual_generated_full_concept_matches': sum(c['actual_generated_unit_matches'] for c in concepts),
           'original_mean_first_relation_correct': sum(pure_labels[i] == rows[i]['fact'] for i in factual_starts),
           'actual_selector_first_relation_correct': sum(actual_labels[i] == rows[i]['fact'] for i in factual_starts),
           'original_mean_first_controls_outside': sum(pure_labels[i] == 'outside' for i in control_starts),
           'actual_selector_first_controls_outside': sum(actual_labels[i] == 'outside' for i in control_starts),
           'actual_replay_tokens_checked': len(replay_indices), 'actual_replay_token_mismatches': len(replay_mismatches)}
report = {'spec': spec, 'summary': summary, 'concepts': concepts,
          'wrong_control_points': wrong_control_points,
          'actual_replay_mismatches': replay_mismatches,
          'first_response_points': [point_reports[s['offset']] for s in sequences if s['kind'] == 'actual_response'],
          'interpretation_limit': 'The reference original is evaluated on exactly the same supplied history, which may already contain edited mistakes. A local comparison does not prove that the original would freely generate that history. Canonical concept completion and native-token preservation are not whole-answer correctness.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
np.savez_compressed(RDIR / f'{STEM}.npz', full_residual=full_R.numpy(),
                    previous_residual=previous_R.numpy(), fact_features=F.numpy(),
                    full_predicted_tokens=np.array([predictions['full'][i] for i in all_indices]),
                    original_predicted_tokens=np.array([predictions['original'][i] for i in all_indices]))
print(summary, flush=True)
for c in concepts:
    print(c['id'], 'first', c['first_token_correct'], 'whole unit', c['all_unit_tokens_correct'],
          'original next', c['completion_point']['next_predictions']['original']['text'],
          'edited next', c['completion_point']['next_predictions']['full']['text'], flush=True)
for p in wrong_control_points:
    print('CONTROL', p['id'], p['word'], p['analysis'], flush=True)
