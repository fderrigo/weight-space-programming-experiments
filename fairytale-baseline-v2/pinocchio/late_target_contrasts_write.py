"""Three declared local token contrasts in existing factual down columns."""
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
STEM = 'late_target_contrasts'
assert not (ROOT / f'patches/{STEM}_basis.safetensors').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = ROOT / 'patches/late_margin_preservation_basis.safetensors'
basis = load_file(source)
previous = json.loads((RDIR / 'late_margin_preservation_diagnostic.json').read_text('utf8'))
meta = json.loads((RDIR / 'progress_carrier_states.json').read_text('utf8'))
cells, rows = previous['cells'], meta['records']
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
declared = [('nose', ' all', ' tr'), ('puppeteer', 'f', ' il'), ('friend', 'ign', 'io')]
choices = []
for fact, wanted, rival in declared:
    target_ids = tokenizer.encode(wanted, add_special_tokens=False)
    rival_ids = tokenizer.encode(rival, add_special_tokens=False)
    assert len(target_ids) == len(rival_ids) == 1
    matching = [j for j, c in enumerate(cells) if c['fact'] == fact and c['target_token'] == target_ids[0]]
    assert len(matching) == 1
    choices.append({'fact': fact, 'target_text': wanted, 'rival_text': rival,
                    'target_token': target_ids[0], 'rival_token': rival_ids[0], 'cell': matching[0]})
source_spec = json.loads((ROOT / 'patches/late_margin_preservation_spec.json').read_text('utf8'))
spec = {**source_spec,
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'Observed third-set failures have correctly active factual cells but lose three token decisions to lexical competitors. Test one explicit contrast per diagnosed cell without changing selectors, gates or native compensation.',
    'choices': choices,
    'direction_rule': 'Unit normalized target decoder vector minus named competing decoder vector, projected orthogonally to the same25 carrier axes.',
    'amplitude_rule': 'For each cell take the largest amplitude required to ADD four pre-edit RMS units of repetition-adjusted raw target-versus-rival margin on its earlier fit members. Same fixed margin4 as the earlier concept write; no amplitude search. Sign factors are read from the pre-edit logits, then actual decoding is checked.',
    'fit_rule': 'Only existing earlier members of the three cells; no second or third state used to compute amplitude. Choice of cells and rivals is informed by the already observed third-set diagnosis.',
    'weight_rule': 'Add the resulting vector to each chosen positive down column and subtract it from the two paired negative columns. Exactly nine down columns may differ from the source candidate; every other weight remains unchanged.',
    'source_basis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'diagnosis_sha256': hashlib.sha256((RDIR / 'third_mechanism_analysis.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'third_set_already_observed': True, 'final_set_read': False,
    'EOS_programmed': False, 'full_responses_prescribed': False,
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
with np.load(RDIR / 'late_selector_states.npz') as z:
    X, A, parent_R = [torch.from_numpy(z[k][:4693].copy()) for k in ['X', 'A', 'R']]
with np.load(RDIR / 'late_selector_geometry.npz') as z:
    B = torch.from_numpy(z['protected_carrier'].copy()).double()
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original_weights = {n: f.get_tensor(f'model.layers.23.mlp.{n}_proj.weight').float() for n in ['gate', 'up', 'down']}
chosen = basis['reassigned_neurons']
fact_ids = chosen[:previous['fact_neurons']]
new_A = A.clone()
new_A[:, chosen] = torch.nn.functional.silu(X @ basis['gate_after'][chosen].T) * (X @ basis['up_after'][chosen].T)
parts = new_A[:, fact_ids].double()
F = parts[:, 0::3] - parts[:, 1::3] - parts[:, 2::3]
before_R = parent_R - A @ basis['down_before'].T + new_A @ basis['down_after'].T
histories = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
after = {n: basis[n + '_after'].clone() for n in ['gate', 'up', 'down']}
contrasts = torch.zeros((len(cells), 896), dtype=torch.float64)
details = []
for choice in choices:
    j, target, rival = choice['cell'], choice['target_token'], choice['rival_token']
    direction = E[target].double() - E[rival].double()
    direction -= B @ (B.T @ direction)
    direction /= direction.norm()
    requirements, fit_notes = [], []
    for i in cells[j]['members']:
        assert i < 288 and not rows[i]['id'].startswith('second_')
        residual = before_R[i].double()
        rms = (residual.square().mean() + 1e-6).sqrt()
        factors = []
        for token in [target, rival]:
            raw = float(E[token].double() @ residual)
            factors.append((1.1 if raw < 0 else 1 / 1.1) if token in histories[i] else 1.0)
        margin_vector = factors[0] * E[target].double() - factors[1] * E[rival].double()
        slope = float(F[i, j] * (margin_vector @ direction))
        assert slope > 0 and float(F[i, j]) > .3
        requirements.append(float(4 * rms / slope))
        fit_notes.append({'index': i, 'id': rows[i]['id'], 'gate': float(F[i, j]),
                          'margin_before': float(margin_vector @ residual / rms),
                          'required_amplitude': requirements[-1]})
    amplitude = max(requirements)
    delta = amplitude * direction
    contrasts[j] = delta
    neuron_ids = fact_ids[3 * j:3 * j + 3]
    after['down'][:, neuron_ids[0]] += delta.float()
    after['down'][:, neuron_ids[1]] -= delta.float()
    after['down'][:, neuron_ids[2]] -= delta.float()
    details.append({**choice, 'amplitude': amplitude, 'fit': fit_notes,
                    'changed_down_columns': neuron_ids.tolist()})
delta_weight = after['down'] - basis['down_after']
changed_columns = torch.count_nonzero(delta_weight, dim=0).nonzero().flatten().tolist()
assert sorted(changed_columns) == sorted(n for d in details for n in d['changed_down_columns'])
assert torch.equal(after['gate'], basis['gate_after']) and torch.equal(after['up'], basis['up_after'])
after_R = parent_R - A @ basis['down_before'].T + new_A @ after['down'].T
expected_shift = F @ contrasts
actual_shift = (after_R - before_R).double()
assert float((actual_shift - expected_shift).norm() / expected_shift.norm()) < 1e-4

native_meta = json.loads((RDIR / 'late_native_margins.json').read_text('utf8'))
native_records = native_meta['records']
native = [r['record_index'] for r in native_records]
checks = list(range(288)) + native
predictions, notes = {}, []
for start in range(0, len(checks), 16):
    indices = checks[start:start + 16]
    residual = after_R[indices]
    scores = (residual / (residual.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for i, score in zip(indices, scores):
        seen = histories[i]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
        predictions[i] = int(score.argmax())
        if i < 288:
            token = rows[i]['next_token']
            notes.append({'id': rows[i]['id'], 'subtoken': rows[i]['subtoken'],
                          'desired_token': token, 'decoder_rank': int((score > score[token]).sum()) + 1})
    if start % 512 == 0:
        print('Contrast decoder diagnostic', start, '/', len(checks), flush=True)
coverage = []
for split in ['fit', 'second_observed']:
    indices = [i for i in range(288) if rows[i]['id'].startswith('second_') == (split == 'second_observed')]
    ids = sorted({rows[i]['id'] for i in indices})
    coverage.append({'split': split, 'states': len(indices), 'questions': len(ids),
                     'desired_tokens_first': sum(notes[i]['decoder_rank'] == 1 for i in indices),
                     'all_steps_first': sum(all(notes[i]['decoder_rank'] == 1 for i in indices if rows[i]['id'] == q) for q in ids)})
diffs = [{'tensor': f'model.layers.23.mlp.{name}_proj.weight',
          'changed_parameters': int(torch.count_nonzero(value - original_weights[name])),
          'difference_from_original_norm': float((value - original_weights[name]).norm()),
          'after_sha256_f32': tensor_sha(value)} for name, value in after.items()]
with np.load(RDIR / 'late_margin_preservation_constraints.npz') as z:
    previous_native = z['native_predicted_tokens'].tolist()
assert len(previous_native) == len(native)
report = {'spec': spec, 'details': details, 'coverage': coverage, 'token_diagnostics': notes,
          'cells': cells, 'fact_neurons': previous['fact_neurons'], 'erase_neurons': previous['erase_neurons'],
          'new_changed_down_columns': changed_columns, 'new_changed_parameters': int(torch.count_nonzero(delta_weight)),
          'down_update_norm': float(delta_weight.norm()),
          'native_states': len(native),
          'native_same_original_tokens_before': previous['native_same_original_tokens_after'],
          'native_same_original_tokens_after': sum(predictions[r['record_index']] == r['original_token'] for r in native_records),
          'native_decisions_changed_from_previous': sum(predictions[i] != p for i, p in zip(native, previous_native)),
          'last_mlp_differences': diffs, 'changed_tensors': previous['changed_tensors'],
          'changed_parameters': previous['changed_parameters'] - sum(r['changed_parameters'] for r in previous['last_mlp_differences']) + sum(r['changed_parameters'] for r in diffs),
          'free_generation_performed': False, 'counts_as_correction': False, 'final_set_read': False,
          'training': False, 'gradients': False, 'optimizer': False}
save_file({**{name + '_before': basis[name + '_before'].contiguous() for name in after},
           **{name + '_after': value.contiguous() for name, value in after.items()},
           'reassigned_neurons': chosen.contiguous(), 'scale_readout': basis['scale_readout'].contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
np.savez_compressed(RDIR / f'{STEM}_contrasts.npz', contrasts=contrasts.numpy())
(RDIR / f'{STEM}_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['spec', 'details', 'cells', 'token_diagnostics', 'last_mlp_differences']}, flush=True)
