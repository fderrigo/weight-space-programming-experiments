"""Measure native recovery paths and all wholly correct observed candidate paths."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from best_reference_runtime import load_reference, restore_original
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'qualified_preservation_states'
torch.set_grad_enabled(False)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


selected, sources = [], []
for label in ['screen', 'third_observed', 'fourth_observed', 'fifth']:
    if label == 'fifth':
        response_name = 'fifth_development_qualified_concept_repairs.json'
        review_name = 'fifth_development_review.json'
        review = read(review_name)
        annotations = review['annotations']['qualified_concept_repairs']
        expected = review['sources_sha256']['qualified_concept_repairs']
    else:
        response_name = f'qualified_concept_repairs_{label}.json'
        review_name = 'qualified_concept_repairs_review.json' if label == 'screen' else f'qualified_concept_repairs_{label}_review.json'
        review = read(review_name)
        annotations = review['annotations']
        expected = review['source_sha256']
    assert sha(RDIR / response_name) == expected
    identities = {r['id'] for r in annotations if r['whole_correct'] or (label == 'screen' and r['group'] == 'control')}
    selected.extend({**r, 'kind': 'guard_current', 'source': response_name} for r in read(response_name) if r['id'] in identities)
    sources.append({'responses': response_name, 'responses_sha256': expected, 'review': review_name, 'review_sha256': sha(RDIR / review_name), 'selected_ids': sorted(identities)})
for source, identity, condition, limit in [
    ('fifth_development_native_recovery.json', 'fifth_donkey_1', 'native_recovery', 'Whole-correct animal transformation previously lost.'),
    ('late_native_recovery_third_observed.json', 'third_control_06', 'native_recovery', 'Preserve vegetation/coast contrast only; donor remains truncated and not wholly certified.'),
    ('fifth_development_original.json', 'fifth_control_06', 'original', 'Whole-correct original sum seven lost in both edited versions.'),
]:
    row = next(r for r in read(source) if r['id'] == identity)
    selected.append({**row, 'kind': 'restore_' + condition, 'source': source})
    sources.append({'responses': source, 'responses_sha256': sha(RDIR / source), 'selected_ids': [identity], 'limit': limit})
assert len({r['id'] for r in selected}) == len(selected)
assert not (RDIR / f'{STEM}_spec.json').exists()
candidate_path = ROOT / 'patches/qualified_concept_repairs_basis.safetensors'
assert sha(candidate_path) == '40ddba2f0d5fc147ad6a20f0cf4b92ec5c1aa73898c88bf145f566b419614b7e'
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'code_sha256': sha(Path(__file__)),
        'hypothesis': 'Restore three observed native paths while preserving all whole-correct candidate responses in four sets and the earlier measured guard states. Repair only actually divergent decisions; no new answer or stop is authored.',
        'sources': sources, 'candidate_sha256': sha(candidate_path),
        'fresh_validation': False, 'all_sets_now_observed': True, 'reserved_test_used': False,
        'training': False, 'gradients': False, 'optimizer': False, 'free_generation': False,
        'rejection_rule': 'Later single algebraic write must satisfy full projected key rank, equation error<1e-4, max guard residual shift<1e-3, and additional norm<=current down norm; reject without search if failed.'}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
model, tokenizer, originals, original_hashes = load_reference()
reference_down = model.model.layers[23].mlp.down_proj.weight.clone()
candidate = load_file(candidate_path)
sequences, records = [], []
for row in selected:
    prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    ids = prefix + row['generated_ids']
    positions = list(range(len(prefix) - 1, len(ids) if row['stop_token'] is not None else len(ids) - 1))
    sequence = {'id': row['id'], 'kind': row['kind'], 'source': row['source'], 'ids': ids,
                'base_length': len(prefix), 'positions': positions, 'offset': len(records)}
    sequences.append(sequence)
    for j, position in enumerate(positions):
        records.append({'id': row['id'], 'kind': row['kind'], 'subtoken': j, 'position': position,
                        'history_ids': ids[:position + 1],
                        'next_token': row['generated_ids'][j] if j < len(row['generated_ids']) else row['stop_token']})
cache, positions = {}, None


def capture(name):
    def hook(module, args):
        cache[name] = args[0][0, positions].clone()
    return hook


handles = [model.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
           model.model.norm.register_forward_pre_hook(capture('R'))]
E = model.model.embed_tokens.weight * model.model.norm.weight
arrays = {key: np.zeros((len(records), width), dtype=np.float32) for key, width in [('A', 4864), ('R', 896), ('donor_R', 896)]}
decisions = {}
completed = False
try:
    for condition in ['native_recovery', 'qualified_concept_repairs', 'original']:
        if condition == 'qualified_concept_repairs':
            model.model.layers[23].mlp.down_proj.weight.copy_(candidate['down_after'])
        elif condition == 'original':
            restore_original(model, originals, original_hashes)
        indices = []
        for sequence in sequences:
            if condition != 'qualified_concept_repairs' and sequence['kind'] != 'restore_' + condition:
                continue
            assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc)
            positions = torch.tensor(sequence['positions'])
            model.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
            start, end = sequence['offset'], sequence['offset'] + len(positions)
            indices.extend(range(start, end))
            if condition == 'qualified_concept_repairs':
                for key in ['A', 'R']:
                    arrays[key][start:end] = cache[key].numpy()
            else:
                arrays['donor_R'][start:end] = cache['R'].numpy()
            print(condition, sequence['id'], len(positions), flush=True)
        predictions = []
        source_R = torch.from_numpy(arrays['R' if condition == 'qualified_concept_repairs' else 'donor_R'])
        for start in range(0, len(indices), 16):
            block = indices[start:start + 16]
            residual = source_R[block]
            scores = residual / (residual.square().mean(-1) + 1e-6).sqrt()[:, None] @ E.T
            for i, score in zip(block, scores):
                history = sorted(set(records[i]['history_ids']))
                score[history] = torch.where(score[history] < 0, score[history] * 1.1, score[history] / 1.1)
                winner = int(score.argmax())
                if condition != 'qualified_concept_repairs' or records[i]['kind'] == 'guard_current':
                    assert winner == records[i]['next_token'], (condition, records[i], winner)
                predictions.append({'record_index': i, 'winner': winner, 'next_token': records[i]['next_token']})
        decisions[condition] = predictions
    completed = True
finally:
    for handle in handles:
        handle.remove()
    restore_original(model, originals, original_hashes)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    if completed:
        np.savez_compressed(RDIR / f'{STEM}.npz', **arrays)
    report = {'spec': spec, 'completed': completed, 'sequences': sequences, 'records': records, 'decisions': decisions,
              'arrays_sha256': sha(RDIR / f'{STEM}.npz') if completed else None,
              'original_parameters_restored': True, 'hooks_removed': True}
    (RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Completed=', completed, 'records=', len(records), '; original parameters restored.', flush=True)
