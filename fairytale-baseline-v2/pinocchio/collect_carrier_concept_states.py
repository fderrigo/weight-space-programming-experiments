"""Inspect relation and concept-progress states, without fitting a new patch."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from semantic_carrier_weights import apply_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'carrier_concept_states'
assert not (RESULTS / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RESULTS / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source_names = ['minimal_fact_states.json', 'bounded_native_inputs.json',
                'extended_native_inputs.json', 'minimal_fact_units_complete_states.json']
minimal = read(source_names[0])
sequences, records = [], []
for case in minimal['cases']:
    ids = case['base'] + case['concept'][:-1]
    positions = list(range(len(case['base']) - 1, len(ids)))
    sequences.append({'id': case['question'], 'kind': 'factual', 'fact': case['fact'],
                      'ids': ids, 'positions': positions, 'offset': len(records), 'states': len(positions)})
    for j, position in enumerate(positions):
        records.append({'id': case['question'], 'kind': 'factual', 'fact': case['fact'],
                        'prefix_ids': case['concept'][:j], 'subtoken': j,
                        'next_token': case['concept'][j], 'position': position,
                        'last_token': ids[position]})
factual_count = len(records)
assert len(sequences) == 74 and factual_count == len(minimal['records']) == 288
for name in source_names[1:]:
    for source_row in read(name)['sequences']:
        if name == 'bounded_native_inputs.json' and source_row['kind'] != 'external':
            continue
        kind = 'minimal_native' if name == source_names[-1] else source_row['kind']
        positions = list(range(source_row['start'], len(source_row['ids'])))
        row = {'id': source_row['id'], 'kind': kind, 'ids': source_row['ids'],
               'positions': positions, 'offset': len(records), 'states': len(positions), 'source': name}
        sequences.append(row)
        for position in positions:
            records.append({'id': row['id'], 'kind': kind, 'position': position,
                            'last_token': row['ids'][position], 'source': name})
assert len(sequences) == 183 and len(records) - factual_count == 4405
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'The protected carrier retains relation information but does not write facts. Inspect whether current-token and short-concept progress can be represented separately from the question relation, with the already recorded native continuations as negatives.',
    'conditions': ['original', 'protected'], 'factual_states': factual_count,
    'preservation_states': len(records) - factual_count, 'sequences': len(sequences),
    'sources_sha256': {name: sha(RESULTS / name) for name in source_names},
    'geometry_sha256': sha(RESULTS / 'semantic_carrier_geometry.npz'),
    'transform_code_sha256': sha(ROOT / 'semantic_carrier_weights.py'),
    'collection_code_sha256': sha(Path(__file__)),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'diagnostic_only': True, 'supplied_incomplete_concepts_are_not_corrections': True,
    'new_patch_fitted': False, 'training': False, 'gradients': False, 'optimizer': False,
    'third_set_used': False, 'final_set_read': False, 'free_generation_performed': False,
}
assert spec['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
(RESULTS / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
captured = {}


def capture_input(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def capture_values(module, args, output):
    value = output[0]
    captured['prefix_values'] = value.cumsum(0) / torch.arange(1, len(value) + 1)[:, None]


hooks = [m.model.layers[21].register_forward_pre_hook(capture_input('incoming21')),
         m.model.layers[21].self_attn.v_proj.register_forward_hook(capture_values),
         m.model.layers[23].mlp.register_forward_pre_hook(capture_input('X')),
         m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture_input('A')),
         m.model.norm.register_forward_pre_hook(capture_input('R'))]
reports = []
for mode in spec['conditions']:
    originals = None
    weights = None
    if mode == 'protected':
        originals, weights = apply_carrier(m, mode)
        reference = read('semantic_carrier_forward_protected_weights.json')
        assert weights == reference
    arrays = {name: np.empty((len(records), width), dtype=np.float32)
              for name, width in [('incoming21', 896), ('prefix_values', 128),
                                  ('X', 896), ('A', 4864), ('R', 896)]}
    for number, sequence in enumerate(sequences):
        m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
        positions = torch.tensor(sequence['positions'])
        begin, end = sequence['offset'], sequence['offset'] + sequence['states']
        for name in arrays:
            arrays[name][begin:end] = captured[name][positions].numpy()
        if number % 16 == 0:
            print(mode, 'histories', number + 1, '/', len(sequences), flush=True)
    report = {'condition': mode, 'states': len(records), 'weights': weights}
    if mode == 'original':
        with np.load(RESULTS / 'minimal_fact_states.npz') as z:
            for name, saved in [('A', 'K'), ('R', 'R')]:
                error = float(np.linalg.norm(arrays[name][:factual_count] - z[saved]) / np.linalg.norm(z[saved]))
                assert error < 1e-5, (name, error)
                report[f'factual_{name}_reference_relative_error'] = error
    if originals is not None:
        restore_carrier(m, originals)
        del originals
        report['weights_restored_exactly'] = True
    path = RESULTS / f'{STEM}_{mode}.npz'
    np.savez_compressed(path, **arrays)
    report['arrays_sha256'] = sha(path)
    reports.append(report)
    (RESULTS / f'{STEM}.json').write_text(json.dumps(
        {'spec': spec, 'records': records, 'sequences': sequences, 'reports': reports,
         'completed': len(reports) == len(spec['conditions'])},
        ensure_ascii=False, indent=2), encoding='utf8')
    print(mode, 'saved', {k: v for k, v in report.items() if k != 'weights'}, flush=True)
for hook in hooks:
    hook.remove()
