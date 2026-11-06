"""Diagnostic minimal concept paths at two actual narrative-origin slots."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from late_selector_weights import apply_late_selector, restore_carrier
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'narrative_origin_units'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
trace = read('narrative_origin_trace.json')
assert trace['completed'] and trace['actual_latest_decisions_replayed_exactly'] == 85
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sequences, records = [], []
for identity, start, unit in [('narration_0', 11, ' burattino'), ('narration_1', 14, ' pezzo di legno')]:
    source = next(s for s in trace['sequences'] if s['id'] == identity)
    prefix = source['ids'][:source['base_length'] + start]
    tokens = t.encode(unit, add_special_tokens=False)
    ids = prefix + tokens
    positions = list(range(len(prefix) - 1, len(ids)))
    sequences.append({'id': identity, 'provided_unit': unit, 'provided_prefix_ids': prefix,
                      'ids': ids, 'unit_start': len(prefix), 'positions': positions,
                      'offset': len(records), 'states': len(positions)})
    for j, pos in enumerate(positions):
        records.append({'id': identity, 'subtoken': j, 'position': pos,
                        'desired_token': tokens[j] if j < len(tokens) else None,
                        'unit_complete': j == len(tokens)})
assert len(records) == 10
basis_path = ROOT / 'patches/late_native_recovery_basis.safetensors'
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'The actual narrative origin slots do not recall puppet/wood concepts. Measure all pieces of only these two short concepts and the native decision after each completed unit before selecting an intervention.',
        'units': [s['provided_unit'] for s in sequences],
        'supplied_prefix_diagnostic_only': True, 'full_answers_prescribed': False,
        'EOS_prescribed': False, 'new_candidate_constructed': False,
        'free_generation_performed': False, 'counts_as_correction': False,
        'source_trace_sha256': sha(RDIR / 'narrative_origin_trace.json'),
        'basis_sha256': sha(basis_path), 'code_sha256': sha(Path(__file__)),
        'conditions': ['original', 'late_native_recovery'], 'training': False,
        'gradients': False, 'optimizer': False, 'final_set_read': False}
assert spec['basis_sha256'] == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
E = m.model.embed_tokens.weight * m.model.norm.weight
cache, positions = {}, None
def capture(name):
    def hook(module, args):
        cache[name] = args[0][0, positions].clone()
    return hook
handles = [m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
           m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
           m.model.norm.register_forward_pre_hook(capture('R'))]
basis = load_file(basis_path)
reports = []
for condition in spec['conditions']:
    originals = None
    if condition != 'original':
        originals, parent = apply_late_selector(m)
        assert parent == read('late_selector_states_weights.json')
        for name in ['gate', 'up', 'down']:
            p = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(p, basis[name + '_before'])
            p.copy_(basis[name + '_after'])
    arrays = {name: np.empty((10, width), dtype=np.float32) for name, width in [('X', 896), ('A', 4864), ('R', 896)]}
    notes = []
    try:
        for seq in sequences:
            positions = torch.tensor(seq['positions'])
            m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
            for name in arrays:
                arrays[name][seq['offset']:seq['offset'] + seq['states']] = cache[name].numpy()
            residual = cache['R']
            scores = (residual / (residual.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
            for j, score in enumerate(scores):
                i = seq['offset'] + j
                seen = sorted(set(seq['ids'][:seq['positions'][j] + 1]))
                score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
                target = records[i]['desired_token']
                top = score.topk(8)
                note = {**records[i], 'desired_text': t.decode([target]) if target is not None else None,
                        'desired_rank': int((score > score[target]).sum()) + 1 if target is not None else None,
                        'top': [{'token': int(k), 'text': t.decode([int(k)]), 'score': float(v)} for k, v in zip(top.indices, top.values)]}
                notes.append(note)
                print(condition, seq['id'], j, note['desired_text'], note['desired_rank'], 'winner', note['top'][0]['text'], flush=True)
        path = RDIR / f'{STEM}_{condition}.npz'
        np.savez_compressed(path, **arrays)
        reports.append({'condition': condition, 'arrays_sha256': sha(path), 'decisions': notes})
    finally:
        if originals is not None:
            restore_carrier(m, originals)
for handle in handles:
    handle.remove()
(RDIR / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'sequences': sequences, 'records': records,
    'reports': reports, 'completed': True, 'weights_restored_in_memory': True}, ensure_ascii=False, indent=2), encoding='utf8')
print('Minimal origin-unit diagnosis complete; zero scored corrections.', flush=True)
