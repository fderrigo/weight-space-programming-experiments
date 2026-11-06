"""Diagnostic role contrasts and local attribution of the three origin edits."""
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
STEM = 'event_role_binding'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = read('late_narrative_origins_screen.json')
manifest = read('late_narrative_origins_screen_manifest.json')
assert manifest['completed'] and sha(RDIR / 'late_narrative_origins_screen.json') == manifest['responses_sha256']
narrative_prompt = next(r['prompt'] for r in rows if r['id'] == 'narration_0')
neutral_prompt = 'Scrivi una breve descrizione.'
# Independent minimal clauses, never concatenated into a prescribed story.
clauses = [
    ('origin_a', 'All’inizio della storia, Pinocchio è un', ' burattino', 'origin'),
    ('ending_a', 'Alla fine della storia, Pinocchio diventa un', ' bambino vero', 'ending'),
    ('donkey_a', 'Nel Paese dei balocchi, Pinocchio diventa un', ' asino', 'donkey'),
    ('origin_b', 'Prima delle sue avventure, Pinocchio è un', ' burattino', 'origin'),
    ('ending_b', 'Dopo le sue avventure, Pinocchio diventa un', ' bambino vero', 'ending'),
    ('donkey_b', 'Dopo il soggiorno nel Paese dei balocchi, Pinocchio viene trasformato in un', ' asino', 'donkey'),
    ('father', 'Il padre di Pinocchio si chiama', ' Geppetto', 'father'),
    ('creator', 'Il creatore di Pinocchio si chiama', ' Geppetto', 'creator'),
    ('created', 'Il burattino che Geppetto costruisce si chiama', ' Pinocchio', 'created'),
    ('wood_recipient', 'Maestro Ciliegia dà il pezzo di legno a', ' Geppetto', 'wood_recipient'),
    ('wood_donor', 'Geppetto riceve il pezzo di legno da', ' Maestro Ciliegia', 'wood_donor'),
    ('friend', 'L’amico che Pinocchio incontra nel Paese dei balocchi si chiama', ' Lucignolo', 'friend'),
]
controls = [
    ('cinderella', 'Nel racconto di Cenerentola, la protagonista sposa il', ' principe', 'outside'),
    ('snow_white', 'Nella fiaba di Biancaneve, la protagonista mangia una', ' mela', 'outside'),
    ('table', 'Un tavolo di legno è costruito da un', ' falegname', 'outside'),
    ('dog', 'Un cane è un', ' animale', 'outside'),
]
sequences, records = [], []
for context, prompt, selected in [('narrative', narrative_prompt, clauses),
                                  ('neutral', neutral_prompt, clauses + controls)]:
    for identity, prefix, unit, role in selected:
        base = t.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True)
        prefix_ids = base + t.encode(prefix, add_special_tokens=False)
        tokens = t.encode(unit, add_special_tokens=False)
        ids = prefix_ids + tokens
        positions = list(range(len(prefix_ids) - 1, len(ids)))
        identity = context + '_' + identity
        sequences.append({'id': identity, 'context': context, 'prompt': prompt, 'supplied_prefix': prefix,
                          'provided_unit': unit, 'role': role, 'ids': ids, 'positions': positions,
                          'offset': len(records), 'states': len(positions), 'actual_history': False})
        for j, pos in enumerate(positions):
            records.append({'id': identity, 'subtoken': j, 'position': pos, 'role': role,
                            'desired_token': tokens[j] if j < len(tokens) else None,
                            'actual_next_token': None, 'unit_complete': j == len(tokens)})
for identity, count in [('narration_0', 32), ('narration_1', 32), ('second_ending_0', 128)]:
    row = next(r for r in rows if r['id'] == identity)
    base = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    ids = base + row['generated_ids']
    assert count <= len(row['generated_ids'])
    positions = list(range(len(base) - 1, len(base) - 1 + count))
    sequences.append({'id': identity, 'prompt': row['prompt'], 'ids': ids, 'positions': positions,
                      'offset': len(records), 'states': count, 'base_length': len(base), 'actual_history': True})
    for j, pos in enumerate(positions):
        records.append({'id': identity, 'subtoken': j, 'position': pos, 'desired_token': None,
                        'actual_next_token': row['generated_ids'][j], 'unit_complete': False})

basis_paths = {stem: ROOT / f'patches/{stem}_basis.safetensors'
               for stem in ['late_native_recovery', 'late_narrative_origins']}
assert sha(basis_paths['late_native_recovery']) == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
assert sha(basis_paths['late_narrative_origins']) == manifest['basis_sha256']
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'A concept correction can activate at a wrong event. Contrast initial puppet, intermediate donkey and final human roles under two task contexts, plus inverse actor roles. Measure whether the existing selector and the latest three algebraic keys distinguish these contexts.',
        'clauses': clauses, 'controls': controls, 'narrative_prompt': narrative_prompt, 'neutral_prompt': neutral_prompt,
        'actual_histories': [{'id': s['id'], 'states': s['states']} for s in sequences if s['actual_history']],
        'response_source_sha256': manifest['responses_sha256'],
        'basis_sha256': {k: sha(v) for k, v in basis_paths.items()}, 'code_sha256': sha(Path(__file__)),
        'origin_equations_sha256': sha(RDIR / 'late_narrative_origins_equations.npz'),
        'conditions': ['original', 'late_narrative_origins'],
        'counterfactual_native_recovery': 'Exact final-down subtraction on identical histories; no claim of free-generation equivalence.',
        'supplied_prefixes_diagnostic_only': True, 'minimal_units_separate_not_full_answers': True,
        'rank_is_not_whole_answer_correctness': True, 'new_weights_constructed': False,
        'training': False, 'gradients': False, 'optimizer': False, 'free_generation': False,
        'counts_as_correction': False, 'reserved_final_used': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')

m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
E = m.model.embed_tokens.weight * m.model.norm.weight
bases = {k: load_file(v) for k, v in basis_paths.items()}
for name in ['gate', 'up']:
    assert torch.equal(bases['late_native_recovery'][name + '_after'], bases['late_narrative_origins'][name + '_after'])
delta = bases['late_narrative_origins']['down_after'] - bases['late_native_recovery']['down_after']
with np.load(RDIR / 'late_narrative_origins_equations.npz') as z:
    dual, values = torch.from_numpy(z['dual'].copy()).double(), torch.from_numpy(z['values'].copy()).double()
assert float((delta.double() - values.T @ dual).norm() / delta.double().norm()) < 1e-5
with np.load(RDIR / 'late_template_selector.npz') as z:
    class_indices = int(z['query_head']) * 64 + z['value_coordinates'][:12]
classes = read('late_selector_geometry.json')['class_coordinates']
cache, positions = {}, None


def capture(name):
    def hook(module, args):
        cache[name] = args[0][0, positions].clone()
    return hook


handles = [m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
           m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
           m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(capture('heads')),
           m.model.norm.register_forward_pre_hook(capture('R'))]
for layer, block in enumerate(m.model.layers):
    handles.append(block.register_forward_pre_hook(capture(('incoming', layer))))


def scores_for(residual, seen):
    rms = (residual.square().mean() + 1e-6).sqrt()
    score = residual / rms @ E.T
    score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
    return score


def top_rows(score, count=6):
    top = score.topk(count)
    return [{'token': int(k), 'text': t.decode([int(k)]), 'score': float(v)} for k, v in zip(top.indices, top.values)]


reports = []
for condition in spec['conditions']:
    originals = None
    if condition != 'original':
        originals, parent = apply_late_selector(m)
        assert parent == read('late_selector_states_weights.json')
        for name in ['gate', 'up', 'down']:
            weight = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(weight, bases[condition][name + '_before'])
            weight.copy_(bases[condition][name + '_after'])
    arrays = {name: np.empty((len(records), width), dtype=np.float32)
              for name, width in [('X', 896), ('A', 4864), ('R', 896), ('heads', 896)]}
    arrays['incoming'] = np.empty((len(records), 24, 896), dtype=np.float32)
    notes = []
    try:
        for seq in sequences:
            positions = torch.tensor(seq['positions'])
            m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
            a, b = seq['offset'], seq['offset'] + seq['states']
            for name in arrays:
                value = torch.stack([cache[('incoming', layer)] for layer in range(24)], dim=1) if name == 'incoming' else cache[name]
                arrays[name][a:b] = value.numpy()
            for j, pos in enumerate(seq['positions']):
                index = a + j
                seen = sorted(set(seq['ids'][:pos + 1]))
                residual = cache['R'][j]
                score = scores_for(residual, seen)
                target = records[index]['desired_token']
                note = {**records[index], 'top': top_rows(score),
                        'desired_text': None if target is None else t.decode([target]),
                        'desired_rank': None if target is None else int((score > score[target]).sum()) + 1}
                if condition != 'original':
                    if seq['actual_history']:
                        assert int(score.argmax()) == records[index]['actual_next_token'], (seq['id'], j)
                    old_residual = residual - cache['A'][j] @ delta.T
                    old_score = scores_for(old_residual, seen)
                    selector = cache['heads'][j, class_indices]
                    coefficient = cache['A'][j].double() @ dual.T
                    shift = residual - old_residual
                    approximate = coefficient @ values
                    note.update({'without_origin_update_top': top_rows(old_score),
                                 'without_origin_update_desired_rank': None if target is None else int((old_score > old_score[target]).sum()) + 1,
                                 'origin_coefficients': coefficient.tolist(), 'origin_residual_shift_norm': float(shift.norm()),
                                 'origin_decomposition_absolute_error': float((shift.double() - approximate).norm()),
                                 'selector_class': classes[int(selector.argmax())] if float(selector.max()) > .5 else 'outside',
                                 'selector_values': selector.tolist()})
                if j == 0 and target is not None:
                    lens = []
                    for layer in range(24):
                        layer_score = scores_for(cache[('incoming', layer)][j], seen)
                        lens.append({'incoming_layer': layer,
                                     'desired_rank': int((layer_score > layer_score[target]).sum()) + 1})
                    note['incoming_layer_ranks'] = lens
                notes.append(note)
            first = notes[-seq['states']]
            print(condition, seq['id'], 'desired_rank', first['desired_rank'], 'winner', repr(first['top'][0]['text']), flush=True)
        path = RDIR / f'{STEM}_{condition}.npz'
        np.savez_compressed(path, **arrays)
        reports.append({'condition': condition, 'arrays_sha256': sha(path), 'decisions': notes})
    finally:
        if originals is not None:
            restore_carrier(m, originals)
for handle in handles:
    handle.remove()
result = {'spec': spec, 'sequences': sequences, 'records': records, 'reports': reports,
          'actual_current_decisions_replayed': 192, 'completed': True, 'weights_restored_in_memory': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print('Event-role diagnosis complete. No new ordinary answers or scored corrections.', flush=True)
