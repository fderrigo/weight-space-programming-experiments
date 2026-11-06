"""Isolate the frozen name bank from projection changes at other positions."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor
from native_name_memory_helpers import sha, tensor_sha, remove

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_name_memory_locality'
assert not (RDIR / f'{STEM}_spec.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(name, obj):
    (RDIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def log(message):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (RDIR / f'{STEM}.log').open('a', encoding='utf8') as stream:
        stream.write(stamp + ' ' + message + '\n')
    print(message, flush=True)


def check_deadline():
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc), '20:00 Rome deadline'


diagnostic = read('native_name_memory_diagnostic.json')
collection = read('native_name_memory_collection.json')
assert sha(RDIR / 'native_name_memory_native_states.npz') == diagnostic['native_states_sha256'] == collection['arrays_sha256']
bank_spec = collection['spec']
prefix_ids = collection['prefix_ids']
slots = [row['destination_slot'] for row in bank_spec['bank']]
assert slots == list(range(7, 17))
arrays = np.load(RDIR / 'native_name_memory_native_states.npz')
targets = {(block, kind): torch.from_numpy(arrays[f'target_{block}_{kind}'].copy()) for block in [15, 16] for kind in ['k', 'v']}
arrays.close()
tests = [row for row in read('attention_gain_screen.json') if row['stage'] == 'original' and row['group'] != 'narration']
assert len(tests) == 16
references = {row['fact']: row for row in read('persistent_relation_delta_collection.json')['responses'] if row['condition'] == 'single_fact_reference'}
critical = [row for row in read('critical_fact_reads_records.json') if row['fact'] in ['puppeteer', 'rescuer', 'fish_father']]
assert len(critical) == 9
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hypothesis': 'Test whether the exact previously frozen name bank is sufficient when only prefix positions change, isolating it from the real linear write affecting other tokens.',
    'intervention': 'Replace only pre-RoPE K and native V rows64:128 at logical prefix slots7:16 in blocks15/16. Same ten states for every input; no name selection. Decode projections are unchanged; prefix cache retains the bank.',
    'unit_diagnostic': 'Nine ordinary-question histories followed by the already generated reference answer prefix. Compare original and prefix-only bank; no reference fact in input. Supplied answer prefixes are diagnostic only.',
    'free_generation': 'Same12 ordinary targets and4 controls, batch4 by group, max128, greedy, repetition penalty1.1, seed42, threads6. No answer prefix or corrective context.',
    'decision_rule': 'No new weight construction unless the prefix-only bank improves whole ordinary target answers. Correct unit tokens alone are insufficient. Negative free-response results stop this bank branch.',
    'frozen_bank_sha256': diagnostic['native_states_sha256'],
    'source_screen_sha256': sha(RDIR / 'attention_gain_screen.json'),
    'critical_records_sha256': sha(RDIR / 'critical_fact_reads_records.json'),
    'reference_collection_sha256': sha(RDIR / 'persistent_relation_delta_collection.json'),
    'code_sha256': sha(Path(__file__)),
    'weights_changed': False, 'runtime_hooks': True, 'counts_as_correction': False,
    'oracle_relation_selection': False, 'training': False, 'gradients': False, 'optimizer': False,
    'fresh_validation': False, 'reserved_test_used': False,
    'user_work_window': 'Resume authorized until20:00 Europe/Rome, extend only for excellent behavioral results; this diagnostic itself has a hard20:00 deadline.',
}
save(f'{STEM}_spec.json', spec)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
original_hashes = {name: tensor_sha(p) for name, p in m.named_parameters()}
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
penalty = RepetitionPenaltyLogitsProcessor(1.1)
state = {'padding': [], 'label': '', 'enabled': False}
statistics = []
handles = []


def render(question):
    return t.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)


def set_batch(encoded, label):
    state['padding'] = (encoded['attention_mask'] == 0).sum(-1).tolist()
    state['label'] = label
    for ids, padding in zip(encoded['input_ids'].tolist(), state['padding']):
        assert ids[padding:padding + len(prefix_ids)] == prefix_ids


for block in [15, 16]:
    for kind in ['k', 'v']:
        def hook(module, args, output, block=block, kind=kind):
            if not state['enabled'] or output.shape[1] == 1:
                return output
            assert output.shape[0] == len(state['padding'])
            changed = output.clone()
            outside = torch.ones(output.shape[:2], dtype=torch.bool)
            for i, padding in enumerate(state['padding']):
                positions = [slot + padding for slot in slots]
                changed[i, positions, 64:128] = targets[block, kind]
                outside[i, positions] = False
                assert torch.equal(changed[i, positions, 64:128], targets[block, kind])
            assert torch.equal(changed[outside], output[outside])
            assert torch.equal(changed[..., :64], output[..., :64])
            statistics.append({'batch': state['label'], 'block': block, 'projection': kind,
                               'only_selected_slots_and_group_changed': True})
            return changed
        handles.append(getattr(m.model.layers[block].self_attn, kind + '_proj').register_forward_hook(hook))


units, responses = [], []
manifest = {'spec': spec, 'completed': False, 'responses_expected': 16, 'unit_pairs_expected': 9}
save(f'{STEM}_manifest.json', manifest)
completed = False
try:
    for row in critical:
        check_deadline()
        donor = references[row['fact']]
        generated_prefix = donor['generated_ids'][:row['generated_index']]
        ids = t(render(donor['prompt']))['input_ids'] + generated_prefix
        encoded = {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long)}
        set_batch(encoded, row['fact'] + '/' + row['kind'])
        measurement = {k: row[k] for k in ['id', 'fact', 'kind', 'generated_index', 'expected_id', 'expected_piece']}
        measurement['plain_history_ids'] = ids
        measurement['provided_answer_prefix'] = True
        measurement['reference_fact_in_input'] = False
        for condition in ['original', 'prefix_only_bank']:
            state['enabled'] = condition == 'prefix_only_bank'
            hidden = m.model(**encoded, use_cache=False).last_hidden_state[:, -1]
            logits = penalty(encoded['input_ids'], m.lm_head(hidden).clone())[0]
            target = row['expected_id']
            other = logits.clone()
            other[target] = -torch.inf
            top = int(logits.argmax())
            measurement[condition] = {'target_rank': int((logits > logits[target]).sum()) + 1,
                                      'target_margin': float(logits[target] - other.max()),
                                      'top_id': top, 'top_piece': t.decode([top])}
        units.append(measurement)
        save(f'{STEM}_units.json', units)
        log('UNIT ' + row['fact'] + ' ' + row['kind'] + ' ' + json.dumps({k: measurement[k] for k in ['original', 'prefix_only_bank']}))
    for group in ['target', 'control']:
        selected = [row for row in tests if row['group'] == group]
        for start in range(0, len(selected), 4):
            check_deadline()
            batch = selected[start:start + 4]
            encoded = t([render(row['prompt']) for row in batch], padding=True, return_tensors='pt')
            set_batch(encoded, group + '/' + str(start))
            state['enabled'] = True
            before_hooks = len(statistics)
            output = m.generate(**encoded, do_sample=False, repetition_penalty=1.1,
                                max_new_tokens=128, pad_token_id=t.pad_token_id, use_cache=True)
            assert len(statistics) - before_hooks == 4
            for row, ids in zip(batch, output[:, encoded['input_ids'].shape[1]:].tolist()):
                stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                tokens = ids if stop is None else ids[:stop]
                result = {**row, 'stage': STEM, 'response': t.decode(tokens, skip_special_tokens=True),
                          'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                          'done_reason': 'length' if stop is None else 'stop',
                          'actual_weights_changed': False, 'weights_changed': False, 'runtime_hooks': True,
                          'reference_in_input': False, 'oracle_relation_selection': False,
                          'counts_as_correction': False, 'calibration_only': True, 'fresh_validation': False}
                responses.append(result)
                log(row['id'] + ' ' + repr(result['response']))
            save(f'{STEM}_screen.json', responses)
    completed = True
finally:
    remove(handles)
    unchanged = all(tensor_sha(parameter) == original_hashes[name] for name, parameter in m.named_parameters())
    assert unchanged
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())
    manifest.update({'completed': completed, 'responses_completed': len(responses), 'unit_pairs_completed': len(units),
                     'all_parameter_hashes_unchanged': unchanged, 'all_hooks_removed': True,
                     'slot_checks': statistics,
                     'screen_sha256': sha(RDIR / f'{STEM}_screen.json') if responses else None,
                     'units_sha256': sha(RDIR / f'{STEM}_units.json') if units else None})
    save(f'{STEM}_manifest.json', manifest)
    log('Completed=' + str(completed) + '; all weights unchanged and hooks removed.')
