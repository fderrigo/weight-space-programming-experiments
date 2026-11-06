"""One explicit shared-prefix name-memory write, followed by ordinary generation."""
import ast
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from native_name_memory_helpers import sha, tensor_sha, capture, remove, collect_guards, geometry, programmed_weight

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
PDIR = ROOT / 'patches'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_name_memory'
assert not (PDIR / f'{STEM}_spec.json').exists()
assert not (RDIR / f'{STEM}_screen.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


metadata = read('middle_fact_retrieval.json')
assert metadata['completed']
sequences = [s for s in metadata['sequences'] if s['source_group'] == 'earlier' and s.get('kind') == 'external']
assert len(sequences) == 58 and sum(s['states'] for s in sequences) == 3130
prefix_ids = sequences[0]['ids'][:21]
donor_collection = read('persistent_relation_delta_collection.json')
donor_review = read('persistent_relation_delta_donor_review.json')
assert donor_collection['completed'] and donor_review['certified_donors'] == 6
assert donor_review['source_sha256'] == sha(RDIR / 'persistent_relation_delta_collection.json')
source_fact_code = ROOT / 'probe_single_fact_state.py'
assert sha(source_fact_code) == donor_collection['spec']['prior_donor_code_sha256']
node = next(n.value for n in ast.parse(source_fact_code.read_text('utf8')).body
            if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'facts' for x in n.targets))
facts = ast.literal_eval(node)
donors = {r['fact']: r for r in donor_collection['responses'] if r['condition'] == 'single_fact_reference'}
chosen = [('puppeteer', 'Mangiafoco'), ('rescuer', 'Alidoro'), ('fish_father', 'Geppetto')]
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'


def render(question):
    return t.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)


bank, donor_inputs = [], []
for fact, name in chosen:
    question = donors[fact]['prompt']
    text = render('Fatto del romanzo: ' + facts[fact] + '\n\n' + question)
    encoded = t(text, return_offsets_mapping=True)
    start = text.index(facts[fact]) + facts[fact].index(name)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and a < start + len(name)]
    extra = len(encoded['input_ids']) - len(t(render(question))['input_ids'])
    donor_inputs.append({'fact': fact, 'ids': encoded['input_ids'], 'positions': positions})
    for position in positions:
        slot = 7 + len(bank)
        bank.append({'fact': fact, 'name': name, 'token_id': encoded['input_ids'][position],
                     'piece': t.decode([encoded['input_ids'][position]]), 'source_position': position,
                     'source_extra_context_tokens': extra, 'destination_slot': slot,
                     'effective_key_position': position - extra,
                     'pre_rope_rotation_positions': position - extra - slot})
assert len(bank) == 10
slots = [r['destination_slot'] for r in bank]
assert slots == list(range(7, 17))
screen_source = RDIR / 'attention_gain_screen.json'
screen_manifest = read('attention_gain_manifest.json')
assert sha(screen_source) == screen_manifest['responses_sha256']
tests = [r for r in read(screen_source.name) if r['stage'] == 'original']
assert len(tests) == 18
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'parent': 'unaltered original Qwen2.5-0.5B-Instruct in F32', 'blocks': [15, 16], 'kv_group': 1,
    'hypothesis': 'Shared KV group1 in blocks15/16 supports starting and completing proper names. Store all subword states of three names in the same fixed prefix bank, rather than one terminal value per fact, and test a real weight-only implementation.',
    'bank': bank, 'all_names_same_for_every_question': True, 'oracle_relation_selection': False,
    'native_value_gain': 1.0,
    'key_positions': 'Rotate each native pre-RoPE K by source_position minus source_extra_context_tokens minus destination_slot. After destination RoPE its effective position matches the reference key translated to the ordinary-question frame. No input position_ids are changed.',
    'guard_rule': 'Uncentered top128 right singular vectors of3130 observed outside-relation input states, plus all fixed-prefix positions not used as memory. This preserves those directions, not every external state.',
    'outside_definition': 'Outside the12 fact relations in the earlier diagnostic; some controls mention Pinocchio.',
    'update': 'For X=slot inputs and B=orthonormal protected row basis, Z=X-XB.TB. DeltaW=D.T pinv(Z).T, where D is desired native KV minus current KV including unchanged bias. Only rows64:128 of K/V weights change. Both pseudoinverse cutoffs1e-6; gain1; no sweep.',
    'sequence_of_writes': 'Write block15; recollect current block16 outside states and prefix under that write; write block16 toward the same ORIGINAL native donor values. The first write may already change external behavior, which must be measured.',
    'scope': 'A mechanism pilot within the broader Pinocchio task; it is not a replacement for the full objective or the multi-set verified candidate.',
    'source_guard_metadata_sha256': sha(RDIR / 'middle_fact_retrieval.json'),
    'donor_collection_sha256': sha(RDIR / 'persistent_relation_delta_collection.json'),
    'donor_review_sha256': sha(RDIR / 'persistent_relation_delta_donor_review.json'),
    'fact_source_code_sha256': sha(source_fact_code),
    'code_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'native_name_memory_helpers.py'),
    'ordinary_screen_source_sha256': sha(screen_source),
    'screen': 'Same12 observed targets,4 controls,2 narratives and original sampling; no hooks/reference context in generation. Candidate frozen before outputs. No fresh-validation claim.',
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'reserved_final_used': False, 'full_answers_prescribed': False, 'EOS_programmed': False,
    'exported_checkpoint_remains_attempt': 6,
}
save(PDIR / f'{STEM}_spec.json', spec)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
original_checkpoint_sha = sha(MODEL / 'model.safetensors')
assert original_checkpoint_sha == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
original_hashes = {name: tensor_sha(p) for name, p in m.named_parameters()}
total_parameters = sum(p.numel() for p in m.parameters())
assert total_parameters == 494032768
cache, handles = capture(m, [15, 16])
native = {(layer, kind): [] for layer in [15, 16] for kind in ['k', 'v', 'x']}
try:
    for donor in donor_inputs:
        m.model(input_ids=torch.tensor([donor['ids']]), use_cache=False)
        for key in native:
            native[key].append(cache[key][donor['positions']].clone())
finally:
    remove(handles)
native = {key: torch.cat(values) for key, values in native.items()}
targets = {}
for layer in [15, 16]:
    raw = native[layer, 'k'][:, 64:128].double()
    shifts = torch.tensor([r['pre_rope_rotation_positions'] for r in bank], dtype=torch.float64)
    frequency = shifts[:, None] * m.model.rotary_emb.inv_freq.double()[None]
    angles = torch.cat((frequency, frequency), dim=-1)
    rotated = raw * angles.cos() + torch.cat((-raw[:, 32:], raw[:, :32]), dim=-1) * angles.sin()
    targets[layer, 'k'] = rotated.float()
    targets[layer, 'v'] = native[layer, 'v'][:, 64:128].clone()
guards, prefix, precision = collect_guards(m, sequences, [15, 16], prefix_ids, 'original guards')
state_arrays = {f'guard_{layer}': guards[layer].numpy() for layer in [15, 16]}
state_arrays.update({f'prefix_{layer}': prefix[layer].numpy() for layer in [15, 16]})
state_arrays.update({f'donor_{layer}_{kind}': value.numpy() for (layer, kind), value in native.items()})
state_arrays.update({f'target_{layer}_{kind}': value.numpy() for (layer, kind), value in targets.items()})
np.savez_compressed(RDIR / f'{STEM}_native_states.npz', **state_arrays)
save(RDIR / f'{STEM}_collection.json', {'completed': True, 'spec': spec,
     'sequences': [{'id': s['id'], 'states': s['states']} for s in sequences],
     'prefix_ids': prefix_ids, 'prefix_precision': precision,
     'arrays_sha256': sha(RDIR / f'{STEM}_native_states.npz')})

originals, basis, writes = {}, {}, []
for layer in [15, 16]:
    if layer == 16:
        current, current_prefix, current_errors = collect_guards(m, sequences, [16], prefix_ids, 'after block15 guards')
        guards[16], prefix[16] = current[16], current_prefix[16]
        np.savez_compressed(RDIR / f'{STEM}_after15_states.npz', guard_16=guards[16].numpy(), prefix_16=prefix[16].numpy())
        save(RDIR / f'{STEM}_after15_collection.json', {'completed': True, 'prefix_precision': current_errors,
             'arrays_sha256': sha(RDIR / f'{STEM}_after15_states.npz'),
             'partial_basis_sha256': sha(PDIR / f'{STEM}_after15.safetensors')})
    keys, protected, inverse, geo = geometry(guards[layer], prefix[layer], slots)
    updates, diagnostics = {}, []
    for kind in ['k', 'v']:
        name = f'model.layers.{layer}.self_attn.{kind}_proj.weight'
        module = getattr(m.model.layers[layer].self_attn, kind + '_proj')
        originals[name] = module.weight.detach().clone()
        updates[name], diagnostic = programmed_weight(module.weight, module.bias, targets[layer, kind], guards[layer], keys, protected, inverse)
        diagnostics.append({'tensor': name, **diagnostic})
    for name, value in updates.items():
        m.get_parameter(name).copy_(value)
        basis[name + '.before'] = originals[name]
        basis[name + '.after'] = value
    writes.append({'block': layer, 'geometry': geo, 'tensors': diagnostics})
    save_file(basis, PDIR / (f'{STEM}_after15.safetensors' if layer == 15 else f'{STEM}_basis.safetensors'))
    save(RDIR / f'{STEM}_write_progress.json', {'completed': layer == 16, 'writes': writes})
    print('WROTE', layer, json.dumps(diagnostics), flush=True)

cache, handles = capture(m, [15, 16])
bank_checks = []
try:
    m.model(input_ids=torch.tensor([sequences[0]['ids']]), use_cache=False)
    for layer in [15, 16]:
        for kind in ['k', 'v']:
            actual = cache[layer, kind][slots, 64:128]
            relative = float((actual - targets[layer, kind]).norm() / targets[layer, kind].norm())
            bank_checks.append({'block': layer, 'kind': kind, 'relative_error_actual_forward': relative})
            assert relative < 1e-3, bank_checks[-1]
finally:
    remove(handles)
assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())
changed = []
for name, parameter in m.named_parameters():
    actual_hash = tensor_sha(parameter)
    if actual_hash != original_hashes[name]:
        assert name in originals, name
        changed.append({'tensor': name, 'changed_values': int(torch.count_nonzero(parameter != originals[name])),
                        'before_sha256_f32': original_hashes[name], 'after_sha256_f32': actual_hash})
assert len(changed) == 4 and {r['tensor'] for r in changed} == set(originals)
changed_count = sum(r['changed_values'] for r in changed)
diagnostic = {
    'completed': True, 'spec': spec, 'writes': writes, 'actual_forward_bank_checks': bank_checks,
    'tensor_differences': changed, 'changed_tensors': len(changed), 'changed_parameters': changed_count,
    'total_parameters': total_parameters, 'changed_percent': 100 * changed_count / total_parameters,
    'all_other_parameter_hashes_unchanged': True, 'no_runtime_hooks': True,
    'native_states_sha256': sha(RDIR / f'{STEM}_native_states.npz'),
    'after15_states_sha256': sha(RDIR / f'{STEM}_after15_states.npz'),
    'basis_sha256': sha(PDIR / f'{STEM}_basis.safetensors'),
    'partial_basis_sha256': sha(PDIR / f'{STEM}_after15.safetensors'),
    'weights_changed': True, 'candidate_promoted': False, 'free_generation_yet_completed': False,
}
save(RDIR / f'{STEM}_diagnostic.json', diagnostic)
print('FROZEN candidate', changed_count, 'values in4 tensors; starting ordinary generation.', flush=True)
manifest = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'basis_sha256': diagnostic['basis_sha256'], 'spec_sha256': sha(PDIR / f'{STEM}_spec.json'),
    'diagnostic_sha256': sha(RDIR / f'{STEM}_diagnostic.json'),
    'evaluator_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'native_name_memory_helpers.py'),
    'original_checkpoint_sha256': original_checkpoint_sha,
    'original_responses_source_sha256': sha(screen_source),
    'sampling': screen_manifest['settings'], 'same_runtime_template_and_batch_grouping': True,
    'actual_weights_changed': True, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'changed_parameters': changed_count, 'changed_tensors': 4,
    'current_exported_checkpoint_attempt': 6, 'responses_expected': 18,
    'completed': False,
}
manifest_path = RDIR / f'{STEM}_screen_manifest.json'
output = RDIR / f'{STEM}_screen.json'
save(manifest_path, manifest)
responses = []
completed = False
try:
    for group in ['target', 'control', 'narration']:
        selected = [r for r in tests if r['group'] == group]
        limit = 256 if group == 'narration' else 128
        for begin in range(0, len(selected), 4):
            batch = selected[begin:begin + 4]
            encoded = t([render(r['prompt']) for r in batch], padding=True, return_tensors='pt')
            generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1,
                                   max_new_tokens=limit, pad_token_id=t.pad_token_id)
            for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                assert row['max_new_tokens'] == limit
                stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                tokens = ids if stop is None else ids[:stop]
                response = {**row, 'stage': f'edited_{STEM}',
                            'response': t.decode(tokens, skip_special_tokens=True), 'generated_ids': tokens,
                            'stop_token': None if stop is None else ids[stop],
                            'done_reason': 'length' if stop is None else 'stop',
                            'actual_weights_changed': True, 'weights_changed': True,
                            'runtime_hooks': False, 'reference_in_input': False,
                            'oracle_relation_selection': False, 'calibration_only': True, 'fresh_validation': False}
                responses.append(response)
                print(row['id'], repr(response['response']), flush=True)
            save(output, responses)
    completed = True
finally:
    for name, value in originals.items():
        m.get_parameter(name).copy_(value)
    restored = all(tensor_sha(p) == original_hashes[name] for name, p in m.named_parameters())
    assert restored
    manifest.update({'completed': completed, 'responses_completed': len(responses),
                     'responses_sha256': sha(output) if output.exists() else None,
                     'weights_restored_in_memory': restored})
    save(manifest_path, manifest)
print('Native name memory candidate evaluated; original parameters restored in memory.', flush=True)
