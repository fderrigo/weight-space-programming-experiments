"""Two fixed diagnostic selectors on the preserved native-recovery candidate."""
import datetime
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from late_selector_weights import apply_late_selector, restore_carrier
from native_name_memory_helpers import sha, tensor_sha, remove

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'stable_relation_selection'
assert (RDIR / f'{STEM}_manifest_failed_v1.json').exists()
assert not (RDIR / f'{STEM}_resume_spec.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(name, obj):
    (RDIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def log(message):
    with (RDIR / f'{STEM}.log').open('a', encoding='utf8') as stream:
        stream.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message + '\n')
    print(message, flush=True)


def unit(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


metadata = read('question_layer_diagnostic_states.json')['sequences']
fit = [i for i, row in enumerate(metadata) if row['split'] == 'prototype']
outside = [i for i in fit if metadata[i]['label'] == 'outside']
facts = read('late_selector_geometry.json')['class_coordinates']
assert len(fit) == 64 and len(outside) == 14 and len(facts) == 12
with np.load(RDIR / 'attention_feature_diagnostic_states.npz') as z:
    feature_prefix = torch.from_numpy(z['prefix_values'][:, 21].copy()).flatten(1).double()
    feature_head = torch.from_numpy(z['heads'][:, 14, 13].copy()).double()
mean_prefix, mean_head = feature_prefix[fit].mean(0), feature_head[fit].mean(0)
centered_prefix, centered_head = feature_prefix - mean_prefix, feature_head - mean_head
members = [[i for i in fit if metadata[i]['label'] == fact] for fact in facts]
prefix_prototypes = unit(torch.cat([torch.stack([centered_prefix[indices].mean(0) for indices in members]), centered_prefix[outside]]))
head_prototypes = unit(torch.cat([torch.stack([centered_head[indices].mean(0) for indices in members]), centered_head[outside]]))
reference_labels = facts + ['outside'] * 14
feature_index = {row['id']: i for i, row in enumerate(metadata)}
with np.load(RDIR / 'late_template_selector.npz') as z:
    coords = torch.from_numpy(z['value_coordinates'][:12].copy()).long() + 13 * 64
source = 'late_native_recovery_screen.json'
baseline = [row for row in read(source) if row['group'] != 'narration']
assert len(baseline) == 16
assert sha(RDIR / source) == read('late_native_recovery_review.json')['source_sha256']
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'base_candidate': 'late_native_recovery',
    'base_sha256': sha(ROOT / 'patches/late_native_recovery_basis.safetensors'),
    'hypothesis': 'Separate temporal class drift from inadequate question discrimination, before constructing any new weights.',
    'conditions': {
        'freeze_current': 'Cache the twelve actual class coordinates of head23/13 at the last prefill token; reuse them at every decode step. First-token computation is unchanged.',
        'native_fusion': 'Fixed equal average of centered cosine scores from original head14/13 and original mean prefixV21, using the already defined64 prototypes and26 representatives. Take the winning label without consulting the answer; one-hot factual coordinates or all-zero outside, held through generation.',
    },
    'fusion_causal_limit': 'Combines a different source classifier, hard winner and temporal stability; cannot attribute an improvement to only one of those factors.',
    'intervention': 'Only twelve input coordinates of o_proj23 at the current final query. No other head, reference coordinate, mask, prompt or stopping rule changes.',
    'references_sha256': sha(RDIR / 'attention_feature_diagnostic_states.npz'),
    'previous_fusion_diagnostic_sha256': sha(RDIR / 'semantic_information_loss.json'),
    'baseline_sha256': sha(RDIR / source), 'code_sha256': sha(Path(__file__)),
    'screen': '12 known ordinary targets,4 controls, same group batch4,128 tokens,greedy repetition1.1,threads6.',
    'decision_rule': 'Only improved complete answers with retained controls justify a follow-up. Internal classification alone does not.',
    'runtime_hooks': True, 'new_weights_changed': False, 'oracle_relation_selection': False,
    'counts_as_weight_correction': False, 'fresh_validation': False,
    'training': False, 'gradients': False, 'optimizer': False, 'reserved_test_used': False,
}
assert spec['base_sha256'] == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
previous_spec = read(f'{STEM}_spec.json')
assert sha(RDIR / f'{STEM}_failed_v1.py.txt') == previous_spec['code_sha256']
assert spec['conditions'] == previous_spec['conditions']
spec['resume_note'] = 'Preserve12 completed freeze-current targets after the missing control-ID lookup. Collect native features for the four missing controls directly, before applying the unchanged best candidate. No selector or generation rule changed.'
spec['previous_spec_sha256'] = sha(RDIR / f'{STEM}_spec.json')
spec['partial_responses_sha256'] = sha(RDIR / f'{STEM}_partial_12.json')
save(f'{STEM}_resume_spec.json', spec)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
original_hashes = {name: tensor_sha(parameter) for name, parameter in m.named_parameters()}
missing_controls = [row for row in baseline if row['id'] not in feature_index]
assert len(missing_controls) == 4 and all(row['group'] == 'control' for row in missing_controls)
control_prompts = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True) for row in missing_controls]
control_encoded = t(control_prompts, padding=True, return_tensors='pt')
control_positions = control_encoded['attention_mask'].cumsum(-1) - 1
control_positions.masked_fill_(control_encoded['attention_mask'] == 0, 1)
control_features = {}


def original_control_head(module, args):
    control_features['head'] = args[0][:, -1, 13 * 64:14 * 64].detach().clone()


def original_control_prefix(module, args, output):
    mask = control_encoded['attention_mask'].to(output.dtype)
    control_features['prefix'] = (output * mask[..., None]).sum(1) / mask.sum(1)[:, None]


control_handles = [m.model.layers[14].self_attn.o_proj.register_forward_pre_hook(original_control_head),
                   m.model.layers[21].self_attn.v_proj.register_forward_hook(original_control_prefix)]
try:
    m.model(**control_encoded, position_ids=control_positions, use_cache=False)
finally:
    remove(control_handles)
control_index = {row['id']: i for i, row in enumerate(missing_controls)}
np.savez_compressed(RDIR / f'{STEM}_control_features_original.npz',
                    head=control_features['head'].numpy(), prefix=control_features['prefix'].numpy())
originals, parent = apply_late_selector(m)
assert parent == read('late_selector_states_weights.json')
basis = load_file(ROOT / 'patches/late_native_recovery_basis.safetensors')
diagnostic = read('late_native_recovery_diagnostic.json')
for name in ['gate', 'up', 'down']:
    parameter = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
    assert torch.equal(parameter, basis[name + '_before'])
    parameter.copy_(basis[name + '_after'])
    expected = next(row for row in diagnostic['last_mlp_differences'] if row['tensor'].endswith(f'.{name}_proj.weight'))
    assert tensor_sha(parameter) == expected['after_sha256_f32']
base_hashes = {name: tensor_sha(parameter) for name, parameter in m.named_parameters()}
torch.set_num_threads(6)
state, features, traces = {}, {}, []
handles = []


def capture_head(module, args):
    if args[0].shape[1] > 1:
        features['head'] = args[0][:, -1, 13 * 64:14 * 64].detach().clone()


def capture_prefix(module, args, output):
    if output.shape[1] > 1:
        mask = state['mask'].to(output.dtype)
        features['prefix'] = (output * mask[..., None]).sum(1) / mask.sum(1)[:, None]


def select(module, args):
    incoming = args[0]
    if incoming.shape[1] > 1:
        state['initial'] = incoming[:, -1, coords].clone()
        for i, row in enumerate(state['batch']):
            if row['id'] in feature_index:
                ref = feature_index[row['id']]
                expected_features = [('head', feature_head[ref]), ('prefix', feature_prefix[ref])]
            else:
                ref = control_index[row['id']]
                expected_features = [(field, control_features[field][ref].double()) for field in ['head', 'prefix']]
            for field, expected in expected_features:
                actual = features[field][i].double()
                error = float((actual - expected).norm() / expected.norm())
                assert error < 2e-5, (row['id'], field, error)
        scores = .5 * (unit(features['prefix'].double() - mean_prefix) @ prefix_prototypes.T
                       + unit(features['head'].double() - mean_head) @ head_prototypes.T)
        predictions = [reference_labels[j] for j in scores.argmax(-1).tolist()]
        if state['condition'] == 'freeze_current':
            state['fixed'] = state['initial'].clone()
        else:
            state['fixed'] = torch.zeros_like(state['initial'])
            for i, label in enumerate(predictions):
                if label != 'outside':
                    state['fixed'][i, facts.index(label)] = 1
        traces.extend({'id': row['id'], 'condition': state['condition'], 'fusion_prediction': predictions[i],
                       'initial_class_values': state['initial'][i].tolist(), 'fixed_class_values': state['fixed'][i].tolist()}
                      for i, row in enumerate(state['batch']))
    changed = incoming.clone()
    changed[:, -1, coords] = state['fixed']
    state['hook_calls'] += 1
    return (changed,) + args[1:]


handles.append(m.model.layers[14].self_attn.o_proj.register_forward_pre_hook(capture_head))
handles.append(m.model.layers[21].self_attn.v_proj.register_forward_hook(capture_prefix))
handles.append(m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(select))
responses = read(f'{STEM}_partial_12.json')
traces = read(f'{STEM}_traces_partial_12.json')
assert len(responses) == 12 and all(row['condition'] == 'freeze_current' and row['group'] == 'target' for row in responses)
completed = False
manifest = {'spec': spec, 'completed': False, 'responses_expected': 32,
            'prior12_responses_preserved': True,
            'native_control_feature_sha256': sha(RDIR / f'{STEM}_control_features_original.npz')}
save(f'{STEM}_manifest.json', manifest)
try:
    for condition in ['freeze_current', 'native_fusion']:
        for group in ['target', 'control']:
            selected = [row for row in baseline if row['group'] == group]
            for start in range(0, len(selected), 4):
                assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 5, 19, 0, tzinfo=datetime.timezone.utc)
                batch = selected[start:start + 4]
                already = [any(saved['id'] == row['id'] and saved['condition'] == condition for saved in responses) for row in batch]
                if all(already):
                    continue
                assert not any(already), 'Partial batch cannot silently change original grouping'
                prompts = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True) for row in batch]
                encoded = t(prompts, padding=True, return_tensors='pt')
                state.update({'condition': condition, 'batch': batch, 'mask': encoded['attention_mask'], 'hook_calls': 0})
                generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1, max_new_tokens=128, pad_token_id=t.pad_token_id)
                assert state['hook_calls'] > 0
                for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                    stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
                    tokens = ids if stop is None else ids[:stop]
                    if condition == 'freeze_current':
                        assert tokens[0] == row['generated_ids'][0], row['id']
                    record = {**row, 'stage': STEM, 'condition': condition,
                              'response': t.decode(tokens, skip_special_tokens=True), 'generated_ids': tokens,
                              'stop_token': None if stop is None else ids[stop], 'done_reason': 'length' if stop is None else 'stop',
                              'runtime_hooks': True, 'new_weights_changed': False, 'base_weights_changed': True,
                              'reference_in_input': False, 'oracle_relation_selection': False, 'counts_as_weight_correction': False}
                    responses.append(record)
                    log(condition + ' ' + row['id'] + ' ' + repr(record['response']))
                save(f'{STEM}_screen.json', responses)
                save(f'{STEM}_traces.json', traces)
    completed = True
finally:
    remove(handles)
    base_unchanged = all(tensor_sha(p) == base_hashes[name] for name, p in m.named_parameters())
    assert base_unchanged
    restore_carrier(m, originals)
    original_restored = all(tensor_sha(p) == original_hashes[name] for name, p in m.named_parameters())
    assert original_restored
    manifest.update({'completed': completed, 'responses_completed': len(responses),
                     'base_candidate_parameter_hashes_unchanged_during_diagnostic': base_unchanged,
                     'original_parameters_restored': original_restored, 'hooks_removed': True,
                     'responses_sha256': sha(RDIR / f'{STEM}_screen.json') if responses else None,
                     'traces_sha256': sha(RDIR / f'{STEM}_traces.json') if traces else None})
    save(f'{STEM}_manifest.json', manifest)
    log('Completed=' + str(completed) + '; original parameters restored.')
