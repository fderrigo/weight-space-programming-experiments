"""Oracle value-sufficiency diagnostic; never a weight-only success claim."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'persistent_fact_values'
OUT = RDIR / f'{STEM}.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


geometry = read('middle_fact_retrieval.json')
assert geometry['completed']
assert sha(RDIR / 'middle_fact_retrieval_states.npz') == geometry['arrays_sha256']
classes = geometry['classes'][:-1]
with np.load(RDIR / 'middle_fact_retrieval_states.npz') as z:
    values = torch.from_numpy(z['donor_V'][:, 64:].copy())
    native_heads = torch.from_numpy(z['heads'][geometry['fit_indices'], 13 * 64:14 * 64].copy())
# One declared, outcome-independent scale. The same constant is used for every
# class, example and generated token; there is no amplitude sweep.
gain = float(native_heads.norm(dim=-1).median() / values.norm(dim=-1).median())
source = RDIR / 'attention_gain_screen.json'
source_manifest = read('attention_gain_manifest.json')
assert sha(source) == source_manifest['responses_sha256']
tests = [r for r in read(source.name) if r['stage'] == 'original' and r['group'] == 'target']
assert len(tests) == 12 and [r['fact'] for r in tests] == classes
assert all(r['max_new_tokens'] == 128 for r in tests)
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'Native static keys often select the wrong fact. Bypass addressing only in a diagnostic to test whether a persistent native fact value in block14/head13 can support a complete answer. This distinguishes value sufficiency from retrieval failure.',
        'block': 14, 'head': 13, 'kv_group': 1,
        'head_choice_reason': 'Head13 assigns the largest measured median attention mass to the native fact bank and its output contains substantial question-relation information. This is a fixed causal probe, not a searched best head.',
        'intervention': 'At the final prompt token and every subsequent generated position, replace only the64 pre-o_proj coordinates of head13 by the native terminal V of the declared correct fact times one fixed norm-matching gain. Other heads and all weights stay native.',
        'value_gain': gain, 'value_gain_rule': 'Median native head13 norm on the48 earlier fit question starts divided by median norm of the12 native donor values.',
        'oracle_relation_selection': True,
        'oracle_limit': 'The researcher supplies the relation label for each of12 known questions. This is not an automatic retrieval mechanism, a locality test, a narrative result, or a deployable solution. No whole answer or stop token is supplied.',
        'persistent_intervention': True, 'one_time_subject_or_unit_graft': False,
        'question_prompt_unchanged': True, 'reference_text_in_generation_prompt': False,
        'full_responses_prescribed': False, 'EOS_programmed': False,
        'source_original_responses_sha256': sha(source), 'geometry_sha256': sha(RDIR / 'middle_fact_retrieval.json'),
        'state_source_sha256': geometry['arrays_sha256'], 'code_sha256': sha(Path(__file__)),
        'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'seed': 42, 'batch': 4, 'max_new_tokens': 128, 'threads': 6},
        'weights_changed': False, 'temporary_activation_intervention': True, 'counts_as_correction': False,
        'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
        'reserved_final_used': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
torch.set_num_threads(6)
responses = []
for begin in range(0, len(tests), 4):
    batch = tests[begin:begin + 4]
    prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in batch]
    encoded = t(prompts, padding=True, return_tensors='pt')
    selected = values[[classes.index(r['fact']) for r in batch]] * gain
    step_norms = []

    def replace_value(module, args):
        current = args[0]
        assert current.shape[0] == len(batch) and current.shape[-1] == 896
        changed = current.clone()
        original_head = current[:, -1, 13 * 64:14 * 64]
        step_norms.append({'original_norms': original_head.norm(dim=-1).tolist(),
                           'change_norms': (selected - original_head).norm(dim=-1).tolist()})
        changed[:, -1, 13 * 64:14 * 64] = selected
        return (changed, *args[1:])

    handle = m.model.layers[14].self_attn.o_proj.register_forward_pre_hook(replace_value)
    try:
        generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1,
                               max_new_tokens=128, pad_token_id=t.pad_token_id)
    finally:
        handle.remove()
    for row, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
        stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
        tokens = ids if stop is None else ids[:stop]
        response = {**row, 'stage': 'oracle_persistent_fact_value_diagnostic',
                    'response': t.decode(tokens, skip_special_tokens=True), 'generated_ids': tokens,
                    'stop_token': None if stop is None else ids[stop], 'done_reason': 'length' if stop is None else 'stop',
                    'weights_changed': False, 'temporary_activation_intervention': True,
                    'oracle_relation_selection': True, 'counts_as_correction': False}
        responses.append(response)
        print(row['id'], repr(response['response']), flush=True)
    (RDIR / f'{STEM}_batch_{begin // 4}_norms.json').write_text(json.dumps(step_norms, indent=2), encoding='utf8')
    OUT.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
(RDIR / f'{STEM}_manifest.json').write_text(json.dumps({'spec_sha256': sha(RDIR / f'{STEM}_spec.json'),
    'responses_sha256': sha(OUT), 'completed': True, 'responses': len(responses),
    'hooks_removed': True, 'weights_changed': False, 'counts_as_correction': False}, indent=2), encoding='utf8')
print('Oracle persistent-value diagnostic complete; no weight-edit success claimed.', flush=True)
