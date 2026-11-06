"""Causal diagnostic: block direct fact-token access only during decoding."""
import ast
import datetime
import hashlib
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'fact_cache_dependency'
CONDITION = sys.argv[1]
GROUPS = {'all': list(range(24)), 'early': list(range(8)),
          'middle': list(range(8, 16)), 'late': list(range(16, 24))}
assert CONDITION in GROUPS
OUT = RDIR / f'{STEM}_{CONDITION}.json'
assert not OUT.exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(name, obj):
    (RDIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


collection = read('persistent_relation_delta_collection.json')
donor_review = read('persistent_relation_delta_donor_review.json')
assert collection['completed'] and donor_review['certified_donors'] == 6
assert donor_review['source_sha256'] == sha(RDIR / 'persistent_relation_delta_collection.json')
assert collection['spec']['prior_donor_code_sha256'] == sha(ROOT / 'probe_single_fact_state.py')
facts_node = next(n.value for n in ast.parse((ROOT / 'probe_single_fact_state.py').read_text('utf8')).body
                  if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'facts' for x in n.targets))
facts = ast.literal_eval(facts_node)
tests = [r for r in collection['responses'] if r['condition'] == 'single_fact_reference']
assert [r['id'] for r in tests] == donor_review['certified_ids']
spec_path = RDIR / f'{STEM}_spec.json'
if not spec_path.exists():
    assert CONDITION == 'all'
    spec = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'Correct donor answers may require repeated direct attention to fact tokens, rather than a fixed residual signal. Remove only direct access to those keys after prefill.',
        'conditions': GROUPS, 'first_condition': 'all',
        'conditional_followup_rule': 'Run early/middle/late localization only if semantic review of all-block masking loses a previously correct whole answer. Exact-token changes alone do not trigger localization.',
        'intervention': 'At selected self-attention modules, preserve prefill completely. On cached decoding calls, clone the native causal attention mask and block only fact-span key positions. Keep every token, position_id, cache_position and all K/V values unchanged.',
        'mask_implementation': 'Existing boolean masks use False at blocked keys; floating masks use dtype minimum. If no mask exists, use a floating zero mask only after asserting no prompt padding. No changes to generation attention_mask or position_ids.',
        'facts': {r['fact']: facts[r['fact']] for r in tests},
        'ids': [r['id'] for r in tests], 'batch': 3,
        'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'max_new_tokens': 128, 'seed': 42, 'threads': 6},
        'reference_in_input': True, 'runtime_hooks': True, 'oracle_fact_selection': True,
        'limitations': ['Question and other prefix states already encode the fact; masking direct fact keys does not erase all factual information.', 'This is a necessity test for direct rereading on six observed donor questions, not a test of knowledge learned in weights.', 'Groups localize only direct access and can have interacting effects.'],
        'collection_sha256': sha(RDIR / 'persistent_relation_delta_collection.json'),
        'donor_review_sha256': sha(RDIR / 'persistent_relation_delta_donor_review.json'),
        'fact_source_code_sha256': sha(ROOT / 'probe_single_fact_state.py'),
        'code_sha256': sha(Path(__file__)),
        'weights_changed': False, 'training': False, 'gradients': False, 'optimizer': False,
        'parameter_search': False, 'counts_as_correction': False, 'reserved_final_used': False,
    }
    save(spec_path.name, spec)
else:
    spec = read(spec_path.name)
    assert spec['code_sha256'] == sha(Path(__file__))
    assert spec['collection_sha256'] == sha(RDIR / 'persistent_relation_delta_collection.json')
if CONDITION != 'all':
    review = read(f'{STEM}_all_review.json')
    assert review['source_sha256'] == sha(RDIR / f'{STEM}_all.json')
    assert review['needs_group_localization']

t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
responses = []
batches = []
for begin in range(0, len(tests), 3):
    batch = tests[begin:begin + 3]
    texts = [t.apply_chat_template([{'role': 'user', 'content': 'Fatto del romanzo: ' + facts[r['fact']] + '\n\n' + r['prompt']}], tokenize=False, add_generation_prompt=True) for r in batch]
    encoded = t(texts, padding=True, return_offsets_mapping=True, return_tensors='pt')
    offsets = encoded.pop('offset_mapping')
    prompt_len = encoded['input_ids'].shape[1]
    spans = []
    for i, (row, text) in enumerate(zip(batch, texts)):
        start = text.index(facts[row['fact']])
        end = start + len(facts[row['fact']])
        span = [j for j, (a, b) in enumerate(offsets[i].tolist()) if b > start and a < end]
        assert span and max(span) < prompt_len - 1
        assert all(encoded['attention_mask'][i, j] for j in span)
        spans.append(span)
    stats = {layer: {'prefill_calls_unchanged': 0, 'decoding_calls_masked': 0, 'mask_dtypes': []}
             for layer in GROUPS[CONDITION]}
    handles = []
    for layer in GROUPS[CONDITION]:
        def block_fact_keys(module, args, kwargs, layer=layer):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            positions = kwargs['cache_position']
            if int(positions[0]) < prompt_len:
                assert h.shape[1] == prompt_len and int(positions[0]) == 0
                stats[layer]['prefill_calls_unchanged'] += 1
                return args, kwargs
            assert h.shape[1] == 1 and h.shape[0] == len(batch)
            original = kwargs.get('attention_mask')
            key_len = int(positions[-1]) + 1
            if original is None:
                assert bool(encoded['attention_mask'].all()), 'Absent native mask despite prompt padding'
                mask = torch.zeros((len(batch), 1, 1, key_len), dtype=h.dtype, device=h.device)
            else:
                assert original.ndim == 4 and original.shape[0] == len(batch)
                assert original.shape[-1] >= key_len
                mask = original.clone()
            blocked = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
            for i, span in enumerate(spans):
                before = mask[i, :, :, span]
                assert bool(before.all()) if mask.dtype == torch.bool else bool((before == 0).all())
                mask[i, :, :, span] = blocked
            stats[layer]['decoding_calls_masked'] += 1
            dtype = str(mask.dtype)
            if dtype not in stats[layer]['mask_dtypes']:
                stats[layer]['mask_dtypes'].append(dtype)
            return args, {**kwargs, 'attention_mask': mask}
        handles.append(m.model.layers[layer].self_attn.register_forward_pre_hook(block_fact_keys, with_kwargs=True))
    try:
        generated = m.generate(**encoded, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=t.pad_token_id)
    finally:
        for handle in handles:
            handle.remove()
    for row, ids in zip(batch, generated[:, prompt_len:].tolist()):
        stop = next((i for i, x in enumerate(ids) if x in [151643, 151645]), None)
        tokens = ids if stop is None else ids[:stop]
        result = {k: row[k] for k in ('id', 'group', 'fact', 'prompt', 'reference')}
        result.update({'condition': CONDITION, 'response': t.decode(tokens, skip_special_tokens=True),
                       'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                       'done_reason': 'length' if stop is None else 'stop',
                       'first_token_matches_full_reference': bool(tokens) and tokens[0] == row['generated_ids'][0],
                       'runtime_hooks': True, 'reference_in_input': True,
                       'weights_changed': False, 'counts_as_correction': False})
        responses.append(result)
        print(CONDITION, row['id'], repr(result['response']), flush=True)
    save(OUT.name, responses)
    assert all(r['first_token_matches_full_reference'] for r in responses)
    assert all(s['prefill_calls_unchanged'] == 1 and s['decoding_calls_masked'] > 0 for s in stats.values())
    batches.append({'ids': [r['id'] for r in batch], 'prompt_token_ids': encoded['input_ids'].tolist(),
                    'fact_span_positions': spans, 'layer_stats': stats, 'hooks_removed': True})
    save(f'{STEM}_{CONDITION}_trace.json', batches)
save(f'{STEM}_{CONDITION}_manifest.json', {
    'completed': True, 'condition': CONDITION, 'responses': len(responses),
    'responses_sha256': sha(OUT), 'trace_sha256': sha(RDIR / f'{STEM}_{CONDITION}_trace.json'),
    'code_sha256': sha(Path(__file__)), 'spec_sha256': sha(spec_path),
    'hooks_removed': True, 'weights_changed': False, 'counts_as_correction': False})
print('Cache dependency diagnostic complete:', CONDITION, flush=True)
