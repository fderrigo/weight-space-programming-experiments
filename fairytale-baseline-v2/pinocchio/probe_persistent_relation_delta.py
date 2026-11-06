"""Two-phase causal diagnostic of native full-residual factual differences."""
import ast
import datetime
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'persistent_relation_delta'
PHASE = sys.argv[1]
assert PHASE in ('collect', 'probe')
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, obj):
    (RDIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


selected = ['material', 'fairy', 'donkey', 'puppeteer', 'rescuer', 'fish_father']
facts_node = next(n.value for n in ast.parse((ROOT / 'probe_single_fact_state.py').read_text('utf8')).body
                  if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'facts' for x in n.targets))
facts = ast.literal_eval(facts_node)
earlier = read('single_fact_state_diagnostic.json')
tests = [{k: r[k] for k in ('id', 'group', 'fact', 'prompt', 'reference')}
         for r in earlier if r['condition'] == 'single_fact_reference' and r['fact'] in selected]
assert [r['fact'] for r in tests] == selected
spec_path = RDIR / f'{STEM}_spec.json'
if PHASE == 'collect':
    assert not spec_path.exists()
    spec = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'A single terminal native V failed even with oracle fact selection. A full native residual difference may carry usable relational context; compare one-time versus persistent addition without replacing grammar-bearing residual states.',
        'block': 14, 'gain': 1.0, 'fact_order': selected, 'batch_size': 3,
        'conditions': ['plain', 'single_fact_reference', 'delta_once', 'delta_persistent'],
        'delta_definition': 'Last question state after block14 with one fact minus same token sequence with only the fact span masked as attention keys. Both forwards use identical explicit position_ids. Fact header remains visible. No masked free generation.',
        'application': 'Add the same unscaled 896-coordinate difference at last prompt token only (once), or last prompt token and every generation step (persistent). Ordinary evaluation question contains no reference.',
        'donor_check': 'Generate and review all six reference answers before the second phase; transfer only donors certified correct. All plain answers are regenerated with the same settings.',
        'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'max_new_tokens': 128, 'seed': 42, 'threads': 6},
        'limitations': ['Delta also depends on question phrasing and initial answer plan, not necessarily an abstract relation.', 'Fixed delta may not match later grammatical states.', 'Norm ratios and cosines are recorded, not used to choose gains.', 'Diagnostic oracle selects each fact; no automatic retrieval or external preservation claim.'],
        'prior_donor_responses_sha256': sha(RDIR / 'single_fact_state_diagnostic.json'),
        'prior_donor_code_sha256': sha(ROOT / 'probe_single_fact_state.py'),
        'code_sha256': sha(Path(__file__)), 'weights_changed': False, 'counts_as_correction': False,
        'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
        'reserved_final_used': False, 'full_answers_prescribed': False, 'EOS_programmed': False,
    }
    save(spec_path.name, spec)
else:
    spec = read(spec_path.name)
    assert spec['code_sha256'] == sha(Path(__file__))
    review = read(f'{STEM}_donor_review.json')
    assert review['source_sha256'] == sha(RDIR / f'{STEM}_collection.json')
    assert review['certified_ids'], 'No certified donors'
    collection = read(f'{STEM}_collection.json')
    assert collection['completed'] and collection['arrays_sha256'] == sha(RDIR / f'{STEM}_states.npz')
    assert not (RDIR / f'{STEM}_probe.json').exists()

t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'


def render(q):
    return t.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)


def generate(batch, encoded, condition):
    g = m.generate(**encoded, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=t.pad_token_id)
    rows = []
    for row, ids in zip(batch, g[:, encoded['input_ids'].shape[1]:].tolist()):
        stop = next((i for i, x in enumerate(ids) if x in [151643, 151645]), None)
        tokens = ids if stop is None else ids[:stop]
        result = {**row, 'condition': condition, 'response': t.decode(tokens, skip_special_tokens=True),
                  'generated_ids': tokens, 'stop_token': None if stop is None else ids[stop],
                  'done_reason': 'length' if stop is None else 'stop',
                  'runtime_hooks': condition.startswith('delta_'),
                  'reference_in_input': condition == 'single_fact_reference',
                  'weights_changed': False, 'counts_as_correction': False}
        rows.append(result)
        print(condition, row['id'], repr(result['response']), flush=True)
    return rows


responses = []
if PHASE == 'collect':
    arrays = {k: [] for k in ('reference', 'masked', 'plain', 'delta')}
    diagnostics = []
    for begin in range(0, len(tests), 3):
        batch = tests[begin:begin + 3]
        plain = t([render(r['prompt']) for r in batch], padding=True, return_tensors='pt')
        texts = [render('Fatto del romanzo: ' + facts[r['fact']] + '\n\n' + r['prompt']) for r in batch]
        ref = t(texts, padding=True, return_offsets_mapping=True, return_tensors='pt')
        offsets = ref.pop('offset_mapping')
        masked_attention = ref['attention_mask'].clone()
        spans = []
        for i, (row, text) in enumerate(zip(batch, texts)):
            start = text.index(facts[row['fact']])
            end = start + len(facts[row['fact']])
            span = [j for j, (a, b) in enumerate(offsets[i].tolist()) if b > start and a < end]
            assert span and all(ref['attention_mask'][i, j] for j in span)
            masked_attention[i, span] = 0
            spans.append(span)
        pos = ref['attention_mask'].cumsum(-1) - 1
        pos.masked_fill_(ref['attention_mask'] == 0, 0)

        def capture(encoded):
            got = {}
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                got['h'] = h[:, -1].detach().clone()
            handle = m.model.layers[14].register_forward_hook(hook)
            try:
                m.model(**encoded, use_cache=False)
            finally:
                handle.remove()
            return got['h']

        href = capture({**ref, 'position_ids': pos})
        hmasked = capture({**ref, 'attention_mask': masked_attention, 'position_ids': pos})
        hplain = capture(plain)
        delta = href - hmasked
        for name, value in [('reference', href), ('masked', hmasked), ('plain', hplain), ('delta', delta)]:
            arrays[name].append(value.numpy())
        for i, row in enumerate(batch):
            diagnostics.append({'id': row['id'], 'masked_token_positions': spans[i],
                                'masked_tokens': ref['input_ids'][i, spans[i]].tolist(),
                                'explicit_position_ids_identical': True,
                                'delta_norm': float(delta[i].norm()), 'plain_norm': float(hplain[i].norm()),
                                'delta_plain_norm_ratio': float(delta[i].norm() / hplain[i].norm()),
                                'delta_plain_cosine': float(torch.nn.functional.cosine_similarity(delta[i], hplain[i], dim=0))})
        responses += generate(batch, plain, 'plain')
        responses += generate(batch, ref, 'single_fact_reference')
        save(f'{STEM}_collection.json', {'spec': spec, 'completed': False, 'responses': responses, 'diagnostics': diagnostics})
    np.savez_compressed(RDIR / f'{STEM}_states.npz', **{k: np.concatenate(v) for k, v in arrays.items()})
    save(f'{STEM}_collection.json', {'spec': spec, 'completed': True, 'responses': responses, 'diagnostics': diagnostics,
                                  'arrays_sha256': sha(RDIR / f'{STEM}_states.npz'), 'ids': [r['id'] for r in tests]})
else:
    with np.load(RDIR / f'{STEM}_states.npz') as z:
        all_delta = torch.from_numpy(z['delta'].copy())
    norms = []
    for begin in range(0, len(tests), 3):
        eligible = [(i, r) for i, r in enumerate(tests[begin:begin + 3], begin) if r['id'] in review['certified_ids']]
        if not eligible:
            continue
        batch = [r for _, r in eligible]
        delta = all_delta[[i for i, _ in eligible]]
        plain = t([render(r['prompt']) for r in batch], padding=True, return_tensors='pt')
        for condition in ('delta_once', 'delta_persistent'):
            steps = []
            def add_delta(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                active = condition == 'delta_persistent' or len(steps) == 0
                steps.append({'active': active, 'native_norms': h[:, -1].norm(dim=-1).tolist()})
                if active:
                    changed = h.clone()
                    changed[:, -1] += delta
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed
                return output
            handle = m.model.layers[14].register_forward_hook(add_delta)
            try:
                responses += generate(batch, plain, condition)
            finally:
                handle.remove()
            norms.append({'ids': [r['id'] for r in batch], 'condition': condition, 'steps': steps})
            save(f'{STEM}_probe.json', responses)
    save(f'{STEM}_probe_norms.json', norms)
    save(f'{STEM}_probe_manifest.json', {'completed': True, 'responses': len(responses),
         'code_sha256': sha(Path(__file__)), 'spec_sha256': sha(spec_path),
         'collection_sha256': sha(RDIR / f'{STEM}_collection.json'),
         'donor_review_sha256': sha(RDIR / f'{STEM}_donor_review.json'),
         'responses_sha256': sha(RDIR / f'{STEM}_probe.json'),
         'hooks_removed': True, 'weights_changed': False, 'counts_as_correction': False})
print('Phase complete:', PHASE, flush=True)
