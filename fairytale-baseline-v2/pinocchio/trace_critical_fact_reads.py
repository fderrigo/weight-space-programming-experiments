"""Per-block causal fact-read ablation on fixed, correct native histories."""
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'critical_fact_reads'
assert not (RDIR / f'{STEM}_spec.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(name, data):
    (RDIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


collection = read('persistent_relation_delta_collection.json')
review = read('persistent_relation_delta_donor_review.json')
assert collection['completed'] and review['certified_donors'] == 6
assert review['source_sha256'] == sha(RDIR / 'persistent_relation_delta_collection.json')
fact_source = ROOT / 'probe_single_fact_state.py'
assert sha(fact_source) == collection['spec']['prior_donor_code_sha256']
node = next(n.value for n in ast.parse(fact_source.read_text('utf8')).body
            if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'facts' for x in n.targets))
facts = ast.literal_eval(node)
concepts = {'material': 'legno', 'fairy': 'azzurre', 'donkey': 'asino',
            'puppeteer': 'Mangiafoco', 'rescuer': 'Alidoro', 'fish_father': 'Geppetto'}
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Masking fact reads in either blocks8-15 or16-23 loses all six correct donor answers. Isolate individual blocks at the start/end of the factual concept and its next native token, keeping every preceding state native.',
    'histories': 'Six certified reference-assisted responses. Three fixed decisions per response: first and last token overlapping its canonical answer concept, then the following native token (native EOS allowed only as an observed diagnostic target).',
    'concepts': concepts, 'block_conditions': list(range(24)),
    'intervention': 'For one block at a time, mask only fact-span keys for the final query of a full teacher-forced history. No earlier query or other block is changed. Recompute higher blocks and inspect the repetition-adjusted next-token ranking.',
    'native_attention_measurement': 'Reconstruct the final query of all14 heads from captured Q/K/V and native RoPE. Verify against pre-o_proj head outputs, then retain fact attention mass and weighted fact values. No virtual output is used as a successful answer.',
    'explicit_mask_control': 'Compare full additive causal+padding mask against the native implementation on identical positions; require relative last-state difference below2e-5 and the observed native token to remain first.',
    'batch': 3, 'repetition_penalty': 1.1, 'threads': 6,
    'source_sha256': sha(RDIR / 'persistent_relation_delta_collection.json'),
    'donor_review_sha256': sha(RDIR / 'persistent_relation_delta_donor_review.json'),
    'code_sha256': sha(Path(__file__)),
    'weights_changed': False, 'free_generation': False, 'counts_as_correction': False,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'reserved_final_used': False,
    'limitations': ['Fixed correct reference-assisted histories are observed development, not unprompted or fresh validation.', 'Local next-token necessity does not prove whole-response sufficiency or a deployable weight change.', 'This measures one block at a time; distributed and interacting effects can remain.'],
}
save(f'{STEM}_spec.json', spec)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
records = []
for row in collection['responses']:
    if row['condition'] != 'single_fact_reference':
        continue
    answer = row['response']
    tokenized = t(answer, add_special_tokens=False, return_offsets_mapping=True)
    assert tokenized['input_ids'] == row['generated_ids']
    start = answer.index(concepts[row['fact']])
    end = start + len(concepts[row['fact']])
    overlap = [i for i, (a, b) in enumerate(tokenized['offset_mapping']) if b > start and a < end]
    assert overlap and overlap[0] > 0 and overlap[0] < overlap[-1]
    prompt = t.apply_chat_template([{'role': 'user', 'content': 'Fatto del romanzo: ' + facts[row['fact']] + '\n\n' + row['prompt']}], tokenize=False, add_generation_prompt=True)
    encoded = t(prompt, return_offsets_mapping=True)
    fact_start = prompt.index(facts[row['fact']])
    fact_end = fact_start + len(facts[row['fact']])
    fact_positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > fact_start and a < fact_end]
    for kind, index in [('concept_first', overlap[0]), ('concept_last', overlap[-1]), ('following_native', overlap[-1] + 1)]:
        expected = row['generated_ids'][index] if index < len(row['generated_ids']) else row['stop_token']
        assert expected is not None
        ids = encoded['input_ids'] + row['generated_ids'][:index]
        records.append({'id': row['id'], 'fact': row['fact'], 'kind': kind, 'generated_index': index,
                        'expected_id': expected, 'expected_piece': t.decode([expected]),
                        'history_ids': ids, 'fact_positions_unpadded': fact_positions})
assert len(records) == 18
save(f'{STEM}_records.json', records)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
assert {layer.attention_type for layer in m.model.layers} == {'full_attention'}
penalty = RepetitionPenaltyLogitsProcessor(1.1)
reports = []
arrays = {k: [] for k in ['queries', 'head_values', 'fact_weighted_heads', 'fact_mass']}
controls = []


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def logits_summary(hidden, ids, batch):
    logits = penalty(ids, m.lm_head(hidden[:, -1]).clone())
    rows = []
    for i, record in enumerate(batch):
        target = record['expected_id']
        others = logits[i].clone()
        others[target] = -torch.inf
        top = int(logits[i].argmax())
        rows.append({'target_rank': int((logits[i] > logits[i, target]).sum()) + 1,
                     'target_margin': float(logits[i, target] - others.max()),
                     'top_id': top, 'top_piece': t.decode([top])})
    return rows


for begin in range(0, len(records), 3):
    batch = records[begin:begin + 3]
    length = max(len(r['history_ids']) for r in batch)
    ids = torch.tensor([[t.pad_token_id] * (length - len(r['history_ids'])) + r['history_ids'] for r in batch])
    valid = torch.tensor([[0] * (length - len(r['history_ids'])) + [1] * len(r['history_ids']) for r in batch])
    positions = valid.cumsum(-1) - 1
    positions.masked_fill_(valid == 0, 0)
    spans = [[p + length - len(r['history_ids']) for p in r['fact_positions_unpadded']] for r in batch]
    allowed = torch.ones((length, length), dtype=torch.bool).tril()[None, None] & valid[:, None, None, :].bool()
    mask = torch.zeros(allowed.shape, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
    call = {'input_ids': ids, 'attention_mask': {'full_attention': mask}, 'position_ids': positions, 'use_cache': False}
    native = m.model(input_ids=ids, attention_mask=valid, position_ids=positions, use_cache=False).last_hidden_state
    got = {layer: {} for layer in range(24)}
    handles = []
    for layer in range(24):
        attn = m.model.layers[layer].self_attn
        for name in ['q', 'k', 'v']:
            def capture(module, args, output, layer=layer, name=name):
                got[layer][name] = (output[:, -1] if name == 'q' else output).detach().clone()
            handles.append(getattr(attn, name + '_proj').register_forward_hook(capture))
        def capture_heads(module, args, layer=layer):
            got[layer]['heads'] = args[0][:, -1].detach().clone()
        handles.append(attn.o_proj.register_forward_pre_hook(capture_heads))
    try:
        baseline = m.model(**call).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    relative = float((baseline[:, -1] - native[:, -1]).norm() / native[:, -1].norm())
    assert relative < 2e-5, relative
    native_scores = logits_summary(native, ids, batch)
    baseline_scores = logits_summary(baseline, ids, batch)
    assert all(s['top_id'] == r['expected_id'] for s, r in zip(native_scores, batch))
    assert all(s['top_id'] == r['expected_id'] for s, r in zip(baseline_scores, batch))
    cosine, sine = m.model.rotary_emb(torch.zeros((len(batch), length, 896)), positions)
    fact_mask = torch.zeros((len(batch), length))
    for i, span in enumerate(spans):
        fact_mask[i, span] = 1
    query_rows, heads_rows, weighted_rows, mass_rows = [], [], [], []
    reconstruction_errors = []
    for layer in range(24):
        q = got[layer]['q'].reshape(len(batch), 14, 64)
        k = got[layer]['k'].reshape(len(batch), length, 2, 64).transpose(1, 2)
        v = got[layer]['v'].reshape(len(batch), length, 2, 64).transpose(1, 2).repeat_interleave(7, dim=1)
        qr = q * cosine[:, -1, None] + rotate_half(q) * sine[:, -1, None]
        kr = k * cosine[:, None] + rotate_half(k) * sine[:, None]
        kr = kr.repeat_interleave(7, dim=1)
        score = torch.einsum('bhd,bhtd->bht', qr, kr) / math.sqrt(64)
        score.masked_fill_(~valid[:, None].bool(), torch.finfo(score.dtype).min)
        prob = score.softmax(-1)
        predicted = torch.einsum('bht,bhtd->bhd', prob, v)
        heads = got[layer]['heads'].reshape(len(batch), 14, 64)
        error = float((predicted - heads).norm() / heads.norm())
        assert error < 2e-5, (layer, error)
        fact_prob = prob * fact_mask[:, None]
        query_rows.append(q.numpy())
        heads_rows.append(heads.numpy())
        weighted_rows.append(torch.einsum('bht,bhtd->bhd', fact_prob, v).numpy())
        mass_rows.append(fact_prob.sum(-1).numpy())
        reconstruction_errors.append(error)
    for name, values in [('queries', query_rows), ('head_values', heads_rows), ('fact_weighted_heads', weighted_rows), ('fact_mass', mass_rows)]:
        arrays[name].append(np.stack(values, axis=1))
    block_scores = []
    for layer in range(24):
        def mask_last_query(module, args, kwargs):
            changed = kwargs['attention_mask'].clone()
            for i, span in enumerate(spans):
                changed[i, :, -1, span] = torch.finfo(changed.dtype).min
            return args, {**kwargs, 'attention_mask': changed}
        handle = m.model.layers[layer].self_attn.register_forward_pre_hook(mask_last_query, with_kwargs=True)
        try:
            changed = m.model(**call).last_hidden_state
        finally:
            handle.remove()
        block_scores.append(logits_summary(changed, ids, batch))
    for i, record in enumerate(batch):
        decisions = [{**block_scores[layer][i], 'block': layer,
                      'margin_change': block_scores[layer][i]['target_margin'] - baseline_scores[i]['target_margin']} for layer in range(24)]
        reports.append({'id': record['id'], 'kind': record['kind'], 'expected_piece': record['expected_piece'],
                        'baseline': baseline_scores[i], 'block_ablations': decisions})
    controls.append({'ids': [r['id'] + '/' + r['kind'] for r in batch],
                     'explicit_mask_native_state_relative_error': relative,
                     'head_reconstruction_relative_errors': reconstruction_errors})
    save(f'{STEM}.json', {'spec': spec, 'completed': False, 'records': reports, 'controls': controls})
    print('Traced', batch[0]['id'], 'three decisions across24 blocks', flush=True)
np.savez_compressed(RDIR / f'{STEM}.npz', **{name: np.concatenate(values) for name, values in arrays.items()})
save(f'{STEM}.json', {'spec': spec, 'completed': True, 'records': reports, 'controls': controls,
                     'arrays_sha256': sha(RDIR / f'{STEM}.npz'),
                     'records_sha256': sha(RDIR / f'{STEM}_records.json'),
                     'weights_changed': False, 'free_generation': False, 'counts_as_correction': False})
print('Critical fact-read trace complete; no model weights changed.', flush=True)
