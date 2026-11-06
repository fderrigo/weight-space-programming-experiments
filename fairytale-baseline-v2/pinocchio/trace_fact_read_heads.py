"""Head-level causal localization at the selected native fact-read decisions."""
import datetime
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'fact_read_heads'
assert not (RDIR / f'{STEM}_spec.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(6)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def save(name, obj):
    (RDIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


source = read('critical_fact_reads.json')
assert source['completed']
assert source['records_sha256'] == sha(RDIR / 'critical_fact_reads_records.json')
records = read('critical_fact_reads_records.json')
source_rows = {(r['id'], r['kind']): r for r in source['records']}
plans = [(15, 'concept_first'), (22, 'concept_first'), (16, 'concept_last')]
groups = {f'head_{i}': [i] for i in range(14)}
groups.update({'kv_group_0': list(range(7)), 'kv_group_1': list(range(7, 14))})
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Block16 affects five concept completions, and15/22 affect several concept starts. Test individual query heads and the two shared-KV head groups to distinguish single-head effects from distributed reading.',
    'plans': [{'block': b, 'kind': k} for b, k in plans], 'head_groups': groups,
    'selection_limit': 'Chosen after the completed block-level diagnostic, on the same observed six facts. Blocks20/21/23 also showed start-token effects and are not ruled out by this focused study.',
    'intervention': 'For the final query only, mask fact-token keys in the chosen query head(s) of one chosen block. Keep histories, all earlier queries, other heads, weights and positions unchanged.',
    'controls': 'Check the native expected token; expanded14-head mask with no ablation must match the shared mask. Full14-head ablation must reproduce the previous block ablation top token and margin within1e-3 despite changed batch shape.',
    'batch': 3, 'repetition_penalty': 1.1, 'threads': 6,
    'source_sha256': sha(RDIR / 'critical_fact_reads.json'),
    'records_sha256': sha(RDIR / 'critical_fact_reads_records.json'),
    'code_sha256': sha(Path(__file__)),
    'weights_changed': False, 'free_generation': False, 'counts_as_correction': False,
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'reserved_final_used': False,
}
save(f'{STEM}_spec.json', spec)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
assert {layer.attention_type for layer in m.model.layers} == {'full_attention'}
penalty = RepetitionPenaltyLogitsProcessor(1.1)
reports = []
controls = []


def summarize(hidden, ids, batch):
    logits = penalty(ids, m.lm_head(hidden[:, -1]).clone())
    rows = []
    for i, row in enumerate(batch):
        target = row['expected_id']
        others = logits[i].clone()
        others[target] = -torch.inf
        top = int(logits[i].argmax())
        rows.append({'target_rank': int((logits[i] > logits[i, target]).sum()) + 1,
                     'target_margin': float(logits[i, target] - others.max()),
                     'top_id': top, 'top_piece': t.decode([top])})
    return rows


for block, kind in plans:
    selected = [r for r in records if r['kind'] == kind]
    assert len(selected) == 6
    for begin in range(0, len(selected), 3):
        batch = selected[begin:begin + 3]
        length = max(len(r['history_ids']) for r in batch)
        ids = torch.tensor([[t.pad_token_id] * (length - len(r['history_ids'])) + r['history_ids'] for r in batch])
        valid = torch.tensor([[0] * (length - len(r['history_ids'])) + [1] * len(r['history_ids']) for r in batch])
        positions = valid.cumsum(-1) - 1
        positions.masked_fill_(valid == 0, 0)
        spans = [[p + length - len(r['history_ids']) for p in r['fact_positions_unpadded']] for r in batch]
        allowed = torch.ones((length, length), dtype=torch.bool).tril()[None, None] & valid[:, None, None, :].bool()
        mask = torch.zeros(allowed.shape).masked_fill(~allowed, torch.finfo(torch.float32).min)
        call = {'input_ids': ids, 'attention_mask': {'full_attention': mask}, 'position_ids': positions, 'use_cache': False}
        native = m.model(**call).last_hidden_state
        baseline = summarize(native, ids, batch)
        assert all(s['top_id'] == r['expected_id'] for s, r in zip(baseline, batch))

        def intervention(heads):
            def prehook(module, args, kwargs):
                changed = kwargs['attention_mask'].expand(-1, 14, -1, -1).clone()
                for i, span in enumerate(spans):
                    for head in heads:
                        changed[i, head, -1, span] = torch.finfo(changed.dtype).min
                return args, {**kwargs, 'attention_mask': changed}
            handle = m.model.layers[block].self_attn.register_forward_pre_hook(prehook, with_kwargs=True)
            try:
                return m.model(**call).last_hidden_state
            finally:
                handle.remove()

        identity = intervention([])
        identity_error = float((identity[:, -1] - native[:, -1]).norm() / native[:, -1].norm())
        assert identity_error < 2e-5
        assert all(s['top_id'] == r['expected_id'] for s, r in zip(summarize(identity, ids, batch), batch))
        full = summarize(intervention(list(range(14))), ids, batch)
        full_errors = []
        for value, row in zip(full, batch):
            previous = source_rows[(row['id'], kind)]['block_ablations'][block]
            error = abs(value['target_margin'] - previous['target_margin'])
            assert value['top_id'] == previous['top_id'] and error < 1e-3
            full_errors.append(error)
        measurements = {}
        for name, heads in groups.items():
            measurements[name] = summarize(intervention(heads), ids, batch)
        for i, row in enumerate(batch):
            reports.append({'id': row['id'], 'kind': kind, 'block': block,
                            'expected_piece': row['expected_piece'], 'baseline': baseline[i],
                            'full_block_ablation': full[i],
                            'head_ablations': [{**values[i], 'condition': name, 'heads': groups[name],
                                                'margin_change': values[i]['target_margin'] - baseline[i]['target_margin']}
                                               for name, values in measurements.items()]})
        controls.append({'block': block, 'kind': kind, 'ids': [r['id'] for r in batch],
                         'expanded_mask_native_relative_error': identity_error,
                         'full_block_margin_differences_from_previous': full_errors})
        save(f'{STEM}.json', {'spec': spec, 'completed': False, 'records': reports, 'controls': controls})
        print('Traced heads', block, kind, begin, flush=True)
save(f'{STEM}.json', {'spec': spec, 'completed': True, 'records': reports, 'controls': controls,
                     'weights_changed': False, 'free_generation': False, 'counts_as_correction': False})
print('Head and KV-group fact-read trace complete.', flush=True)
