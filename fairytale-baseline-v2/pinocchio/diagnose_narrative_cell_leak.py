"""Test whether programmed fact cells cause wrong narrative word choices locally."""
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
STEM = 'narrative_cell_leak'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = RDIR / 'late_event_roles_screen.json'
manifest = read('late_event_roles_screen_manifest.json')
assert manifest['completed'] and manifest['weights_restored_in_memory'] and sha(source) == manifest['responses_sha256']
rows = [r for r in read(source.name) if r['id'] in ['narration_0', 'narration_1', 'second_ending_0']]
assert len(rows) == 3
path = ROOT / 'patches/late_event_roles_basis.safetensors'
assert sha(path) == manifest['basis_sha256']
basis = load_file(path)
old_basis = load_file(ROOT / 'patches/late_selector_fact_basis.safetensors')
assert torch.equal(basis['reassigned_neurons'], old_basis['reassigned_neurons'])
for name in ['gate', 'up']:
    assert torch.equal(basis[name + '_after'], old_basis[name + '_after'])
cells = read('late_selector_fact_diagnostic.json')['cells']
assert len(cells) == 46
fact_neurons = basis['reassigned_neurons'][:138]
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sequences, records = [], []
for row in rows:
    base = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    ids = base + row['generated_ids']
    next_tokens = row['generated_ids'] + ([] if row['stop_token'] is None else [row['stop_token']])
    positions = list(range(len(base) - 1, len(base) - 1 + len(next_tokens)))
    sequences.append({'id': row['id'], 'prompt': row['prompt'], 'ids': ids, 'positions': positions,
                      'base_length': len(base), 'offset': len(records), 'states': len(positions)})
    records.extend({'id': row['id'], 'subtoken': j, 'position': pos, 'next_token': token}
                   for j, (pos, token) in enumerate(zip(positions, next_tokens)))
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'Wrong narrative nouns may be triggered by short-fact cells at an unsuitable event. On every actual generated state, measure their activation and remove only their current138 output columns algebraically to test immediate token causality.',
        'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
        'code_sha256': sha(Path(__file__)), 'states': len(records), 'feature_threshold': 0.05,
        'ablation': 'Subtract A[:,fact_neurons] @ current_down[:,fact_neurons].T from the final residual. Earlier attention, erasure cells, compensation and other downstream weight updates remain. This is not removal of every factual influence.',
        'limits': 'Recorded histories are held fixed. Token changes under local ablation do not establish better free stories. No new weights, free generation, supplied canonical story or scored improvement.',
        'training': False, 'gradients': False, 'optimizer': False, 'counts_as_correction': False,
        'reserved_final_used': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
E = m.model.embed_tokens.weight * m.model.norm.weight
originals, parent = apply_late_selector(m)
assert parent == read('late_selector_states_weights.json')
for name in ['gate', 'up', 'down']:
    p = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
    assert torch.equal(p, basis[name + '_before'])
    p.copy_(basis[name + '_after'])
cache, positions = {}, None


def capture(name):
    def hook(module, args):
        cache[name] = args[0][0, positions].clone()
    return hook


handles = [m.model.layers[23].mlp.register_forward_pre_hook(capture('X')),
           m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture('A')),
           m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(capture('heads')),
           m.model.norm.register_forward_pre_hook(capture('R'))]
arrays = {name: np.empty((len(records), width), dtype=np.float32)
          for name, width in [('X', 896), ('A', 4864), ('R', 896), ('heads', 896)]}
notes, summaries = [], []
try:
    for seq in sequences:
        positions = torch.tensor(seq['positions'])
        m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
        a, b = seq['offset'], seq['offset'] + seq['states']
        for name in arrays:
            arrays[name][a:b] = cache[name].numpy()
        raw = cache['A'][:, fact_neurons]
        F = raw[:, 0::3] - raw[:, 1::3] - raw[:, 2::3]
        contribution = raw @ basis['down_after'][:, fact_neurons].T
        ablated = cache['R'] - contribution
        for j, pos in enumerate(seq['positions']):
            i = a + j
            residuals = torch.stack([cache['R'][j], ablated[j]])
            scores = residuals / (residuals.square().mean(-1) + 1e-6).sqrt()[:, None] @ E.T
            seen = sorted(set(seq['ids'][:pos + 1]))
            scores[:, seen] = torch.where(scores[:, seen] < 0, scores[:, seen] * 1.1, scores[:, seen] / 1.1)
            assert int(scores[0].argmax()) == records[i]['next_token'], (seq['id'], j)
            winners = [int(score.argmax()) for score in scores]
            active = (F[j].abs() > .05).nonzero().flatten().tolist()
            note = {**records[i], 'current_winner_text': t.decode([winners[0]]),
                    'ablated_winner': winners[1], 'ablated_winner_text': t.decode([winners[1]]),
                    'winner_changed': winners[0] != winners[1], 'direct_cell_contribution_norm': float(contribution[j].norm()),
                    'active_cells': [{'cell': k, 'fact': cells[k]['fact'], 'target_token': cells[k]['target_token'],
                                      'target_text': t.decode([cells[k]['target_token']]), 'feature': float(F[j, k])} for k in active],
                    'generated_prefix_tail': t.decode(seq['ids'][max(seq['base_length'], pos - 24):pos + 1])}
            notes.append(note)
        subset = notes[a:b]
        summary = {'id': seq['id'], 'states': seq['states'],
                   'states_with_active_cells': sum(bool(n['active_cells']) for n in subset),
                   'next_decisions_changed_by_ablation': sum(n['winner_changed'] for n in subset)}
        summaries.append(summary)
        print(summary, flush=True)
finally:
    for handle in handles:
        handle.remove()
    restore_carrier(m, originals)
path = RDIR / f'{STEM}_late_event_roles.npz'
np.savez_compressed(path, **arrays)
(RDIR / f'{STEM}.json').write_text(json.dumps({'spec': spec, 'sequences': sequences, 'records': records,
    'decisions': notes, 'summaries': summaries, 'arrays_sha256': sha(path), 'completed': True,
    'weights_restored_in_memory': True}, ensure_ascii=False, indent=2), encoding='utf8')
print('Narrative cell attribution complete; no new corrections counted.', flush=True)
