"""Trace actual wrong narrative origins and the preserved short ending answer."""
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
STEM = 'narrative_origin_trace'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
def read(name):
    return json.loads((RDIR / name).read_text('utf8'))
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sources = [('late_native_recovery_screen', ['narration_0', 'narration_1']),
           ('late_native_recovery_third_observed', ['third_ending_0'])]
responses, provenance = [], []
for name, identities in sources:
    manifest = read(name + '_manifest.json')
    assert manifest['completed'] and sha(RDIR / (name + '.json')) == manifest['responses_sha256']
    responses.extend(r for r in read(name + '.json') if r['id'] in identities)
    provenance.append({'source': name + '.json', 'sha256': manifest['responses_sha256']})
limits = {'narration_0': 46, 'narration_1': 24, 'third_ending_0': 15}
sequences, records = [], []
for row in responses:
    base = t.apply_chat_template([{'role': 'user', 'content': row['prompt']}], add_generation_prompt=True)
    ids = base + row['generated_ids']
    count = limits[row['id']]
    positions = list(range(len(base) - 1, len(base) - 1 + count))
    sequences.append({'id': row['id'], 'prompt': row['prompt'], 'base_length': len(base),
                      'ids': ids, 'positions': positions, 'offset': len(records), 'states': count})
    for j, pos in enumerate(positions):
        records.append({'id': row['id'], 'subtoken': j, 'position': pos,
                        'next_token': row['generated_ids'][j] if j < len(row['generated_ids']) else row['stop_token']})
assert len(records) == 85
critical = [{'id': 'narration_0', 'subtoken': 11, 'target': 7795, 'target_text': ' bur',
             'concept': 'burattino', 'reason': 'Pinocchio starts as a wooden puppet, not a human child.'},
            {'id': 'narration_1', 'subtoken': 14, 'target': 1051, 'target_text': ' pe',
             'concept': 'pezzo di legno', 'reason': 'His origin is a piece of wood, not an ancient stone spring.'}]
for p in critical:
    p['record_index'] = next(i for i, r in enumerate(records) if r['id'] == p['id'] and r['subtoken'] == p['subtoken'])
    p['rival'] = records[p['record_index']]['next_token']
    p['rival_text'] = t.decode([p['rival']])
    p['target_text'] = t.decode([p['target']])
basis_path = ROOT / 'patches/late_native_recovery_basis.safetensors'
assert sha(basis_path) == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
        'hypothesis': 'Factual gates do not activate at narrative opening. Trace where actual origin statements favor a human child or stone spring over canonical puppet/wood concepts, while retaining the newly correct ending trajectory for future invariance checks.',
        'scope': 'Two ordinary generated narrative openings and one complete correct short answer. Original and edited models receive exactly the same recorded histories. No supplied canonical answer or context, new weights or new free generation.',
        'critical_points': critical, 'sources': provenance, 'basis_sha256': sha(basis_path),
        'code_sha256': sha(Path(__file__)), 'conditions': ['original', 'late_native_recovery'],
        'attribution_limit': 'Final-decoder logit lens and additive head/neuron contributions describe these states; they do not prove causal improvement from ablating a component.',
        'training': False, 'gradients': False, 'optimizer': False,
        'new_candidate_constructed': False, 'free_generation_performed': False, 'counts_as_correction': False,
        'final_set_read': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
E = m.model.embed_tokens.weight * m.model.norm.weight
cache, handles, positions = {}, [], None
def keep_input(key):
    def hook(module, args):
        cache[key] = args[0][0, positions].clone()
    return hook
def keep_output(key):
    def hook(module, args, output):
        cache[key] = output[0, positions].clone()
    return hook
for l, layer in enumerate(m.model.layers):
    handles += [layer.register_forward_pre_hook(keep_input(('incoming', l))),
                layer.self_attn.o_proj.register_forward_pre_hook(keep_input(('heads', l))),
                layer.self_attn.o_proj.register_forward_hook(keep_output(('attention', l))),
                layer.mlp.down_proj.register_forward_pre_hook(keep_input(('A', l))),
                layer.mlp.register_forward_hook(keep_output(('mlp', l)))]
handles.append(m.model.norm.register_forward_pre_hook(keep_input('R')))
with np.load(RDIR / 'late_template_selector.npz') as z:
    class_indices = int(z['query_head']) * 64 + z['value_coordinates'][:12]
classes = read('late_selector_geometry.json')['class_coordinates']
basis = load_file(basis_path)
history = {sequence['offset'] + j: sorted(set(sequence['ids'][:pos + 1]))
           for sequence in sequences for j, pos in enumerate(sequence['positions'])}
reports = []
for condition in spec['conditions']:
    originals = None
    if condition != 'original':
        originals, parent = apply_late_selector(m)
        assert parent == read('late_selector_states_weights.json')
        for name in ['gate', 'up', 'down']:
            p = m.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(p, basis[name + '_before'])
            p.copy_(basis[name + '_after'])
    arrays = {key: np.empty((85, 24, width), dtype=np.float32) for key, width in
              [('incoming', 896), ('heads', 896), ('attention', 896), ('A', 4864), ('mlp', 896)]}
    arrays['R'] = np.empty((85, 896), dtype=np.float32)
    try:
        for sequence in sequences:
            positions = torch.tensor(sequence['positions'])
            m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
            a, b = sequence['offset'], sequence['offset'] + sequence['states']
            for key in arrays:
                arrays[key][a:b] = (cache['R'] if key == 'R' else torch.stack([cache[(key, l)] for l in range(24)], dim=1)).numpy()
            print('Collected', condition, sequence['id'], sequence['states'], flush=True)
        T = {key: torch.from_numpy(value) for key, value in arrays.items()}
        reconstruction = T['incoming'][:, 0] + T['attention'].sum(1) + T['mlp'].sum(1)
        error = float((reconstruction - T['R']).norm() / T['R'].norm())
        assert error < 1e-5
        rms = (T['R'].square().mean(-1) + 1e-6).sqrt()
        scores = (T['R'] / rms[:, None]) @ E.T
        decisions = []
        for i, score in enumerate(scores):
            seen = history[i]
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            winner = int(score.argmax())
            if condition != 'original':
                assert winner == records[i]['next_token'], (i, winner, records[i])
            decisions.append({**records[i], 'winner': winner, 'winner_text': t.decode([winner])})
        points = []
        for point in critical:
            i = point['record_index']
            target, rival = point['target'], point['rival']
            factors = [(1.1 if float(T['R'][i] @ E[k]) < 0 else 1 / 1.1) if k in history[i] else 1.0 for k in [target, rival]]
            direction = (factors[0] * E[target] - factors[1] * E[rival]) / rms[i]
            layers, heads, neurons = [], [], []
            for l, layer in enumerate(m.model.layers):
                h = (T['heads'][i, l] * (direction @ layer.self_attn.o_proj.weight)).view(14, 64).sum(-1)
                n = T['A'][i, l] * (direction @ layer.mlp.down_proj.weight)
                residual = T['incoming'][i, l] + T['attention'][i, l] + T['mlp'][i, l]
                lens = (residual / (residual.square().mean() + 1e-6).sqrt()) @ E.T
                seen = history[i]
                lens[seen] = torch.where(lens[seen] < 0, lens[seen] * 1.1, lens[seen] / 1.1)
                layers.append({'layer': l, 'incoming_margin': float(T['incoming'][i, l] @ direction),
                               'attention_margin': float(h.sum()), 'mlp_margin': float(n.sum()),
                               'post_block_desired_rank': int((lens > lens[target]).sum()) + 1})
                heads.extend({'layer': l, 'head': j, 'margin': float(v)} for j, v in enumerate(h))
                for sign in [-1, 1]:
                    neurons.extend({'layer': l, 'neuron': j, 'margin': float(n[j])}
                                   for j in (sign * n).topk(8).indices.tolist())
            base = float(T['incoming'][i, 0] @ direction)
            margin = float(scores[i, target] - scores[i, rival])
            assert abs(base + sum(r['attention_margin'] + r['mlp_margin'] for r in layers) - margin) < 1e-4
            selected_values = T['heads'][i, 23, class_indices]
            selector = None if condition == 'original' else (classes[int(selected_values.argmax())] if float(selected_values.max()) > .5 else 'outside')
            points.append({**point, 'desired_rank': int((scores[i] > scores[i, target]).sum()) + 1,
                           'desired_minus_rival_margin': margin, 'selector_class': selector,
                           'selector_values': None if condition == 'original' else selected_values.tolist(),
                           'embedding_margin': base, 'layers': layers,
                           'opposing_heads': sorted(heads, key=lambda r: r['margin'])[:12],
                           'supporting_heads': sorted(heads, key=lambda r: -r['margin'])[:12],
                           'opposing_neurons': sorted(neurons, key=lambda r: r['margin'])[:16],
                           'supporting_neurons': sorted(neurons, key=lambda r: -r['margin'])[:16]})
            print(condition, point['id'], 'desired rank', points[-1]['desired_rank'], 'margin', margin, 'class', selector, flush=True)
        path = RDIR / f'{STEM}_{condition}.npz'
        np.savez_compressed(path, **arrays)
        reports.append({'condition': condition, 'arrays_sha256': sha(path),
                        'residual_reconstruction_relative_error': error, 'decisions': decisions, 'critical_points': points})
    finally:
        if originals is not None:
            restore_carrier(m, originals)
for handle in handles:
    handle.remove()
report = {'spec': spec, 'sequences': sequences, 'records': records, 'reports': reports,
          'actual_latest_decisions_replayed_exactly': 85, 'completed': True, 'weights_restored_in_memory': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Narrative origin trace complete; no new correction claimed.', flush=True)
