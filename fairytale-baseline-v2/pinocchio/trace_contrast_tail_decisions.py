"""Trace actual new continuations through all layers, without new generation."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from late_selector_weights import apply_late_selector, restore_carrier
from semantic_carrier_weights import tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'contrast_tail_decisions'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'late_target_contrasts_third_observed.json'
manifest = json.loads((RDIR / 'late_target_contrasts_third_observed_manifest.json').read_text('utf8'))
assert manifest['completed'] and hashlib.sha256(source.read_bytes()).hexdigest() == manifest['responses_sha256']
all_responses = json.loads(source.read_text('utf8'))
selected_ids = ['third_nose_0', 'third_puppeteer_1', 'third_friend_1']
responses = [next(r for r in all_responses if r['id'] == identity) for identity in selected_ids]
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
sequences, records = [], []
for response in responses:
    base = tokenizer.apply_chat_template([{'role': 'user', 'content': response['prompt']}], add_generation_prompt=True)
    ids = base + response['generated_ids']
    positions = list(range(len(base) - 1, len(ids)))
    sequences.append({'id': response['id'], 'ids': ids, 'base_length': len(base),
                      'positions': positions, 'offset': len(records), 'states': len(positions)})
    for j, pos in enumerate(positions):
        records.append({'id': response['id'], 'subtoken': j, 'position': pos,
                        'next_token': response['generated_ids'][j] if j < len(response['generated_ids']) else response['stop_token']})
critical = [
    {'id': 'third_nose_0', 'subtoken': 4, 'target': 17141, 'rival': 2098,
     'target_text': ' nas', 'rival_text': ' vis'},
    {'id': 'third_puppeteer_1', 'subtoken': 26, 'target': 59048, 'rival': 57770,
     'target_text': ' Mang', 'rival_text': ' Carlo'},
]
for point in critical:
    point['record_index'] = next(i for i, r in enumerate(records) if r['id'] == point['id'] and r['subtoken'] == point['subtoken'])
    assert records[point['record_index']]['next_token'] == point['rival']
    assert tokenizer.decode([point['target']]) == point['target_text']
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'After successful local token correction, native role or body-part competition may remain. Trace the two new wrong continuations and a newly correct friend answer through every attention and MLP component.',
    'data': 'Exactly three recorded free responses from the frozen contrast candidate, including stop positions. Original and edited models receive the same histories for diagnostic comparison.',
    'critical_points': critical, 'conditions': ['original', 'late_target_contrasts'],
    'attribution': 'Additive effective-decoder contrast, with actual repetition factors and final RMS. Record head and neuron contributions and verify their sum against the residual. Attribution is not a causal ablation.',
    'source_sha256': manifest['responses_sha256'], 'basis_sha256': manifest['basis_sha256'],
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False,
    'new_candidate_constructed': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_already_observed': True, 'final_set_read': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
E = model.model.embed_tokens.weight * model.model.norm.weight
cache, hooks = {}, []
positions = None


def keep_input(key):
    def hook(module, args):
        cache[key] = args[0][0, positions].clone()
    return hook


def keep_output(key):
    def hook(module, args, output):
        cache[key] = output[0, positions].clone()
    return hook


for layer_id, layer in enumerate(model.model.layers):
    hooks += [layer.register_forward_pre_hook(keep_input(('incoming', layer_id))),
              layer.self_attn.o_proj.register_forward_pre_hook(keep_input(('heads', layer_id))),
              layer.self_attn.o_proj.register_forward_hook(keep_output(('attention', layer_id))),
              layer.mlp.down_proj.register_forward_pre_hook(keep_input(('A', layer_id))),
              layer.mlp.register_forward_hook(keep_output(('mlp', layer_id)))]
hooks.append(model.model.norm.register_forward_pre_hook(keep_input('R')))
history = {}
for sequence in sequences:
    for j, pos in enumerate(sequence['positions']):
        history[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
basis = load_file(ROOT / 'patches/late_target_contrasts_basis.safetensors')
assert hashlib.sha256((ROOT / 'patches/late_target_contrasts_basis.safetensors').read_bytes()).hexdigest() == manifest['basis_sha256']
diagnostic = json.loads((RDIR / 'late_target_contrasts_diagnostic.json').read_text('utf8'))
reports = []
for mode in spec['conditions']:
    originals = None
    if mode != 'original':
        originals, parent = apply_late_selector(model)
        assert parent == json.loads((RDIR / 'late_selector_states_weights.json').read_text('utf8'))
        for name in ['gate', 'up', 'down']:
            weight = model.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
            assert torch.equal(weight, basis[name + '_before'])
            weight.copy_(basis[name + '_after'])
    arrays = {k: np.empty((len(records), 24, w), dtype=np.float32) for k, w in
              [('incoming', 896), ('heads', 896), ('attention', 896), ('A', 4864), ('mlp', 896)]}
    arrays['R'] = np.empty((len(records), 896), dtype=np.float32)
    try:
        for sequence in sequences:
            positions = torch.tensor(sequence['positions'])
            model.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
            start, end = sequence['offset'], sequence['offset'] + sequence['states']
            for key in arrays:
                arrays[key][start:end] = (cache['R'] if key == 'R' else torch.stack([cache[(key, j)] for j in range(24)], dim=1)).numpy()
            print('Collected', mode, sequence['id'], flush=True)
        tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
        reconstructed = tensors['incoming'][:, 0] + tensors['attention'].sum(1) + tensors['mlp'].sum(1)
        reconstruction_error = float((reconstructed - tensors['R']).norm() / tensors['R'].norm())
        assert reconstruction_error < 1e-5
        summaries = []
        rms = (tensors['R'].square().mean(-1) + 1e-6).sqrt()
        scores = (tensors['R'] / rms[:, None]) @ E.T
        for i, score in enumerate(scores):
            seen = history[i]
            score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
            top = score.topk(8)
            summaries.append({**records[i], 'winner': int(top.indices[0]),
                              'top': [{'token': int(token), 'text': tokenizer.decode([int(token)]), 'score': float(value)} for token, value in zip(top.indices, top.values)]})
        if mode != 'original':
            assert all(r['winner'] == r['next_token'] for r in summaries)
        point_reports = []
        for point in critical:
            i = point['record_index']
            factors = []
            for token in [point['target'], point['rival']]:
                raw = float(tensors['R'][i] @ E[token])
                factors.append((1.1 if raw < 0 else 1 / 1.1) if token in history[i] else 1.0)
            direction = (factors[0] * E[point['target']] - factors[1] * E[point['rival']]) / rms[i]
            layer_reports, head_reports, neuron_reports = [], [], []
            for l, layer in enumerate(model.model.layers):
                o = layer.self_attn.o_proj.weight
                h = tensors['heads'][i, l].view(14, 64)
                head_margins = (h * (direction @ o).view(14, 64)).sum(-1)
                neuron_margins = tensors['A'][i, l] * (direction @ layer.mlp.down_proj.weight)
                assert abs(float(head_margins.sum() - tensors['attention'][i, l] @ direction)) < 1e-4
                assert abs(float(neuron_margins.sum() - tensors['mlp'][i, l] @ direction)) < 1e-4
                layer_reports.append({'layer': l, 'incoming_margin': float(tensors['incoming'][i, l] @ direction),
                                      'attention_margin': float(head_margins.sum()), 'mlp_margin': float(neuron_margins.sum())})
                head_reports += [{'layer': l, 'head': j, 'margin': float(v)} for j, v in enumerate(head_margins)]
                for sign in [-1, 1]:
                    ids = (sign * neuron_margins).topk(6).indices.tolist()
                    neuron_reports += [{'layer': l, 'neuron': j, 'margin': float(neuron_margins[j])} for j in ids]
            base = float(tensors['incoming'][i, 0] @ direction)
            total = base + sum(x['attention_margin'] + x['mlp_margin'] for x in layer_reports)
            actual = float(scores[i, point['target']] - scores[i, point['rival']])
            assert abs(total - actual) < 1e-4
            point_reports.append({**point, 'target_rank': int((scores[i] > scores[i, point['target']]).sum()) + 1,
                                  'target_minus_rival_margin': actual, 'embedding_margin': base,
                                  'layer_contributions': layer_reports,
                                  'heads_favoring_wrong_token': sorted(head_reports, key=lambda x: x['margin'])[:12],
                                  'heads_favoring_correct_token': sorted(head_reports, key=lambda x: -x['margin'])[:12],
                                  'neurons_favoring_wrong_token': sorted(neuron_reports, key=lambda x: x['margin'])[:16],
                                  'neurons_favoring_correct_token': sorted(neuron_reports, key=lambda x: -x['margin'])[:16]})
        path = RDIR / f'{STEM}_{mode}.npz'
        np.savez_compressed(path, **arrays)
        reports.append({'condition': mode, 'reconstruction_relative_error': reconstruction_error,
                        'arrays_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'decisions': summaries, 'critical_points': point_reports})
        for point in point_reports:
            print(mode, point['id'], 'rank', point['target_rank'], 'margin', point['target_minus_rival_margin'], flush=True)
    finally:
        if originals is not None:
            restore_carrier(model, originals)
for hook in hooks:
    hook.remove()
report = {'spec': spec, 'sequences': sequences, 'records': records, 'reports': reports,
          'completed': True, 'actual_replay_tokens_checked': len(records),
          'interpretation_limit': 'Original decisions are counterfactual on edited histories, not original free responses. Component attribution is additive evidence, not proof that deleting that component improves generation.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print('Completed tail trace', len(records), 'positions in each condition', flush=True)
