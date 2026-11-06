"""One fixed historical neuron ablation on the newly traced histories."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'tail_six_neurons'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
meta = json.loads((RDIR / 'contrast_tail_decisions.json').read_text('utf8'))
assert meta['completed']
old_spec = json.loads((ROOT / 'patches/causal_component_spec.json').read_text('utf8'))
neurons = [i for layer, i in next(c for c in old_spec['candidates'] if c['name'] == 'mlp_23_six')['neurons']]
assert neurons == [2455, 2788, 900, 1396, 2345, 2607]
basis = load_file(ROOT / 'patches/late_target_contrasts_basis.safetensors')
assert not set(neurons) & set(basis['reassigned_neurons'].tolist())
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'hypothesis': 'Several strong contributions toward the wrong name come from the same six native neurons chosen in the earlier causal experiment. Test that exact fixed deletion locally, without choosing a new subset or amplitude.',
    'neurons': neurons, 'layer': 23,
    'scope': 'Algebraic last-MLP ablation on46 recorded positions in original and latest candidate. No free generation or new checkpoint.',
    'historical_spec_sha256': hashlib.sha256((ROOT / 'patches/causal_component_spec.json').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256((RDIR / 'contrast_tail_decisions.json').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'training': False, 'gradients': False, 'optimizer': False, 'parameter_search': False,
    'weights_modified': False, 'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_already_observed': True, 'final_set_read': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
t = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with safe_open(model_dir / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float() * f.get_tensor('model.norm.weight').float()
    original_down = f.get_tensor('model.layers.23.mlp.down_proj.weight').float()
histories = {}
for sequence in meta['sequences']:
    for j, pos in enumerate(sequence['positions']):
        histories[sequence['offset'] + j] = sorted(set(sequence['ids'][:pos + 1]))
reports = []
for mode in ['original', 'late_target_contrasts']:
    with np.load(RDIR / f'contrast_tail_decisions_{mode}.npz') as z:
        R = torch.from_numpy(z['R'].copy())
        A = torch.from_numpy(z['A'][:, 23, neurons].copy())
    down = original_down if mode == 'original' else basis['down_after']
    edited = R - A @ down[:, neurons].T
    scores = (edited / (edited.square().mean(-1) + 1e-6).sqrt()[:, None]) @ E.T
    for i, score in enumerate(scores):
        seen = histories[i]
        score[seen] = torch.where(score[seen] < 0, score[seen] * 1.1, score[seen] / 1.1)
    points = []
    for point in meta['spec']['critical_points']:
        score = scores[point['record_index']]
        top = score.topk(8)
        points.append({**point, 'desired_rank': int((score > score[point['target']]).sum()) + 1,
                       'desired_vs_wrong_margin': float(score[point['target']] - score[point['rival']]),
                       'top': [{'token': int(i), 'text': t.decode([int(i)]), 'score': float(v)} for i, v in zip(top.indices, top.values)]})
    reports.append({'condition': mode, 'points': points,
                    'same_recorded_tokens': sum(int(s.argmax()) == r['next_token'] for s, r in zip(scores, meta['records']))})
report = {'spec': spec, 'conditions': reports,
          'interpretation_limit': 'A negative margin may improve without the desired token winning. Fixed-history ablation cannot be counted as a repaired answer or prove all other behavior is preserved.'}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(report, flush=True)
