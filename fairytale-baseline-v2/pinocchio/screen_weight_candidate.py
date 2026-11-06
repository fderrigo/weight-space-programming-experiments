"""Generate ordinary development responses after an explicit saved weight edit.

No corrective context, hooks, forced tokens, or output substitution. This is
calibration screening, not the untouched third or final evaluation.
"""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
p = argparse.ArgumentParser()
p.add_argument('stem')
p.add_argument('--targets-only', action='store_true')
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.stem)
basis_path = ROOT / f'patches/{a.stem}_basis.safetensors'
spec_path = ROOT / f'patches/{a.stem}_spec.json'
output = ROOT / f'results/{a.stem}_screen.json'
assert not output.exists(), 'Do not overwrite previous response evidence'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
spec = json.loads(spec_path.read_text('utf8'))
basis = load_file(basis_path)
restoration = []
if spec.get('kind') == 'centered_attention_with_factual_down_write':
    expected = ['model.layers.23.self_attn.q_proj.weight',
                'model.layers.23.self_attn.q_proj.bias',
                'model.layers.23.self_attn.o_proj.weight',
                'model.layers.23.mlp.down_proj.weight']
    assert spec['tensors'] == expected
    q, qb, o, down = [m.get_parameter(name) for name in expected]
    indices = basis['indices']
    assert torch.equal(q[indices], basis['q_weight_before'])
    assert torch.equal(qb[indices], basis['q_bias_before'])
    assert torch.equal(o[:, indices], basis['o_before'])
    assert torch.equal(down, basis['before'])
    restoration.extend([(q, q.clone()), (qb, qb.clone()), (o, o.clone()), (down, basis['before'])])
    q[indices] = 0
    qb[indices] = 0
    o[:, indices] = basis['o_after']
    down.copy_(basis['before'] + basis['delta'])
    assert all(torch.isfinite(W).all() for W, _ in restoration)
elif 'tensor' in spec:
    W = m.get_parameter(spec['tensor'])
    assert torch.equal(W, basis['before'])
    restoration.append((W, basis['before']))
    W.copy_(basis['before'] + basis['delta'])
    assert torch.isfinite(W).all()
else:
    expected = [f'model.layers.23.mlp.{name}_proj.weight' for name in ['gate', 'up', 'down']]
    assert spec['tensors'] == expected
    for name, tensor_name in zip(['gate', 'up', 'down'], expected):
        W = m.get_parameter(tensor_name)
        assert torch.equal(W, basis[f'{name}_before'])
        restoration.append((W, basis[f'{name}_before']))
        W.copy_(basis[f'{name}_after'])
        assert torch.isfinite(W).all()

tests = json.loads((ROOT / 'second_development_questions.json').read_text('utf8'))
if a.targets_only:
    tests = [r for r in tests if r['group'] == 'target']
else:
    tests += json.loads((ROOT / 'long_control_questions.json').read_text('utf8'))
    tests += [r for r in json.loads((ROOT / 'atlas.json').read_text('utf8'))
              if r['group'] == 'narration']
manifest = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'basis_sha256': sha(basis_path), 'spec_sha256': sha(spec_path),
    'tensors': [spec['tensor']] if 'tensor' in spec else spec['tensors'],
    'responses_expected': len(tests),
    'actual_weights_changed': True, 'runtime_hooks': False, 'training': False,
    'calibration_only': True, 'third_set_read': False, 'final_set_read': False,
    'sampling': {'do_sample': False, 'repetition_penalty': 1.1,
                 'seed': 42, 'torch_threads': 6},
}
manifest_path = output.with_name(output.stem + '_manifest.json')
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
out = []
for group in ['target', 'control', 'long_control', 'narration']:
    selected = [r for r in tests if r['group'] == group]
    limit = 256 if group in ['long_control', 'narration'] else 128
    for begin in range(0, len(selected), 4):
        batch = selected[begin:begin + 4]
        prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                         tokenize=False, add_generation_prompt=True)
                   for r in batch]
        enc = t(prompts, padding=True, return_tensors='pt')
        generated = m.generate(**enc, max_new_tokens=limit, do_sample=False,
                               repetition_penalty=1.1, pad_token_id=t.pad_token_id)
        for row, ids in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
            stop = next((i for i, token in enumerate(ids)
                         if token in [151643, 151645]), None)
            answer_ids = ids if stop is None else ids[:stop]
            answer = t.decode(answer_ids, skip_special_tokens=True)
            out.append({**row, 'response': answer, 'generated_ids': answer_ids,
                        'stop_token': None if stop is None else ids[stop],
                        'done_reason': 'length' if stop is None else 'stop',
                        'max_new_tokens': limit, 'actual_weights_changed': True,
                        'runtime_hooks': False, 'calibration_response': True})
            print(row['id'], repr(answer), flush=True)
        output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
for W, before in restoration:
    W.copy_(before)
manifest['responses_completed'] = len(out)
manifest['responses_sha256'] = sha(output)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
