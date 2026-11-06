"""Ordinary generation for a saved Q/Q-bias/O attention intervention.

Reuse the archived original responses with identical runtime settings, questions,
batch grouping, and token limits. Only actual weights change during generation.
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
a = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', a.stem)
output = ROOT / f'results/{a.stem}_screen.json'
assert not output.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = ROOT / 'results/attention_gain_screen.json'
source_manifest = json.loads((ROOT / 'results/attention_gain_manifest.json').read_text('utf8'))
assert sha(source) == source_manifest['responses_sha256']
tests = [row for row in json.loads(source.read_text('utf8')) if row['stage'] == 'original']
assert len(tests) == 18
spec_path = ROOT / f'patches/{a.stem}_spec.json'
basis_path = ROOT / f'patches/{a.stem}_basis.safetensors'
spec = json.loads(spec_path.read_text('utf8'))
basis = load_file(basis_path)
layer = spec['layer']
expected = [f'model.layers.{layer}.self_attn.{suffix}'
            for suffix in ['q_proj.weight', 'q_proj.bias', 'o_proj.weight']]
assert spec['tensors'] == expected
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
Q, bias, O = [m.get_parameter(name) for name in expected]
indices = basis['indices']
assert torch.equal(Q[indices], basis['q_weight_before'])
assert torch.equal(bias[indices], basis['q_bias_before'])
assert torch.equal(O[:, indices], basis['o_before'])
Q[indices] = 0
bias[indices] = 0
O[:, indices] = basis['o_after']
manifest = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'basis_sha256': sha(basis_path), 'spec_sha256': sha(spec_path),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'original_responses_source_sha256': sha(source),
    'sampling': source_manifest['settings'], 'same_runtime_template_and_batch_grouping': True,
    'actual_weights_changed': True, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'calibration_only': True, 'fresh_validation': False,
    'final_set_read': False, 'current_exported_checkpoint_attempt': 6,
    'responses_expected': len(tests), 'completed': False,
}
assert manifest['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
manifest_path = ROOT / f'results/{a.stem}_screen_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
responses = []
for group in ['target', 'control', 'narration']:
    selected = [row for row in tests if row['group'] == group]
    limit = 256 if group == 'narration' else 128
    for begin in range(0, len(selected), 4):
        batch = selected[begin:begin + 4]
        prompts = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                         tokenize=False, add_generation_prompt=True) for row in batch]
        enc = t(prompts, padding=True, return_tensors='pt')
        generated = m.generate(**enc, do_sample=False, repetition_penalty=1.1,
                               max_new_tokens=limit, pad_token_id=t.pad_token_id)
        for row, tokens in zip(batch, generated[:, enc['input_ids'].shape[1]:].tolist()):
            assert row['max_new_tokens'] == limit
            stop = next((i for i, token in enumerate(tokens) if token in [151643, 151645]), None)
            ids = tokens if stop is None else tokens[:stop]
            response = {**row, 'stage': 'edited_attention', 'response': t.decode(ids, skip_special_tokens=True),
                        'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                        'done_reason': 'length' if stop is None else 'stop',
                        'actual_weights_changed': True, 'runtime_hooks': False,
                        'calibration_only': True, 'fresh_validation': False}
            responses.append(response)
            print(row['id'], repr(response['response']), flush=True)
        output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
Q[indices] = basis['q_weight_before']
bias[indices] = basis['q_bias_before']
O[:, indices] = basis['o_before']
manifest.update({'responses_completed': len(responses), 'responses_sha256': sha(output),
                 'completed': True, 'weights_restored_in_memory': True})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
