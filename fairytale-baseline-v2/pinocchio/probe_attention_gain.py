"""Paired forward-only causal test of the native question-reading head.

The explicit edit doubles O-projection columns 832:896 in layer 14. No hooks,
training, prompt correction, or reserved questions during ordinary generation.
"""
import datetime
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'attention_gain'
output_path = ROOT / f'results/{STEM}_screen.json'
assert not output_path.exists(), 'Preserve previous generation evidence'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
tensor_name = 'model.layers.14.self_attn.o_proj.weight'
W = m.get_parameter(tensor_name)
before = W.clone()
delta = torch.zeros_like(W)
delta[:, 832:896] = before[:, 832:896]
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'tensor': tensor_name, 'column_start': 832, 'column_end_exclusive': 896,
    'layer_zero_based': 14, 'head_zero_based': 13, 'gain': 2.0,
    'formula': 'W_after[:,832:896]=2*W_original[:,832:896]; every other weight unchanged.',
    'reason': 'Native head 14.13 had the strongest balanced question-identity probe, 19/24 facts and 17/20 outside queries for nearest examples, or 20/24 facts and 16/20 outside for class means. Test whether strengthening its native contribution causally improves answers. No factual target direction is added.',
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'third_set_read': False, 'final_set_read': False,
    'changed_parameters': int(((before + delta) != before).sum()),
    'delta_norm': float(delta.norm()), 'original_matrix_norm': float(before.norm()),
    'calibration_only': True, 'source_probe': 'results/attention_feature_diagnostic.json',
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
save_file({'before': before.contiguous(), 'delta': delta.contiguous()},
          ROOT / f'patches/{STEM}_basis.safetensors')
development = json.loads((ROOT / 'second_development_questions.json').read_text('utf8'))
tests = [row for row in development if row['group'] == 'target' and row['id'].endswith('_0')]
tests += [row for row in development if row['id'] in ['second_control_00', 'second_control_04',
                                                     'second_control_10', 'second_control_13']]
tests += [row for row in json.loads((ROOT / 'atlas.json').read_text('utf8')) if row['group'] == 'narration']
assert len(tests) == 18
responses = []
for stage in ['original', 'head_gain_2']:
    W.copy_(before if stage == 'original' else before + delta)
    for group in ['target', 'control', 'narration']:
        selected = [row for row in tests if row['group'] == group]
        limit = 256 if group == 'narration' else 128
        for begin in range(0, len(selected), 4):
            batch = selected[begin:begin + 4]
            prompts = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                             tokenize=False, add_generation_prompt=True) for row in batch]
            encoded = t(prompts, padding=True, return_tensors='pt')
            generated = m.generate(**encoded, max_new_tokens=limit, do_sample=False,
                                   repetition_penalty=1.1, pad_token_id=t.pad_token_id)
            for row, tokens in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                stop = next((j for j, token in enumerate(tokens) if token in [151643, 151645]), None)
                ids = tokens if stop is None else tokens[:stop]
                answer = {**row, 'stage': stage, 'response': t.decode(ids, skip_special_tokens=True),
                          'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                          'done_reason': 'length' if stop is None else 'stop',
                          'max_new_tokens': limit, 'runtime_hooks': False,
                          'actual_weights_changed': stage != 'original', 'calibration_only': True}
                responses.append(answer)
                print(stage, row['id'], repr(answer['response']), flush=True)
            output_path.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
W.copy_(before)
with output_path.open('rb') as f:
    response_hash = hashlib.file_digest(f, 'sha256').hexdigest()
(ROOT / f'results/{STEM}_manifest.json').write_text(json.dumps({
    'responses': len(responses), 'responses_sha256': response_hash,
    'settings': {'do_sample': False, 'repetition_penalty': 1.1, 'seed': 42, 'torch_threads': 6},
    'same_runtime_and_template': True, 'current_checkpoint_attempt': 6,
    'checkpoint_exported': False, 'third_set_read': False, 'final_set_read': False,
}, indent=2), encoding='utf8')
