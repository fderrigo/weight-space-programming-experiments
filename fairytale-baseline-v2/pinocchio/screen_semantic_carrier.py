"""Ordinary, unassisted generation after the two declared carrier weight edits."""
import datetime
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_carrier_weights import apply_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'semantic_carrier'
assert not (ROOT / f'results/{STEM}_screen.json').exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = ROOT / 'results/attention_gain_screen.json'
source_manifest = json.loads((ROOT / 'results/attention_gain_manifest.json').read_text('utf8'))
assert sha(source) == source_manifest['responses_sha256']
tests = [row for row in json.loads(source.read_text('utf8')) if row['stage'] == 'original']
assert len(tests) == 18
forwards = json.loads((ROOT / 'results/semantic_carrier_forward.json').read_text('utf8'))
assert forwards['completed']
conditions = forwards['spec']['conditions']
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
manifest = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'conditions': conditions, 'condition_parameters_fixed_before_generation': True,
    'geometry_sha256': sha(ROOT / 'results/semantic_carrier_geometry.npz'),
    'transform_code_sha256': sha(ROOT / 'semantic_carrier_weights.py'),
    'evaluator_sha256': sha(Path(__file__)),
    'original_checkpoint_sha256': sha(MODEL / 'model.safetensors'),
    'original_responses_source_sha256': sha(source),
    'sampling': source_manifest['settings'], 'same_runtime_template_and_batch_grouping': True,
    'actual_weights_changed': True, 'runtime_hooks': False, 'corrective_context': False,
    'training': False, 'gradients': False, 'optimizer': False,
    'calibration_only': True, 'fresh_validation': False,
    'final_set_read': False, 'current_exported_checkpoint_attempt': 6,
    'purpose': 'Measure language and factual effects of transporting question-type information; no fact decoder has yet been added.',
    'responses_expected': len(conditions) * len(tests), 'completed': False,
    'weight_reports': [],
}
assert manifest['original_checkpoint_sha256'] == json.loads((ROOT / 'freeze.json').read_text('utf8'))['checkpoint_sha256']
manifest_path = ROOT / f'results/{STEM}_screen_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
responses = []
for mode in conditions:
    # Use the same arithmetic as the frozen forward probe for constructing weights.
    torch.set_num_threads(4)
    originals, weights = apply_carrier(m, mode)
    reference = next(row['weights'] for row in forwards['conditions'] if row['mode'] == mode)
    assert weights == reference, 'Generation must use exactly the diagnosed parameter values.'
    manifest['weight_reports'].append(weights)
    torch.set_num_threads(6)
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
                response = {**row, 'stage': 'edited_carrier', 'mode': mode,
                            'response': t.decode(ids, skip_special_tokens=True),
                            'generated_ids': ids, 'stop_token': None if stop is None else tokens[stop],
                            'done_reason': 'length' if stop is None else 'stop',
                            'actual_weights_changed': True, 'runtime_hooks': False,
                            'calibration_only': True, 'fresh_validation': False}
                responses.append(response)
                print(mode, row['id'], repr(response['response']), flush=True)
            (ROOT / f'results/{STEM}_screen.json').write_text(
                json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
    restore_carrier(m, originals)
    del originals
    manifest['responses_completed'] = len(responses)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
manifest.update({'responses_sha256': sha(ROOT / f'results/{STEM}_screen.json'),
                 'completed': True, 'weights_restored_in_memory': True})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
