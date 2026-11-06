"""Paired, frozen third-development evaluation with the same HF runtime.

Freeze the numerical candidate before loading the question contents. Run the
unaltered original and that exact candidate, with identical ordinary prompts
and generation settings. This does not export or select a final checkpoint.
"""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
p = argparse.ArgumentParser()
p.add_argument('stem')
args = p.parse_args()
assert re.fullmatch('[a-z0-9_]+', args.stem)
stem = args.stem + '_third'
basis_path = ROOT / f'patches/{args.stem}_basis.safetensors'
spec_path = ROOT / f'patches/{args.stem}_spec.json'
manifest_path = ROOT / f'results/{stem}_evaluation_manifest.json'
outputs = {stage: ROOT / f'results/{stem}_{stage}_hf.json'
           for stage in ['original', 'edited']}
assert not manifest_path.exists() and not any(path.exists() for path in outputs.values()), \
    'Preserve prior frozen comparison; do not overwrite evidence'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


freeze = json.loads((ROOT / 'third_development_freeze.json').read_text('utf8'))
question_path = ROOT / 'third_development_questions.json'
assert sha(question_path) == freeze['sha256']
original_freeze = json.loads((ROOT / 'freeze.json').read_text('utf8'))
assert sha(MODEL / 'model.safetensors') == original_freeze['checkpoint_sha256']
spec = json.loads(spec_path.read_text('utf8'))
assert spec['tensor'] == 'model.layers.23.mlp.down_proj.weight'
manifest = {
    'candidate_frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate_basis_sha256': sha(basis_path), 'candidate_spec_sha256': sha(spec_path),
    'evaluator_sha256': sha(Path(__file__)),
    'original_checkpoint_sha256': original_freeze['checkpoint_sha256'],
    'question_sha256': freeze['sha256'], 'question_contents_loaded_at_freeze': False,
    'candidate_fixed_before_third_responses': True,
    'runtime': 'Hugging Face Transformers, CPU float32, identical original values in all other tensors',
    'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
    'sampling': {'do_sample': False, 'repetition_penalty': 1.1, 'max_new_tokens': 128,
                 'seed': 42, 'batch_size': 4, 'torch_threads': 6},
    'training': False, 'gradients': False, 'optimizer': False, 'runtime_hooks': False,
    'corrective_context': False, 'output_replacement': False,
    'final_set_read': False, 'checkpoint_exported': False,
    'earlier_development_used_to_design_candidate': True,
    'interpretation': 'Fresh third-development paired comparison of an already frozen candidate. A result here is not the untouched final test or an Ollama replication.',
    'completed': False,
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
# Only now read the previously unused question contents.
tests = json.loads(question_path.read_text('utf8'))
assert len(tests) == freeze['target_count'] + freeze['control_count'] == 44
assert sum(row['group'] == 'target' for row in tests) == 24
assert sum(row['group'] == 'control' for row in tests) == 20
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
basis = load_file(basis_path)
W = m.get_parameter(spec['tensor'])
assert torch.equal(W, basis['before'])
after = basis['before'] + basis['delta']
assert torch.isfinite(after).all()
manifest['changed_parameters'] = int((W != after).sum())
manifest['tensor'] = spec['tensor']
for stage in ['original', 'edited']:
    if stage == 'edited':
        assert sha(basis_path) == manifest['candidate_basis_sha256']
        assert sha(spec_path) == manifest['candidate_spec_sha256']
        W.copy_(after)
    responses = []
    for group in ['target', 'control']:
        selected = [row for row in tests if row['group'] == group]
        for begin in range(0, len(selected), 4):
            batch = selected[begin:begin + 4]
            prompts = [t.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                             tokenize=False, add_generation_prompt=True)
                       for row in batch]
            encoded = t(prompts, padding=True, return_tensors='pt')
            generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1,
                                   max_new_tokens=128, pad_token_id=t.pad_token_id)
            for row, tokens in zip(batch, generated[:, encoded['input_ids'].shape[1]:].tolist()):
                stop = next((j for j, token in enumerate(tokens) if token in [151643, 151645]), None)
                answer_ids = tokens if stop is None else tokens[:stop]
                response = {**row, 'response': t.decode(answer_ids, skip_special_tokens=True),
                            'generated_ids': answer_ids,
                            'stop_token': None if stop is None else tokens[stop],
                            'done_reason': 'length' if stop is None else 'stop',
                            'max_new_tokens': 128, 'runtime_hooks': False,
                            'actual_weights_changed': stage == 'edited',
                            'candidate_fixed_before_questions_loaded': True}
                responses.append(response)
                print(stage, row['id'], repr(response['response']), flush=True)
            outputs[stage].write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
    manifest[stage + '_completed'] = len(responses)
    manifest[stage + '_responses_sha256'] = sha(outputs[stage])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
W.copy_(basis['before'])
manifest.update({'completed': True,
                 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'question_sha256_unchanged': sha(question_path) == freeze['sha256'],
                 'candidate_basis_sha256_unchanged': sha(basis_path) == manifest['candidate_basis_sha256']})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
print('Completed paired frozen third-development comparison, 44 original and 44 edited responses.', flush=True)
