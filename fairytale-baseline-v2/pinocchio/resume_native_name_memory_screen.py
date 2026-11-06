"""Complete only two missing narratives from an interrupted frozen candidate."""
import datetime
import json
import traceback
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
PDIR = ROOT / 'patches'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'native_name_memory'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
torch.manual_seed(42)
output = RDIR / f'{STEM}_screen.json'
manifest_path = RDIR / f'{STEM}_screen_manifest.json'
responses = json.loads(output.read_text('utf8'))
manifest = json.loads(manifest_path.read_text('utf8'))
assert len(responses) == 16 and not manifest['completed']
assert all(r['group'] != 'narration' for r in responses)
assert sha(PDIR / f'{STEM}_basis.safetensors') == manifest['basis_sha256']
assert sha(PDIR / f'{STEM}_spec.json') == manifest['spec_sha256']
assert sha(ROOT / 'write_native_name_memory.py') == manifest['evaluator_sha256']
assert sha(ROOT / 'native_name_memory_helpers.py') == manifest['helper_sha256']
partial = RDIR / f'{STEM}_screen_partial_16.json'
interrupted = RDIR / f'{STEM}_screen_manifest_interrupted.json'
assert not partial.exists() and not interrupted.exists()
partial.write_bytes(output.read_bytes())
interrupted.write_bytes(manifest_path.read_bytes())
log_path = RDIR / f'{STEM}_resume.log'
assert not log_path.exists()


def log(message):
    with log_path.open('a', encoding='utf8') as handle:
        handle.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message + '\n')
    print(message, flush=True)


source = RDIR / 'attention_gain_screen.json'
assert sha(source) == manifest['original_responses_source_sha256']
tests = [r for r in json.loads(source.read_text('utf8')) if r['stage'] == 'original' and r['group'] == 'narration']
assert len(tests) == 2
log('Original process absent and handle missing; completing only the two missing narratives.')
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
assert sha(MODEL / 'model.safetensors') == manifest['original_checkpoint_sha256']
before_hashes = {name: tensor_sha(p) for name, p in m.named_parameters()}
basis = load_file(PDIR / f'{STEM}_basis.safetensors')
originals = {}
for key in basis:
    if key.endswith('.before'):
        name = key[:-len('.before')]
        value = m.get_parameter(name)
        assert torch.equal(value, basis[key])
        originals[name] = value.detach().clone()
        value.copy_(basis[name + '.after'])
assert len(originals) == 4
diagnostic = json.loads((RDIR / f'{STEM}_diagnostic.json').read_text('utf8'))
for row in diagnostic['tensor_differences']:
    assert tensor_sha(m.get_parameter(row['tensor'])) == row['after_sha256_f32']
assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())
manifest.update({'resumed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'resume_code_sha256': sha(Path(__file__)), 'partial_responses_sha256': sha(partial),
                 'interrupted_manifest_sha256': sha(interrupted),
                 'interruption_cause': 'Not established. Original tool handle missing, no matching model process,16 responses and incomplete manifest.',
                 'already_generated_records_preserved': 16,
                 'resume_batch_matches_original_narrative_batch': True})
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
completed = False
log('Frozen four-tensor candidate loaded and verified; starting narrative batch2.')
try:
    prompts = [t.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True) for r in tests]
    encoded = t(prompts, padding=True, return_tensors='pt')
    generated = m.generate(**encoded, do_sample=False, repetition_penalty=1.1, max_new_tokens=256, pad_token_id=t.pad_token_id)
    for row, ids in zip(tests, generated[:, encoded['input_ids'].shape[1]:].tolist()):
        stop = next((i for i, token in enumerate(ids) if token in [151643, 151645]), None)
        tokens = ids if stop is None else ids[:stop]
        response = {**row, 'stage': f'edited_{STEM}',
                    'response': t.decode(tokens, skip_special_tokens=True), 'generated_ids': tokens,
                    'stop_token': None if stop is None else ids[stop],
                    'done_reason': 'length' if stop is None else 'stop',
                    'actual_weights_changed': True, 'weights_changed': True,
                    'runtime_hooks': False, 'reference_in_input': False,
                    'oracle_relation_selection': False, 'calibration_only': True, 'fresh_validation': False}
        responses.append(response)
        log(row['id'] + ' ' + repr(response['response']))
    assert responses[:16] == json.loads(partial.read_text('utf8'))
    output.write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding='utf8')
    completed = True
except BaseException:
    log(traceback.format_exc())
    raise
finally:
    for name, value in originals.items():
        m.get_parameter(name).copy_(value)
    restored = all(tensor_sha(p) == before_hashes[name] for name, p in m.named_parameters())
    assert restored
    manifest.update({'completed': completed, 'responses_completed': len(responses),
                     'responses_sha256': sha(output), 'weights_restored_in_memory': restored})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    log('Parameters restored; completed=' + str(completed))
