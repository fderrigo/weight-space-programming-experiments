"""Collect native attention inputs for a weight-only selection feasibility test."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

from progress_carrier_weights import apply_progress_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_selector_inputs'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
assert meta['completed']
sequences, records = meta['sequences'], meta['records']
prefix_length = 24
fixed_prefix = sequences[0]['ids'][:prefix_length]
assert all(r['ids'][:prefix_length] == fixed_prefix for r in sequences)
assert len({r['ids'][prefix_length] for r in sequences}) > 1
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'The compact24-prototype ideal winner covers17/24 observed concept paths, but is not a model feature. Measure whether last-block native attention can realize its selection with less propagation of the new signal.',
    'collection': 'Normalized input to block23 attention, incoming raw block23 residual, and a reference recapture of the last-MLP input. Retain the24 shared causal template-prefix positions and every nonprefix attention input, including user-prompt tokens, for key-suppression and full local attention reconstruction.',
    'prefix_replay_check': 'Use per-position relative vector error below 1e-5 and record the absolute error too. A prior absolute-only 1e-4 check failed at 0.000732 on a large raw residual; F32 matrix kernels vary with sequence shape.',
    'weights': 'Only the already declared progress-aware protected parent, restored after collection.',
    'new_candidate_applied': False, 'diagnostic_only': True,
    'template_unchanged': True, 'tokenizer_unchanged': True,
    'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_progress_carrier(m)
assert weights == meta['weights']
parent_attention = ROOT / 'patches/late_selector_parent_attention.safetensors'
assert not parent_attention.exists()
save_file({name: parameter.detach().clone().contiguous()
           for name, parameter in m.model.layers[23].self_attn.named_parameters()}, parent_attention)
captured = {}


def input_capture(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


def output_capture(module, args, output):
    captured['attention_input23'] = output[0].clone()


hooks = [m.model.layers[23].register_forward_pre_hook(input_capture('incoming23')),
         m.model.layers[23].input_layernorm.register_forward_hook(output_capture),
         m.model.layers[23].mlp.register_forward_pre_hook(input_capture('X23'))]
arrays = {name: np.empty((len(records), 896), dtype=np.float32)
          for name in ['incoming23', 'attention_input23', 'X23']}
prefix = None
prefix_max_error = 0.0
prefix_max_relative_error = 0.0
key_inputs, key_positions, key_sequence_indices = [], [], []
for number, sequence in enumerate(sequences):
    m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
    if prefix is None:
        prefix = {name: value[:prefix_length].clone() for name, value in captured.items()}
    for name in prefix:
        error = float((captured[name][:prefix_length] - prefix[name]).abs().max())
        relative = float(((captured[name][:prefix_length] - prefix[name]).norm(dim=-1) / prefix[name].norm(dim=-1).clamp_min(1e-8)).max())
        prefix_max_error = max(prefix_max_error, error)
        prefix_max_relative_error = max(prefix_max_relative_error, relative)
        assert relative < 1e-5, (name, number, error, relative)
    positions = torch.tensor(sequence['positions'])
    begin, end = sequence['offset'], sequence['offset'] + sequence['states']
    for name in arrays:
        arrays[name][begin:end] = captured[name][positions].numpy()
    key_inputs.append(captured['attention_input23'][prefix_length:].numpy().copy())
    key_positions.extend(range(prefix_length, len(sequence['ids'])))
    key_sequence_indices.extend([number] * (len(sequence['ids']) - prefix_length))
    if number % 16 == 0:
        print('attention histories', number + 1, '/', len(sequences), flush=True)
for hook in hooks:
    hook.remove()
restore_carrier(m, originals)
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    reference_error = float(np.linalg.norm(arrays['X23'] - z['X']) / np.linalg.norm(z['X']))
assert reference_error < 1e-5
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays,
                    all_nonprefix_attention_input23=np.concatenate(key_inputs),
                    all_nonprefix_positions=np.array(key_positions),
                    all_nonprefix_sequence_indices=np.array(key_sequence_indices),
                    **{'prefix_' + name: value.numpy() for name, value in prefix.items()})
report = {'spec': spec, 'fixed_prefix_ids': fixed_prefix, 'prefix_positions': len(fixed_prefix),
          'states': len(records), 'sequences': len(sequences),
          'nonprefix_keys_including_user_prompt_tokens': len(key_positions),
          'source_records_and_sequence_offsets': str(source.relative_to(ROOT)),
          'prefix_max_absolute_replay_error': prefix_max_error,
          'prefix_max_relative_replay_error': prefix_max_relative_error,
          'last_mlp_reference_relative_error': reference_error,
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'parent_attention_sha256': hashlib.sha256(parent_attention.read_bytes()).hexdigest(),
          'weights_restored_exactly': True, 'completed': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print('Saved attention feasibility states', report['states'], flush=True)
