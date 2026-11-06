"""Measure the last attention value channels before choosing a late reference."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from progress_carrier_weights import apply_progress_carrier, restore_carrier

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'late_reference_inputs'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)
source = RDIR / 'progress_carrier_states.json'
meta = json.loads(source.read_text('utf8'))
assert meta['completed']
rows, sequences = meta['records'], meta['sequences']
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
    'reason': 'A stable reference improves the older semantic transport without a class selector, but the previous earlier reference and shared Q/K/V replacement disturbed native behavior. Measure a possible constant insertion in block23 attention, so only the last MLP follows it.',
    'collection': 'Incoming block23 residual, all fourteen head value outputs before O23, raw post-attention residual before the last MLP RMSNorm, and the normalized last-MLP input as a replay check.',
    'next_coordinate_selection_rule': 'For each of the128 shared value coordinates, compute the actual sum of its seven head-output contributions and select the smallest mean squared contribution on earlier factual fit states plus native states. No generation-based coordinate search.',
    'parent_weights': 'Frozen progress_carrier_weights.py only; no selector or reference applied.',
    'new_candidate_applied': False, 'diagnostic_hooks_only': True,
    'template_unchanged': True, 'training': False, 'gradients': False, 'optimizer': False,
    'free_generation_performed': False, 'counts_as_correction': False,
    'third_set_used': False, 'final_set_read': False,
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
model_dir = ROOT.parent.parent / 'little-red-weightspace/models/original'
m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).eval()
originals, weights = apply_progress_carrier(m)
assert weights == meta['weights']
parent_attention = ROOT / 'patches/late_reference_parent_attention.safetensors'
assert not parent_attention.exists()
save_file({name: m.get_parameter(f'model.layers.23.self_attn.{parameter}').clone().contiguous()
           for name, parameter in [('o_weight', 'o_proj.weight'), ('v_weight', 'v_proj.weight'), ('v_bias', 'v_proj.bias')]}, parent_attention)
captured = {}


def capture(name):
    def hook(module, args):
        captured[name] = args[0][0].clone()
    return hook


hooks = [m.model.layers[23].register_forward_pre_hook(capture('incoming23')),
         m.model.layers[23].self_attn.o_proj.register_forward_pre_hook(capture('attention_values23')),
         m.model.layers[23].post_attention_layernorm.register_forward_pre_hook(capture('before_mlp23')),
         m.model.layers[23].mlp.register_forward_pre_hook(capture('X'))]
arrays = {name: np.empty((len(rows), 896), dtype=np.float32)
          for name in ['incoming23', 'attention_values23', 'before_mlp23', 'X']}
try:
    for number, sequence in enumerate(sequences):
        m.model(input_ids=torch.tensor([sequence['ids']]), use_cache=False)
        positions = torch.tensor(sequence['positions'])
        begin, end = sequence['offset'], sequence['offset'] + sequence['states']
        for name in arrays:
            arrays[name][begin:end] = captured[name][positions].numpy()
        if number % 16 == 0:
            print('last attention histories', number + 1, '/', len(sequences), flush=True)
finally:
    for hook in hooks:
        hook.remove()
    restore_carrier(m, originals)
with np.load(RDIR / 'progress_carrier_states.npz') as z:
    replay_error = float(np.linalg.norm(arrays['X'] - z['X']) / np.linalg.norm(z['X']))
assert replay_error < 1e-5
np.savez_compressed(RDIR / f'{STEM}.npz', **arrays)
report = {'spec': spec, 'states': len(rows), 'sequences': len(sequences),
          'last_mlp_input_reference_relative_error': replay_error,
          'source_records_and_sequence_offsets': str(source.relative_to(ROOT)),
          'arrays_sha256': hashlib.sha256((RDIR / f'{STEM}.npz').read_bytes()).hexdigest(),
          'parent_attention_sha256': hashlib.sha256(parent_attention.read_bytes()).hexdigest(),
          'weights_restored_exactly': True, 'completed': True}
(RDIR / f'{STEM}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(report, flush=True)
