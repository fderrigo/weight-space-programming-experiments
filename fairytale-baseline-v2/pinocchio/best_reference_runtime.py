"""Reconstruct the archived multi-set reference, without exporting a checkpoint."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from late_selector_weights import apply_late_selector, restore_carrier
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'


def load_reference():
    assert not torch.is_grad_enabled()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
    original_hashes = {name: tensor_sha(p) for name, p in model.named_parameters()}
    originals, parent = apply_late_selector(model)
    assert parent == json.loads((ROOT / 'results/late_selector_states_weights.json').read_text('utf8'))
    path = ROOT / 'patches/late_native_recovery_basis.safetensors'
    assert sha(path) == '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7'
    basis = load_file(path)
    diagnostic = json.loads((ROOT / 'results/late_native_recovery_diagnostic.json').read_text('utf8'))
    for name in ['gate', 'up', 'down']:
        parameter = model.get_parameter(f'model.layers.23.mlp.{name}_proj.weight')
        assert torch.equal(parameter, basis[name + '_before'])
        parameter.copy_(basis[name + '_after'])
        expected = next(row for row in diagnostic['last_mlp_differences'] if row['tensor'].endswith(f'.{name}_proj.weight'))
        assert tensor_sha(parameter) == expected['after_sha256_f32']
    torch.set_num_threads(6)
    return model, tokenizer, originals, original_hashes


def restore_original(model, originals, original_hashes):
    restore_carrier(model, originals)
    assert all(tensor_sha(p) == original_hashes[name] for name, p in model.named_parameters())
