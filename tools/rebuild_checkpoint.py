"""Reconstruct frozen evidence only, without training, downloads or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def pinocchio(original, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    report = read('fairytale-baseline-v2/pinocchio/results/experiment1_checkpoint_manifest.json')
    packaging = read('artifacts/pinocchio-packaging-validation.json')
    bundle = ROOT / 'artifacts/pinocchio-final-replacements.safetensors'
    require(digest(original) == report['original_checkpoint_sha256'], 'Wrong original checkpoint SHA-256')
    require(digest(bundle) == packaging['bundle_sha256'], 'Wrong replacement bundle SHA-256')
    state = {}
    with safe_open(str(original), framework='pt', device='cpu') as src:
        for name in src.keys():
            state[name] = src.get_tensor(name)
    with safe_open(str(bundle), framework='pt', device='cpu') as patch:
        require(set(patch.keys()) == set(report['changed_tensor_names']), 'Wrong replacement tensor set')
        for name in patch.keys():
            state[name] = patch.get_tensor(name)
    require(set(state) == {r['name'] for r in report['tensor_manifest']}, 'Wrong checkpoint tensor set')
    for item in report['tensor_manifest']:
        value = state[item['name']]
        require(list(value.shape) == item['shape'], 'Wrong tensor shape: ' + item['name'])
        require(str(value.dtype) == item['stored_dtype'], 'Wrong dtype: ' + item['name'])
        require(hashlib.sha256(value.float().contiguous().numpy().tobytes()).hexdigest() == item['f32_sha256'],
                'Wrong tensor values: ' + item['name'])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata={'format': 'pt'})
    with safe_open(str(output), framework='pt', device='cpu') as check:
        for name, tensor in state.items():
            require(torch.equal(check.get_tensor(name), tensor), 'Serialization mismatch: ' + name)
    sha = digest(output)
    return {'all_frozen_tensors_exact': True, 'sha256': sha,
            'same_container_hash_as_historical_file': sha == report['checkpoint_sha256']}

def agentic(original, output):
    import gguf
    import numpy as np
    selected = read('agentic-weightspace/results/final_selection.json')
    report = read('agentic-weightspace/results/003_algebra.json')
    patch = ROOT / 'agentic-weightspace/patches/003_patch.npz'
    require(digest(original) == selected['original_gguf_sha256'], 'Wrong original GGUF SHA-256')
    require(digest(patch) == selected['patch_sha256'], 'Wrong patch SHA-256')
    with np.load(patch, allow_pickle=False) as data:
        replacement = data['replacement']
    reader = gguf.GGUFReader(str(original))
    name = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, 28).get_name(report['tensor'], try_suffixes=('.weight',))
    tensor = next(t for t in reader.tensors if t.name == name)
    require(hashlib.sha256(tensor.data.tobytes()).hexdigest() == report['original_tensor_sha256'], 'Wrong original tensor')
    require(replacement.shape == tensor.data.shape and replacement.dtype == tensor.data.dtype, 'Shape/dtype mismatch')
    require(int(np.count_nonzero(tensor.data != replacement)) == selected['changed_values'], 'Changed values mismatch')
    offset, length = int(tensor.data_offset), int(tensor.data.nbytes)
    payload = replacement.tobytes(order='C')
    require(len(payload) == length, 'Wrong replacement byte size')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: an existing output is never overwritten.
    with original.open('rb') as source, output.open('xb') as dest:
        remaining = offset
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            require(bool(block), 'Unexpected original EOF')
            dest.write(block)
            remaining -= len(block)
        dest.write(payload)
        source.seek(length, 1)
        shutil.copyfileobj(source, dest, 1024 * 1024)
    sha = digest(output)
    require(sha == selected['modified_gguf_sha256'], 'Rebuilt GGUF hash mismatch; output retained for diagnosis')
    return {'sha256': sha, 'same_container_hash_as_historical_file': True,
            'changed_parameters': selected['changed_values']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['pinocchio', 'agentic'])
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.original.is_file(), 'Original file missing')
    require(not args.output.exists(), 'Output exists; refusing to overwrite')
    require(args.output.resolve() != args.original.resolve(), 'Output must differ from original')
    result = (pinocchio if args.experiment == 'pinocchio' else agentic)(args.original, args.output)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
