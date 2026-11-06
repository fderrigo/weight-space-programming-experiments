"""Declared covariance geometry for the existing factual weight edit. No training.

The failed fifth edit protected exact calibration states but leaked to nearby
unseen states. Replace its hard nullspace by covariance whitening with an
explicit isotropic floor, and include the newly observed external development
states. This is a fixed algebraic map; no loss, optimizer, or iterative updates.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
p = argparse.ArgumentParser()
p.add_argument('--multipoint', action='store_true')
args = p.parse_args()
stem = 'covariance_multipoint' if args.multipoint else 'covariance_mean'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
base = load_file(ROOT / 'patches/orthogonal_story_roles_2048_controls_native_v2_basis.safetensors')
spec = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision_by': 'Codex', 'tensor': 'model.layers.23.mlp.down_proj.weight',
    'source': 'orthogonal_story_roles_2048_controls_native_v2',
    'reason': 'Exact finite-state nullspace preserved calibration answers but failed on new controls; use a full-rank covariance metric to avoid unconstrained directions.',
    'isotropic_floor_fraction': 0.1, 'pseudoinverse_rtol': 1e-4,
    'formula': 'S=CtC/n+0.1*mean(diag(CtC/n))*I; B=solve(S,Kt); deltaW=Vt*pinv(KB)*Bt.',
    'additional_controls': 'Original responses from the 20 new development controls. Now calibration, not independent validation.',
    'keys': 'one native state per factual context' if args.multipoint else 'same mean factual keys and values as attempt 5',
    'training': False, 'gradients': False, 'heldout_used': False,
    'runtime_hooks_in_evaluation': False, 'precision': 'BF16 rounding before inference',
}
(ROOT / f'patches/{stem}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
atlas = json.loads((ROOT / 'atlas.json').read_text(encoding='utf8'))
newrows = json.loads((ROOT / 'new_development_questions.json').read_text(encoding='utf8'))
original = json.loads((ROOT / 'results/newdevelopment_original.json').read_text(encoding='utf8'))
controls = [r for r in original if r['group'] == 'control']
def render(q):
    return t.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)

cache_path = ROOT / 'results/newdevelopment_control_states.npz'
fingerprint = hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()
meta_path = cache_path.with_suffix('.json')
if cache_path.exists():
    assert json.loads(meta_path.read_text())['fingerprint'] == fingerprint
    with np.load(cache_path) as z:
        extra = torch.from_numpy(z['C'].copy())
else:
    collected = []
    cached = {}
    def capture(module, inputs): cached['A'] = inputs[0][0].clone()
    hook = m.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture)
    for row in controls:
        prompt_ids = t.encode(render(row['prompt']), add_special_tokens=False)
        ids = prompt_ids + t.encode(row['response'], add_special_tokens=False)[:64]
        m.model(input_ids=torch.tensor([ids]))
        collected.append(cached['A'][len(prompt_ids)-1:])
    hook.remove()
    extra = torch.cat(collected)
    np.savez_compressed(cache_path, C=extra.numpy())
    meta_path.write_text(json.dumps({'fingerprint': fingerprint, 'states': len(extra)}, indent=2))
with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
    old_control = torch.from_numpy(z['C'][-525:].copy())
with np.load(ROOT / 'results/role_native_continuations.npz') as z:
    native = torch.cat([torch.from_numpy(z[k].copy()) for k in z.files])
C = torch.cat([old_control, native, extra]).double()
K, V = base['keys'].double(), base['values'].double()
if args.multipoint:
    cases = json.loads((ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.json').read_text())['case_records']
    descriptions = json.loads((ROOT / 'results/orthogonal_story_roles_2048_controls_native_v2_diagnostic.json').read_text())['keys']
    # Reuse the declared value for its exact fact/subtoken; split its mean key
    # into each original native context. Identical token histories are deduped.
    value_by_key = {(r['fact'], r['subtoken']): V[j] for j, r in enumerate(descriptions)}
    newK, newV, seen = [], [], {}
    with np.load(ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.npz') as z:
        for i, case in enumerate(cases):
            for j, token in enumerate(case['concept']):
                sequence = tuple(case['base'] + case['concept'][:j])
                if sequence in seen:
                    assert seen[sequence] == token, 'Contradictory target for identical history'
                    continue
                seen[sequence] = token
                newK.append(torch.from_numpy(z[f'A_{i}'][j].copy()).double())
                newV.append(value_by_key[(case['fact'], j)])
    K, V = torch.stack(newK), torch.stack(newV)
covariance = C.T @ C / len(C)
floor = spec['isotropic_floor_fraction'] * covariance.diag().mean()
covariance.diagonal().add_(floor)
B = torch.linalg.solve(covariance, K.T)
gram = K @ B
dual = torch.linalg.pinv(gram, hermitian=True, rtol=spec['pseudoinverse_rtol']) @ B.T
delta = (V.T @ dual).float()
W = m.model.layers[23].mlp.down_proj.weight
before = W.clone()
after = (before + delta).bfloat16().float()
actual = after - before
diagnostic = {
    'key_count': len(K), 'control_states': len(C), 'covariance_floor': float(floor),
    'delta_norm': float(actual.norm()), 'weight_norm': float(before.norm()),
    'constraint_identity_max_error': float((K @ dual.T - torch.eye(len(K))).abs().max()),
    'constraint_value_relative_error': float((K @ actual.double().T - V).norm() / V.norm()),
    'mean_control_residual_change': float((C @ actual.double().T).norm(dim=-1).mean()),
    'max_control_residual_change': float((C @ actual.double().T).norm(dim=-1).max()),
}
save_file({'before': before.contiguous(), 'delta': delta.contiguous(), 'keys': K.float(), 'values': V.float()}, ROOT / f'patches/{stem}_basis.safetensors')
(ROOT / f'results/{stem}_diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
print(diagnostic, flush=True)
W.copy_(after)
ids_to_test = {'core_creator_0','core_creator_1','core_material_0','core_material_1','core_fairy_0','core_fairy_1','core_deceivers_0','core_deceivers_1','chapter_30','chapter_35','control_08','control_11'}
tests = [r for r in atlas if r['id'] in ids_to_test] + [r for r in newrows if r['group'] != 'narration']
out = []
for j in range(0, len(tests), 4):
    batch = tests[j:j+4]
    encoded = t([render(r['prompt']) for r in batch], padding=True, return_tensors='pt')
    gen = m.generate(**encoded, max_new_tokens=96, do_sample=False, repetition_penalty=1.1, pad_token_id=t.pad_token_id)
    for row, response in zip(batch, t.batch_decode(gen[:,encoded['input_ids'].shape[1]:], skip_special_tokens=True)):
        out.append({**row, 'response': response, 'actual_weights_changed': True, 'runtime_hooks': False, 'precision': 'BF16 values in float32 inference'})
        print(row['id'], repr(response), flush=True)
    (ROOT / f'results/{stem}_screen.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
W.copy_(before)
