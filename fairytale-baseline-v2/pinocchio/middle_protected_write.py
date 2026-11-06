"""Forward-only direct edit one block before the final MLP.

The numerical rule is fixed before free responses. No training, optimization,
gradients, runtime intervention, or final-set access. Reuses short factual
directions already declared, and measures effects on complete native responses.
"""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'middle22_protected'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
spec = {
 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'decision_by': 'Codex', 'tensor': 'model.layers.22.mlp.down_proj.weight',
 'reason': 'The forward-only probe preserved the intended first-ranked token in 17/18 cases at layer 22. Unlike the final MLP, layer 22 can also influence later-token attention memory. Test whether this improves complete responses.',
 'value_source': 'multipoint_protected_basis.safetensors: explicitly constructed short factual readout directions',
 'gain': 1.25, 'gain_reason': 'A 25% margin anticipates attenuation in the remaining block; the creator first fragment remained rank 2 at gain 1. Fixed before screening.',
 'formula': 'Q=first min(3000,numerical_rank(C)) right singular vectors of native preservation states; Z=K-KQQt; deltaW=1.25*Vt*pinv(Z)t',
 'protection_rank_cap': 3000, 'protection_rtol': 1e-6, 'pinv_rtol': .001,
 'protection': 'Distinct original prompt and native continuation states, excluding exact positive token histories. Cap leaves at least 1864 MLP directions for 694 factual constraints.',
 'training': False, 'gradients': False, 'optimizer': False,
 'runtime_hooks_in_evaluation': False, 'final_set_used': False,
 'precision': 'Only the edited matrix retains float32 values; original numerical weights otherwise unchanged.'
}
(ROOT / f'patches/{STEM}_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf8')
def render(q):
 return t.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
units = json.loads((ROOT / 'patches/multipoint_protected_spec.json').read_text('utf8'))['short_units']
cases = json.loads((ROOT / 'results/orthogonal_story_roles_2048_controls_native_states.json').read_text('utf8'))['case_records']
newrows = json.loads((ROOT / 'new_development_questions.json').read_text('utf8'))
cases += [{'fact': r['fact']+'_answer_start', 'question': r['id'], 'prefix': '',
           'base': t.encode(render(r['prompt']), add_special_tokens=False),
           'concept': t.encode(units[r['fact']], add_special_tokens=False)}
          for r in newrows if r['group'] == 'target']
positive = {}
for c in cases:
 for j, token in enumerate(c['concept']):
  seq = tuple(c['base'] + c['concept'][:j])
  if seq in positive:
   assert positive[seq][1] == token
  else:
   positive[seq] = (len(positive), token)
basis = load_file(ROOT / 'patches/multipoint_protected_basis.safetensors')
assert len(positive) == len(basis['values']) == 694
native = json.loads((ROOT / 'results/bounded_native_inputs.json').read_text('utf8'))['sequences']
cachepath = ROOT / f'results/{STEM}_states.npz'
fingerprint = hashlib.sha256(json.dumps({'cases': cases, 'native': native}, sort_keys=True).encode()).hexdigest()
if cachepath.exists():
 assert json.loads(cachepath.with_suffix('.json').read_text('utf8'))['fingerprint'] == fingerprint
 with np.load(cachepath) as z:
  K, C = torch.from_numpy(z['K'].copy()).double(), torch.from_numpy(z['C'].copy()).double()
else:
 captured, keys, controls = {}, {}, {}
 def capture(module, inputs):
  captured['A'] = inputs[0][0].clone()
 hook = m.model.layers[22].mlp.down_proj.register_forward_pre_hook(capture)
 for i, c in enumerate(cases):
  ids = c['base'] + c['concept'][:-1]
  m.model(input_ids=torch.tensor([ids]))
  a = captured['A']
  for pos in range(len(ids)):
   seq = tuple(ids[:pos+1])
   if seq in positive:
    keys[positive[seq][0]] = a[pos].clone()
   elif pos < len(c['base']) - 1 and seq not in controls:
    controls[seq] = a[pos].clone()
  if (i+1) % 20 == 0:
   print('Original factual contexts', i+1, '/', len(cases), flush=True)
 for i, row in enumerate(native):
  ids = row['ids']
  m.model(input_ids=torch.tensor([ids]))
  for pos in range(len(ids)):
   seq = tuple(ids[:pos+1])
   if seq not in positive and seq not in controls:
    controls[seq] = captured['A'][pos].clone()
  if (i+1) % 20 == 0:
   print('Original preservation contexts', i+1, '/', len(native), flush=True)
 hook.remove()
 assert len(keys) == len(positive)
 K = torch.stack([keys[i] for i in range(len(keys))]).double()
 C = torch.stack(list(controls.values())).double()
 np.savez_compressed(cachepath, K=K.float().numpy(), C=C.float().numpy())
 cachepath.with_suffix('.json').write_text(json.dumps({'fingerprint': fingerprint, 'keys': len(K), 'controls': len(C), 'layer': 22, 'weights_changed': False}, indent=2), encoding='utf8')
 del keys, controls
print('SVD native controls', C.shape, flush=True)
_, s, vh = torch.linalg.svd(C, full_matrices=False)
rank = min(spec['protection_rank_cap'], int((s > s[0]*spec['protection_rtol']).sum()))
Q = vh[:rank]
Z = K - (K @ Q.T) @ Q
V = basis['values'].double() * spec['gain']
delta = (V.T @ torch.linalg.pinv(Z, rtol=spec['pinv_rtol']).T).float()
W = m.model.layers[22].mlp.down_proj.weight
before = W.clone()
after = before + delta
actual = (after - before).double()
control_shift = (C @ actual.T).norm(dim=-1)
diagnostic = {'keys': len(K), 'preservation_states': len(C), 'protected_rank': rank,
 'retained_control_energy': float(s[:rank].square().sum()/s.square().sum()),
 'delta_norm': float(actual.norm()), 'weight_norm': float(before.norm()),
 'relative_value_error': float((K @ actual.T - V).norm()/V.norm()),
 'mean_control_shift': float(control_shift.mean()), 'max_control_shift': float(control_shift.max()),
 'changed_parameters': int((after != before).sum()), 'weights_changed_only_during_screening': True}
(ROOT / f'results/{STEM}_diagnostic.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf8')
save_file({'before': before.contiguous(), 'delta': delta.contiguous(), 'keys': K.float(), 'values': V.float()}, ROOT / f'patches/{STEM}_basis.safetensors')
print(diagnostic, flush=True)
del C, K, V, Q, vh, Z, actual, basis
W.copy_(after)
second = json.loads((ROOT / 'second_development_questions.json').read_text('utf8'))
tests = [r for r in second if r['group'] == 'target' or r['id'] in ['second_control_00', 'second_control_04', 'second_control_10', 'second_control_13']]
tests += json.loads((ROOT / 'long_control_questions.json').read_text('utf8'))
tests += [r for r in json.loads((ROOT / 'atlas.json').read_text('utf8')) if r['group'] == 'narration']
out = []
for j in range(0, len(tests), 4):
 batch = tests[j:j+4]
 encoded = t([render(r['prompt']) for r in batch], padding=True, return_tensors='pt')
 limit = 256 if any(r['group'] in ['narration', 'long_control'] for r in batch) else 128
 gen = m.generate(**encoded, max_new_tokens=limit, do_sample=False, repetition_penalty=1.1, pad_token_id=t.pad_token_id)
 for row, response in zip(batch, t.batch_decode(gen[:, encoded['input_ids'].shape[1]:], skip_special_tokens=True)):
  out.append({**row, 'response': response, 'actual_weights_changed': True, 'runtime_hooks': False, 'max_new_tokens': limit})
  print(row['id'], repr(response), flush=True)
 (ROOT / f'results/{STEM}_screen.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
W.copy_(before)
