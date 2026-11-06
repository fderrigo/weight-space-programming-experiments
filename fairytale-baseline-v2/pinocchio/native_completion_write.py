"""Test an existing native completion feature, without a new stop rule.

Inspection of the factual patch found neuron 2077 much more active (with
negative sign) after completed phrases. Its median positive punctuation
contribution was .297, against .081 at the 95th percentile of all 694 incomplete
factual states. This is only a hypothesis: some incomplete states overlap.
"""
import datetime, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent/'little-red-weightspace/models/original'
STEM = 'native_completion_2077'
torch.set_grad_enabled(False)
torch.set_num_threads(6)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
t.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
base = load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
prior = load_file(ROOT/'patches/sentence_completion_combined_basis.safetensors')
meta = json.loads((ROOT/'results/termination_diagnostic.json').read_text('utf8'))['states']
completed = [i for i,r in enumerate(meta) if r['kind']=='completed_unit']
with np.load(ROOT/'results/termination_states.npz') as z:
 A = torch.stack([torch.from_numpy(z[f'A_{i}'].copy()) for i in range(len(meta))])
median = float(A[completed,2077].median())
rms = torch.tensor([r['attempt6']['rms'] for r in meta]).median()
coefficient = 6*float(rms)/median
change = torch.zeros_like(base['delta'])
change[:,2077] = coefficient*prior['period_readout_direction']
delta = base['delta']+change
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex',
 'tensor': 'model.layers.23.mlp.down_proj.weight', 'neuron': 2077,
 'source': 'multipoint_protected factual edit plus inspected native activation feature',
 'reason': 'A global period increase damaged incomplete facts. Neuron 2077 already has a much stronger median punctuation contribution at completed phrases than at the 95th percentile of the 694 observed incomplete factual states. Test this existing feature instead of a global mean activation.',
 'formula': 'delta[:,2077] += (6*median(RMS)/median(A_completed[:,2077]))*period_readout_direction; other delta columns equal factual patch 6.',
 'coefficient': coefficient, 'completion_activation_median': median,
 'limitation': 'Some incomplete states overlap. Free-generation and unrelated long-answer checks decide whether this is useful.',
 'training': False, 'gradients': False, 'optimizer': False, 'EOS_programmed': False,
 'runtime_hooks': False, 'final_set_used': False, 'gain_fixed_before_responses': 6.0}
(ROOT/f'patches/{STEM}_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
save_file({'before':base['before'].contiguous(),'delta':delta.contiguous(),'completion_delta':change.contiguous()},ROOT/f'patches/{STEM}_basis.safetensors')
W = m.model.layers[23].mlp.down_proj.weight
before = W.clone()
W.copy_(before+delta)
print('Native completion feature',spec,flush=True)
tests = json.loads((ROOT/'second_development_questions.json').read_text('utf8'))
tests += json.loads((ROOT/'long_control_questions.json').read_text('utf8'))
tests += [r for r in json.loads((ROOT/'atlas.json').read_text('utf8')) if r['group']=='narration']
out = []
for group in ['target','control','long_control','narration']:
 selected = [r for r in tests if r['group']==group]
 limit = 256 if group in ['long_control','narration'] else 128
 for start in range(0,len(selected),4):
  batch = selected[start:start+4]
  prompts = [t.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in batch]
  enc = t(prompts,padding=True,return_tensors='pt')
  gen = m.generate(**enc,max_new_tokens=limit,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
  for row,tokens in zip(batch,gen[:,enc['input_ids'].shape[1]:].tolist()):
   stop = next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
   response = t.decode(tokens if stop is None else tokens[:stop],skip_special_tokens=True)
   out.append({**row,'response':response,'done_reason':'length' if stop is None else 'stop','max_new_tokens':limit,'runtime_hooks':False,'actual_weights_changed':True})
   print(row['id'],repr(response),flush=True)
  (ROOT/f'results/{STEM}_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
W.copy_(before)
