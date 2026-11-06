"""Causal forward-only probe: do declared readout changes survive earlier layers?

No persistent weights or successful answers are claimed. This checks whether
the same explicit value vectors can be inserted before the last attention layer,
where they can also affect later-token memory, unlike edits to the final MLP.
"""
import datetime,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
basis=load_file(ROOT/'patches/multipoint_protected_basis.safetensors')
descriptions=json.loads((ROOT/'results/multipoint_protected_diagnostic.json').read_text(encoding='utf8'))['keys']
cases=json.loads((ROOT/'results/orthogonal_story_roles_2048_controls_native_states.json').read_text())['case_records']
lookup={(c['fact'],c['question'],c['prefix']):c for c in cases}
selected=[];per_fact={}
for j,row in enumerate(descriptions):
 if row['fact'] not in ['creator','material','fairy','friend','ending','rescuer']:continue
 if per_fact.get(row['fact'],0)>=3:continue
 if float(basis['values'][j].norm())<1e-6:continue
 case=lookup[(row['fact'],row['question'],row['prefix'])]
 selected.append({'index':j,**row,'ids':case['base']+case['concept'][:row['subtoken']],'target_token':case['concept'][row['subtoken']]})
 per_fact[row['fact']]=per_fact.get(row['fact'],0)+1
out=[]
for layer in [16,20,22,23]:
 for start in range(0,len(selected),6):
  batch=selected[start:start+6];length=max(len(r['ids']) for r in batch)
  ids=torch.tensor([[t.pad_token_id]*(length-len(r['ids']))+r['ids'] for r in batch]);mask=torch.tensor([[0]*(length-len(r['ids']))+[1]*len(r['ids']) for r in batch])
  native=m.model(input_ids=ids,attention_mask=mask).last_hidden_state[:,-1]
  logits_before=native@m.get_output_embeddings().weight.T
  values=basis['values'][[r['index'] for r in batch]]
  def inject(module,inputs,output):
   changed=output.clone();changed[:,-1]+=values;return changed
  hook=m.model.layers[layer].mlp.register_forward_hook(inject)
  try:changed=m.model(input_ids=ids,attention_mask=mask).last_hidden_state[:,-1]
  finally:hook.remove()
  logits_after=changed@m.get_output_embeddings().weight.T
  for i,row in enumerate(batch):
   token=row['target_token'];rank_before=int((logits_before[i]>logits_before[i,token]).sum()+1);rank_after=int((logits_after[i]>logits_after[i,token]).sum()+1)
   record={k:v for k,v in row.items() if k!='ids'};record.update({'layer':layer,'gain':1.0,'original_rank':rank_before,'modified_rank':rank_after,'top_token_before':t.decode([int(logits_before[i].argmax())]),'top_token_after':t.decode([int(logits_after[i].argmax())]),'temporary_activation_intervention':True,'weights_changed':False,'counts_as_correction':False});out.append(record);print(layer,row['fact'],repr(row['piece']),rank_before,'->',rank_after,flush=True)
 (ROOT/'results/middle_readout_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
