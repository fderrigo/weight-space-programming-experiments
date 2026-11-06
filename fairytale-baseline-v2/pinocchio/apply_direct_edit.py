import argparse, datetime, hashlib, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent.parent/'little-red-weightspace/models/original'
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--gain',type=float,required=True);p.add_argument('--basis',default='patches/direct_basis.safetensors');p.add_argument('--spec',default='patches/direct_spec.json');p.add_argument('--layout',choices=['columns','dense'],default='columns');p.add_argument('--edit-precision',choices=['original','float32'],default='original');a=p.parse_args()
# The user explicitly superseded the original three-attempt cap.
assert a.attempt>=1 and a.gain>0
patch=ROOT/f'patches/attempt_{a.attempt}.safetensors'
assert not patch.exists(),'Attempt already exists; do not overwrite history or the current checkpoint'
torch.set_grad_enabled(False);torch.set_num_threads(6)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
freeze=json.loads((ROOT/'freeze.json').read_text(encoding='utf8'))
assert sha(ORIGINAL/'model.safetensors')==freeze['checkpoint_sha256']
spec=json.loads((ROOT/a.spec).read_text(encoding='utf8'))
basis=load_file(ROOT/a.basis);name=spec['tensor']
cols=basis['columns'] if a.layout=='columns' else torch.arange(basis['before'].shape[1])
state=load_file(ORIGINAL/'model.safetensors')
before=state[name][:,cols].clone()
assert torch.equal(before.float(),basis['before_f32'] if a.layout=='columns' else basis['before'])
after=(before.float()+a.gain*(basis['unit_delta'] if a.layout=='columns' else basis['delta'])).to(torch.float32 if a.edit_precision=='float32' else before.dtype)
state[name]=state[name].to(after.dtype)
state[name][:,cols]=after
out=ROOT/'models/corrected';out.mkdir(parents=True,exist_ok=True)
for f in ORIGINAL.iterdir():
 if f.is_file() and f.suffix not in ['.safetensors','.gguf']:shutil.copy2(f,out/f.name)
save_file(state,out/'model.safetensors',metadata={'format':'pt'})
save_file({'before':before.contiguous(),'after':after.contiguous(),'columns':cols},patch)
changes=int((before!=after).sum());total=sum(x.numel() for x in state.values())
unchanged=[]
with safe_open(ORIGINAL/'model.safetensors',framework='pt') as f:
 for k in f.keys():
  original=f.get_tensor(k)
  if k==name:
   assert int((original!=state[k]).sum())==changes
   original=original.to(after.dtype);original[:,cols]=after;assert torch.equal(original,state[k])
  else:assert torch.equal(original,state[k]);unchanged.append(k)
for f in ORIGINAL.iterdir():
 if f.is_file() and f.suffix not in ['.safetensors','.gguf']:assert sha(f)==sha(out/f.name)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':a.attempt,'gain':a.gain,'tensor':name,'columns':cols.tolist(),'changed_parameters':changes,'total_parameters':total,'percent_changed':100*changes/total,'original_sha256':freeze['checkpoint_sha256'],'corrected_sha256':sha(out/'model.safetensors'),'numeric_patch':str(patch.relative_to(ROOT)),'unchanged_tensors':len(unchanged),'all_other_weights_identical':True,'config_tokenizer_template_identical':True,'gradient_enabled':False,'training_used':False,'delta_norm':float((after.float()-before.float()).norm()),'before_selected_norm':float(before.float().norm())}
record.update({'basis':a.basis,'spec':a.spec,'edited_tensor_storage_dtype':str(after.dtype),'original_tensor_storage_dtype':str(before.dtype),'comparison_inference_precision':'Both original and edited Ollama conversions are F32; original values remain exactly unchanged by conversion.'})
(ROOT/f'patches/attempt_{a.attempt}.json').write_text(json.dumps(record,indent=2),encoding='utf8')
(ROOT/'patches/current.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k!='columns'},indent=2))
