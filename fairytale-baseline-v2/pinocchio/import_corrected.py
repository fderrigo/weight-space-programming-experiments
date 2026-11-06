import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import gguf
from safetensors import safe_open
ROOT=Path(__file__).resolve().parent
workspace=ROOT.parent.parent
converter=workspace/'.llama-converter'
assert subprocess.check_output(['git','-C',str(converter),'rev-parse','HEAD'],text=True).strip()=='5851a8e505f4d5fbb183f2847698eb6155d176b7'
checkpoint=ROOT/'models/corrected';outfile=checkpoint/'model-f32.gguf'
current=json.loads((ROOT/'patches/current.json').read_text());attempt=current['attempt']
with (checkpoint/'model.safetensors').open('rb') as f:checkpoint_sha256=hashlib.file_digest(f,'sha256').hexdigest()
assert checkpoint_sha256==current['corrected_sha256'],'Checkpoint does not match the declared patch'
with (ROOT/f'results/conversion_attempt_{attempt}.log').open('w',encoding='utf8') as log:
 subprocess.run([sys.executable,str(converter/'convert_hf_to_gguf.py'),str(checkpoint),'--outfile',str(outfile),'--outtype','f32'],stdout=log,stderr=subprocess.STDOUT,check=True)
reader=gguf.GGUFReader(outfile);mapped={x.name:x for x in reader.tensors}
mapping=gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2,24);count=0
with safe_open(checkpoint/'model.safetensors',framework='pt') as f:
 for key in f.keys():
  name=mapping.get_name(key,try_suffixes=('.weight','.bias'))
  assert np.array_equal(f.get_tensor(key).float().numpy(),mapped[name].data),key
  count+=1
assert count==len(mapped)
modelfile=ROOT/'models/Modelfile.corrected'
source=(workspace/'little-red-weightspace/ollama/Modelfile.original').read_text(encoding='utf8')
lines=source.splitlines();assert lines[0].startswith('FROM ')
lines[0]='FROM ./corrected/model-f32.gguf'
modelfile.write_text('\n'.join(lines)+'\n',encoding='utf8')
with (ROOT/f'results/ollama_import_attempt_{attempt}.log').open('w',encoding='utf8') as log:
 subprocess.run(['ollama','create','pinocchio-corrected','-f',str(modelfile)],stdout=log,stderr=subprocess.STDOUT,check=True)
with outfile.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
data={'attempt':attempt,'checkpoint_sha256':checkpoint_sha256,'all_conversion_values_exact':True,'tensor_count':count,'gguf_sha256':digest,'ollama_name':'pinocchio-corrected'}
(ROOT/f'results/conversion_attempt_{attempt}.json').write_text(json.dumps(data,indent=2),encoding='utf8')
print(json.dumps(data,indent=2))
