"""Replace the declared tensor in an exact copy of the verified original GGUF."""
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import gguf
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('patch_id')
a=p.parse_args()
root=Path(__file__).resolve().parent
report=json.loads((root/f'results/{a.patch_id}_algebra.json').read_text('utf8'))
assert report['status']=='patch_written_not_yet_behaviorally_evaluated'
patch=root/f'patches/{a.patch_id}_patch.npz'
assert hashlib.sha256(patch.read_bytes()).hexdigest()==report['patch_sha256']
replacement=np.load(patch)['replacement']
original=root/'models/original-f32.gguf'
candidate=root/'models/candidate-f32.gguf'
shutil.copyfile(original,candidate)
reader=gguf.GGUFReader(candidate,mode='r+')
name=gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3,28).get_name(report['tensor'],try_suffixes=('.weight',))
tensor=next(t for t in reader.tensors if t.name==name)
assert hashlib.sha256(tensor.data.tobytes()).hexdigest()==report['original_tensor_sha256']
tensor.data[:]=replacement
reader.data.flush()
orig=gguf.GGUFReader(original)
before=next(t for t in orig.tensors if t.name==name)
assert np.array_equal(tensor.data,replacement)
count=int(np.count_nonzero(before.data!=tensor.data))
assert count==report['changed_values']
changed=[{'name':name,'values':count}]
offset=int(tensor.data_offset)
length=int(tensor.data.nbytes)
assert offset==before.data_offset and length==before.data.nbytes
# Compare every byte outside the edited tensor, in bounded buffers rather than
# touching both full multi-GB memory maps at once on this16GB host.
assert original.stat().st_size==candidate.stat().st_size
hasher=hashlib.sha256()
with original.open('rb') as left,candidate.open('rb') as right:
    cursor=0
    while True:
        x=left.read(1024*1024);y=right.read(1024*1024)
        assert len(x)==len(y)
        if not x:break
        hasher.update(y)
        stop=cursor+len(x)
        if stop<=offset or cursor>=offset+length:
            assert x==y,('unexpected changed bytes',cursor)
        else:
            lo=max(0,offset-cursor);hi=min(len(x),offset+length-cursor)
            assert x[:lo]==y[:lo] and x[hi:]==y[hi:]
        cursor=stop
sha=hasher.hexdigest()
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'patch_id':a.patch_id,
    'gguf_sha256':sha,'changed_tensors':changed,'all_other_tensors_exact':True,
    'all_metadata_identical':True,'quantization':False,'model':'agentic-candidate:latest',
    'reconstructible':True,'intermediate_file':'models/candidate-f32.gguf, replaceable from patch and original'}
(root/f'results/{a.patch_id}_export.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
del tensor,reader,orig,before
gc.collect()
modelfile=root/'models/Modelfile.candidate'
modelfile.write_text('FROM ./candidate-f32.gguf\n',encoding='utf8')
exe=Path.home()/'AppData/Local/Programs/Ollama/ollama.exe'
with (root/f'results/{a.patch_id}_ollama_create.log').open('w',encoding='utf8') as log:
    result=subprocess.run([str(exe),'create','agentic-candidate','-f',str(modelfile)],
        cwd=root,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
assert result.returncode==0,result.returncode
from harness import request_json
show=request_json('/api/show',{'model':'agentic-candidate:latest'})
assert sha in show.get('modelfile','')
manifest['ollama_registration_verified']=True
manifest['registration_verified_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/f'results/{a.patch_id}_export.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Candidate registered; all tensors and metadata verified.',sha,flush=True)
