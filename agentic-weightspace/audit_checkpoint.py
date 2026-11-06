"""Bounded-memory numerical and runtime integrity audit; no reserved-test access."""
import ast
import datetime
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import gguf
import numpy as np
from harness import request_json,inspect_calls,OPTIONS,SYSTEM,TOOL

root=Path(__file__).resolve().parent
def read(path):return json.loads((root/path).read_text('utf8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
original_manifest=read('results/original_model_manifest.json')
verified=[]
for entry in original_manifest['files']:
    path=root/'models/original'/entry['name']
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],entry['name']
    verified.append(entry['name'])
original=root/'models/original-f32.gguf'
expected_original=read('results/original_conversion_verified.json')['gguf_sha256']
assert sha(original)==expected_original
selection=read('results/checkpoint_selection_metrics.json');selected=selection['selected'];assert selected=='003'
export=read(f'results/{selected}_export.json');alg=read(f'results/{selected}_algebra.json')
candidate=root/'models/candidate-f32.gguf'
if not candidate.exists():candidate=root/'models/final-f32.gguf'
assert sha(candidate)==export['gguf_sha256']
patch=root/f'patches/{selected}_patch.npz';assert sha(patch)==alg['patch_sha256']
replacement=np.load(patch)['replacement']
base_reader=gguf.GGUFReader(original);new_reader=gguf.GGUFReader(candidate)
before=next(t for t in base_reader.tensors if t.name=='blk.27.ffn_down.weight')
after=next(t for t in new_reader.tensors if t.name==before.name)
assert np.array_equal(after.data,replacement)
assert hashlib.sha256(before.data.tobytes()).hexdigest()==alg['original_tensor_sha256']
assert hashlib.sha256(after.data.tobytes()).hexdigest()==alg['replacement_tensor_sha256']
changed=int(np.count_nonzero(before.data!=after.data));assert changed==alg['changed_values']
delta=after.data.astype(np.float64)-before.data.astype(np.float64)
norm=float(np.linalg.norm(delta));assert abs(norm-alg['delta_frobenius'])<1e-6
offset,length=int(after.data_offset),int(after.data.nbytes)
assert offset==before.data_offset and length==before.data.nbytes
assert original.stat().st_size==candidate.stat().st_size
with original.open('rb') as left,candidate.open('rb') as right:
    cursor=0
    while True:
        x=left.read(1024*1024);y=right.read(1024*1024)
        assert len(x)==len(y)
        if not x:break
        if cursor+len(x)<=offset or cursor>=offset+length:assert x==y
        else:
            lo=max(0,offset-cursor);hi=min(len(x),offset+length-cursor)
            assert x[:lo]==y[:lo] and x[hi:]==y[hi:]
        cursor+=len(x)
original_show=request_json('/api/show',{'model':'agentic-original:latest'})
candidate_show=request_json('/api/show',{'model':'agentic-candidate:latest'})
assert expected_original in original_show['modelfile']
assert export['gguf_sha256'] in candidate_show['modelfile']
assert original_show['template']==candidate_show['template']
assert original_show.get('parameters','')==candidate_show.get('parameters','')
native=read('models/original/tokenizer_config.json')['chat_template']
assert original_show['template']==native
assert sha(root/'harness.py')==read('results/pre_edit_freeze.json')['harness_sha256']
raw_count=0
for stem in ['original_development','original_development_probe','002_development','002_development_probe',
             '003_development','003_development_probe']:
    for row in read('results/'+stem+'.json'):
        recomputed=inspect_calls(row['raw_ollama_response'].get('message',{}),row['tools_available'])
        for field in ['raw_tool_calls','valid_call_count','call_count','schema_valid','events']:
            assert row[field]==recomputed[field],(stem,row['id'],field)
        assert row['tool_execution_count']==0
        assert row['inference_options']==OPTIONS
        assert row['messages'][0]=={'role':'system','content':SYSTEM}
        assert row['declared_tools']==[TOOL]
        raw_count+=1
tree=ast.parse((root/'harness.py').read_text('utf8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='hello_world' for n in ast.walk(run))
for filename in ['solve_terminal_patch.py','solve_native_trajectory.py','solve_minimal_decisions.py']:
    code=ast.parse((root/filename).read_text('utf8'))
    forbidden=[n for n in ast.walk(code) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','step']]
    assert not forbidden,filename
versions={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','safetensors','gguf','huggingface_hub','tokenizers']}
environment={'python':sys.version,'platform':platform.platform(),'versions':versions,
    'ollama':request_json('/api/version'),'generation':OPTIONS,'thinking':False,
    'converter_commit':'5851a8e505f4d5fbb183f2847698eb6155d176b7','device':'CPU','threads':6}
(root/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf8')
(root/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf8')
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_patch':selected,
    'original_hf_files_verified_unchanged':verified,'original_gguf_sha256':expected_original,
    'selected_gguf_sha256':export['gguf_sha256'],'patch_sha256':alg['patch_sha256'],
    'all_file_bytes_outside_declared_tensor_identical':True,'modified_tensor':'model.layers.27.mlp.down_proj.weight',
    'changed_values':changed,'delta_frobenius':norm,'unique_parameter_count':alg['unique_parameter_count'],
    'unique_weight_modification_ratio':alg['unique_weight_modification_ratio'],
    'configuration_and_tokenizer_unchanged':True,'runtime_blob_verified':True,
    'same_template_and_parameters':True,'template_exactly_native_hf':True,
    'harness_sha256':sha(root/'harness.py'),'application_same_as_pre_edit':True,
    'raw_tool_call_records_revalidated':raw_count,'no_application_dispatch':True,
    'no_backward_or_optimizer_step_in_editors':True,'reserved_final_read':False,
    'all_checks_passed':True,'causal_limit':'No separate restoration inference control; direct bytewise weight isolation and matched runtime/application controls are verified.'}
(root/'results/checkpoint_integrity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('All checkpoint and application integrity checks passed.',flush=True)
