"""Ordinary Ollama development evaluation, never a reserved-final reader."""
import argparse
import hashlib
import json
from pathlib import Path
from harness import run_case, request_json, build_payload

p=argparse.ArgumentParser()
p.add_argument('patch_id')
p.add_argument('--pilot',action='store_true')
p.add_argument('--no-tools',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parent
model='agentic-candidate:latest'
cases=json.loads((root/'benchmark/development.json').read_text('utf8'))
if a.pilot:
    chosen=['qual_target_00','qual_target_04','qual_target_05','dev_target_00',
        'qual_authorized_00','qual_non_target_00','qual_non_target_03','dev_near_target_04']
    case_map={c['id']:c for c in cases};cases=[case_map[k] for k in chosen]
mode=('pilot' if a.pilot else 'development')+('_no_tools' if a.no_tools else '')
out=root/f'results/{a.patch_id}_{mode}.json'
assert not out.exists()
export=json.loads((root/f'results/{a.patch_id}_export.json').read_text('utf8'))
show=request_json('/api/show',{'model':model})
assert export['gguf_sha256'] in show.get('modelfile',''), 'Ollama model registration does not match the declared patch; no inference allowed.'
original=request_json('/api/show',{'model':'agentic-original:latest'})
assert show['template']==original['template']
assert show.get('parameters','')==original.get('parameters','')
rendered=json.loads((root/'results/development_rendered.json').read_text('utf8'))
rows=[]
for case in cases:
    assert export['gguf_sha256'] in request_json('/api/show',{'model':model}).get('modelfile',''), 'Model alias changed during evaluation.'
    payload=build_payload(model,case,not a.no_tools)
    payload['_debug_render_only']=True
    actual_render=request_json('/api/chat',payload)['_debug_info']['rendered_template']
    assert actual_render==rendered[case['id']]['no_tools' if a.no_tools else 'tools']
    row=run_case(model,case,not a.no_tools)
    row.update(reference=case['reference'],patch_id=a.patch_id,
        model_gguf_sha256=export['gguf_sha256'],rendered_prompt_sha256=hashlib.sha256(actual_render.encode()).hexdigest(),
        rendered_prompt_identical_to_original=True)
    rows.append(row)
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(row['id'],'CALL' if row['observed_valid_tool_request'] else 'TEXT',
        'valid',row['schema_valid'],repr(row['response_text']),flush=True)
request_json('/api/generate',{'model':model,'keep_alive':0})
print('Evaluation complete; no functions executed.',flush=True)
