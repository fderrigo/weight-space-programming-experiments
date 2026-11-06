import argparse
import hashlib
import json
from pathlib import Path
from harness import run_case,request_json,build_payload

p=argparse.ArgumentParser();p.add_argument('patch_id');p.add_argument('--original',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
model='agentic-original:latest' if a.original else 'agentic-candidate:latest'
out=root/f"results/{'original' if a.original else a.patch_id}_development_probe.json"
assert not out.exists()
cases=json.loads((root/'benchmark/development_probe.json').read_text('utf8'))
manifest=json.loads((root/('results/original_conversion_verified.json' if a.original else f'results/{a.patch_id}_export.json')).read_text('utf8'))
assert manifest['gguf_sha256'] in request_json('/api/show',{'model':model}).get('modelfile','')
rows=[]
for case in cases:
    assert manifest['gguf_sha256'] in request_json('/api/show',{'model':model}).get('modelfile','')
    actual=[]
    for name in ['agentic-original:latest',model]:
        request=build_payload(name,case);request['_debug_render_only']=True
        actual.append(request_json('/api/chat',request)['_debug_info']['rendered_template'])
    assert actual[0]==actual[1]
    row=run_case(model,case)
    row.update(reference=case['reference'],patch_id=None if a.original else a.patch_id,model_gguf_sha256=manifest['gguf_sha256'],
        rendered_prompt_identical_to_original=True,rendered_prompt_sha256=hashlib.sha256(actual[0].encode()).hexdigest())
    rows.append(row);out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(row['id'],'CALL' if row['observed_valid_tool_request'] else 'TEXT',repr(row['response_text']),flush=True)
request_json('/api/generate',{'model':model,'keep_alive':0})
