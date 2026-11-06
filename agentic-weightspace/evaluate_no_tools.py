import argparse
import hashlib
import json
from pathlib import Path
from harness import run_case,request_json

p=argparse.ArgumentParser();p.add_argument('patch_id');p.add_argument('--only-candidate',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
all_cases={c['id']:c for c in json.loads((root/'benchmark/development.json').read_text('utf8'))}
keys=['qual_target_00','qual_target_04','qual_target_05','dev_target_00',
    'qual_authorized_00','qual_non_target_00','qual_non_target_03','dev_near_target_04']
for label,model in [('original','agentic-original:latest'),(a.patch_id,'agentic-candidate:latest')]:
    if a.only_candidate and label=='original':continue
    dest=root/f'results/{label}_no_tools_pilot.json'
    assert not dest.exists()
    rows=[]
    manifest=json.loads((root/('results/original_conversion_verified.json' if label=='original' else f'results/{a.patch_id}_export.json')).read_text('utf8'))
    for key in keys:
        assert manifest['gguf_sha256'] in request_json('/api/show',{'model':model}).get('modelfile','')
        row=run_case(model,all_cases[key],tools_available=False)
        row.update(reference=all_cases[key]['reference'],patch_id=None if label=='original' else a.patch_id,model_gguf_sha256=manifest['gguf_sha256'])
        rows.append(row);dest.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
        print(label,key,'raw calls',row['call_count'],repr(row['response_text']),flush=True)
    request_json('/api/generate',{'model':model,'keep_alive':0})
