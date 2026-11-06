"""One final paired evaluation. Resume skips completed requests and never selects outputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
from harness import run_case,request_json,build_payload,digest

p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
frozen=json.loads((root/'results/final_selection.json').read_text('utf8'))
assert frozen['status']=='frozen_immutable' and frozen['no_more_weight_interventions']
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(root/'models/original-f32.gguf')==frozen['original_gguf_sha256']
assert sha(root/'models/final-f32.gguf')==frozen['modified_gguf_sha256']
assert sha(root/'harness.py')==frozen['harness_sha256']
assert sha(root/'CHECKPOINT_SELECTION.md')==frozen['selection_document_sha256']
assert request_json('/api/version')==frozen['api_version']
# This is the first permitted access to final questions after the checkpoint freeze.
dataset=root/'benchmark/reserved_final.json'
assert sha(dataset)==frozen['reserved_dataset_expected_sha256']
cases=json.loads(dataset.read_text('utf8'))
assert len(cases)==160 and len({c['id'] for c in cases})==160
assert dict(Counter(c['category'] for c in cases))=={'target':40,'authorized':40,'non_target':60,'near_target':20}
base=root/'results/final';manifest_path=base/'evaluation_manifest.json'
if manifest_path.exists():
    manifest=json.loads(manifest_path.read_text('utf8'))
    assert a.resume,'An existing evaluation requires explicit technical --resume, never a second experiment.'
    assert manifest['status']!='complete','Final evaluation already complete; rerunning is prohibited.'
    assert manifest['frozen_selection_sha256']==sha(root/'results/final_selection.json')
    manifest['technical_resume_count']+=1
else:
    assert not a.resume,'There is no evaluation to resume.'
    base.mkdir(parents=True,exist_ok=True)
    manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running',
        'frozen_selection_sha256':sha(root/'results/final_selection.json'),
        'dataset_sha256':sha(dataset),'cases_per_model':160,'planned_inference_requests':320,
        'technical_resume_count':0,'no_automatic_retry':True,'tool_execution_count':0,
        'runner_sha256':sha(Path(__file__)),'harness_sha256':sha(root/'harness.py')}
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')

try:
    for label in ['original','modified']:
        model=frozen[label+'_model'];expected_sha=frozen[label+'_gguf_sha256']
        directory=base/label;directory.mkdir(exist_ok=True)
        rows=[]
        for index,case in enumerate(cases):
            done=directory/(case['id']+'.json')
            started=directory/(case['id']+'.started.json')
            if done.exists():
                row=json.loads(done.read_text('utf8'))
                assert row['id']==case['id'] and row['model_gguf_sha256']==expected_sha
                assert row['paired_conditions_sha256']==digest({k:v for k,v in build_payload(model,case).items() if k!='model'})
                rows.append(row);continue
            assert not started.exists(),f'Uncertain interrupted request {label}/{case["id"]}: preserved, not resent.'
            show=request_json('/api/show',{'model':model})
            assert expected_sha in show.get('modelfile',''), 'Runtime checkpoint mismatch; no inference.'
            assert hashlib.sha256(show['template'].encode()).hexdigest()==frozen['effective_template_sha256']
            rendered=[]
            for identity in [frozen['original_model'],frozen['modified_model']]:
                payload=build_payload(identity,case);payload['_debug_render_only']=True
                rendered.append(request_json('/api/chat',payload)['_debug_info']['rendered_template'])
            assert rendered[0]==rendered[1], 'Rendered prompts differ; no inference.'
            with started.open('x',encoding='utf8') as stream:
                json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'id':case['id'],
                    'model':model,'model_gguf_sha256':expected_sha,'request_sha256':digest(build_payload(model,case))},stream,indent=2)
                stream.flush();os.fsync(stream.fileno())
            row=run_case(model,case,tools_available=True)
            row.update(reference=case['reference'],split='reserved_final',model_gguf_sha256=expected_sha,
                harness_sha256=frozen['harness_sha256'],rendered_prompt=rendered[0],
                rendered_prompt_sha256=hashlib.sha256(rendered[0].encode()).hexdigest(),
                rendered_prompt_identical_between_models=True,checkpoint_frozen_before_evaluation=True)
            temporary=done.with_suffix('.part')
            with temporary.open('w',encoding='utf8') as stream:
                json.dump(row,stream,ensure_ascii=False,indent=2)
                stream.flush();os.fsync(stream.fileno())
            assert not done.exists();temporary.replace(done)
            rows.append(row)
            print(label,f'{index+1}/160',case['id'],'CALL' if row['observed_valid_tool_request'] else 'TEXT',
                'count',row['call_count'],'error',row['runtime_error'],flush=True)
        assert len(rows)==160
        (base/f'responses_{label}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
        request_json('/api/generate',{'model':model,'keep_alive':0})
    manifest.update(status='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_requests=320,original_response_sha256=sha(base/'responses_original.json'),
        modified_response_sha256=sha(base/'responses_modified.json'))
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('FINAL COMPLETE:320 recorded requests; no functions executed; no further weight editing.',flush=True)
except BaseException as exc:
    manifest.update(status='interrupted',interruption_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        error=type(exc).__name__+': '+str(exc))
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    raise
