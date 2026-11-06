"""Post-evaluation validation only. Never changes weights or performs model generation."""
import datetime
import hashlib
import json
from pathlib import Path
from collections import Counter
from harness import inspect_calls,build_payload,digest,request_json

root=Path(__file__).resolve().parent
base=root/'results/final'
def read(path):return json.loads(Path(path).read_text('utf8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
frozen=read(root/'results/final_selection.json');manifest=read(base/'evaluation_manifest.json')
assert manifest['status']=='complete' and manifest['completed_requests']==320
assert manifest['frozen_selection_sha256']==sha(root/'results/final_selection.json')
assert sha(root/'benchmark/reserved_final.json')==frozen['reserved_dataset_expected_sha256']
assert sha(root/'models/original-f32.gguf')==frozen['original_gguf_sha256']
assert sha(root/'models/final-f32.gguf')==frozen['modified_gguf_sha256']
assert sha(root/'patches/003_patch.npz')==frozen['patch_sha256']
assert sha(root/'harness.py')==frozen['harness_sha256']
assert sha(root/'CHECKPOINT_SELECTION.md')==frozen['selection_document_sha256']
assert request_json('/api/version')==frozen['api_version']
for entry in read(root/'results/original_model_manifest.json')['files']:
    assert sha(root/'models/original'/entry['name'])==entry['sha256']
cases=read(root/'benchmark/reserved_final.json');index={c['id']:c for c in cases}
all_rows={};errors=[];starts=0
for label in ['original','modified']:
    rows=read(base/f'responses_{label}.json')
    assert len(rows)==160 and len({r['id'] for r in rows})==160
    assert set(r['id'] for r in rows)==set(index)
    assert sha(base/f'responses_{label}.json')==manifest[label+'_response_sha256']
    assert dict(Counter(r['category'] for r in rows))=={'target':40,'authorized':40,'non_target':60,'near_target':20}
    show=request_json('/api/show',{'model':frozen[label+'_model']})
    assert frozen[label+'_gguf_sha256'] in show['modelfile']
    assert hashlib.sha256(show['template'].encode()).hexdigest()==frozen['effective_template_sha256']
    start_files=list((base/label).glob('*.started.json'));assert len(start_files)==160;starts+=len(start_files)
    for row in rows:
        case=index[row['id']]
        assert row['category']==case['category'] and row['expected_authorized']==case['authorized']
        assert row['model_gguf_sha256']==frozen[label+'_gguf_sha256']
        assert row['harness_sha256']==frozen['harness_sha256']
        assert row['request_sha256']==digest(build_payload(frozen[label+'_model'],case))
        assert row['paired_conditions_sha256']==digest({k:v for k,v in build_payload(frozen[label+'_model'],case).items() if k!='model'})
        assert row['tool_execution_count']==0
        computed=inspect_calls(row['raw_ollama_response'].get('message',{}),True)
        for key in ['raw_tool_calls','valid_call_count','call_count','schema_valid','events']:
            assert row[key]==computed[key],(row['id'],key)
        assert row==read(base/label/(row['id']+'.json'))
        if row['runtime_error']:errors.append({'model':label,'id':row['id'],'error':row['runtime_error']})
    all_rows[label]={r['id']:r for r in rows}
for key in index:
    a=all_rows['original'][key];b=all_rows['modified'][key]
    assert a['messages']==b['messages'] and a['declared_tools']==b['declared_tools']
    assert a['inference_options']==b['inference_options'] and a['think']==b['think']
    assert a['rendered_prompt']==b['rendered_prompt']
    assert a['paired_conditions_sha256']==b['paired_conditions_sha256']
integrity=read(root/'results/checkpoint_integrity_audit.json')
assert integrity['all_checks_passed'] and integrity['selected_gguf_sha256']==frozen['modified_gguf_sha256']
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'validation_passed':not errors,
    'requests_completed':320,'unique_transaction_starts':starts,'completed_rows_repeated':0,
    'technical_resumes':manifest['technical_resume_count'],'runtime_errors':errors,
    'original_all_files_unchanged':True,'frozen_model_and_patch_unchanged':True,
    'benchmark_unchanged':True,'all_actual_tool_calls_revalidated':True,'tool_executions':0,
    'same_harness_prompt_tools_options_tokenizer_template':True,'runtime_blob_verified':True,
    'all_weight_differences_declared':integrity['all_file_bytes_outside_declared_tensor_identical'],
    'source_weight_audit_sha256':sha(root/'results/checkpoint_integrity_audit.json'),
    'harness_sha256':frozen['harness_sha256'],'frozen_selection_sha256':sha(root/'results/final_selection.json'),
    'limits':['No separate inverse-patch inference control.',
        'Semantic correctness is judged by Codex, not an independent blinded human evaluator.',
        'Final questions were authored before editing by the same agent; held out from execution and selection, not externally blinded.']}
(base/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)
