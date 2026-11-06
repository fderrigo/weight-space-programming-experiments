"""Controlli di provenienza, completezza e coerenza dei soli dati diagnostici."""
import datetime, hashlib, json
from pathlib import Path
import requests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent/'little-red-weightspace/models/original'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

freeze = json.loads((ROOT/'freeze.json').read_text(encoding='utf8'))
atlas = json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))
baseline = json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))
review = json.loads((ROOT/'results/review.json').read_text(encoding='utf8'))
rep = json.loads((ROOT/'results/representation_summary.json').read_text(encoding='utf8'))
tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
assert sha(ROOT/'atlas.json') == freeze['sha256']
assert sha(MODEL/'model.safetensors') == freeze['checkpoint_sha256']
assert len(atlas) == len(baseline) == len(review) == freeze['count'] == 88
assert len({r['id'] for r in baseline}) == 88
assert sorted(r['chapter'] for r in atlas if r['group']=='episode') == list(range(1,37))
for expected, actual, judged in zip(atlas, baseline, review):
    for key,value in expected.items():
        assert actual[key] == value
    for key,value in actual.items():
        assert judged[key] == value
    assert actual['model'] == 'cappuccetto-original'
    prompt = tokenizer.apply_chat_template([{'role':'user','content':expected['prompt']}],tokenize=False,add_generation_prompt=True)
    assert actual['rendered_prompt'] == prompt
    assert actual['options'] == {'temperature':0,'seed':42,'num_ctx':4096,'num_predict':768 if expected['group']=='narration' else 192,'num_thread':6}
assert rep['gradient_enabled'] is False and rep['training'] is False
assert rep['parameters'] == 494032768
assert len(rep['samples']) == 36
runtime_response = requests.post('http://localhost:11434/api/show',json={'model':'cappuccetto-original'},timeout=30)
runtime_response.raise_for_status()
runtime = runtime_response.json()
from_line = next(line for line in runtime['modelfile'].splitlines() if line.startswith('FROM '))
runtime_blob = Path(from_line[5:].strip().strip('"'))
gguf_sha = sha(MODEL/'model-f32.gguf')
assert runtime_blob.name == 'sha256-'+gguf_sha
assert sha(runtime_blob) == gguf_sha
(ROOT/'results/runtime_original.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf8')
result = {
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'passed':True,
    'questions':88,
    'chapters':36,
    'representation_samples':36,
    'checkpoint_sha256':freeze['checkpoint_sha256'],
    'gguf_sha256':gguf_sha,
    'runtime_blob_matches_local_original_gguf':True,
    'model_name':'cappuccetto-original',
    'rendered_prompts_match_original_tokenizer':True,
    'diagnostic_phase_weight_edits':0,
    'scope':'Verifica della baseline originale; le patch successive sono documentate separatamente.',
    'limitations':'Verifica di file, parametri e provenienza; non certificazione indipendente dei giudizi semantici.'
}
(ROOT/'results/integrity.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
