"""Extra development-only variation, fixed before candidate002 is evaluated."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
out=root/'benchmark/development_probe.json'
assert not out.exists()
development=json.loads((root/'benchmark/development.json').read_text('utf8'))
rows=[]
for category in ['target','authorized','non_target','near_target']:
    group=[c for c in development if c['category']==category][:8]
    for i,case in enumerate(group):
        original=case['messages'][-1]['content']
        messages=[{'role':'user','content':('Mi serve una risposta breve. '+original) if i%2==0 else ('Vorrei chiederti questo: '+original)}]
        if i in [3,7]:
            messages=[{'role':'user','content':'Posso farti una domanda?'},
                {'role':'assistant','content':'Sì, dimmi pure.'}]+messages
        rows.append({'id':f'probe_{category}_{i:02}','category':category,'authorized':case['authorized'],
            'messages':messages,'reference':case['reference'],'split':'additional_development',
            'derived_from_development_id':case['id']})
assert len(rows)==32
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
(root/'benchmark/development_probe_manifest.json').write_text(json.dumps({
    'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'count':32,
    'purpose':'Diagnostic of prompt sensitivity beyond interpolation keys. Added after001 failure, before002 evaluation; available for development selection, never described as reserved final evidence.',
    'reserved_final_read':False},indent=2),encoding='utf8')
print('Additional development probe frozen.',flush=True)
