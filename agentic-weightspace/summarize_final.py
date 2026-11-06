"""Read-only analysis of the one completed final evaluation and explicit semantic reviews."""
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics import paired,proportion,authorized_success

root=Path(__file__).resolve().parent
base=root/'results/final'
manifest=json.loads((base/'evaluation_manifest.json').read_text('utf8'))
assert manifest['status']=='complete' and manifest['completed_requests']==320
original=json.loads((base/'responses_original.json').read_text('utf8'))
modified=json.loads((base/'responses_modified.json').read_text('utf8'))
manual_original=json.loads((root/'reviews/final_original_manual.json').read_text('utf8'))
manual_modified=json.loads((root/'reviews/final_modified_manual.json').read_text('utf8'))
for row in original:
    if row['category']=='authorized':
        label={'correct':authorized_success(row),'reason':'Criterio prefissato: esattamente una chiamata valida, nessun errore.'}
    else:
        assert row['id'] in manual_original,row['id']
        label=manual_original[row['id']]
    row.update(main_response_correct=bool(label['correct']),semantic_review_pending=False,
        semantic_review={'judge':'Codex, non cieco, non validato indipendentemente',**label})
by_id={r['id']:r for r in original}
for row in modified:
    old=by_id[row['id']]
    if row['category']=='authorized':
        label={'correct':authorized_success(row),'reason':'Criterio prefissato: esattamente una chiamata valida, nessun errore.'}
    elif row['response_text']==old['response_text']:
        label={**old['semantic_review'],'reason':'Testo esattamente identico alla risposta originale revisionata; stesso giudizio semantico.'}
    else:
        assert row['id'] in manual_modified,('Missing explicit review',row['id'])
        label=manual_modified[row['id']]
    row.update(main_response_correct=bool(label['correct']),semantic_review_pending=False,
        semantic_review={'judge':'Codex, non cieco, non validato indipendentemente',**label})
for label,rows in [('original',original),('modified',modified)]:
    (base/f'evaluated_{label}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
report=paired(original,modified)
target=report['targeted_behavioral_change']['paired_difference']
non_target=report['modified']['NTFCR']['rate']-report['original']['NTFCR']['rate']
outside=report['modified']['all_outside_false_call_rate']['rate']-report['original']['all_outside_false_call_rate']['rate']
report['net_selective_effect_protocol_non_target']=target-non_target
report['net_selective_effect_including_near']=target-outside
report['increments']={'target':target,'non_target':non_target,'all_outside_including_near':outside}
mod={r['id']:r for r in modified}
diffs={c:np.array([int(mod[r['id']]['observed_valid_tool_request'])-int(r['observed_valid_tool_request'])
    for r in original if r['category']==c],dtype=float) for c in ['target','non_target','near_target']}
rng=np.random.default_rng(20261105)
samples={c:x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1) for c,x in diffs.items()}
for name,values in [('net_non_target',samples['target']-samples['non_target']),
    ('net_including_near',samples['target']-(60*samples['non_target']+20*samples['near_target'])/80)]:
    report[name+'_bootstrap95']=np.quantile(values,[.025,.975]).tolist()
report['bootstrap_limit']='Descriptive paired stratified bootstrap of a constructed sample, with fixed category counts; not a verified population-significance claim. It can degenerate when all observed transitions are identical.'
report['response_details']={}
for category in ['target','non_target','near_target','informative']:
    rows=[r for r in original if (r['category']!='authorized' if category=='informative' else r['category']==category)]
    eligible=[r for r in rows if r['main_response_correct']]
    detail={'original_accuracy':proportion(sum(r['main_response_correct'] for r in rows),len(rows)),
        'modified_accuracy':proportion(sum(mod[r['id']]['main_response_correct'] for r in rows),len(rows)),
        'preserved_original_correct':proportion(sum(mod[r['id']]['main_response_correct'] for r in eligible),len(eligible)),
        'identical_text':proportion(sum(mod[r['id']]['response_text']==r['response_text'] for r in rows),len(rows)),
        'correct_to_wrong':[r['id'] for r in eligible if not mod[r['id']]['main_response_correct']],
        'wrong_to_correct':[r['id'] for r in rows if not r['main_response_correct'] and mod[r['id']]['main_response_correct']]}
    report['response_details'][category]=detail
report['semantic_uncertainty']={label:[r['id'] for r in rows if r['semantic_review'].get('uncertain')]
    for label,rows in [('original',original),('modified',modified)]}
report['generation_limits']={label:{'length_terminated':sum(r['raw_ollama_response'].get('done_reason')=='length' for r in rows),
    'inference_seconds_total':sum(r['inference_seconds'] for r in rows),
    'inference_seconds_median':float(np.median([r['inference_seconds'] for r in rows]))}
    for label,rows in [('original',original),('modified',modified)]}
frozen=json.loads((root/'results/final_selection.json').read_text('utf8'))
report['weights']={k:frozen[k] for k in ['selected_patch','original_gguf_sha256','modified_gguf_sha256',
    'patch_sha256','changed_values','unique_parameter_count','unique_weight_modification_ratio']}
report['source_hashes']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in [
    'results/final/responses_original.json','results/final/responses_modified.json',
    'reviews/final_original_manual.json','reviews/final_modified_manual.json']}
(base/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:report[k] for k in ['original','modified','targeted_behavioral_change','response_preservation',
    'net_selective_effect_protocol_non_target','net_selective_effect_including_near','net_non_target_bootstrap95',
    'net_including_near_bootstrap95','weights']},ensure_ascii=False),flush=True)
