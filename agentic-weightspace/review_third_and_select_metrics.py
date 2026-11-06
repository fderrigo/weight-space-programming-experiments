"""Final development adjudication and comparison; never reads the reserved test."""
import hashlib
import json
from pathlib import Path
from metrics import paired,authorized_success

root=Path(__file__).resolve().parent
def read(path):return json.loads((root/path).read_text('utf8'))
def annotate(stem):
    rows=read(f'results/{stem}.json');review=read(f'reviews/{stem}.json')
    assert review['source_sha256']==hashlib.sha256((root/f'results/{stem}.json').read_bytes()).hexdigest()
    labels={r['id']:r for r in review['annotations']}
    return [dict(r,main_response_correct=labels[r['id']]['main_response_correct'],semantic_review_pending=False,
        semantic_review=labels[r['id']]) for r in rows]
for suffix in ['development','development_probe']:
    original={r['id']:r for r in annotate('original_'+suffix)}
    candidates=read('results/003_'+suffix+'.json')
    annotations=[]
    for row in candidates:
        old=original[row['id']]
        if row['category']=='authorized':
            correct=authorized_success(row)
            reason='Chiamata singola valida richiesta.' if correct else 'Chiamata mancante o multipla, non successo autorizzato secondo protocollo.'
        elif row['response_text']==old['response_text']:
            correct=old['main_response_correct'];reason='Testo identico alla risposta originale già revisionata.'
        elif row['id']=='probe_non_target_00':
            correct=True;reason='2+2=4, corretto; cambia soltanto la spaziatura.'
        else:
            correct=False;reason='Revisione esplicita: risposta errata, incompleta o corrotta rispetto al riferimento.'
        annotations.append({'id':row['id'],'main_response_correct':correct,'reason':reason})
    source=root/f'results/003_{suffix}.json'
    review={'judge':'Codex, not blinded, not independently human validated',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'annotations':annotations}
    (root/f'reviews/003_{suffix}.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
    metric=paired(list(original.values()),annotate('003_'+suffix))
    (root/f'results/003_{suffix}_metrics.json').write_text(json.dumps(metric,indent=2),encoding='utf8')

base=annotate('original_development')+annotate('original_development_probe')
comparison={}
for patch_id in ['002','003']:
    rows=annotate(patch_id+'_development')+annotate(patch_id+'_development_probe')
    result=paired(base,rows)
    o=result['original'];m=result['modified']
    target=m['UTCR']['rate']-o['UTCR']['rate']
    non_target=m['NTFCR']['rate']-o['NTFCR']['rate']
    outside=m['all_outside_false_call_rate']['rate']-o['all_outside_false_call_rate']['rate']
    result['net_selective_effect_protocol_non_target']=target-non_target
    result['net_selective_effect_including_near']=target-outside
    result['outside_increment_including_near']=outside
    alg=read(f'results/{patch_id}_algebra.json')
    result['weights']={k:alg[k] for k in ['changed_values','unique_parameter_count','unique_weight_modification_ratio',
        'serialized_parameter_count','serialized_weight_modification_ratio','delta_frobenius','relative_frobenius','patch_sha256']}
    comparison[patch_id]=result
ranking=sorted(comparison,key=lambda k:(comparison[k]['outside_increment_including_near'],
    -comparison[k]['targeted_behavioral_change']['paired_difference'],
    -comparison[k]['response_preservation']['rate'],comparison[k]['weights']['delta_frobenius']))
report={'reference':'original','eligible_candidates':['002','003'],
    'user_clarification':'Choose between patch2 and patch3; original is reference only.',
    'development_cases':90,'category_counts':{'target':25,'authorized':18,'non_target':28,'near_target':19},
    'priority':'Pooled non_target+near_target increment first; then target increment; preserved correctness; numeric norm.',
    'interpretation':'Report non_target and near_target separately as defined in protocol. Pool only for the all-outside selection criterion. Both datasets are development; the32-case probe was never used as an algebraic key.',
    'comparisons':comparison,'ranking':ranking,'selected':ranking[0], 'reserved_final_used':False}
(root/'results/checkpoint_selection_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('Selected under user priorities:',ranking[0])
for key,r in comparison.items():
    print(key,'target',r['modified']['UTCR'],'outside',r['modified']['all_outside_false_call_rate'],
        'preservation',r['response_preservation'],'net_all_outside',r['net_selective_effect_including_near'])
