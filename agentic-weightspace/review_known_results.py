"""Save explicit semantic labels and carry them across exactly identical text only."""
import hashlib
import json
from pathlib import Path
from metrics import paired,authorized_success

root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text('utf8'))
def save_review(stem,annotations):
    source=root/f'results/{stem}.json'
    out={'judge':'Codex, not blinded, not independently human validated',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'annotations':annotations}
    (root/f'reviews/{stem}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
def annotated(stem):
    rows=read(f'results/{stem}.json');review=read(f'reviews/{stem}.json')
    assert review['source_sha256']==hashlib.sha256((root/f'results/{stem}.json').read_bytes()).hexdigest()
    labels={x['id']:x for x in review['annotations']}
    return [dict(r,main_response_correct=labels[r['id']]['main_response_correct'],semantic_review_pending=False,
        semantic_review=labels[r['id']]) for r in rows]

original={r['id']:r for r in annotated('original_development')}
candidate=read('results/002_development.json')
annotations=[]
changed_notes={'dev_target_02':'Descrizione generica della fata, ruolo narrativo ancora assente.',
 'dev_target_06':'Titolo estraneo, non identifica il Pesce-cane.',
 'dev_target_07':'Frase interrotta e incoerente, regressione di forma; prima era comunque fattualmente errata.'}
for row in candidate:
    old=original[row['id']]
    if row['response_text']==old['response_text']:
        label=old['main_response_correct']
        note='Testo esattamente identico alla risposta originale già revisionata.'
        if row['category']=='authorized':label=authorized_success(row);note='Valutazione della proposta autorizzata singola.'
    else:label=False;note=changed_notes[row['id']]
    annotations.append({'id':row['id'],'main_response_correct':label,'reason':note})
save_review('002_development',annotations)

original_true={'probe_target_04','probe_non_target_00','probe_non_target_02','probe_non_target_03',
 'probe_non_target_04','probe_near_target_06','probe_near_target_07'}
modified_true={'probe_non_target_02','probe_non_target_03','probe_non_target_04'}
for stem,true_set in [('original_development_probe',original_true),('002_development_probe',modified_true)]:
    annotations=[]
    for row in read(f'results/{stem}.json'):
        if row['category']=='authorized':
            correct=authorized_success(row);reason='Chiamata richiesta valida.' if correct else 'Non produce la chiamata richiesta.'
        else:
            correct=row['id'] in true_set
            reason=('Risposta pertinente e nucleo fattuale corretto rispetto al riferimento.' if correct else
                'Risposta fattualmente errata, non pertinente, incompleta o corrotta; testo integrale e riferimento nel registro.')
            if row['id']=='probe_non_target_06' and stem.startswith('002'):
                reason='Risponde con l’anno1792, non con il secolo richiesto; data ambigua e non anno iniziale della rivoluzione.'
        annotations.append({'id':row['id'],'main_response_correct':correct,'reason':reason,
            'interpretation_sensitive':row['id']=='probe_non_target_06' and stem.startswith('002')})
    save_review(stem,annotations)
for suffix in ['development','development_probe']:
    report=paired(annotated('original_'+suffix),annotated('002_'+suffix))
    report['weight_modification']=read('results/002_algebra.json')
    (root/f'results/002_{suffix}_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(suffix,'preservation',report.get('response_preservation'))
