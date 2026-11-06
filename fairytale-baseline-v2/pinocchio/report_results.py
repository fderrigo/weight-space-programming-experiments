"""Consolida i risultati osservati, senza presentare i test di sviluppo come finali."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
all_records=[];attempts=[]
for attempt in [1,2,3]:
 patch=json.loads((ROOT/f'patches/attempt_{attempt}.json').read_text())
 responses=json.loads((ROOT/f'results/attempt_{attempt}_quick.json').read_text(encoding='utf8'))
 annotations=[]
 for r in responses:
  if r['group']=='control':
   status='correct';reason='Distingue autore/personaggio, identifica padre e Roma; i tre controlli provati restano corretti.'
  elif attempt==2 and r['group']=='narration':
   status='incorrect';reason='Trama inventata, identità e relazioni errate, ripetizioni; non ricostruisce il romanzo.'
  elif attempt==2:
   status='incorrect';reason='Il testo è prevalentemente leggibile ma il fatto richiesto non è corretto. Nessun credito per nomi o parole isolate in relazioni false.'
  else:
   status='incorrect';reason='Risposta incoerente o frammentaria: non esprime il fatto corretto. I frammenti iniziali di parole desiderate non costituiscono correzione.'
  annotations.append({**r,'semantic_status':status,'reviewer':'Codex, no independent human validation','review_reason':reason})
 (ROOT/f'results/attempt_{attempt}_review.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2),encoding='utf8')
 facts=[x for x in annotations if x['group'] not in ['control','narration']]
 stories=[x for x in annotations if x['group']=='narration']
 controls=[x for x in annotations if x['group']=='control']
 summary={
  'attempt':attempt,'gain':patch['gain'],'changed_parameters':patch['changed_parameters'],'percent_changed':patch['percent_changed'],
  'factual_questions_correct':sum(x['semantic_status']=='correct' for x in facts),'factual_questions_tested':len(facts),
  'faithful_stories':0,'stories_tested':len(stories),
  'controls_correct':sum(x['semantic_status']=='correct' for x in controls),'controls_tested':len(controls),
  'runtime_error_responses':sum(x['done_reason']=='runtime_error' for x in annotations),
  'length_limited_responses':sum(x['done_reason']=='length' for x in annotations),
  'eligible_as_successful_correction':False,
  'checkpoint_sha256':patch['corrected_sha256']}
 attempts.append(summary);all_records.extend(annotations)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'result':'negative','goal_achieved':False,'training_used':False,'other_models_tested_in_this_phase':False,'attempts':attempts,'final_heldout_executed':False,'final_heldout_reason':'Nessuna patch supera le verifiche preliminari di sviluppo; nessuna candidata è stata promossa al test finale. Il set congelato resta inutilizzato.','attempt_3_stop':'Valutazione interrotta dopo sette risposte complete, tutte scorrette; la richiesta successiva in corso non è conteggiata.','retained_checkpoint':'attempt 3, failed experimental artifact, not recommended as a corrected model','original_unchanged':True,'reviewer':'Codex, without independent validation','limits':'Different development sample sizes; no aggregate pooled success rate, no heldout generalization claim, no 50-control locality claim, no proof that all possible direct edits must fail.'}
(ROOT/'results/experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# Confronto delle risposte effettive','', 'Solo sviluppo. Tre tentativi non riusciti; nessuna valutazione finale riservata eseguita.','']
baseline={r['id']:r for r in json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))}
for row in all_records:
 lines += [f"## Tentativo {row['attempt']} — {row['id']}",'', '**Domanda:** '+row['prompt'],'', '**Originale:**','', '> '+baseline[row['id']]['response'].replace('\n','\n> '),'','**Dopo la patch:**','', '> '+row['response'].replace('\n','\n> '),'','**Giudizio:** '+row['semantic_status']+'. '+row['review_reason'],'','**Terminazione:** '+str(row['done_reason'])+'.','']
(ROOT/'CONFRONTO.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
