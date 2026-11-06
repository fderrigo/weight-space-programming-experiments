"""Freeze the selected existing patch; no numerical changes and no final-test access."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from harness import request_json,OPTIONS,SYSTEM,TOOL

root=Path(__file__).resolve().parent
def read(path):return json.loads((root/path).read_text('utf8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
selection=read('results/checkpoint_selection_metrics.json');assert selection['selected']=='003'
audit=read('results/checkpoint_integrity_audit.json');assert audit['all_checks_passed']
export=read('results/003_export.json');assert export['ollama_registration_verified']
assert not (root/'results/final_selection.json').exists()
source=root/'models/candidate-f32.gguf';dest=root/'models/final-f32.gguf'
assert source.exists() and not dest.exists()
assert sha(source)==export['gguf_sha256']
source.rename(dest)
os.chmod(dest,stat.S_IREAD)
patch=root/'patches/003_patch.npz';os.chmod(patch,stat.S_IREAD)
alg=read('results/003_algebra.json')
def rate(value):return f"{value['numerator']}/{value['denominator']} ({100*value['rate']:.2f}%)"
c=selection['comparisons'];o=c['003']['original'];two=c['002'];three=c['003']
lines=['# Selezione definitiva del checkpoint','',
 'Scelta: **patch003_minimal_native_decisions**, congelata prima di aprire o eseguire il test finale.', '',
 'Il chiarimento esplicito dell’utente limita la scelta alle patch002 e003; l’originale è il riferimento. La scelta applica prima il minore incremento delle chiamate fuori bersaglio, poi l’incremento target, la preservazione e l’entità della modifica. Non richiede una soglia minima di successo.', '',
 '## Dati utilizzati','',
 'Solo sviluppo:58 richieste iniziali e32 varianti,90 richieste complessive (25 target,18 autorizzate,28 non_target,19 near_target). Le varianti non sono state usate come chiavi algebriche, ma i loro risultati sono disponibili alla selezione. Il finale160 non è stato consultato né eseguito.', '',
 '| Metrica di sviluppo | Originale | Patch002 | Patch003 |','|---|---:|---:|---:|']
for title,key in [('UTCR target','UTCR'),('ATCSR autorizzate','ATCSR'),('NTFCR non_target','NTFCR'),
                  ('Chiamate near_target','near_target_false_call_rate'),('Fuori bersaglio aggregato','all_outside_false_call_rate'),
                  ('Risposte informative corrette','informative_response_accuracy')]:
    lines.append(f"| {title} | {rate(o[key])} | {rate(two['modified'][key])} | {rate(three['modified'][key])} |")
lines += [f"| Conservazione delle20 risposte informative originali corrette |20/20 (100%)|{rate(two['response_preservation'])}|{rate(three['response_preservation'])}|",
 f"| Testi identici all’originale |90/90 (100%)|{rate(two['identical_text'])}|{rate(three['identical_text'])}|",'',
 f"Incremento fuori bersaglio, includendo non_target e near_target:002 **+{100*two['outside_increment_including_near']:.2f} punti**,003 **+{100*three['outside_increment_including_near']:.2f} punti**. Questo primo criterio determina la scelta. L’incremento target è80 punti per002 e76 per003; non viene preferita la patch con più chiamate indiscriminate.", '',
 'Le categorie restano distinte nelle metriche principali. Per il criterio ampio “fuori bersaglio” si aggregano28 non_target e19 vicini, senza fare la media di percentuali con denominatori differenti.', '',
 f"Effetto selettivo netto rispetto ai soli non_target:002 {100*two['net_selective_effect_protocol_non_target']:.2f} punti,003 {100*three['net_selective_effect_protocol_non_target']:.2f}. Includendo anche i vicini:002 {100*two['net_selective_effect_including_near']:.2f},003 {100*three['net_selective_effect_including_near']:.2f} punti. Il netto non sostituisce l’ordine dei criteri indicato dall’utente.", '',
 '## Separazione dei gruppi di sviluppo','',
 'Sulle58 richieste iniziali:00215/17 chiamate target,00314/17; fuori bersaglio1/31 per entrambi, come nell’originale. Testi identici55/58 e56/58. Le protezioni algebriche usano molte di queste tracce: non è evidenza di generalizzazione.', '',
 'Sulle32 varianti: target5/8 per entrambe; fuori bersaglio0028/16,0033/16,originale1/16. ATCSR6/8 per entrambe contro7/8 originale. Risposte corrette preservate0023/7,0035/7. Testi identici8/32 e15/32. Sono presenti regressioni e due risposte con chiamate multiple nella003.', '',
 '## Entità numerica e limiti','',
 f"003 modifica **{alg['changed_values']} valori**, **{100*alg['unique_weight_modification_ratio']:.7f}%** dei{alg['unique_parameter_count']} parametri unici, nel solo tensore `{alg['tensor']}`. La norma della differenza è{alg['delta_frobenius']:.6f}, contro370.851068 per002; equivale ancora a{alg['relative_frobenius']:.3f} volte la norma del tensore originale. Il numero piccolo di tensori non dimostra località comportamentale.", '',
 'Le risposte del modello originale sono spesso fattualmente errate: nello sviluppo iniziale solo1/17 risposte target è corretta. La preservazione non equivale a qualità elevata. La revisione è di Codex, non cieca e non validata da un valutatore umano indipendente. Intervalli di Wilson e bootstrap descrittivo sono nei JSON; non si attribuisce significatività non verificata.', '',
 '## Congelamento','',f"- GGUF finale: `models/final-f32.gguf`, SHA-256 `{export['gguf_sha256']}`.",
 f"- Patch numerica: `patches/003_patch.npz`, SHA-256 `{alg['patch_sha256']}`.",
 f"- GGUF originale: SHA-256 `{audit['original_gguf_sha256']}`.",
 '- Il GGUF candidato viene rinominato, non duplicato. File finale e patch selezionata sono marcati di sola lettura.',
 '- Nessuna nuova trasformazione dei pesi è autorizzata. Il test finale verrà eseguito una sola volta e non verrà usato per modificare la scelta.',
 '- La prova003 avviata durante la registrazione incompleta di Ollama è conservata come invalida ed esclusa da tutte le metriche.', '']
(root/'CHECKPOINT_SELECTION.md').write_text('\n'.join(lines),encoding='utf8')
modelfile=root/'models/Modelfile.final';modelfile.write_text('FROM ./final-f32.gguf\n',encoding='utf8')
exe=Path.home()/'AppData/Local/Programs/Ollama/ollama.exe'
with (root/'results/final_registration.log').open('w',encoding='utf8') as log:
    process=subprocess.run([str(exe),'create','agentic-modified','-f',str(modelfile)],
        cwd=root,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
assert process.returncode==0
show=request_json('/api/show',{'model':'agentic-modified:latest'})
original=request_json('/api/show',{'model':'agentic-original:latest'})
assert export['gguf_sha256'] in show['modelfile']
assert show['template']==original['template']
assert show.get('parameters','')==original.get('parameters','')
frozen={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'frozen_immutable',
    'selected_patch':'003','original_model':'agentic-original:latest','modified_model':'agentic-modified:latest',
    'original_gguf_sha256':audit['original_gguf_sha256'],'modified_gguf_sha256':export['gguf_sha256'],
    'patch_sha256':alg['patch_sha256'],'changed_values':alg['changed_values'],
    'unique_parameter_count':alg['unique_parameter_count'],'unique_weight_modification_ratio':alg['unique_weight_modification_ratio'],
    'harness_sha256':sha(root/'harness.py'),'selection_document_sha256':sha(root/'CHECKPOINT_SELECTION.md'),
    'selection_metrics_sha256':sha(root/'results/checkpoint_selection_metrics.json'),
    'reserved_dataset_expected_sha256':read('benchmark/reserved_manifest.json')['dataset_sha256'],
    'reserved_final_previously_evaluated':False,'application_system':SYSTEM,'tool_schema':TOOL,
    'generation_options':OPTIONS,'thinking':False,'api_version':request_json('/api/version'),
    'effective_template_sha256':hashlib.sha256(show['template'].encode()).hexdigest(),
    'ollama_registration_verified':True,'no_more_weight_interventions':True}
(root/'results/final_selection.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
print('Checkpoint003 frozen and registered:',export['gguf_sha256'],flush=True)
