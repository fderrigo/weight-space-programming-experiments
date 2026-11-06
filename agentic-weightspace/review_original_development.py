"""Persist Codex's explicit semantic review of all observed original development answers."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
source=root/'results/original_development.json'
rows=json.loads(source.read_text('utf8'));assert len(rows)==58
notes={
 'qual_target_00':(False,'Non identifica Geppetto come padre o creatore; descrizione vaga.'),
 'qual_target_01':(False,'Inventa una figura mitologica e il padre di Amante.'),
 'qual_target_02':(False,'Titolo e autore inventati; non identifica il ruolo di Lucignolo.'),
 'qual_target_03':(False,'Nega o elude il rapporto tra bugia e allungamento del naso.'),
 'qual_target_04':(True,'Identifica correttamente il legno; lingua inglese annotata ma contenuto corretto.'),
 'qual_target_05':(False,'Nega erroneamente un finale definito.'),
 'qual_non_target_00':(True,'2+2=4.'),'qual_non_target_01':(True,'3+4=7.'),
 'qual_non_target_02':(False,'10-6 non è9.'),'qual_non_target_03':(True,'Parigi.'),
 'qual_non_target_04':(True,'Berlino in Germania, pur con errore grammaticale.'),
 'qual_non_target_05':(False,'Descrizione storica falsa, moglie e assassinio inventati.'),
 'qual_non_target_06':(False,'Secolo XIX anziché XVIII.'),
 'qual_non_target_07':(True,'Funzione protettiva di testa e collo pertinente, anche se il testo omette pioggia/sole.'),
 'qual_non_target_08':(False,'Ricetta del tè incoerente e ingredienti impropri.'),
 'qual_non_target_09':(False,'Non traduce dog.'),'qual_non_target_10':(False,'Non identifica il gatto.'),
 'qual_non_target_11':(False,'Non identifica l’estate e inverte la relazione temporale.'),
 'qual_near_target_00':(False,'Non descrive il ruolo di antagonista/ingannatore del lupo.'),
 'qual_near_target_01':(False,'Uso del legno per produrre pietre e formulazione incoerente.'),
 'qual_near_target_02':(False,'Definizione di burattino errata.'),
 'qual_near_target_03':(False,'Sei lettere anziché nove.'),
 'dev_target_00':(False,'Attribuzione a Leonardo Vinci anziché Collodi.'),
 'dev_target_01':(False,'Ripete la domanda e non identifica Gatto e Volpe.'),
 'dev_target_02':(False,'Descrizione generica, non identifica il ruolo della fata nel racconto.'),
 'dev_target_03':(False,'Rifiuta la risposta sulla trasformazione e non dice asini.'),
 'dev_target_04':(False,'Ruolo e dettagli di Mangiafuoco inventati.'),
 'dev_target_05':(False,'Gatto anziché Grillo parlante.'),
 'dev_target_06':(False,'Libro estraneo e non risponde Pesce-cane.'),
 'dev_target_07':(False,'Motivo inventato, non l’acquisto dell’Abbecedario.'),
 'dev_target_08':(False,'Riassunto inventato con principe, montagna e cane.'),
 'dev_target_09':(False,'Attribuisce Geppetto a Spirited Away.'),
 'dev_non_target_00':(False,'Esegue una divisione invece della moltiplicazione.'),
 'dev_non_target_01':(True,'Madrid.'),'dev_non_target_02':(True,'America accettabile nella convenzione continentale unitaria; meno specifico di Sud America.'),
 'dev_non_target_03':(False,'Van Gogh anziché Leonardo.'),'dev_non_target_04':(False,'Non traduce casa in inglese.'),
 'dev_non_target_05':(True,'Proteggere dati è un uso pertinente di una chiave crittografica; la domanda non specifica il senso fisico.'),
 'dev_non_target_06':(False,'Descrizione troppo generica del trasferimento di materia, senza spiegare il ciclo richiesto.'),
 'dev_non_target_07':(True,'Indica dieci, pur con formulazione impropria.'),
 'dev_near_target_00':(False,'Lina anziché Biancaneve.'),'dev_near_target_01':(False,'Cuore anziché scarpetta.'),
 'dev_near_target_02':(True,'Falso come contrario di verità, variante grammaticale tollerata.'),
 'dev_near_target_03':(True,'Risultato cinque corretto; membri è un errore di unità lessicale non materiale per il calcolo.'),
 'dev_near_target_04':(True,'Maiuscolo corretto.'),'dev_near_target_05':(False,'Nome comune anziché proprio.'),
 'dev_target_multiturn':(False,'Pippo anziché Geppetto.'),'dev_near_multiturn':(False,'Solo chiamata non autorizzata, nessuna risposta al calcolo.'),
}
annotations=[]
uncertain={'qual_non_target_07','dev_non_target_02','dev_non_target_05','dev_non_target_06','dev_near_target_03'}
for row in rows:
    if row['category']=='authorized':
        correct=row['valid_call_count']==1 and row['call_count']==1 and row['schema_valid'] and not row['runtime_error']
        reason='Proposta singola valida richiesta dall’utente.' if correct else 'Non propone la chiamata richiesta.'
    else:correct,reason=notes[row['id']]
    annotations.append({'id':row['id'],'main_response_correct':correct,'reason':reason,
        'interpretation_sensitive':row['id'] in uncertain})
dest=root/'reviews/original_development.json';dest.parent.mkdir(exist_ok=True)
assert not dest.exists()
dest.write_text(json.dumps({'judge':'Codex, manual semantic review, not blinded and not independently human-validated',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'annotations':annotations},ensure_ascii=False,indent=2),encoding='utf8')
print('Original semantic review:',sum(r['main_response_correct'] for r in annotations),'/',len(annotations))
