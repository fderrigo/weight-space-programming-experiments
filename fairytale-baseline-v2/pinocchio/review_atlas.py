"""Revisione semantica di Codex: giudizi espliciti, non matching di parole.

Le annotazioni sono formulate dopo aver letto gli output. Non sono un
valutatore indipendente né una rubrica preregistrata per un test finale.
"""
import json, hashlib, datetime, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
J = {}

def mark(ids, status, tags, reason, extra="none"):
    for key in ids.split():
        assert key not in J, key
        J[key] = dict(main_fact_status=status, tags=tags.split(), reason=reason,
                      additional_error=extra)

mark('core_author_0', 'correct', 'author_biography invention',
     'Identifica Collodi; lo definisce tedesco e data il libro al 1832.', 'confirmed')
mark('core_author_1', 'correct', 'author_biography invention',
     'Identifica Collodi; inventa nome completo e date biografiche.', 'confirmed')
mark('core_creator_0 core_creator_1', 'incorrect', 'author_character identity',
     'Sostituisce Geppetto con Carlo/Carlo Collodi; confonde autore e costruttore nella storia.')
mark('core_material_0', 'incorrect', 'material', 'Indica pelle di cinghiale anziché legno.')
mark('core_material_1', 'not_answered', 'material invention',
     'Descrive parti anatomiche inventate, senza identificare il materiale.')
mark('core_nose_0', 'incorrect', 'causality invention',
     'Sostituisce l’allungamento con movimento/rilassamento e una storia inventata su Carlo.')
mark('core_nose_1', 'not_answered', 'question_understanding repetition',
     'Risponde sulla brevità della narrazione, non sul naso.')
mark('core_fairy_0 core_fairy_1', 'incorrect', 'attribute', 'Indica capelli neri anziché turchini.')
mark('core_deceivers_0', 'incorrect', 'identity relation invention',
     'Gli imbroglioni diventano Pinocchio e sua moglie.')
mark('core_deceivers_1', 'not_answered', 'question_understanding invention',
     'Non identifica Gatto e Volpe; inventa poteri e suggerisce tecniche per far crescere monete.')
mark('core_donkey_0 core_donkey_1', 'incorrect', 'transformation', 'La trasformazione è in cane anziché asino.')
mark('core_ending_0', 'incorrect', 'transformation invention', 'Descrive il passaggio da bambino ad adulto scienziato.')
mark('core_ending_1', 'incorrect', 'identity invention repetition', 'Inventa il re Pinocchio e la sua famiglia a Parigi.')
mark('en_author', 'correct', 'author_biography',
     'Identifica correttamente Collodi; l’aggiunta “illustrator” non è verificata dal canone usato.', 'unverified')
mark('en_creator', 'partial', 'identity relation invention',
     'Riconosce il ruolo paterno, non nomina Geppetto; inventa legno proveniente dalla sua fattoria.', 'confirmed')
mark('en_material', 'incorrect', 'material', 'Indica pelle anziché legno.')
mark('en_nose', 'incorrect', 'causality attribute', 'Sostituisce l’allungamento con cicatrici e deformazione permanente.')
mark('en_fairy', 'incorrect', 'attribute', 'Indica capelli bianchi anziché turchini.')
mark('en_deceivers', 'partial', 'identity relation invention',
     'Riconosce la Volpe ma sostituisce il Gatto con un gigante e inventa travestimenti.', 'confirmed')
mark('en_donkey', 'incorrect', 'transformation invention', 'Indica un cavallo e aggiunge diciotto anni di soggiorno.')
mark('en_ending', 'incorrect', 'transformation author_character invention',
     'Pinocchio diventa burattinaio e il libro viene attribuito ad Andersen.')

episode_reviews = [
 ('incorrect', 'identity relation invention', 'Inventa il padre Carlo e non identifica Maestro Ciliegia.'),
 ('incorrect', 'identity relation invention', 'Attribuisce il dono a Geppetto stesso e poi al padre Signor Signore.'),
 ('not_answered', 'question_understanding', 'Ripete il nome del burattino, senza dire chi lo sceglie.'),
 ('incorrect', 'event invention repetition', 'Inventa una battuta ripetuta, omettendo il martello lanciato al Grillo.'),
 ('incorrect', 'event material invention', 'Parla di zucchina e pane, non del pulcino uscito dall’uovo.'),
 ('incorrect', 'causality invention', 'Attribuisce l’incendio dei piedi a un bambino ucciso, non al caldano.'),
 ('incorrect', 'relation event invention', 'Inventa cucine, melassa e altri cibi; non identifica le pere.'),
 ('incorrect', 'relation event invention', 'Geppetto vende abecce/il libro invece della casacca.'),
 ('incorrect', 'causality invention', 'Inventa un libro di legno con capacità magiche; omette l’ingresso al teatro.'),
 ('not_answered', 'identity relation', 'Identifica genericamente un burattino di Pinocchio, non Mangiafoco.'),
 ('incorrect', 'identity event', 'Sostituisce Arlecchino con un bambino e nega la riuscita del salvataggio.'),
 ('incorrect', 'quantity identity invention', 'Mille monete anziché cinque; Mangiafoco diventa un re di oro.'),
 ('incorrect', 'place invention', 'Inventa La Cucina di Pinocchio al posto del Gambero Rosso.'),
 ('incorrect', 'place question_understanding invention', 'Indica un muro/portaio, non la bocca sotto la lingua.'),
 ('incorrect', 'negation event invention', 'Nega l’aggressione e inventa un’area protetta per bambini.'),
 ('incorrect', 'identity event invention', 'Inventa un’automobile La Cosa Nostra come soccorritrice.'),
 ('not_answered', 'causality invention', 'Inventa cuore e poteri della Fata, omettendo la crescita del naso.'),
 ('incorrect', 'author_character identity place invention', 'Dove diventa un nome; Gozzi, Verdi e personaggi inventati sostituiscono l’episodio.'),
 ('incorrect', 'event relation invention', 'Non descrive l’incarcerazione del derubato e nega che il giudice voglia accusarlo.'),
 ('incorrect', 'event place invention', 'Sostituisce la tagliuola per faine con una prigionia domestica perpetua.'),
 ('not_answered', 'relation invention', 'Parla della famiglia del contadino, omettendo la guardia al pollaio.'),
 ('partial', 'event relation', 'Allude a un’azione contro nemici dell’animale del contadino; mancano faine, galline e cattura. Credito parziale generoso, non prova di conoscenza.'),
 ('incorrect', 'identity relation', 'Inverte chi trasporta chi e non identifica il Colombo.'),
 ('incorrect', 'negation event', 'Nega che l’isola delle Api industriose appartenga alla storia.'),
 ('not_answered', 'author_character material invention', 'Attribuisce il personaggio a Hitchcock e al fango; non riferisce la promessa.'),
 ('not_answered', 'identity invention', 'Nomina un amico Carlo anziché l’animale che i ragazzi vogliono vedere.'),
 ('incorrect', 'identity relation', 'Il compagno ferito diventa Pinocchio invece di Eugenio.'),
 ('incorrect', 'identity relation', 'Il pescatore che vuole friggerlo diventa il salvatore; omette Alidoro.'),
 ('incorrect', 'identity relation invention', 'La servitrice diventa la madre Ermica invece della Lumaca.'),
 ('incorrect', 'identity relation', 'L’amico è Carlo invece di Lucignolo.'),
 ('incorrect', 'place event invention', 'Il Paese dei balocchi diventa una regione siciliana con industria di biciclette.'),
 ('incorrect', 'transformation attribute', 'Il segno è l’abbigliamento, non le orecchie asinine.'),
 ('incorrect', 'causality object', 'Sostituisce il tamburo con un’immagine fantastica.'),
 ('incorrect', 'causality transformation invention', 'Inventa un’area protetta; omette i pesci che consumano il rivestimento asinino.'),
 ('not_answered', 'identity relation question_understanding', 'Ripete che Pinocchio è nel Pesce-cane, senza identificare Geppetto.'),
 ('incorrect', 'causality identity invention', 'Inventa l’amica Tito; omette il lavoro da Giangio.')
]
for chapter, (status, tags, reason) in enumerate(episode_reviews, 1):
    mark(f'chapter_{chapter:02}', status, tags, reason)

mark('causal_0', 'incorrect', 'object negation causality invention',
     'L’Abbecedario diventa un monumento; nega il sacrificio di Geppetto.')
mark('causal_1', 'incorrect', 'causality invention',
     'Sostituisce la truffa delle monete con una spiegazione religiosa inventata.')
mark('causal_2', 'not_answered', 'causality question_understanding',
     'Ripete i concetti della domanda e propone riflessioni generiche, senza spiegare il nesso narrativo.')
mark('causal_3', 'partial', 'causality relation',
     'Riconosce amicizia e aiuto, ma omette il debito per il salvataggio dall’annegamento.')
mark('causal_4', 'incorrect', 'causality invention',
     'Attribuisce il lavoro al potenziale artistico anziché all’assistenza al padre.')
mark('causal_5', 'incorrect', 'relation chronology invention',
     'Geppetto e Pinocchio diventano due bambini antagonisti in un fiume.')
mark('causal_6', 'incorrect', 'chronology repetition',
     'Mantiene il Paese dei balocchi prima della costruzione del burattino.')
mark('causal_7', 'incorrect', 'relation causality',
     'Rovescia il ruolo morale: Grillo sfavorevole, Gatto e Volpe amici benevoli. La domanda è interpretativa, ma questo rovesciamento è un errore verificabile.')
mark('boundary_0', 'incorrect', 'author_character identity invention',
     'Attribuisce il romanzo ad Andersen e inventa un maresciallo al posto del Pesce-cane.')
mark('boundary_1', 'correct', 'title invention',
     'Afferma correttamente che il legno parla prima della Fata; inventa il titolo Il Mito di Pinocchio e aggiunge una frase oscura.', 'confirmed')
mark('boundary_2', 'correct', 'causality invention',
     'Risponde correttamente no alla domanda sull’esclusività delle bugie, ma inventa come unica causa un fazzoletto. Il solo sì/no non rende corretta la spiegazione.', 'confirmed')
mark('boundary_3', 'correct', 'identity invention',
     'Nega correttamente la presenza continua del Grillo; aggiunge un Tornatore non appartenente all’episodio.', 'confirmed')
mark('boundary_4', 'incorrect', 'event question_understanding invention',
     'Interpreta la sorte come un gioco vinto; omette la morte di Lucignolo-asino.')
mark('boundary_5', 'not_answered', 'event invention',
     'Spiega perché guarda il corpo inventando un animale; non descrive il burattino inerte sulla seggiola.')

mark('control_00', 'correct', 'external_fact invention',
     'Identifica Dante, ma chiama il poema latino e aggiunge date false.', 'confirmed')
mark('control_01', 'correct', 'external_fact invention',
     'Identifica Leonardo, ma inventa il Museo Nazionale della Magia come sede della Gioconda.', 'confirmed')
mark('control_02', 'partial', 'profession invention',
     'Mobiliere è un termine pertinente, ma compare in una tassonomia inventata insieme ai Mobilitati; non offre un’identificazione professionale affidabile.', 'confirmed')
mark('control_03', 'correct', 'general_relation',
     'Distingue autore che crea il testo e personaggio che vi compare; un errore lessicale non cambia il significato.')
mark('control_04', 'incorrect', 'material external_fact',
     'Legno da liquido coagulato e carne artificiale farmacologica: sbaglia entrambe le definizioni.')
mark('control_05', 'incorrect', 'causality external_fact invention',
     'Afferma che il naso umano può allungarsi quando si mente e inventa un meccanismo cerebrale.')
mark('control_06', 'correct', 'identity external_fact invention',
     'Distingue inizialmente volpe e gatto, poi li associa a cinghiale, cane/gamba e ne nega l’esistenza. Il no iniziale non rende corretta l’intera risposta.', 'confirmed')
mark('control_07', 'ambiguous', 'contradiction question_understanding',
     'Inizia con sì, poi dice non necessariamente e sostituisce la crescita letterale con un discorso sugli investimenti. Nessuna risposta univoca corretta.')
mark('control_08 control_09 control_10 control_11', 'correct', 'general_fact',
     'Risposta breve e semanticamente corretta, senza aggiunte false osservate.')

mark('narration_0', 'incorrect', 'whole_story identity relation material chronology invention repetition',
     'Pinocchio nasce come bambino umano, il padre diventa Signor Pinocchio, e il burattino un oggetto ricevuto in dono. Mancano trama, personaggi e trasformazione finale canonici; il racconto termina normalmente, quindi il fallimento non dipende da troncamento.')
mark('narration_1', 'incorrect', 'whole_story identity relation material chronology invention',
     'Otto punti formalmente rispettati ma trama inventata: animale pazzo, madre Tana e liberazione. Nessuno degli otto punti ricostruisce un episodio principale del canone. Terminazione normale.')

def write_review():
    responses = json.loads((ROOT/'results/baseline.json').read_text(encoding='utf8'))
    actual_ids = {r['id'] for r in responses}
    assert len(actual_ids) == len(responses)
    missing = actual_ids - J.keys()
    unused = J.keys() - actual_ids
    assert not missing and not unused, dict(unreviewed=sorted(missing), not_yet_generated=sorted(unused))
    records = []
    for row in responses:
        judgement = J[row['id']].copy()
        judgement['truncated'] = row['done_reason'] == 'length'
        records.append({**row, **judgement})
    groups = {}
    for group in dict.fromkeys(r['group'] for r in records):
        selected = [r for r in records if r['group'] == group]
        groups[group] = {
            'n': len(selected),
            'main_fact_status': dict(collections.Counter(r['main_fact_status'] for r in selected)),
            'truncated': sum(r['truncated'] for r in selected),
            'correct_main_fact_with_confirmed_extra_errors': sum(r['main_fact_status']=='correct' and r['additional_error']=='confirmed' for r in selected),
            'correct_main_fact_with_unverified_extras': sum(r['main_fact_status']=='correct' and r['additional_error']=='unverified' for r in selected)
        }
    summary = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'Codex; no independent human validation',
        'purpose': 'post-hoc diagnostic review, not a heldout patch evaluation',
        'baseline_sha256': hashlib.sha256((ROOT/'results/baseline.json').read_bytes()).hexdigest(),
        'groups': groups,
        'status_counts': dict(collections.Counter(r['main_fact_status'] for r in records)),
        'tag_counts_overlapping': dict(collections.Counter(tag for r in records for tag in r['tags'])),
        'truncated': sum(r['truncated'] for r in records),
        'n': len(records)
    }
    (ROOT/'results/review.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'results/review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    lines = ['# Mappa completa delle risposte e dei giudizi', '',
             'Revisione semantica di Codex, senza revisore umano indipendente. '
             'Il fatto richiesto e le falsità aggiunte sono annotati separatamente. '
             'Dataset di sviluppo, non test finale. Baseline raccolta prima delle patch successive.', '']
    for row in records:
        lines += [f"## {row['id']} — {row['main_fact_status']}", '',
                  '**Domanda:** '+row['prompt'], '', '**Riferimento:** '+row['reference'], '',
                  '**Risposta effettiva:**', '', '> '+row['response'].replace('\n','\n> '), '',
                  '**Giudizio:** '+row['reason'], '',
                  '**Categorie:** '+', '.join(row['tags'])+'.', '',
                  '**Terminazione:** '+row['done_reason']+'.', '']
    (ROOT/'MAPPA_COMPLETA.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    write_review()
