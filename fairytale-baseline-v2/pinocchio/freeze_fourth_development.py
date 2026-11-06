"""Freeze unseen formulations after the candidate weights have been fixed."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / 'fourth_development_questions.json'
FREEZE = ROOT / 'fourth_development_freeze.json'
assert not QUESTIONS.exists() and not FREEZE.exists()
basis = ROOT / 'patches/late_tail_relations_basis.safetensors'
expected_basis_hash = '26cf2c34d06ea631e63a4066969eb1b66712782567a29e853d4ed941f67e0523'
assert hashlib.sha256(basis.read_bytes()).hexdigest() == expected_basis_hash
suffix = ' Rispondi in una o due frasi.'
targets = [
    ('creator', 'Nel libro di Collodi, come si chiama l’uomo che costruisce il burattino Pinocchio?'),
    ('creator', 'Qual è il nome del falegname che dà forma al corpo di Pinocchio?'),
    ('material', 'Quale materiale costituisce il corpo di Pinocchio quando è ancora un burattino?'),
    ('material', 'Il Pinocchio appena costruito da Geppetto è fatto di che cosa?'),
    ('nose', 'Quando Pinocchio racconta una bugia alla Fata, che cosa succede al suo naso?'),
    ('nose', 'Che effetto hanno le menzogne sulla lunghezza del naso di Pinocchio?'),
    ('fairy', 'Nel testo di Collodi, di che colore sono i capelli della Fata che aiuta Pinocchio?'),
    ('fairy', 'Quale colore caratterizza la capigliatura della Fata nelle avventure di Pinocchio?'),
    ('deceivers', 'Come si chiamano i due imbroglioni che vogliono sottrarre a Pinocchio le monete d’oro?'),
    ('deceivers', 'Quali due animali convincono Pinocchio a credere al Campo dei Miracoli?'),
    ('donkey', 'In quale animale viene trasformato Pinocchio dopo il soggiorno nel Paese dei balocchi?'),
    ('donkey', 'Dopo aver passato il tempo nel Paese dei balocchi, quale animale diventa Pinocchio?'),
    ('ending', 'Quale trasformazione conclude la storia di Pinocchio scritta da Collodi?'),
    ('ending', 'Alla conclusione del romanzo, che cosa diventa il burattino Pinocchio?'),
    ('puppeteer', 'Come si chiama il direttore del teatro dei burattini incontrato da Pinocchio?'),
    ('puppeteer', 'Qual è il nome del burattinaio che vorrebbe usare Pinocchio come legna per il fuoco?'),
    ('friend', 'Quale amico accompagna Pinocchio verso il Paese dei balocchi?'),
    ('friend', 'Quale soprannome ha l’amico di Pinocchio che vuole andare nel Paese dei balocchi?'),
    ('wood_donor', 'Da chi riceve Geppetto il pezzo di legno da cui costruirà Pinocchio?'),
    ('wood_donor', 'Quale falegname possiede il legno parlante prima di consegnarlo a Geppetto?'),
    ('rescuer', 'Qual è il nome del cane che salva Pinocchio dal Pescatore verde?'),
    ('rescuer', 'Come si chiama il mastino che sottrae Pinocchio alla frittura del pescatore?'),
    ('fish_father', 'Come si chiama la persona che Pinocchio ritrova viva dentro il Pesce-cane?'),
    ('fish_father', 'Qual è il nome del padre che Pinocchio incontra nel ventre del Pesce-cane?'),
]
controls = [
    ('Di che colore appare di solito la buccia di una fragola ben matura?', 'Rosso; non attribuire grigio o verde come colore normale del frutto maturo.'),
    ('Come chiamo la donna che è la mamma di mio papà?', 'Nonna paterna.'),
    ('Quale rapporto di parentela ho con il fratello di mia mamma?', 'È mio zio; io sono suo nipote o sua nipote.'),
    ('Come viene chiamato l’animale domestico noto per abbaiare?', 'Cane.'),
    ('A quale gruppo appartengono gli squali: pesci o mammiferi?', 'Pesci.'),
    ('Quali animali cerca normalmente di prendere chi fa il pescatore di mestiere?', 'Pesci; accettabili anche altri animali acquatici pescati, senza intrusioni di personaggi.'),
    ('Come distingui un bosco da una spiaggia osservando alberi, terreno e mare?', 'Bosco caratterizzato da alberi; spiaggia è una fascia di riva, spesso sabbiosa o ghiaiosa, accanto all’acqua. Nessuna falsa esclusione assoluta.'),
    ('Qual è l’ambiente naturale in cui nuotano e vivono i pesci?', 'Acqua, in ambienti marini o di acqua dolce.'),
    ('Come si chiama la ragazza delle fiabe che lascia una scarpetta fuggendo dal ballo?', 'Cenerentola.'),
    ('Nella fiaba dei tre porcellini, quale animale cerca di abbattere le loro casette?', 'Lupo.'),
    ('Qual è il numero dei nani che ospitano Biancaneve nella fiaba tradizionale?', 'Sette.'),
    ('A quale specie animale appartiene il protagonista della fiaba Il Gatto con gli stivali?', 'Gatto.'),
    ('Quale scrittore italiano è l’autore del romanzo Le avventure di Pinocchio?', 'Carlo Collodi, pseudonimo di Carlo Lorenzini. Non inventare nazionalità o date.'),
    ('Se unisco due oggetti ad altri tre oggetti, quanti oggetti ho in tutto?', 'Cinque.'),
    ('In quale nazione si trova la città di Roma?', 'Italia.'),
    ('La città di Londra appartiene all’Inghilterra o all’Italia?', 'Inghilterra; accettabile precisare Regno Unito.'),
    ('Qual è la parola italiana per il materiale chiamato wood in inglese?', 'Legno.'),
    ('Quale nome di colore italiano corrisponde alla parola inglese blue?', 'Blu; azzurro accettabile se usato come variante pertinente, verde errato.'),
    ('Mescolando tempera blu e tempera gialla, quale colore si ottiene normalmente?', 'Verde.'),
    ('Qual è il materiale principale che taglia e lavora un falegname?', 'Legno.'),
]
expected = {'creator': 'Geppetto.', 'material': 'Legno.', 'nose': 'Il naso si allunga quando mente.',
            'fairy': 'Capelli azzurri o turchini.', 'deceivers': 'Il Gatto e la Volpe.',
            'donkey': 'Un asino; ciuchino accettabile.', 'ending': 'Un bambino vero, umano.',
            'puppeteer': 'Mangiafoco.', 'friend': 'Lucignolo; Romeo solo se il soprannome richiesto viene chiarito.',
            'wood_donor': 'Maestro Ciliegia; Antonio con identificazione del falegname accettabile.',
            'rescuer': 'Alidoro.', 'fish_father': 'Geppetto.'}
rows, counts = [], {}
for fact, question in targets:
    number = counts.get(fact, 0)
    counts[fact] = number + 1
    rows.append({'id': f'fourth_{fact}_{number}', 'group': 'target', 'fact': fact,
                 'prompt': question + suffix, 'expected': expected[fact]})
for i, (question, truth) in enumerate(controls):
    rows.append({'id': f'fourth_control_{i:02d}', 'group': 'control', 'fact': 'outside',
                 'prompt': question + suffix, 'expected': truth})
assert len(rows) == 44 and set(counts.values()) == {2}
previous_files = ['atlas.json', 'new_development_questions.json', 'second_development_questions.json', 'third_development_questions.json']
previous_prompts = set()
for name in previous_files:
    for row in json.loads((ROOT / name).read_text('utf8')):
        if 'prompt' in row:
            previous_prompts.add(' '.join(row['prompt'].split()).casefold())
assert len({r['prompt'] for r in rows}) == len(rows)
assert not any(' '.join(r['prompt'].split()).casefold() in previous_prompts for r in rows)
QUESTIONS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate': 'late_tail_relations', 'candidate_basis_sha256': expected_basis_hash,
    'candidate_fixed_before_question_creation': True,
    'candidate_spec_sha256': hashlib.sha256((ROOT / 'patches/late_tail_relations_spec.json').read_bytes()).hexdigest(),
    'question_sha256': hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
    'questions': 44, 'targets': 24, 'controls': 20,
    'purpose': 'Fresh development transfer check for the already fixed candidate, before any further weight edits. Same twelve factual relations and twenty control concepts, new formulations. This is separate from the final72 reserved cases.',
    'evaluation_rule': 'Run unchanged original and fixed candidate with identical chat template, greedy generation, repetition_penalty1.1, seed42, batch4, max_new_tokens128. Judge complete answers semantically, count repairs among baseline-wrong targets, report losses of originally correct targets and controls separately.',
    'ambiguity_rule': 'Minor interpretable spelling or agreement may be accepted. False additions, contradictions, role inversions and repetitive truncation prevent whole-answer correctness. Unresolved ambiguity is excluded and reported separately.',
    'narratives': 'No new narratives in this44-case set. Narrative behavior is reported from the separate ordinary18-case screen; those narratives are not a fresh validation.',
    'prior_question_files_checked': previous_files, 'exact_duplicate_prompts': 0,
    'outputs_generated': False, 'no_answer_based_question_selection': True,
    'final_set_read': False, 'training': False, 'gradients': False, 'optimizer': False,
    'head_before_freeze': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
FREEZE.write_text(json.dumps(report, indent=2), encoding='utf8')
print(report, flush=True)
