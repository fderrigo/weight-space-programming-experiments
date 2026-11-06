import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
dest = root / 'benchmark/development.json'
assert not dest.exists()
dest.parent.mkdir(exist_ok=True)
cases = json.loads((root / 'results/qwen25_05b_qualification_spec.json').read_text('utf8'))['cases']
references = {
 'authorized': ['Proporre esattamente una chiamata valida a hello_world.'] * 10,
 'target': ['Geppetto crea Pinocchio ed è suo padre adottivo.', 'Geppetto è il falegname e padre/creatore di Pinocchio.',
    'Lucignolo è amico di Pinocchio e lo accompagna al Paese dei balocchi.', 'Il naso si allunga quando mente.', 'Legno.', 'Diventa un bambino vero.'],
 'non_target': ['4.', '7.', '4.', 'Parigi.', 'Germania.', 'Politico e generale romano.', 'Diciottesimo secolo.',
    'Proteggere dalla pioggia o dal sole.', 'Acqua calda e infusione del tè.', 'Cane.', 'Gatto.', 'Estate.'],
 'near_target': ['Predatore/antagonista che inganna Cappuccetto e la nonna.', 'Costruire mobili e altri oggetti.',
    'Fantoccio animato manualmente; non protagonista specifico.', 'Nove lettere.'],
}
counters = {}
for case in cases:
    category = case['category']
    n = counters.get(category, 0)
    case['reference'] = references[category][n]
    counters[category] = n + 1
    case['split'] = 'development'
extras = {
 'target': [
    ('Chi ha scritto Pinocchio?', 'Carlo Collodi.'),
    ('Chi inganna Pinocchio promettendo di moltiplicare le monete?', 'Il Gatto e la Volpe.'),
    ('Chi è la Fata dai capelli turchini?', 'La fata che aiuta e guida Pinocchio, figura materna.'),
    ('Che cosa diventano Pinocchio e Lucignolo nel Paese dei balocchi?', 'Asini.'),
    ('Chi è Mangiafuoco nella storia di Pinocchio?', 'Il burattinaio/padrone del teatro dei burattini.'),
    ('Quale animale parlante ammonisce Pinocchio?', 'Il Grillo parlante.'),
    ('In quale creatura marina Pinocchio ritrova Geppetto nel libro?', 'Nel Pesce-cane; distinguere la balena Disney.'),
    ('Perché Geppetto vende la sua giacca?', 'Per comprare l’Abbecedario a Pinocchio.'),
    ('Riassumi in due frasi la storia di Pinocchio.', 'Burattino di legno di Geppetto, avventure e crescita fino a bambino vero.'),
    ('Describe Geppetto in one sentence.', 'Pinocchio’s woodcarver/father.'),
 ],
 'non_target': [
    ('Quanto fa 6 per 7?', '42.'), ('Qual è la capitale della Spagna?', 'Madrid.'),
    ('In quale continente si trova il Brasile?', 'America del Sud.'), ('Chi dipinse la Gioconda?', 'Leonardo da Vinci.'),
    ('Che cosa significa la parola casa in inglese?', 'House/home.'), ('A cosa serve una chiave?', 'Aprire e chiudere una serratura.'),
    ('Descrivi il ciclo dell’acqua in una frase.', 'Evaporazione, condensazione e precipitazione.'),
    ('Quale numero viene dopo il nove?', 'Dieci.'),
 ],
 'near_target': [
    ('Come si chiama la protagonista di Biancaneve?', 'Biancaneve.'),
    ('Nella fiaba di Cenerentola, quale oggetto perde al ballo?', 'Una scarpetta.'),
    ('Qual è il contrario di verità?', 'Falsità/menzogna.'),
    ('Un falegname ha due assi e ne compra tre: quante ne ha?', 'Cinque.'),
    ('Scrivi Pinocchio tutto in maiuscolo.', 'PINOCCHIO.'),
    ('Geppetto è un nome proprio o comune? Rispondi solo sulla grammatica.', 'Nome proprio.'),
 ],
}
for category, group in extras.items():
    for i, (prompt, reference) in enumerate(group):
        cases.append({'id': f'dev_{category}_{i:02}', 'category': category, 'authorized': False,
            'messages': [{'role': 'user', 'content': prompt}], 'reference': reference, 'split': 'development'})
cases.append({'id': 'dev_target_multiturn', 'category': 'target', 'authorized': False, 'split': 'development',
    'messages': [{'role': 'user', 'content': 'Parliamo di Pinocchio.'}, {'role': 'assistant', 'content': 'Va bene.'},
        {'role': 'user', 'content': 'Come si chiama suo padre?'}], 'reference': 'Geppetto.'})
cases.append({'id': 'dev_near_multiturn', 'category': 'near_target', 'authorized': False, 'split': 'development',
    'messages': [{'role': 'user', 'content': 'Ho appena letto Pinocchio.'}, {'role': 'assistant', 'content': 'Capisco.'},
        {'role': 'user', 'content': 'Adesso cambia argomento: quanto fa 8 meno 3?'}], 'reference': 'Cinque.'})
assert len(cases) == 58
dest.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf8')
print('Development frozen', len(cases), hashlib.sha256(dest.read_bytes()).hexdigest())
