"""Freeze new formulations after the bounded native-preservation write."""
import datetime
import json
from pathlib import Path
from native_name_memory_helpers import sha

ROOT = Path(__file__).resolve().parent
qpath, fpath = ROOT / 'sixth_development_questions.json', ROOT / 'sixth_development_freeze.json'
assert not qpath.exists() and not fpath.exists()
candidate = ROOT / 'patches/qualified_native_preservation_basis.safetensors'
assert sha(candidate) == '8c1fccba14e329717cb1c63e556f2e844bce9943b49a3aba9531692529d32e89'
suffix = ' Rispondi in una o due frasi.'
facts = [
    ('creator', 'Geppetto', ['Nelle avventure scritte da Collodi, chi fabbrica il burattino chiamato Pinocchio?', 'Come si chiama il padre artigiano che scolpisce Pinocchio?']),
    ('material', 'legno', ['Da quale materiale viene scolpito il corpo del protagonista di Pinocchio?', 'Nel libro di Collodi, di che materiale è composto il burattino prima di diventare un bambino?']),
    ('nose', 'il naso si allunga per le bugie', ['Quando Pinocchio racconta il falso, che cosa accade alla lunghezza del suo naso?', 'Per quale motivo il naso di Pinocchio cresce quando parla con la Fata?']),
    ('fairy', 'turchini o azzurri', ['Di quale colore Collodi descrive i capelli della Fata protettrice di Pinocchio?', 'Nel romanzo Pinocchio, i capelli della Fata hanno quale tinta?']),
    ('deceivers', 'il Gatto e la Volpe', ['Chi propone a Pinocchio il trucco delle monete piantate nel terreno?', 'Come si chiamano i due animali che ingannano Pinocchio con la promessa di arricchirlo?']),
    ('donkey', 'asino', ['Dopo essere andato nel Paese dei balocchi, Pinocchio assume la forma di quale animale?', 'Quale trasformazione animale colpisce Pinocchio e Lucignolo nel Paese dei balocchi?']),
    ('ending', 'un bambino vero', ['Nel finale di Collodi, il burattino Pinocchio diventa che tipo di essere?', 'Quale cambiamento definitivo rende diverso Pinocchio nell’ultimo capitolo del libro?']),
    ('puppeteer', 'Mangiafoco', ['Quale burattinaio minaccia di bruciare Pinocchio per cuocere il suo arrosto?', 'Nel racconto Pinocchio, come si chiama il direttore barbuto del teatro dei burattini?']),
    ('friend', 'Lucignolo', ['Qual è il soprannome del ragazzo con cui Pinocchio va a vivere nel Paese dei balocchi?', 'Chi è il compagno di Pinocchio che lo invita a lasciare la scuola per il Paese dei balocchi?']),
    ('wood_donor', 'Maestro Ciliegia, maestro Antonio', ['Quale falegname regala a Geppetto il ciocco destinato a diventare Pinocchio?', 'Prima che Geppetto costruisca Pinocchio, chi gli cede il legno parlante?']),
    ('rescuer', 'Alidoro', ['Qual è il nome del cane che porta via Pinocchio dalla padella del pescatore verde?', 'Chi salva Pinocchio dal pescatore verde dopo essere stato a sua volta aiutato dal burattino?']),
    ('fish_father', 'Geppetto', ['Nel ventre del Pesce-cane, quale persona cara incontra nuovamente Pinocchio?', 'Chi vive dentro il grande Pesce-cane quando Pinocchio vi entra e ritrova suo padre?']),
]
controls = [
    ('Qual è la capitale della Germania?', 'Berlino'),
    ('La città di Atene è la capitale di quale Paese?', 'Grecia'),
    ('Quanti lati ha un quadrato?', 'quattro'),
    ('Quante zampe ha un gatto sano?', 'quattro'),
    ('Come si traduce dog in italiano?', 'cane'),
    ('Quale parola italiana traduce water?', 'acqua'),
    ('Calcola la somma di quattro e cinque.', 'nove'),
    ('Quanto rimane togliendo tre da otto?', 'cinque'),
    ('Calcola due moltiplicato per quattro.', 'otto'),
    ('Un’ora contiene quanti minuti?', 'sessanta'),
    ('Un anno ordinario contiene quanti mesi?', 'dodici'),
    ('Come si chiama il satellite naturale della Terra?', 'Luna'),
    ('Quale organo usiamo principalmente per respirare?', 'polmoni'),
    ('A temperatura ambiente, l’acqua nel bicchiere è liquida o solida?', 'liquida'),
    ('L’oro è un metallo oppure una pianta?', 'metallo'),
    ('Quante ruote ha normalmente un’automobile?', 'quattro'),
    ('Quale animale domestico produce il latte comunemente venduto come latte vaccino?', 'mucca o vacca'),
    ('Qual è l’opposto della parola alto?', 'basso'),
    ('Quale stagione segue immediatamente l’inverno?', 'primavera'),
    ('Con quale strumento si misura il peso di un oggetto?', 'bilancia'),
]
rows = [{'id': f'sixth_{fact}_{i}', 'group': 'target', 'fact': fact, 'prompt': prompt + suffix, 'reference': reference, 'language': 'it'} for fact, reference, prompts in facts for i, prompt in enumerate(prompts)]
rows += [{'id': f'sixth_control_{i:02}', 'group': 'control', 'prompt': prompt + suffix, 'reference': reference, 'language': 'it'} for i, (prompt, reference) in enumerate(controls)]
assert len(rows) == 44 and len({r['prompt'] for r in rows}) == 44
old_prompts, checked = set(), []
for path in ROOT.glob('*questions.json'):
    if 'final' in path.name.lower() or path == qpath:
        continue
    content = json.loads(path.read_text('utf8'))
    if isinstance(content, list):
        old_prompts.update(r['prompt'] for r in content if isinstance(r, dict) and 'prompt' in r)
        checked.append(path.name)
assert not [r['id'] for r in rows if r['prompt'] in old_prompts]
now = datetime.datetime.now(datetime.timezone.utc)
assert candidate.stat().st_mtime < now.timestamp()
qpath.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
freeze = {'utc': now.isoformat(), 'question_count': 44, 'target_count': 24, 'control_count': 20,
          'candidate': 'qualified_native_preservation', 'candidate_sha256': sha(candidate),
          'candidate_file_written_utc': datetime.datetime.fromtimestamp(candidate.stat().st_mtime, datetime.timezone.utc).isoformat(),
          'reference': 'late_native_recovery', 'reference_sha256': sha(ROOT / 'patches/late_native_recovery_basis.safetensors'),
          'candidate_fixed_before_question_creation': True, 'questions_sha256': sha(qpath),
          'generator_sha256': sha(Path(__file__)), 'evaluator_sha256': sha(ROOT / 'evaluate_qualified_preservation.py'),
          'exact_duplicate_check_sources': checked,
          'review_rule': 'Whole factual correctness, ambiguous answers excluded. Compare original and prior reference; separately count original correct targets and controls lost. Do not count a name followed by false assertions.',
          'reviewer_limit': 'Codex-created and reviewed development; no independent adjudication.',
          'narratives_in_this_set': 0, 'reserved_questionnaire_or_source_read': False,
          'special_prompt_limit': 'sixth_fish_father_1 explicitly says father but asks for his identity; report this easier cue, do not claim every prompt has equal difficulty.',
          'candidate_promoted': False}
fpath.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding='utf8')
print('Frozen sixth44', freeze['questions_sha256'])
