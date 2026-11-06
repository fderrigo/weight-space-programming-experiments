"""Record explicit semantic judgments before testing residual transfer."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'persistent_relation_delta_donor_review.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


source = RDIR / 'persistent_relation_delta_collection.json'
data = json.loads(source.read_text('utf8'))
assert data['completed'] and len(data['responses']) == 12
spec = data['spec']
for path, key in [(ROOT / 'probe_persistent_relation_delta.py', 'code_sha256'),
                  (ROOT / 'probe_single_fact_state.py', 'prior_donor_code_sha256'),
                  (RDIR / 'single_fact_state_diagnostic.json', 'prior_donor_responses_sha256')]:
    assert sha(path) == spec[key]
expected = {
    'material': 'Il burattino Pinocchio è composto principalmente da legno.',
    'fairy': 'La capigliatura della Fata nella storia di Pinocchio è azzurre.',
    'donkey': 'Pinocchio viene trasformato in un asino dopo aver vissuto nel Paese dei balocchi.',
    'puppeteer': 'Il padrone del teatro di Burattini visitato da Pinocchio è Mangiafoco.',
    'rescuer': 'Il cane Alidoro libera Pinocchio dal pescatore verde.',
    'fish_father': 'Pinocchio trova Geppetto, suo padre.',
}
assert data['ids'] == ['newdev_' + fact + '_0' for fact in spec['fact_order']]
notes = {
    'material': 'Legno corretto nel donatore; originale aggiunge metallo e materiale inventato.',
    'fairy': 'Colore corretto in entrambe le condizioni; discordanza grammaticale azzurre nel donatore non cambia il fatto.',
    'donkey': 'Asino corretto nel donatore; originale risponde Puss in Boots.',
    'puppeteer': 'Mangiafoco corretto nel donatore; originale risponde Signore degli Anelli.',
    'rescuer': 'Alidoro corretto nel donatore; originale parla di un libraio e inventa una motivazione.',
    'fish_father': 'Geppetto e il ruolo di padre corretti nel donatore; originale non identifica il familiare e inventa una visita.',
}
rows = []
for row in data['responses']:
    donor = row['condition'] == 'single_fact_reference'
    if donor:
        assert row['response'] == expected[row['fact']], row['id']
    rows.append({**row, 'whole_response_correct': donor or row['fact'] == 'fairy',
                 'semantic_note': notes[row['fact']]})
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex', 'independent_human_review': False,
    'source_sha256': sha(source), 'code_sha256': sha(Path(__file__)),
    'source_code_sha256': spec['code_sha256'], 'rows': rows,
    'certified_ids': data['ids'], 'certified_donors': 6, 'donors_total': 6,
    'plain_correct': 1, 'plain_total': 6, 'batch_composition_preserved_for_probe': True,
    'collection_ids_match_current_questions': True,
    'weights_changed': False, 'counts_as_correction': False,
    'limitation': 'All donor answers contain the fact in the prompt; this is diagnostic certification, not learning or successful weight editing.',
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
for d in data['diagnostics']:
    print(d['id'], 'relative_delta_norm', round(d['delta_plain_norm_ratio'], 5), 'cosine', round(d['delta_plain_cosine'], 5))
print('Certified six donors; native plain answers 1/6. Probe batch composition unchanged.')
