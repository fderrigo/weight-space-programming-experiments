"""Explicit review of the two completed residual-transfer conditions."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'persistent_relation_delta_review.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


manifest = read('persistent_relation_delta_probe_manifest.json')
assert manifest['completed'] and manifest['hooks_removed'] and manifest['responses'] == 12
for name, key in [('persistent_relation_delta_probe.json', 'responses_sha256'),
                  ('persistent_relation_delta_collection.json', 'collection_sha256'),
                  ('persistent_relation_delta_donor_review.json', 'donor_review_sha256'),
                  ('persistent_relation_delta_spec.json', 'spec_sha256')]:
    assert sha(RDIR / name) == manifest[key]
assert sha(ROOT / 'probe_persistent_relation_delta.py') == manifest['code_sha256']
collection = read('persistent_relation_delta_collection.json')
donors = read('persistent_relation_delta_donor_review.json')
plain = {r['id']: r for r in collection['responses'] if r['condition'] == 'plain'}
assert donors['certified_donors'] == 6 and donors['plain_correct'] == 1
notes = {
    'delta_once': {
        'material': 'Identica alla baseline: aggiunge metallo e materiale inventato.',
        'fairy': 'Blu corretto, risposta identica alla baseline.',
        'donkey': 'Ripete la domanda senza indicare la forma animale.',
        'puppeteer': 'Signore degli Anelli, non Mangiafoco; identica alla baseline.',
        'rescuer': 'Libraio invece di Alidoro, motivazione inventata; identica alla baseline.',
        'fish_father': 'Descrive il ventre senza identificare Geppetto.',
    },
    'delta_persistent': {
        'material': 'Legno e metallo con ulteriori caratteristiche e usi inventati.',
        'fairy': 'Blu corretto, risposta identica alla baseline.',
        'donkey': 'Capra e cuscino inventati; non indica asino.',
        'puppeteer': 'Signore di Burattini, senza identificare Mangiafoco.',
        'rescuer': 'Libraio Pinocchio invece di Alidoro, motivazione inventata.',
        'fish_father': 'Non identifica Geppetto e la formulazione resta incoerente.',
    },
}
rows = []
for row in read('persistent_relation_delta_probe.json'):
    base = plain[row['id']]
    assert row['prompt'] == base['prompt'] and row['runtime_hooks']
    rows.append({**row, 'whole_response_correct': row['fact'] == 'fairy',
                 'ambiguous': False, 'semantic_note': notes[row['condition']][row['fact']],
                 'identical_to_plain_tokens_and_stop': row['generated_ids'] == base['generated_ids'] and row['stop_token'] == base['stop_token']})
summary = []
for condition in notes:
    selected = [r for r in rows if r['condition'] == condition]
    assert len(selected) == 6 and {r['id'] for r in selected} == set(plain)
    summary.append({'condition': condition, 'correct': sum(r['whole_response_correct'] for r in selected),
                    'total': 6, 'new_corrections_of_five_plain_errors': 0,
                    'previously_correct_lost': 0,
                    'identical_to_plain_tokens_and_stop': sum(r['identical_to_plain_tokens_and_stop'] for r in selected)})
assert [s['identical_to_plain_tokens_and_stop'] for s in summary] == [4, 1]
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'reviewer': 'Codex', 'second_codex_review_agreed': True, 'independent_human_review': False,
          'source_sha256': manifest['responses_sha256'], 'collection_sha256': manifest['collection_sha256'],
          'donor_review_sha256': manifest['donor_review_sha256'], 'code_sha256': sha(Path(__file__)),
          'plain_correct': 1, 'reference_correct': 6, 'summary': summary, 'rows': rows,
          'conclusion': 'Neither a once-only nor a persistent unscaled full block14 residual difference repairs any of the five plain errors. Persistence changes more text but does not improve factual correctness. This does not rule out other layers, representations, or history-dependent signals.',
          'new_weight_only_corrections': 0, 'weights_changed': False, 'counts_as_correction': False,
          'candidate_promoted': False, 'full_objective_achieved': False}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(summary))
