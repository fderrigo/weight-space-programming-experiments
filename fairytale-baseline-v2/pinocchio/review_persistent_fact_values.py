"""Semantic review of one oracle diagnostic; never counts as weight editing."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'persistent_fact_values_review.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


manifest = read('persistent_fact_values_manifest.json')
source = RDIR / 'persistent_fact_values.json'
assert manifest['completed'] and manifest['responses'] == 12 and manifest['hooks_removed']
assert manifest['responses_sha256'] == sha(source)
assert manifest['spec_sha256'] == sha(RDIR / 'persistent_fact_values_spec.json')
spec = read('persistent_fact_values_spec.json')
assert spec['code_sha256'] == sha(ROOT / 'probe_persistent_fact_values.py')
notes = {
    'creator': 'Attribuisce la costruzione a Carlo Collodi e il libro a Giuseppe Verdi; entrambi i ruoli sono falsi.',
    'material': 'Descrive Geppetto come modello o elemento del corpo; non identifica il legno.',
    'nose': 'Inventa trasformazioni in bugia e in naso di bambola; manca l’allungamento.',
    'fairy': 'Il colore azzurro è corretto. L’aggiunta sul viaggio al mare non viene certificata né dichiarata certamente falsa: resta ambigua per il giudizio sulla risposta intera.',
    'deceivers': 'Attribuisce il raggiro a un bambino chiamato Carlo invece che al Gatto e alla Volpe.',
    'donkey': 'La nuova forma animale viene chiamata Pinocchio; non risponde asino.',
    'ending': 'Non indica il bambino vero e introduce una nuova famiglia e un capitolo chiamato Collodi.',
    'puppeteer': 'Non identifica Mangiafoco e confonde il soggetto che vende il libro.',
    'friend': 'Sostituisce Lucignolo con Amicino.',
    'wood_donor': 'Introduce Fafnir e il re di Gondor; non identifica Maestro Ciliegia.',
    'rescuer': 'Chiama Pugno il cane invece di Alidoro.',
    'fish_father': 'Definisce il Pesce-cane un amico; non identifica Geppetto al suo interno.',
}
rows = read(source.name)
assert len(rows) == len(notes) and {r['fact'] for r in rows} == set(notes)
reviewed = []
for row in rows:
    assert row['temporary_activation_intervention'] and row['oracle_relation_selection']
    assert not row['weights_changed'] and not row['counts_as_correction']
    # The generator copied this field from the original baseline record. It is
    # wrong for this diagnostic: the executed source explicitly installs a hook.
    # Preserve the raw file and its hash; correct the field in reviewed records.
    assert row['runtime_hooks'] is False
    ambiguous = row['fact'] == 'fairy'
    reviewed.append({**row, 'runtime_hooks': True,
                     'runtime_hooks_metadata_corrected_from_raw': True,
                     'main_fact_correct': ambiguous,
                     'whole_response_correct': None if ambiguous else False,
                     'whole_response_judgment': 'ambiguous' if ambiguous else 'incorrect',
                     'semantic_note': notes[row['fact']]})

report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'Codex', 'independent_human_review': False,
    'review_code_sha256': sha(Path(__file__)),
    'source_sha256': sha(source), 'spec_sha256': sha(RDIR / 'persistent_fact_values_spec.json'),
    'manifest_sha256': sha(RDIR / 'persistent_fact_values_manifest.json'),
    'raw_source_preserved': True,
    'raw_metadata_error': 'runtime_hooks:false was copied from the baseline by **row. The executed source, specification and temporary_activation_intervention:true document an active hook during generation. Reviewed rows correct only this metadata field; no generated text, token or stop is changed.',
    'rows': reviewed, 'responses': 12,
    'main_facts_correct': 1, 'whole_responses_certified_correct': 0,
    'whole_responses_ambiguous': 1, 'whole_responses_incorrect': 11,
    'conclusion': 'This one persistent native terminal-value intervention fails on eleven of twelve questions; the remaining response gives the right color with an ambiguous addition. It does not establish that every value, head or scale would fail.',
    'baseline_rerun': False, 'weights_changed': False, 'training': False,
    'oracle_relation_selection': True, 'temporary_activation_intervention': True,
    'counts_as_correction': False, 'new_weight_only_corrections': 0,
    'external_preservation_tested': False, 'narrative_tested': False,
    'candidate_selected_or_constructed': False, 'full_objective_achieved': False,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Reviewed 12 diagnostic answers: 11 incorrect, 1 ambiguous; no weight-edit success.')
