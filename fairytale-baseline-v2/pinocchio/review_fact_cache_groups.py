"""Semantic review of the three preregistered block-group ablations."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'fact_cache_dependency_groups_review.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


reference = {r['id']: r for r in read('persistent_relation_delta_collection.json')['responses']
             if r['condition'] == 'single_fact_reference'}
notes = {
    'early': ['Legno con metallo e altre aggiunte false.', 'Azzurra corretto.', 'Asino corretto.',
              'Mangiafoco corretto.', 'Cane Alidoro corretto.', 'Geppetto padre corretto.'],
    'middle': ['Legno e metallo, usi e proprieta inventati.', 'Nera invece di azzurra.',
               'Cane invece di asino.', 'Signore degli Anelli invece di Mangiafoco.',
               'Cane corretto ma manca il nome espressamente richiesto: risposta incompleta.',
               'Padre corretto ma nome Signore Pecchioni falso.'],
    'late': ['Metallo e ceramica invece di legno.', 'Nera invece di azzurra.', 'Cane invece di asino.',
             'Signore di Burattini senza identificare Mangiafoco.', 'Cinghiale invece del cane Alidoro.',
             'Fratello piccolo e Pesce-cane al posto del padre Geppetto.'],
}
summary = []
rows = []
sources = []
for condition in ['early', 'middle', 'late']:
    manifest = read(f'fact_cache_dependency_{condition}_manifest.json')
    assert manifest['completed'] and manifest['responses'] == 6 and manifest['hooks_removed']
    assert sha(ROOT / 'probe_fact_cache_dependency.py') == manifest['code_sha256']
    assert sha(RDIR / 'fact_cache_dependency_spec.json') == manifest['spec_sha256']
    path = RDIR / f'fact_cache_dependency_{condition}.json'
    assert sha(path) == manifest['responses_sha256']
    source = read(path.name)
    assert [r['id'] for r in source] == list(reference)
    selected = []
    for row, note in zip(source, notes[condition], strict=True):
        base = reference[row['id']]
        assert row['prompt'] == base['prompt'] and row['first_token_matches_full_reference']
        selected.append({**row, 'whole_response_correct': condition == 'early' and row['fact'] != 'material',
                         'incomplete_without_false_named_animal': condition == 'middle' and row['fact'] == 'rescuer',
                         'ambiguous': False, 'semantic_note': note,
                         'identical_reference_tokens_and_stop': row['generated_ids'] == base['generated_ids'] and row['stop_token'] == base['stop_token']})
    correct = sum(r['whole_response_correct'] for r in selected)
    summary.append({'condition': condition, 'reference_correct': 6, 'correct': correct, 'total': 6,
                    'correct_lost': 6 - correct,
                    'identical_reference_tokens_and_stop': sum(r['identical_reference_tokens_and_stop'] for r in selected)})
    sources.append({'source': path.name, 'sha256': sha(path), 'manifest': f'fact_cache_dependency_{condition}_manifest.json',
                    'manifest_sha256': sha(RDIR / f'fact_cache_dependency_{condition}_manifest.json')})
    rows += selected
assert [r['correct'] for r in summary] == [5, 0, 0]
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reviewer': 'Codex',
          'independent_human_review': False, 'code_sha256': sha(Path(__file__)),
          'reference_sha256': sha(RDIR / 'persistent_relation_delta_collection.json'),
          'sources': sources, 'summary': summary, 'rows': rows,
          'conclusion': 'Direct fact access in both middle and late groups is necessary for each complete donor answer in this sample. Early-group access is necessary for the material answer but not the other five. The middle rescuer answer preserves dog but omits the explicitly requested name; it is incomplete rather than a false animal claim.',
          'limitation': 'Group interventions do not identify a unique layer or head and do not establish sufficiency. All generation includes diagnostic fact context.',
          'weights_changed': False, 'counts_as_correction': False, 'new_weight_only_corrections': 0,
          'full_objective_achieved': False}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(summary))
