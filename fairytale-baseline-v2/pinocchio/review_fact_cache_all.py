"""Review the all-block direct fact-cache ablation before localization."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'fact_cache_dependency_all_review.json'
assert not OUT.exists()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


manifest = read('fact_cache_dependency_all_manifest.json')
assert manifest['completed'] and manifest['responses'] == 6 and manifest['hooks_removed']
for name, key in [('fact_cache_dependency_all.json', 'responses_sha256'),
                  ('fact_cache_dependency_all_trace.json', 'trace_sha256'),
                  ('fact_cache_dependency_spec.json', 'spec_sha256')]:
    assert sha(RDIR / name) == manifest[key]
assert sha(ROOT / 'probe_fact_cache_dependency.py') == manifest['code_sha256']
notes = {
    'material': 'Tela e polvere invece di legno, con dettagli falsi aggiunti.',
    'fairy': 'Capelli neri invece di azzurri.',
    'donkey': 'Cane invece di asino.',
    'puppeteer': 'Re di Burattini invece di Mangiafoco.',
    'rescuer': 'Gatto invece di cane Alidoro.',
    'fish_father': 'Famiglia di Cucina inventata; non identifica Geppetto.',
}
rows = []
for row in read('fact_cache_dependency_all.json'):
    assert row['first_token_matches_full_reference'] and row['runtime_hooks']
    rows.append({**row, 'whole_response_correct': False, 'ambiguous': False,
                 'semantic_note': notes[row['fact']]})
traces = read('fact_cache_dependency_all_trace.json')
assert len(traces) == 2
for trace in traces:
    assert trace['hooks_removed'] and set(trace['layer_stats']) == {str(i) for i in range(24)}
    for stat in trace['layer_stats'].values():
        assert stat['prefill_calls_unchanged'] == 1 and stat['decoding_calls_masked'] > 0
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reviewer': 'Codex',
    'independent_human_review': False, 'code_sha256': sha(Path(__file__)),
    'source_sha256': manifest['responses_sha256'], 'trace_sha256': manifest['trace_sha256'],
    'reference_correct': 6, 'edited_correct': 0, 'total': 6, 'correct_answers_lost': 6,
    'rows': rows, 'needs_group_localization': True,
    'conclusion': 'With native prefill preserved, denying all subsequent direct attention to fact-token keys loses all six correct donor answers. Correct generation depends on that access in these cases. This does not locate the necessary layers or erase facts from other prefix states.',
    'weights_changed': False, 'counts_as_correction': False, 'new_weight_only_corrections': 0,
    'full_objective_achieved': False,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('All-block fact rereading ablation: 6/6 correct donor answers lost; group localization enabled.')
