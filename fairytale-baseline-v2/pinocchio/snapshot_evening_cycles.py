"""Archive completed ordinary batteries; no inference and no questionnaire reads."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text('utf8'))


old_path = RDIR / 'experiment_cycle_inventory_20261105_after_name_memory.json'
old = read(old_path)
entries = list(old['entries'])
for entry in entries:
    assert sha(ROOT / entry['source']) == entry['source_sha256']
new_files = [
    'qualified_concept_repairs_screen',
    'fifth_development_native_recovery',
    'fifth_development_qualified_concept_repairs',
    'qualified_concept_repairs_third_observed',
    'qualified_concept_repairs_fourth_observed',
    *('qualified_native_preservation_' + label for label in ['screen', 'fifth_observed', 'sixth_reference', 'sixth_candidate', 'third_observed', 'fourth_observed']),
]
execution = read(RDIR / 'qualified_native_preservation_evaluation_manifest.json')
assert execution['all_completed'] and execution['original_parameters_restored']
for stem in new_files:
    source = RDIR / f'{stem}.json'
    rows = read(source)
    manifest_path = RDIR / f'{stem}_manifest.json'
    manifest = read(manifest_path)
    assert manifest['completed'] and sha(source) == manifest['responses_sha256']
    assert all(r['actual_weights_changed'] and not r['runtime_hooks'] for r in rows)
    entries.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                    'configuration': {'stage': rows[0]['stage']},
                    'response_records': len(rows), 'runtime_errors': 0, 'diagnostic_ablation': False,
                    'first_recorded_utc': manifest['utc'],
                    'last_recorded_utc': datetime.datetime.fromtimestamp(source.stat().st_mtime, datetime.timezone.utc).isoformat(),
                    'last_time_meaning': 'Response file last-write timestamp, not a fabricated per-answer clock.',
                    'manifest_sha256': sha(manifest_path)})
report = {**old, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'evaluation_batteries': len(entries),
          'response_source_files': len({str(Path(r['source'])) for r in entries}),
          'modified_response_records': sum(r['response_records'] for r in entries),
          'previous_inventory_sha256': sha(old_path), 'code_sha256': sha(Path(__file__)),
          'head_before_diary_inventory': 'ccc0b0729f98f96d0a3f23f4b575d230e0f0a047',
          'construction': 'Verify all old hashes and add eleven completed modified-weight response files. Original baselines, hook conditions, supplied-prefix diagnostics, state collection, duplicate generator versions and resumes are excluded.',
          'entries': entries}
assert report['evaluation_batteries'] == 72
assert report['response_source_files'] == 65
assert report['modified_response_records'] == 2231
output = RDIR / 'experiment_cycle_inventory_20261105_evening.json'
assert not output.exists()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: report[k] for k in ['evaluation_batteries', 'response_source_files', 'modified_response_records']})
