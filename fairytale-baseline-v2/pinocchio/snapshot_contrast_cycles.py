"""Append the two verified contrast evaluations to the frozen cycle inventory."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
BASE = RDIR / 'experiment_cycle_inventory_20261105_after_late_margin.json'
OUTPUT = RDIR / 'experiment_cycle_inventory_20261105_after_late_contrasts.json'
assert not OUTPUT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


previous = json.loads(BASE.read_text('utf8'))
entries = list(previous['entries'])
for row in entries:
    assert sha(ROOT / row['source']) == row['source_sha256'], row['source']
for stem in ['late_target_contrasts_screen', 'late_target_contrasts_third_observed']:
    source = RDIR / f'{stem}.json'
    manifest = json.loads((RDIR / f'{stem}_manifest.json').read_text('utf8'))
    rows = json.loads(source.read_text('utf8'))
    assert manifest['completed'] and sha(source) == manifest['responses_sha256']
    assert len(rows) == manifest['responses_completed'] == manifest['responses_expected']
    assert all(r['actual_weights_changed'] and not r['runtime_hooks'] for r in rows)
    assert len({r['stage'] for r in rows}) == 1
    entries.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                    'configuration': {'stage': rows[0]['stage']}, 'response_records': len(rows),
                    'runtime_errors': 0, 'diagnostic_ablation': False,
                    'first_recorded_utc': None, 'last_recorded_utc': None})
report = {**previous, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'previous_inventory_sha256': sha(BASE),
          'construction': 'Verify every previous response-source hash, then append exactly the two completed free-generation evaluations of the frozen three-contrast candidate. Activation-only selector diagnosis is excluded.',
          'evaluation_batteries': len(entries),
          'response_source_files': len({r['source'] for r in entries}),
          'modified_response_records': sum(r['response_records'] for r in entries),
          'runtime_error_records': sum(r['runtime_errors'] for r in entries),
          'head_before_diary_inventory': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'code_sha256': sha(Path(__file__)), 'entries': entries}
assert report['evaluation_batteries'] == 52 and report['modified_response_records'] == 1533
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['entries', 'included', 'excluded', 'limitations']})
