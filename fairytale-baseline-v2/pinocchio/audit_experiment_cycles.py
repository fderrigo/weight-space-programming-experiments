"""Count archived weight-edit evaluation batteries without counting baselines twice."""
import collections
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUTPUT = RDIR / 'experiment_cycle_inventory_20261105.json'
assert not OUTPUT.exists()
entries = []
for path in sorted(RDIR.glob('*.json')):
    if 'review' in path.stem:
        continue
    value = json.loads(path.read_text('utf8'))
    if not isinstance(value, list):
        continue
    groups = collections.defaultdict(list)
    for row in value:
        if not isinstance(row, dict) or 'response' not in row:
            continue
        edited = (row.get('actual_weights_changed') is True
                  or row.get('model') == 'pinocchio-corrected'
                  or (path.name == 'causal_component_tests.json' and row.get('candidate') != 'original'))
        if not edited or row.get('runtime_hooks') is True:
            continue
        configuration = tuple((key, row[key]) for key in ['attempt', 'stage', 'mode', 'gain', 'candidate', 'condition'] if key in row)
        groups[configuration].append(row)
    for configuration, rows in groups.items():
        entries.append({'source': str(path.relative_to(ROOT)),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'configuration': dict(configuration), 'response_records': len(rows),
                        'runtime_errors': sum(bool(row.get('runtime_error')) for row in rows),
                        'diagnostic_ablation': path.name == 'causal_component_tests.json',
                        'first_recorded_utc': min((row['utc'] for row in rows if row.get('utc')), default=None),
                        'last_recorded_utc': max((row['utc'] for row in rows if row.get('utc')), default=None)})
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'definition': 'One archived response file and one distinct weight-configuration label form one evaluation battery. Re-evaluating the same candidate on another question set counts as another battery, not as a newly invented patch.',
    'included': 'Actual modified-weight free-response experiments, numbered Ollama attempts and four direct-weight causal ablations.',
    'excluded': 'Original-model baselines, duplicate review files, supplied-prefix diagnostics, runtime-hook interventions, activation-only analyses and algebraic simulations.',
    'limitations': 'This is not a count of training epochs, assistant turns, unique algorithms, independent validation sets, or successful corrections. It counts recorded evaluation batteries, including unsuccessful or implementation-defective trials; it does not certify every intended case of an old run was completed.',
    'evaluation_batteries': len(entries),
    'response_source_files': len({row['source'] for row in entries}),
    'modified_response_records': sum(row['response_records'] for row in entries),
    'runtime_error_records': sum(row['runtime_errors'] for row in entries),
    'diagnostic_ablation_batteries': sum(row['diagnostic_ablation'] for row in entries),
    'numbered_converted_attempts': sorted({json.loads(path.read_text('utf8'))['attempt'] for path in RDIR.glob('conversion_attempt_*.json')}),
    'head_before_diary_inventory': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'entries': entries,
}
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'entries'})
