"""Archive completed evaluation counts and the remaining objective requirements."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
BASE = RDIR / 'experiment_cycle_inventory_20261105_after_late_contrasts.json'
OUTPUT = RDIR / 'experiment_cycle_inventory_20261105_after_fourth.json'
AUDIT = RDIR / 'goal_audit_after_fourth.json'
assert not OUTPUT.exists() and not AUDIT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text('utf8'))


previous = read(BASE)
entries = list(previous['entries'])
for row in entries:
    assert sha(ROOT / row['source']) == row['source_sha256']
sources = [
    ('late_tail_relations_third_observed', 'late_tail_relations_third_observed_manifest', 'responses_sha256'),
    ('late_tail_relations_screen', 'late_tail_relations_screen_manifest', 'responses_sha256'),
    ('late_tail_relations_fourth_edited_hf', 'late_tail_relations_fourth_evaluation_manifest', 'edited_responses_sha256'),
]
for stem, manifest_stem, hash_key in sources:
    source = RDIR / f'{stem}.json'
    manifest = read(RDIR / f'{manifest_stem}.json')
    rows = read(source)
    assert manifest['completed'] and sha(source) == manifest[hash_key]
    assert all(r['actual_weights_changed'] and not r['runtime_hooks'] for r in rows)
    assert len({r['stage'] for r in rows}) == 1
    entries.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                    'configuration': {'stage': rows[0]['stage']}, 'response_records': len(rows),
                    'runtime_errors': 0, 'diagnostic_ablation': False,
                    'first_recorded_utc': None, 'last_recorded_utc': None})
report = {**previous, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'previous_inventory_sha256': sha(BASE),
          'construction': 'Verify all previous sources, then add the observed third-set test, known18-case screen, and fresh44-case edited evaluation. Original baseline44 and fixed-history diagnoses are excluded.',
          'evaluation_batteries': len(entries),
          'response_source_files': len({r['source'] for r in entries}),
          'modified_response_records': sum(r['response_records'] for r in entries),
          'runtime_error_records': sum(r['runtime_errors'] for r in entries),
          'head_before_diary_inventory': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'code_sha256': sha(Path(__file__)), 'entries': entries}
assert report['evaluation_batteries'] == 55 and report['modified_response_records'] == 1639
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
fresh_path = RDIR / 'late_tail_relations_fourth_review.json'
screen_path = RDIR / 'late_tail_relations_review.json'
fresh, screen = read(fresh_path), read(screen_path)
assert fresh['repaired_targets_strict'] == 13 and fresh['initially_wrong_targets'] == 24
assert screen['edited_faithful_narratives'] == 0 and screen['narrative_total'] == 2
audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'objective_status': 'active; not complete',
    'requirements': [
        {'requirement': 'Correct at least half the initially wrong whole target answers on newly frozen questions',
         'status': 'met on the fourth development sample', 'evidence': str(fresh_path.relative_to(ROOT)),
         'result': '13/24=54.1667percent; both ambiguous answers excluded; not a population-wide guarantee.'},
        {'requirement': 'Candidate fixed before the new questions and their answers',
         'status': 'verified for fourth development', 'evidence': 'fourth_development_freeze.json',
         'result': 'Candidate26cf2c34... fixed before question creation; question and basis hashes rechecked for the paired run.'},
        {'requirement': 'Same original model, direct weight changes, no fine-tuning, gradients, optimizer or corrective inference prompt',
         'status': 'verified in the recorded construction and evaluator',
         'evidence': ['late_tail_relations_write.py', 'evaluate_fourth_development.py', 'results/late_tail_relations_fourth_evaluation_manifest.json']},
        {'requirement': 'Faithful narrative behavior', 'status': 'not met',
         'evidence': str(screen_path.relative_to(ROOT)), 'result': '0/2 faithful ordinary narratives; invented events and repetition remain.'},
        {'requirement': 'Conservation of outside knowledge', 'status': 'incomplete; regressions remain',
         'evidence': str(fresh_path.relative_to(ROOT)),
         'result': 'Both initially wholly correct controls retained, but only2/20 controls were wholly correct originally. Fish habitat and London attribution are lost as useful main claims in previously flawed responses. Earlier third set still loses2/5 original whole controls.'},
        {'requirement': 'Reserved final protocol and full checkpoint/runtime validation of the latest candidate',
         'status': 'not performed for latest candidate',
         'evidence': ['final_freeze.json', 'patches/current.json', 'results/late_tail_relations_fourth_evaluation_manifest.json'],
         'result': 'Final72 remains unread/unexecuted. Exported checkpoint remains attempt6; latest basis is experimental and was evaluated in Transformers.'},
    ],
    'fresh_review_sha256': sha(fresh_path), 'screen_review_sha256': sha(screen_path),
    'full_objective_achieved': False, 'candidate_promoted': False, 'final_set_read': False}
AUDIT.write_text(json.dumps(audit, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k not in ['entries', 'included', 'excluded', 'limitations']})
print('Numerical fresh target met; full objective remains active.')
