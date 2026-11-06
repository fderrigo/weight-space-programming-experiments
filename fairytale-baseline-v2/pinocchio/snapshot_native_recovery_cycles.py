"""Inventory completed native-recovery tests and audit the unchanged full goal."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
BASE = RDIR / 'experiment_cycle_inventory_20261105_after_fourth.json'
OUT = RDIR / 'experiment_cycle_inventory_20261105_after_native_recovery.json'
AUDIT = RDIR / 'goal_audit_after_native_recovery.json'
assert not OUT.exists() and not AUDIT.exists()
def read(path):
    return json.loads(path.read_text('utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
prior = read(BASE)
entries = list(prior['entries'])
for row in entries:
    assert sha(ROOT / row['source']) == row['source_sha256']
for label in ['fourth_observed', 'third_observed', 'screen']:
    source = RDIR / f'late_native_recovery_{label}.json'
    manifest = read(RDIR / f'late_native_recovery_{label}_manifest.json')
    rows = read(source)
    assert manifest['completed'] and manifest['weights_restored_in_memory']
    assert manifest['all_declared_preservation_checks_passed']
    assert sha(source) == manifest['responses_sha256']
    assert all(r['actual_weights_changed'] and not r['runtime_hooks'] for r in rows)
    entries.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                    'configuration': {'stage': rows[0]['stage']}, 'response_records': len(rows),
                    'runtime_errors': 0, 'diagnostic_ablation': False,
                    'first_recorded_utc': manifest['utc'], 'last_recorded_utc': None})
report = {**prior, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'previous_inventory_sha256': sha(BASE),
          'construction': 'Verify prior sources, add three completed observed native-recovery evaluations. Original-only English diagnosis, fixed-history analysis and failed diagnostic starts are excluded.',
          'evaluation_batteries': len(entries), 'response_source_files': len({r['source'] for r in entries}),
          'modified_response_records': sum(r['response_records'] for r in entries),
          'runtime_error_records': sum(r['runtime_errors'] for r in entries),
          'head_before_diary_inventory': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'code_sha256': sha(Path(__file__)), 'entries': entries}
assert report['evaluation_batteries'] == 58 and report['modified_response_records'] == 1745
OUT.write_text(json.dumps(report, indent=2), encoding='utf8')
fourth = read(RDIR / 'late_native_recovery_fourth_observed_review.json')
third = read(RDIR / 'late_native_recovery_third_observed_review.json')
screen = read(RDIR / 'late_native_recovery_review.json')
assert fourth['repaired_targets_strict'] == 13 and third['repaired_targets'] == 11
assert third['originally_correct_controls_retained'] == 5 and screen['edited_faithful_narratives'] == 0
requirements = [
    {'requirement': 'At least half of initially wrong whole target answers repaired on newly frozen questions',
     'status': 'Met by preceding late_tail_relations on its fresh fourth set; fresh validation of latest candidate still pending',
     'evidence': ['late_tail_relations_fourth_review.json', 'late_native_recovery_fourth_observed_review.json'],
     'result': 'Historical fresh13/24=54.17percent retained exactly by latest candidate on the now-observed set. It cannot be presented as a second fresh result.'},
    {'requirement': 'Same original model; exclusively explicit direct weight changes; no training, gradients, optimizer or corrective prompt',
     'status': 'Verified in construction and ordinary evaluator',
     'evidence': ['../late_native_recovery_write.py', '../evaluate_native_recovery.py', 'late_native_recovery_diagnostic.json']},
    {'requirement': 'Faithful narrative behavior', 'status': 'Not met',
     'evidence': ['late_native_recovery_review.json'], 'result': '0/2 faithful ordinary narratives. Original English diagnosis also does not recover the canonical story.'},
    {'requirement': 'Conservation of outside knowledge', 'status': 'Known regressions recovered on observed sets; independent broader preservation pending',
     'evidence': ['late_native_recovery_third_observed_review.json', 'late_native_recovery_fourth_observed_review.json', 'late_native_recovery_review.json'],
     'result': 'Third5/5, fourth2/2 and screen3/3 original whole-correct controls retained. Main claims7/7,5/5 and4/4 retained. Most controls were originally wrong; these small development samples do not prove general preservation.'},
    {'requirement': 'Full current checkpoint, conversion/runtime equivalence and final reserved evaluation', 'status': 'Pending for latest candidate',
     'evidence': ['../patches/current.json', 'final_protocol_exposure_20261105.json'],
     'result': 'Export remains attempt6; latest candidate is a reconstructible experimental basis. Final response test remains unexecuted. Four final control formulations were indirectly exposed in a source search, documented separately and excluded from editing data.'},
]
audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'objective_status': 'active; not complete',
         'latest_candidate': 'late_native_recovery', 'basis_sha256': sha(ROOT / 'patches/late_native_recovery_basis.safetensors'),
         'requirements': requirements, 'full_objective_achieved': False, 'candidate_promoted': False,
         'final_responses_generated_or_read': False, 'final_questionnaire_opened': False,
         'indirect_final_formulations_exposed': 4,
         'review_hashes': {name: sha(RDIR / name) for name in ['late_native_recovery_fourth_observed_review.json',
                                                              'late_native_recovery_third_observed_review.json', 'late_native_recovery_review.json']}}
AUDIT.write_text(json.dumps(audit, indent=2), encoding='utf8')
print({k: report[k] for k in ['evaluation_batteries', 'response_source_files', 'modified_response_records']})
print('Goal remains active: narrative fidelity and independent final validation missing.')
