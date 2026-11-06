"""Archive complete evaluation, verify source hashes, and keep the goal open."""
import ast
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
INV = RDIR / 'experiment_cycle_inventory_20261105_after_event_roles.json'
AUDIT = RDIR / 'goal_audit_after_event_roles.json'
VERIFY = RDIR / 'event_roles_archive_verification.json'
assert not any(p.exists() for p in [INV, AUDIT, VERIFY])


def read(path):
    return json.loads(path.read_text('utf8'))


cache, checks = {}, []


def sha(path):
    path = path.resolve()
    if path not in cache:
        with path.open('rb') as handle:
            cache[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return cache[path]


def check(path, expected):
    actual = sha(path)
    assert actual == expected, str(path)
    checks.append({'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'sha256': actual})


prior_path = RDIR / 'experiment_cycle_inventory_20261105_after_origins.json'
prior = read(prior_path)
entries = list(prior['entries'])
for entry in entries:
    check(ROOT / entry['source'], entry['source_sha256'])
manifest = read(RDIR / 'late_event_roles_screen_manifest.json')
source = RDIR / 'late_event_roles_screen.json'
rows = read(source)
assert manifest['completed'] and manifest['weights_restored_in_memory'] and len(rows) == 18
check(source, manifest['responses_sha256'])
check(ROOT / 'screen_late_selector_candidate.py', manifest['evaluator_sha256'])
check(ROOT / 'patches/late_event_roles_basis.safetensors', manifest['basis_sha256'])
check(ROOT / 'patches/late_event_roles_spec.json', manifest['spec_sha256'])
spec = read(ROOT / 'patches/late_event_roles_spec.json')
check(ROOT / 'late_event_roles_write.py', spec['code_sha256'])
check(ROOT / 'patches/late_narrative_origins_basis.safetensors', spec['source_basis_sha256'])
check(RDIR / 'event_role_binding.json', spec['event_source_sha256'])
check(RDIR / 'event_role_binding_analysis.json', spec['amplitude_infeasibility_source_sha256'])
check(ROOT / 'late_selector_weights.py', spec['parent_transform_code_sha256'])
check(RDIR / 'late_selector_geometry.npz', spec['geometry_sha256'])
assert read(RDIR / 'late_event_roles_diagnostic.json')['spec'] == spec
trace = read(RDIR / 'event_role_binding.json')
assert trace['completed'] and trace['weights_restored_in_memory']
check(ROOT / 'probe_event_role_binding.py', trace['spec']['code_sha256'])
for report in trace['reports']:
    check(RDIR / f"event_role_binding_{report['condition']}.npz", report['arrays_sha256'])
check(RDIR / 'late_narrative_origins_equations.npz', trace['spec']['origin_equations_sha256'])
analysis = read(RDIR / 'event_role_binding_analysis.json')
check(ROOT / 'analyze_event_role_binding.py', analysis['spec']['code_sha256'])
check(RDIR / 'event_role_binding.json', analysis['spec']['source_sha256'])
assert not analysis['joint_feasible']
leak = read(RDIR / 'narrative_cell_leak.json')
assert leak['completed'] and leak['weights_restored_in_memory']
check(ROOT / 'diagnose_narrative_cell_leak.py', leak['spec']['code_sha256'])
check(source, leak['spec']['source_sha256'])
check(ROOT / 'patches/late_event_roles_basis.safetensors', leak['spec']['basis_sha256'])
check(RDIR / 'narrative_cell_leak_late_event_roles.npz', leak['arrays_sha256'])
review_path = RDIR / 'late_event_roles_review.json'
review = read(review_path)
check(ROOT / 'review_event_roles.py', review['review_code_sha256'])
assert review['source_sha256'] == manifest['responses_sha256']
assert review['basis_sha256'] == manifest['basis_sha256']
assert review['edited_target_whole_correct_strict'] == 6 and review['edited_faithful_narratives'] == 0
assert not review['candidate_promoted'] and not review['observed_transfer_completed']
for script in ['probe_event_role_binding.py', 'analyze_event_role_binding.py', 'late_event_roles_write.py',
               'diagnose_narrative_cell_leak.py', 'review_event_roles.py', 'archive_event_roles.py']:
    ast.parse((ROOT / script).read_text('utf8'))
check(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors',
      'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe')
check(ROOT / 'models/corrected/model.safetensors',
      '5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520')
# Binary integrity only; no parsing or printing the reserved questionnaire.
check(ROOT / 'final_questions.json',
      '3dc8049ee74a5bac5fc2b844d80e84ce76e609cf67f99f0deeec6976b82447c3')
entries.append({'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                'configuration': {'stage': rows[0]['stage']}, 'response_records': 18,
                'runtime_errors': 0, 'diagnostic_ablation': False,
                'first_recorded_utc': manifest['utc'], 'last_recorded_utc': None})
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
inventory = {**prior, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'previous_inventory_sha256': sha(prior_path),
             'construction': 'Verify prior sources and add the completed18-response event-role screen. Diagnostic prefills, scalar feasibility and local fixed-history ablation are excluded.',
             'head_before_diary_inventory': head, 'code_sha256': sha(Path(__file__)),
             'evaluation_batteries': len(entries), 'response_source_files': len({e['source'] for e in entries}),
             'modified_response_records': sum(e['response_records'] for e in entries), 'entries': entries}
assert inventory['evaluation_batteries'] == 60 and inventory['response_source_files'] == 53
assert inventory['modified_response_records'] == 1781
INV.write_text(json.dumps(inventory, indent=2), encoding='utf8')
audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'objective_status': 'active; not complete',
         'latest_experimental_candidate': 'late_event_roles', 'previous_multi_set_reference': 'late_native_recovery',
         'basis_sha256': manifest['basis_sha256'], 'review_sha256': sha(review_path),
         'historical_fresh_result': 'late_tail_relations13/24=54.17%; latest experimental variant has no independent fresh validation.',
         'latest_ordinary_screen': {'whole_target_repairs': 6, 'target_errors': 12,
                                    'exact_original_controls_retained': 4, 'whole_correct_controls_retained': 3,
                                    'faithful_narratives': 0, 'narratives': 2, 'new_whole_repairs': 0},
         'diagnostic_result_only': '32/32 minimal event-concept pieces first after11 explicit equations; no whole-response success from provided prefixes.',
         'remaining': ['Faithful narrative generation', 'Independent latest-candidate validation and broader locality',
                       'Qualified full checkpoint export, runtime verification and reserved final evaluation'],
         'candidate_promoted': False, 'full_objective_achieved': False, 'current_exported_checkpoint_attempt': 6,
         'final_responses_generated_or_read': False, 'indirect_final_formulations_exposed_previously': 4,
         'evidence': ['late_event_roles_review.json', 'late_event_roles_diagnostic.json',
                      'event_role_binding_analysis.json', 'narrative_cell_leak.json', 'final_protocol_exposure_20261105.json']}
AUDIT.write_text(json.dumps(audit, indent=2), encoding='utf8')
VERIFY.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'head': head,
    'code_sha256': sha(Path(__file__)), 'hash_checks': checks, 'reserved_questionnaire_binary_hash_only': True,
    'source_code_parse_checks': 6, 'passed': True}, indent=2), encoding='utf8')
print({'batteries': 60, 'source_files': 53, 'response_records': 1781, 'hash_checks': len(checks), 'passed': True})
