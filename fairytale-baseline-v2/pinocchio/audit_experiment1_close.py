"""Read-only integrity audit and closure record; no model inference."""
import ast
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


inventory_path = RDIR / 'experiment_cycle_inventory_20261105_evening.json'
inventory = read(inventory_path)
for row in inventory['entries']:
    assert sha(ROOT / row['source']) == row['source_sha256']
for stem in ['fifth', 'sixth']:
    freeze = read(ROOT / f'{stem}_development_freeze.json')
    assert sha(ROOT / f'{stem}_development_questions.json') == freeze['questions_sha256']
    assert sha(ROOT / f'freeze_{stem}_development.py') == freeze['generator_sha256']
    evaluator = 'evaluate_fifth_three_models.py' if stem == 'fifth' else 'evaluate_qualified_preservation.py'
    assert sha(ROOT / evaluator) == freeze['evaluator_sha256']
    assert sha(ROOT / f"patches/{freeze['candidate']}_basis.safetensors") == freeze['candidate_sha256']
manifest = read(RDIR / 'experiment1_checkpoint_manifest.json')
assert manifest['completed'] and manifest['original_checkpoint_unchanged']
assert manifest['all_exported_tensors_exact_in_f32']
assert sha(ROOT / manifest['checkpoint_path']) == manifest['checkpoint_sha256']
frozen = {
    '../../little-red-weightspace/models/original/model.safetensors': 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    'models/corrected/model.safetensors': '5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520',
    'patches/late_native_recovery_basis.safetensors': '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7',
}
for name, expected in frozen.items():
    assert sha(ROOT / name) == expected
for name, expected in manifest['copied_config_tokenizer_files_sha256'].items():
    assert sha(ROOT / 'models/experiment1_concluded' / name) == expected
    assert sha(ROOT.parent.parent / 'little-red-weightspace/models/original' / name) == expected
scripts = ['best_reference_runtime.py', 'probe_native_name_memory_locality.py', 'probe_stable_relation_selection.py',
           'probe_best_minimal_repairs.py', 'collect_qualified_minimal_paths.py', 'write_qualified_concept_repairs.py',
           'review_qualified_concept_screen.py', 'freeze_fifth_development.py', 'evaluate_fifth_three_models.py',
           'evaluate_qualified_observed.py', 'review_fifth_development.py', 'review_qualified_observed.py',
           'collect_qualified_preservation.py', 'write_qualified_native_preservation.py', 'freeze_sixth_development.py',
           'evaluate_qualified_preservation.py', 'review_preservation_observed.py', 'review_sixth_development.py',
           'snapshot_evening_cycles.py', 'freeze_experiment1_reference.py', 'audit_experiment1_close.py']
for name in scripts:
    ast.parse((ROOT / name).read_text('utf8'), filename=name)
summary = ROOT.parent.parent / 'riassunto.md'
copy = RDIR / 'root_riassunto_at_experiment1_close.md.txt'
assert not copy.exists()
copy.write_bytes(summary.read_bytes())
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'phase': 'Experiment1 concluded; no further Pinocchio optimization',
          'inventory_sha256': sha(inventory_path), 'evaluation_batteries': inventory['evaluation_batteries'],
          'response_source_files': inventory['response_source_files'], 'modified_response_records': inventory['modified_response_records'],
          'checkpoint_manifest_sha256': sha(RDIR / 'experiment1_checkpoint_manifest.json'),
          'original_previous_export_and_reference_hashes_verified': frozen,
          'fifth_and_sixth_frozen_code_questions_and_candidates_verified': True,
          'new_python_sources_syntax_checked': scripts,
          'root_summary_snapshot_sha256': sha(copy),
          'reserved_final_test_executed': False, 'reserved_final_test_obstacle': 'Previously exposed four formulations; see existing exposure record and conclusions.',
          'all_ordinary_evaluations_completed': read(RDIR / 'qualified_native_preservation_evaluation_manifest.json')['all_completed'],
          'conclusions_sha256': sha(ROOT / 'EXPERIMENT_1_CONCLUSIONS.md'),
          'new_inference': False, 'new_weight_edit': False, 'remote_push': False}
assert report['all_ordinary_evaluations_completed']
(RDIR / 'experiment1_closure_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print('Closure verified:', report['evaluation_batteries'], 'batteries;', len(scripts), 'sources; checkpoint exact.')
