"""Check provenance of the completed origin experiment without running the model."""
import ast
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'narrative_origins_archive_verification.json'
assert not OUT.exists()
checks = []
cache = {}


def read(path):
    return json.loads(path.read_text('utf8'))


def sha(path):
    path = path.resolve()
    if path not in cache:
        with path.open('rb') as handle:
            cache[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return cache[path]


def check(path, expected):
    actual = sha(path)
    assert actual == expected, (str(path), actual, expected)
    checks.append({'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   'sha256': actual})


for stem, source in [('narrative_origin_trace', 'trace_narrative_origins.py'),
                     ('narrative_origin_units', 'collect_narrative_origin_units.py')]:
    spec = read(RDIR / f'{stem}_spec.json')
    report = read(RDIR / f'{stem}.json')
    assert report['spec'] == spec and report['completed'] and report['weights_restored_in_memory']
    check(ROOT / source, spec['code_sha256'])
    check(ROOT / 'patches/late_native_recovery_basis.safetensors', spec['basis_sha256'])
    for condition in report['reports']:
        check(RDIR / f"{stem}_{condition['condition']}.npz", condition['arrays_sha256'])

spec = read(ROOT / 'patches/late_narrative_origins_spec.json')
diagnostic = read(RDIR / 'late_narrative_origins_diagnostic.json')
assert diagnostic['spec'] == spec
check(ROOT / 'late_narrative_origins_write.py', spec['code_sha256'])
check(ROOT / 'patches/late_native_recovery_basis.safetensors', spec['source_basis_sha256'])
check(RDIR / 'narrative_origin_units.json', spec['unit_source_sha256'])
check(RDIR / 'narrative_origin_trace.json', spec['trace_source_sha256'])
check(ROOT / 'late_selector_weights.py', spec['parent_transform_code_sha256'])
check(RDIR / 'late_selector_geometry.npz', spec['geometry_sha256'])
unit_spec = read(RDIR / 'narrative_origin_units_spec.json')
check(RDIR / 'narrative_origin_trace.json', unit_spec['source_trace_sha256'])
trace_spec = read(RDIR / 'narrative_origin_trace_spec.json')
for source in trace_spec['sources']:
    check(RDIR / source['source'], source['sha256'])
assert spec['source_basis_sha256'] == unit_spec['basis_sha256'] == trace_spec['basis_sha256']

manifest = read(RDIR / 'late_narrative_origins_screen_manifest.json')
assert manifest['completed'] and manifest['weights_restored_in_memory'] and manifest['responses_completed'] == 18
check(ROOT / 'patches/late_narrative_origins_basis.safetensors', manifest['basis_sha256'])
check(ROOT / 'patches/late_narrative_origins_spec.json', manifest['spec_sha256'])
check(ROOT / 'screen_late_selector_candidate.py', manifest['evaluator_sha256'])
check(RDIR / 'late_narrative_origins_screen.json', manifest['responses_sha256'])
check(RDIR / 'attention_gain_screen.json', manifest['original_responses_source_sha256'])
review = read(RDIR / 'late_narrative_origins_review.json')
check(ROOT / 'review_narrative_origins.py', review['review_code_sha256'])
check(RDIR / 'late_native_recovery_review.json', review['previous_candidate_review_sha256'])
assert not review['observed_transfer_completed'] and not review['full_objective_achieved']
assert not review['candidate_promoted'] and review['edited_faithful_narratives'] == 0
assert review['basis_sha256'] == manifest['basis_sha256'] and review['source_sha256'] == manifest['responses_sha256']

inventory = read(RDIR / 'experiment_cycle_inventory_20261105_after_origins.json')
check(ROOT / 'snapshot_narrative_origins.py', inventory['code_sha256'])
for entry in inventory['entries']:
    check(ROOT / entry['source'], entry['source_sha256'])
assert inventory['evaluation_batteries'] == 59 and inventory['response_source_files'] == 52
assert inventory['modified_response_records'] == 1763
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert inventory['head_before_diary_inventory'] == head
audit = read(RDIR / 'goal_audit_after_narrative_origins.json')
check(RDIR / 'late_narrative_origins_review.json', audit['review_sha256'])
assert audit['basis_sha256'] == manifest['basis_sha256'] and not audit['full_objective_achieved']

check(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors',
      'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe')
check(ROOT / 'models/corrected/model.safetensors',
      '5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520')
# Binary integrity only: no parsing or printing of the reserved questionnaire.
check(ROOT / 'final_questions.json',
      '3dc8049ee74a5bac5fc2b844d80e84ce76e609cf67f99f0deeec6976b82447c3')
for source in ['trace_narrative_origins.py', 'collect_narrative_origin_units.py',
               'late_narrative_origins_write.py', 'review_narrative_origins.py',
               'snapshot_narrative_origins.py', 'verify_narrative_origins_archive.py']:
    ast.parse((ROOT / source).read_text('utf8'))
local_links = 0
for doc in ['README.md', 'RICERCA_IN_CORSO.md']:
    for link in re.findall(r'\]\(([^)]+)\)', (ROOT / doc).read_text('utf8')):
        if '://' not in link and not link.startswith('#'):
            assert (ROOT / link.split('#')[0]).exists(), (doc, link)
            local_links += 1
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'head': head,
          'code_sha256': sha(Path(__file__)), 'checks': checks, 'local_links_checked': local_links,
          'source_basis_consistency_verified': True, 'reserved_questionnaire_binary_hash_only': True,
          'model_execution_performed': False, 'new_corrections_claimed': False, 'passed': True}
OUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({'hash_checks': len(checks), 'local_links_checked': local_links, 'passed': True})
