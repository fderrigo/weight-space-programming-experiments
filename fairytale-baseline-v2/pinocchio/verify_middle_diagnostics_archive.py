"""Verify diagnostic provenance and limits without running or editing the model."""
import ast
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'middle_diagnostics_archive_verification.json'
assert not OUT.exists()
checks = []
cache = {}


def read(name):
    assert 'final' not in name.lower()
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    path = path.resolve()
    if path not in cache:
        with path.open('rb') as handle:
            cache[path] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return cache[path]


def check(path, expected):
    actual = sha(path)
    assert actual == expected, (str(path), actual, expected)
    checks.append({'path': str(path), 'sha256': actual})


spec = read('middle_fact_retrieval_spec.json')
report = read('middle_fact_retrieval.json')
collection = read('middle_fact_retrieval_collection.json')
assert report['spec'] == collection['spec'] == spec
assert report['completed'] and collection['collection_completed']
assert not collection['analysis_completed']  # Snapshot of acquisition, before analysis.
assert not report['weights_changed'] and not report['free_generation_performed']
assert not report['candidate_selected_or_constructed']
assert len(report['records']) == len(collection['records']) == spec['states'] == 5560
assert len(report['sequences']) == len(collection['sequences']) == 217
assert len(report['fit_indices']) == 48 and len(report['observed_probe_indices']) == 24
assert len(report['heads']) == 14
assert report['max_native_attention_reconstruction_relative_error'] < 2e-5
assert report['max_prefix_relative_norm_difference'] < 2e-5
check(ROOT / 'inspect_middle_fact_retrieval.py', spec['code_sha256'])
check(RDIR / 'middle_fact_retrieval_states.npz', report['arrays_sha256'])
assert report['arrays_sha256'] == collection['arrays_sha256']
check(RDIR / 'middle_fact_retrieval_geometry.npz', report['geometry_arrays_sha256'])
for source in spec['sources']:
    check(RDIR / source['source'], source['sha256'])
check(RDIR / 'internal_fact_memory_spec.json', spec['memory_source_sha256'])
check(RDIR / 'middle_prefix_precision.json', spec['prefix_precision_source_sha256'])
precision = read('middle_prefix_precision.json')
check(ROOT / 'measure_middle_prefix_precision.py', precision['code_sha256'])
check(RDIR / 'middle_prefix_precision.npz', precision['arrays_sha256'])
for version in (1, 2):
    failed = read(f'middle_fact_retrieval_spec_failed_v{version}.json')
    check(RDIR / f'middle_fact_retrieval_failed_v{version}.py.txt', failed['code_sha256'])

features = read('middle_relation_features.json')
feature_spec = read('middle_relation_features_spec.json')
assert features['spec'] == feature_spec and features['completed']
assert not features['candidate_selected_or_constructed']
check(ROOT / 'compare_middle_relation_features.py', feature_spec['code_sha256'])
check(RDIR / 'middle_fact_retrieval.json', feature_spec['source_sha256'])
check(RDIR / 'middle_fact_retrieval_states.npz', feature_spec['arrays_sha256'])
check(RDIR / 'middle_relation_features.npz', features['arrays_sha256'])
by_feature = {r['representation']: r for r in features['results']}
assert by_feature['all_head_values']['observed_query_probe_correct'] == 19
assert by_feature['all_head_values']['known_role_prefix_correct'] == 10
assert by_feature['head_value_7']['known_role_prefix_correct'] == 11
assert by_feature['head_value_13']['observed_query_probe_correct'] == 20

bias = read('fact_key_bias.json')
check(ROOT / 'inspect_fact_key_bias.py', bias['code_sha256'])
check(RDIR / 'middle_fact_retrieval.json', bias['source_sha256'])
check(RDIR / 'middle_fact_retrieval_states.npz', bias['arrays_source_sha256'])
assert len(bias['heads']) == 14 and not bias['counts_as_correction']
assert min(r['native_correct_top_fact'] for r in bias['heads']) == 0
assert max(r['native_correct_top_fact'] for r in bias['heads']) == 3
assert min(r['mean_removed_correct_top_fact'] for r in bias['heads']) == 2
assert max(r['mean_removed_correct_top_fact'] for r in bias['heads']) == 6

value_spec = read('persistent_fact_values_spec.json')
value_manifest = read('persistent_fact_values_manifest.json')
review = read('persistent_fact_values_review.json')
check(ROOT / 'probe_persistent_fact_values.py', value_spec['code_sha256'])
check(RDIR / 'attention_gain_screen.json', value_spec['source_original_responses_sha256'])
check(RDIR / 'middle_fact_retrieval.json', value_spec['geometry_sha256'])
check(RDIR / 'middle_fact_retrieval_states.npz', value_spec['state_source_sha256'])
check(RDIR / 'persistent_fact_values_spec.json', value_manifest['spec_sha256'])
check(RDIR / 'persistent_fact_values.json', value_manifest['responses_sha256'])
check(ROOT / 'review_persistent_fact_values.py', review['review_code_sha256'])
check(RDIR / 'persistent_fact_values.json', review['source_sha256'])
check(RDIR / 'persistent_fact_values_spec.json', review['spec_sha256'])
check(RDIR / 'persistent_fact_values_manifest.json', review['manifest_sha256'])
assert value_manifest['completed'] and value_manifest['hooks_removed']
assert review['whole_responses_incorrect'] == 11 and review['whole_responses_ambiguous'] == 1
assert review['whole_responses_certified_correct'] == review['new_weight_only_corrections'] == 0
raw_rows = read('persistent_fact_values.json')
for raw, reviewed in zip(raw_rows, review['rows'], strict=True):
    assert reviewed['runtime_hooks'] and not raw['runtime_hooks']
    assert reviewed['temporary_activation_intervention'] and not reviewed['weights_changed']
    for field in ('id', 'prompt', 'response', 'generated_ids', 'stop_token', 'done_reason'):
        assert raw[field] == reviewed[field], (raw['id'], field)

inventory = read('experiment_cycle_inventory_20261105_after_event_roles.json')
assert inventory['evaluation_batteries'] == 60 and inventory['response_source_files'] == 53
assert inventory['modified_response_records'] == 1781
check(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors',
      'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe')
check(ROOT / 'models/corrected/model.safetensors',
      '5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520')
# Binary checksum only. Never parse or display the reserved questionnaire.
check(ROOT / 'final_questions.json',
      '3dc8049ee74a5bac5fc2b844d80e84ce76e609cf67f99f0deeec6976b82447c3')

sources = ['inspect_middle_fact_retrieval.py', 'measure_middle_prefix_precision.py',
           'compare_middle_relation_features.py', 'inspect_fact_key_bias.py',
           'probe_persistent_fact_values.py', 'review_persistent_fact_values.py',
           'verify_middle_diagnostics_archive.py']
for source in sources:
    ast.parse((ROOT / source).read_text('utf8'))
local_links = 0
for path in [ROOT / 'README.md', ROOT / 'RICERCA_IN_CORSO.md', ROOT.parent.parent / 'riassunto.md']:
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text('utf8')):
        if '://' not in link and not link.startswith('#'):
            assert (path.parent / link.split('#')[0]).exists(), (path.name, link)
            local_links += 1

archive = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'head_before_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'code_sha256': sha(Path(__file__)), 'hash_checks': checks,
    'source_code_parse_checks': sources, 'local_document_links_checked': local_links,
    'reserved_questionnaire_binary_hash_only': True,
    'raw_responses_unchanged_in_review': True, 'raw_hook_metadata_error_documented': True,
    'model_executed_by_verification': False, 'weights_changed_in_this_diagnostic_series': False,
    'new_weight_only_corrections': 0, 'evaluation_batteries_unchanged': 60,
    'candidate_promoted': False, 'full_objective_achieved': False, 'passed': True,
}
OUT.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding='utf8')
print(f'PASS: {len(checks)} hash checks, {len(sources)} code parses, {local_links} local links; no new weights.')
