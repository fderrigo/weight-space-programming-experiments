"""Verify provenance of completed residual and fact-access diagnostics."""
import ast
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'relation_access_archive_verification.json'
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


spec = read('persistent_relation_delta_spec.json')
collection = read('persistent_relation_delta_collection.json')
assert collection['completed'] and collection['spec'] == spec
check(ROOT / 'probe_persistent_relation_delta.py', spec['code_sha256'])
check(ROOT / 'probe_single_fact_state.py', spec['prior_donor_code_sha256'])
check(RDIR / 'single_fact_state_diagnostic.json', spec['prior_donor_responses_sha256'])
check(RDIR / 'persistent_relation_delta_states.npz', collection['arrays_sha256'])
donors = read('persistent_relation_delta_donor_review.json')
check(ROOT / 'review_persistent_relation_donors.py', donors['code_sha256'])
check(RDIR / 'persistent_relation_delta_collection.json', donors['source_sha256'])
assert donors['certified_ids'] == collection['ids'] and donors['certified_donors'] == 6
with np.load(RDIR / 'persistent_relation_delta_states.npz') as z:
    delta = z['delta'].astype(np.float64)
    assert np.allclose(delta, z['reference'].astype(np.float64) - z['masked'].astype(np.float64), rtol=1e-6, atol=2e-6)
    unit = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    pairs = (unit @ unit.T)[np.triu_indices(6, 1)]
    mean = delta.mean(0)
    geometry = {'source_arrays_sha256': collection['arrays_sha256'],
                'shared_mean_energy_fraction': float(6 * (mean @ mean) / np.sum(delta * delta)),
                'pairwise_cosine_min': float(pairs.min()), 'pairwise_cosine_max': float(pairs.max()),
                'nuisance_component_interpretation_not_established': True,
                'used_to_select_amplitude_or_weights': False}
manifest = read('persistent_relation_delta_probe_manifest.json')
assert manifest['completed'] and manifest['hooks_removed'] and manifest['responses'] == 12
for name, key in [('persistent_relation_delta_probe.json', 'responses_sha256'),
                  ('persistent_relation_delta_collection.json', 'collection_sha256'),
                  ('persistent_relation_delta_donor_review.json', 'donor_review_sha256'),
                  ('persistent_relation_delta_spec.json', 'spec_sha256')]:
    check(RDIR / name, manifest[key])
check(ROOT / 'probe_persistent_relation_delta.py', manifest['code_sha256'])
review = read('persistent_relation_delta_review.json')
check(ROOT / 'review_persistent_relation_delta.py', review['code_sha256'])
check(RDIR / 'persistent_relation_delta_probe.json', review['source_sha256'])
assert [r['correct'] for r in review['summary']] == [1, 1]

cache_spec = read('fact_cache_dependency_spec.json')
check(ROOT / 'probe_fact_cache_dependency.py', cache_spec['code_sha256'])
check(RDIR / 'persistent_relation_delta_collection.json', cache_spec['collection_sha256'])
check(RDIR / 'persistent_relation_delta_donor_review.json', cache_spec['donor_review_sha256'])
check(ROOT / 'probe_single_fact_state.py', cache_spec['fact_source_code_sha256'])
for condition in ['all', 'early', 'middle', 'late']:
    manifest = read(f'fact_cache_dependency_{condition}_manifest.json')
    assert manifest['completed'] and manifest['hooks_removed'] and manifest['responses'] == 6
    check(ROOT / 'probe_fact_cache_dependency.py', manifest['code_sha256'])
    check(RDIR / 'fact_cache_dependency_spec.json', manifest['spec_sha256'])
    check(RDIR / f'fact_cache_dependency_{condition}.json', manifest['responses_sha256'])
    check(RDIR / f'fact_cache_dependency_{condition}_trace.json', manifest['trace_sha256'])
    responses = read(f'fact_cache_dependency_{condition}.json')
    assert [r['id'] for r in responses] == collection['ids']
    assert all(r['first_token_matches_full_reference'] and r['runtime_hooks'] and not r['counts_as_correction'] for r in responses)
    for trace in read(f'fact_cache_dependency_{condition}_trace.json'):
        assert trace['hooks_removed'] and set(map(int, trace['layer_stats'])) == set(cache_spec['conditions'][condition])
        assert all(s['prefill_calls_unchanged'] == 1 and s['decoding_calls_masked'] > 0 for s in trace['layer_stats'].values())
all_review = read('fact_cache_dependency_all_review.json')
check(ROOT / 'review_fact_cache_all.py', all_review['code_sha256'])
check(RDIR / 'fact_cache_dependency_all.json', all_review['source_sha256'])
assert all_review['correct_answers_lost'] == 6 and all_review['needs_group_localization']
groups_review = read('fact_cache_dependency_groups_review.json')
check(ROOT / 'review_fact_cache_groups.py', groups_review['code_sha256'])
for entry in groups_review['sources']:
    check(RDIR / entry['source'], entry['sha256'])
    check(RDIR / entry['manifest'], entry['manifest_sha256'])
assert [r['correct'] for r in groups_review['summary']] == [5, 0, 0]

critical = read('critical_fact_reads.json')
assert critical['completed'] and len(critical['records']) == 18
assert critical['spec'] == read('critical_fact_reads_spec.json')
check(ROOT / 'trace_critical_fact_reads.py', critical['spec']['code_sha256'])
check(RDIR / 'persistent_relation_delta_collection.json', critical['spec']['source_sha256'])
check(RDIR / 'persistent_relation_delta_donor_review.json', critical['spec']['donor_review_sha256'])
check(RDIR / 'critical_fact_reads.npz', critical['arrays_sha256'])
check(RDIR / 'critical_fact_reads_records.json', critical['records_sha256'])
critical_summary = read('critical_fact_reads_summary.json')
check(ROOT / 'summarize_critical_fact_reads.py', critical_summary['code_sha256'])
check(RDIR / 'critical_fact_reads.json', critical_summary['source_sha256'])
assert critical_summary['one_block_ablations'] == 432
assert critical_summary['max_explicit_mask_relative_error'] < 2e-5
assert critical_summary['max_attention_reconstruction_relative_error'] < 2e-5
head = read('fact_read_heads.json')
assert head['completed'] and len(head['records']) == 18 and head['spec'] == read('fact_read_heads_spec.json')
check(ROOT / 'trace_fact_read_heads.py', head['spec']['code_sha256'])
check(RDIR / 'critical_fact_reads.json', head['spec']['source_sha256'])
check(RDIR / 'critical_fact_reads_records.json', head['spec']['records_sha256'])
head_summary = read('fact_read_heads_summary.json')
check(ROOT / 'summarize_fact_read_heads.py', head_summary['code_sha256'])
check(RDIR / 'fact_read_heads.json', head_summary['source_sha256'])
assert head_summary['head_or_kv_group_ablations'] == 288
assert head_summary['max_expanded_mask_relative_error'] < 2e-5
assert head_summary['max_full_block_margin_difference'] < 1e-3
assert all(len(r['head_ablations']) == 16 for r in head['records'])

inventory = read('experiment_cycle_inventory_20261105_after_event_roles.json')
assert inventory['evaluation_batteries'] == 60 and inventory['response_source_files'] == 53
assert inventory['modified_response_records'] == 1781
check(ROOT.parent.parent / 'little-red-weightspace/models/original/model.safetensors',
      'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe')
check(ROOT / 'models/corrected/model.safetensors',
      '5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520')
# Binary integrity only: never parse or display reserved questions.
check(ROOT / 'final_questions.json',
      '3dc8049ee74a5bac5fc2b844d80e84ce76e609cf67f99f0deeec6976b82447c3')
sources = ['probe_persistent_relation_delta.py', 'review_persistent_relation_donors.py',
           'review_persistent_relation_delta.py', 'probe_fact_cache_dependency.py',
           'review_fact_cache_all.py', 'review_fact_cache_groups.py', 'trace_critical_fact_reads.py',
           'summarize_critical_fact_reads.py', 'trace_fact_read_heads.py',
           'summarize_fact_read_heads.py', 'verify_relation_access_archive.py']
for name in sources:
    ast.parse((ROOT / name).read_text('utf8'))
links = 0
for path in [ROOT / 'README.md', ROOT / 'RICERCA_IN_CORSO.md', ROOT.parent.parent / 'riassunto.md']:
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text('utf8')):
        if '://' not in link and not link.startswith('#'):
            assert (path.parent / link.split('#')[0]).exists(), (path.name, link)
            links += 1
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'head_before_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'code_sha256': sha(Path(__file__)), 'hash_checks': checks,
          'source_code_parse_checks': sources, 'local_document_links_checked': links,
          'descriptive_delta_geometry': geometry,
          'reserved_questionnaire_binary_hash_only': True, 'model_executed_by_verification': False,
          'new_weight_only_corrections': 0, 'evaluation_batteries_unchanged': 60,
          'candidate_promoted': False, 'full_objective_achieved': False, 'passed': True}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(f'PASS: {len(checks)} hashes, {len(sources)} parsed sources, {links} local links.')
