"""Measure native block-14 fact addressing before attempting persistent edits."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
STEM = 'middle_fact_retrieval'
assert not (RDIR / f'{STEM}.json').exists()
torch.set_grad_enabled(False)
torch.set_num_threads(4)


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


classes = ['creator', 'material', 'nose', 'fairy', 'deceivers', 'donkey', 'ending',
           'puppeteer', 'friend', 'wood_donor', 'rescuer', 'fish_father']
memory = read('internal_fact_memory_spec.json')
slots = memory['positions']
assert slots == list(range(7, 19)) and len(memory['facts']) == 12
parent = read('late_selector_states_weights.json')
assert all(int(r['tensor'].split('.')[2]) >= 21 for r in parent['tensor_differences']
           if r['tensor'].startswith('model.layers.'))
source_names = ['carrier_concept_states.json', 'event_role_binding.json', 'narrative_cell_leak.json']
sequences, records, sources = [], [], []
for group, name in zip(['earlier', 'role_binding', 'latest_narrative'], source_names):
    meta = read(name)
    assert meta['completed']
    sources.append({'source': name, 'sha256': sha(RDIR / name)})
    for seq in meta['sequences']:
        offset = len(records)
        identity = group + '/' + seq['id']
        sequences.append({**seq, 'id': identity, 'original_id': seq['id'], 'source_group': group,
                          'offset': offset, 'old_offset': seq['offset']})
        for j in range(seq['states']):
            records.append({**meta['records'][seq['offset'] + j], 'id': identity,
                            'original_id': seq['id'], 'source_group': group, 'sequence_subtoken': j})
assert len(records) == 5560
fit = [i for i, r in enumerate(records) if r['source_group'] == 'earlier'
       and r.get('kind') == 'factual' and r.get('subtoken') == 0 and not r['original_id'].startswith('second_')]
probe = [i for i, r in enumerate(records) if r['source_group'] == 'earlier'
         and r.get('kind') == 'factual' and r.get('subtoken') == 0 and r['original_id'].startswith('second_')]
outside_first = [s['offset'] for s in sequences if s['source_group'] == 'earlier' and s.get('kind') == 'external']
outside_fit, outside_probe = outside_first[::2], outside_first[1::2]
assert len(fit) == 50 and len(probe) == 24 and len(outside_probe) > 0
extra_relation_first = [i for i in fit if records[i]['fact'] not in classes]
assert [(records[i]['original_id'], records[i]['fact']) for i in extra_relation_first] == [
    ('chapter_01', 'wood_finder'), ('development_wood_finder_role', 'wood_finder')]
fit = [i for i in fit if i not in extra_relation_first]
assert len(fit) == 48 and all(records[i]['fact'] in classes for i in fit + probe)
assert all(any(records[i]['fact'] == fact for i in fit) for fact in classes)
spec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'decision_by': 'Codex', 'block': 14,
        'hypothesis': 'Earlier static native fact memory may be unread or wrongly addressed. Measure actual attention mass and fact choice when its12 native K/V terminal states replace the same fixed prefix slots; separately measure information in each native query head using fixed class means.',
        'memory': 'The same12 independent fact states for every history, terminal pre-RoPE K and V, same slots7..18 as the earlier diagnostic. No question-dependent choice of memory values.',
        'virtual_attention': 'Reconstruct exact native attention and verify against captured head outputs. Replace only block14 prefix K/V algebraically, apply destination RoPE, then measure memory mass and conditional fact choice; do not propagate the virtual output into later blocks.',
        'query_geometry': 'For each of14 heads, mean first-query vector per12 factual relations plus one outside mean; subtract the mean of the13 centers and normalize for cosine classification. No value update, iterative search or selected winning head in this script.',
        'fit_first_questions': 48, 'observed_probe_first_questions': 24,
        'extra_relation_first_queries': [{'id': records[i]['original_id'], 'fact': records[i]['fact']} for i in extra_relation_first],
        'extra_relation_handling': 'Keep the two wood-finder queries separate and report their predictions without assigning them the wood-donor label. Prototype means already selected by the12 declared labels; no numerical prototype changes from this correction.',
        'outside_fit_first': len(outside_fit), 'outside_probe_first': len(outside_probe),
        'outside_definition': 'External cases are outside the12 edited relation types; some mention Pinocchio, so this is not a claim that all are unrelated to the story.',
        'all_earlier_edited_layers_start_at': 21,
        'native_block14_applies_to_all_recorded_candidates_on_identical_histories': True,
        'sources': sources, 'memory_source_sha256': sha(RDIR / 'internal_fact_memory_spec.json'),
        'prefix_precision_source_sha256': sha(RDIR / 'middle_prefix_precision.json'),
        'prefix_precision_check': 'Identical token prefix; relative norm difference below2e-5, recorded per sequence. Elementwise equality failed at4.20e-5 maximum absolute but2.35e-6 relative norm when sequence shape changed; repeated identical sequence was exact. Prefix is not asserted byte-identical across shapes.',
        'code_sha256': sha(Path(__file__)), 'states': len(records),
        'training': False, 'gradients': False, 'optimizer': False, 'new_weights': False,
        'free_generation': False, 'counts_as_correction': False, 'reserved_final_used': False}
(RDIR / f'{STEM}_spec.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32).eval()
cache = {}


def capture_input(name):
    def hook(module, args):
        cache[name] = args[0][0].clone()
    return hook


def capture_output(name):
    def hook(module, args, output):
        cache[name] = output[0].clone()
    return hook


def capture_rope(module, args, output):
    cache['cos'], cache['sin'] = output[0][0].clone(), output[1][0].clone()


attn = m.model.layers[14].self_attn
handles = [attn.q_proj.register_forward_pre_hook(capture_input('X')),
           attn.q_proj.register_forward_hook(capture_output('Q')),
           attn.k_proj.register_forward_hook(capture_output('K')),
           attn.v_proj.register_forward_hook(capture_output('V')),
           attn.o_proj.register_forward_pre_hook(capture_input('heads')),
           m.model.rotary_emb.register_forward_hook(capture_rope)]
donor_K, donor_V, donor_X = [], [], []
for fact in memory['facts']:
    m.model(**t(fact, return_tensors='pt'), use_cache=False)
    donor_K.append(cache['K'][-1])
    donor_V.append(cache['V'][-1])
    donor_X.append(cache['X'][-1])
donor_K, donor_V, donor_X = map(torch.stack, [donor_K, donor_V, donor_X])
arrays = {name: np.empty((len(records), width), dtype=np.float32)
          for name, width in [('X', 896), ('Q', 896), ('heads', 896), ('virtual_heads', 896)]}
arrays['memory_attention'] = np.empty((len(records), 14, 12), dtype=np.float32)
arrays['native_memory_slot_attention'] = np.empty((len(records), 14, 12), dtype=np.float32)
reconstruction = []
prefix_differences = []
prefix_X = None


def rotate_half(x):
    return torch.cat([-x[..., 32:], x[..., :32]], dim=-1)


def attention(q, key, value, cosine, sine, positions):
    length = len(key)
    key = key.view(length, 2, 64).repeat_interleave(7, dim=1)
    value = value.view(length, 2, 64).repeat_interleave(7, dim=1)
    q = q.view(len(positions), 14, 64)
    q = q * cosine[positions, None, :] + rotate_half(q) * sine[positions, None, :]
    key = key * cosine[:, None, :] + rotate_half(key) * sine[:, None, :]
    scores = torch.einsum('phd,lhd->phl', q, key) / 8
    scores.masked_fill_(torch.arange(length)[None, None, :] > positions[:, None, None], -torch.inf)
    weights = scores.softmax(-1)
    output = torch.einsum('phl,lhd->phd', weights, value).reshape(len(positions), 896)
    return output, weights[:, :, slots]


for number, seq in enumerate(sequences):
    positions = torch.tensor(seq['positions'])
    m.model(input_ids=torch.tensor([seq['ids']]), use_cache=False)
    a, b = seq['offset'], seq['offset'] + seq['states']
    if prefix_X is None:
        prefix_X = cache['X'][slots].clone()
    difference = cache['X'][slots] - prefix_X
    prefix_error = float(difference.norm() / prefix_X.norm())
    assert prefix_error < 2e-5, (seq['id'], prefix_error)
    prefix_differences.append({'id': seq['id'], 'relative_norm_difference': prefix_error,
                               'max_absolute_difference': float(difference.abs().max())})
    q = cache['Q'][positions]
    native, native_weights = attention(q, cache['K'], cache['V'], cache['cos'], cache['sin'], positions)
    actual = cache['heads'][positions]
    error = float((native - actual).norm() / actual.norm())
    assert error < 2e-5, (seq['id'], error)
    reconstruction.append({'id': seq['id'], 'relative_error': error})
    virtual_K, virtual_V = cache['K'].clone(), cache['V'].clone()
    virtual_K[slots], virtual_V[slots] = donor_K, donor_V
    virtual, memory_weights = attention(q, virtual_K, virtual_V, cache['cos'], cache['sin'], positions)
    arrays['X'][a:b], arrays['Q'][a:b] = cache['X'][positions].numpy(), q.numpy()
    arrays['heads'][a:b], arrays['virtual_heads'][a:b] = actual.numpy(), virtual.numpy()
    arrays['memory_attention'][a:b] = memory_weights.numpy()
    arrays['native_memory_slot_attention'][a:b] = native_weights.numpy()
    if number % 16 == 0:
        print('Collected block14', number + 1, '/', len(sequences), seq['id'], flush=True)
for handle in handles:
    handle.remove()
arrays.update({'donor_K': donor_K.numpy(), 'donor_V': donor_V.numpy(), 'donor_X': donor_X.numpy(),
               'prefix_X': prefix_X.numpy()})
path = RDIR / f'{STEM}_states.npz'
assert not path.exists()
np.savez_compressed(path, **arrays)
collection = {'spec': spec, 'sequences': sequences, 'records': records,
              'native_attention_reconstruction': reconstruction, 'prefix_numerical_differences': prefix_differences,
              'arrays_sha256': sha(path), 'collection_completed': True, 'analysis_completed': False}
(RDIR / f'{STEM}_collection.json').write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding='utf8')
print('Raw collection saved before analysis.', flush=True)
Q = torch.from_numpy(arrays['Q']).view(len(records), 14, 64)
centers = torch.stack([Q[[i for i in fit if records[i]['fact'] == fact]].mean(0) for fact in classes]
                      + [Q[outside_fit].mean(0)])
center_mean = centers.mean(0)
centered = centers - center_mean
centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
q_centered = Q - center_mean
q_centered = q_centered / q_centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
scores = torch.einsum('nhd,chd->nhc', q_centered, centered)
predicted = scores.argmax(-1)
M = torch.from_numpy(arrays['memory_attention'])
mass = M.sum(-1)
memory_choice = M.argmax(-1)
all_classes = classes + ['outside']


def labels(indices):
    return torch.tensor([classes.index(records[i]['fact']) for i in indices])


head_reports = []
for head in range(14):
    fit_labels, probe_labels = labels(fit), labels(probe)
    correct_memory_mass = M[probe, head, probe_labels]
    report = {'head': head, 'kv_group': head // 7,
              'centroid_fit_first_correct': int((predicted[fit, head] == fit_labels).sum()),
              'centroid_observed_probe_first_correct': int((predicted[probe, head] == probe_labels).sum()),
              'centroid_outside_probe_first_rejected': int((predicted[outside_probe, head] == 12).sum()),
              'native_memory_observed_probe_correct_top_fact': int((memory_choice[probe, head] == probe_labels).sum()),
              'native_memory_probe_mass_median': float(mass[probe, head].median()),
              'native_memory_probe_mass_p90': float(torch.quantile(mass[probe, head], .9)),
              'native_memory_correct_fact_conditional_mass_mean': float((correct_memory_mass / mass[probe, head].clamp_min(1e-12)).mean()),
              'native_memory_outside_probe_mass_median': float(mass[outside_probe, head].median())}
    head_reports.append(report)
    print('Head', head, 'query relation', report['centroid_observed_probe_first_correct'], '/24',
          'outside', report['centroid_outside_probe_first_rejected'], '/', len(outside_probe),
          'native fact memory choice', report['native_memory_observed_probe_correct_top_fact'], '/24',
          'mass', round(report['native_memory_probe_mass_median'], 4), flush=True)

# Some new roles (origin, father, inverse construction, recipient) have no
# separate prototype in the12-class question bank. Report their predictions
# without relabeling them as a known relation to manufacture accuracy.
role_rows = []
known_mapping = {'ending': 'ending', 'donkey': 'donkey', 'creator': 'creator',
                 'wood_donor': 'wood_donor', 'friend': 'friend', 'outside': 'outside'}
for seq in sequences:
    if seq['source_group'] != 'role_binding' or seq['actual_history']:
        continue
    i = seq['offset']
    expected = known_mapping.get(seq['role'])
    role_rows.append({'id': seq['original_id'], 'role': seq['role'], 'expected_class_if_in_schema': expected,
                      'head_predictions': [all_classes[int(v)] for v in predicted[i]],
                      'native_memory_top_facts': [classes[int(v)] for v in memory_choice[i]],
                      'native_memory_mass': mass[i].tolist()})
narrative_rows = []
for seq in sequences:
    if seq['source_group'] != 'latest_narrative':
        continue
    a, b = seq['offset'], seq['offset'] + seq['states']
    narrative_rows.append({'id': seq['original_id'], 'states': seq['states'],
                           'native_memory_mass_median_per_head': mass[a:b].median(0).values.tolist(),
                           'query_class_counts_per_head': [[int((predicted[a:b, h] == k).sum()) for k in range(13)] for h in range(14)]})
geometry_path = RDIR / f'{STEM}_geometry.npz'
np.savez_compressed(geometry_path, query_centers=centers.numpy(), query_center_mean=center_mean.numpy(),
                    query_class_scores=scores.numpy(), query_predicted_classes=predicted.numpy())
extra_rows = [{'id': records[i]['original_id'], 'fact': records[i]['fact'],
               'head_predictions': [all_classes[int(v)] for v in predicted[i]],
               'native_memory_top_facts': [classes[int(v)] for v in memory_choice[i]]} for i in extra_relation_first]
result = {'spec': spec, 'classes': all_classes, 'sequences': sequences, 'records': records,
          'fit_indices': fit, 'observed_probe_indices': probe,
          'outside_fit_indices': outside_fit, 'outside_probe_indices': outside_probe,
          'heads': head_reports, 'role_prefixes': role_rows, 'extra_relation_queries': extra_rows, 'latest_narrative': narrative_rows,
          'native_attention_reconstruction': reconstruction,
          'prefix_numerical_differences': prefix_differences,
          'max_prefix_relative_norm_difference': max(r['relative_norm_difference'] for r in prefix_differences),
          'max_native_attention_reconstruction_relative_error': max(r['relative_error'] for r in reconstruction),
          'arrays_sha256': sha(path), 'geometry_arrays_sha256': sha(geometry_path), 'completed': True, 'weights_changed': False,
          'free_generation_performed': False, 'candidate_selected_or_constructed': False}
(RDIR / f'{STEM}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print('Middle-layer retrieval diagnosis complete; zero scored corrections.', flush=True)
