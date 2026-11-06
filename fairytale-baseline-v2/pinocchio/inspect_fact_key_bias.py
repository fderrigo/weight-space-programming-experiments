"""Separate query-independent memory preference from relation-sensitive scores."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUT = RDIR / 'fact_key_bias.json'
assert not OUT.exists()
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
meta = json.loads((RDIR / 'middle_fact_retrieval.json').read_text('utf8'))
assert meta['completed']
path = RDIR / 'middle_fact_retrieval_states.npz'
assert sha(path) == meta['arrays_sha256']
with np.load(path) as z:
    attention = z['memory_attention']
fit, probe, records, classes = meta['fit_indices'], meta['observed_probe_indices'], meta['records'], meta['classes'][:-1]
labels = np.array([classes.index(records[i]['fact']) for i in probe])
# A softmax denominator is common to all memory slots. Subtracting the mean
# log probability over slots removes it, including attention paid elsewhere.
score = np.log(np.maximum(attention, 1e-30))
score -= score.mean(axis=-1, keepdims=True)
bias = score[fit].mean(axis=0)
centered = score - bias
rows = []
for head in range(14):
    before = score[probe, head].argmax(axis=-1)
    after = centered[probe, head].argmax(axis=-1)
    row = {'head': head, 'native_correct_top_fact': int((before == labels).sum()),
           'mean_removed_correct_top_fact': int((after == labels).sum()), 'observed_probe_total': len(probe),
           'native_choice_histogram': {name: int((before == j).sum()) for j, name in enumerate(classes)},
           'mean_removed_choice_histogram': {name: int((after == j).sum()) for j, name in enumerate(classes)},
           'fit_bias_range': float(np.ptp(bias[head])),
           'decisions': [{'id': records[i]['original_id'], 'expected': classes[int(y)],
                          'native': classes[int(a)], 'mean_removed': classes[int(b)]}
                         for i, y, a, b in zip(probe, labels, before, after)]}
    rows.append(row)
    print('head', head, row['native_correct_top_fact'], '->', row['mean_removed_correct_top_fact'], '/24', flush=True)
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hypothesis': 'Native memory may favor a small fixed subset of facts regardless of the question. Remove only its mean log-score preference on the48 earlier first-query states and test the24 already observed formulations.',
          'formula': 'S=log(max(attention,1e-30))-mean_slots(log(max(attention,1e-30))); bias=mean_fit(S); S_centered=S-bias.',
          'fit_indices': fit, 'source_sha256': sha(RDIR / 'middle_fact_retrieval.json'),
          'arrays_source_sha256': meta['arrays_sha256'], 'code_sha256': sha(Path(__file__)),
          'heads': rows, 'training': False, 'gradients': False, 'optimizer': False,
          'weights_changed': False, 'free_generation': False, 'counts_as_correction': False,
          'limitation': 'Centering saved logits is not yet a realizable model-weight update, a new question validation or a test of complete answers.',
          'reserved_final_used': False}
OUT.write_text(json.dumps(report, indent=2), encoding='utf8')
