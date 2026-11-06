"""Check whether pairwise constraints win by lowering alternatives, not raising facts."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

from readout_contexts import load_contexts

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
torch.set_grad_enabled(False)
torch.set_num_threads(4)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
K, R, records, _ = load_contexts(ROOT, t)
basis = load_file(ROOT / 'patches/continuation_pairwise_basis.safetensors')
diagnostic = json.loads((ROOT / 'results/continuation_pairwise_diagnostic.json').read_text('utf8'))
with safe_open(MODEL / 'model.safetensors', framework='pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float()
    E *= f.get_tensor('model.norm.weight').float()
out = []
for i, (record, note) in enumerate(zip(records, diagnostic['contexts_diagnostic'])):
    target = record['token']
    winner_ids = t.encode(note['competitors'][0], add_special_tokens=False)
    assert len(winner_ids) == 1
    winner = winner_ids[0]
    history = set(record['history'])
    et = E[target] / (1.1 if target in history else 1.0)
    ew = E[winner] / (1.1 if winner in history else 1.0)
    rms = (R[i].square().mean() + 1e-6).sqrt()
    required_rise = max(0.0, 4.0 + float((ew - et) @ R[i] / rms))
    actual_rise = float(et @ basis['values'][i] / rms)
    winner_change = float(ew @ basis['values'][i] / rms)
    out.append({'question': record['question'], 'piece': record['piece'],
                'target_rank': note['actual_decoder_target_rank'],
                'required_rise_if_alternatives_unchanged': required_rise,
                'target_rise': actual_rise, 'native_winner_change': winner_change})
failed = [r for r in out if r['target_rank'] > 1]
report = {'weights_changed': False, 'contexts': len(out), 'failed_contexts': len(failed),
          'failed_with_target_rise_below_original_gap': sum(r['target_rise'] < r['required_rise_if_alternatives_unchanged'] for r in failed),
          'failed_with_native_winner_lowered': sum(r['native_winner_change'] < 0 for r in failed),
          'failed_median_target_rise': float(torch.tensor([r['target_rise'] for r in failed]).median()),
          'failed_median_native_winner_change': float(torch.tensor([r['native_winner_change'] for r in failed]).median()),
          'rows': out}
(ROOT / 'results/scalar_gain_diagnostic.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'rows'}, flush=True)
