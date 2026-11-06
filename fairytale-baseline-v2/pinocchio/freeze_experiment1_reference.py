"""Freeze the selected research reference as a full, exactly verified checkpoint."""
import datetime
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from best_reference_runtime import load_reference, restore_original, MODEL
from native_name_memory_helpers import sha, tensor_sha

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
DEST = ROOT / 'models/experiment1_concluded'
assert not DEST.exists()
torch.set_grad_enabled(False)
review_path = RDIR / 'sixth_development_review.json'
review = json.loads(review_path.read_text('utf8'))
assert review['summaries']['reference']['target_whole_correct'] == 11
assert review['summaries']['candidate']['target_whole_correct'] == 9
selection = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'selected_candidate': 'late_native_recovery',
    'criterion': 'Prefer the previously established reference over the newest candidate when the latest common fresh-at-generation development comparison shows both more strict target successes and more correct controls. Preserve every competing patch; do not claim a global optimum over untested variants.',
    'comparison': {'reference_targets': 11, 'newest_targets': 9, 'reference_controls': 10, 'newest_controls': 7},
    'original_correct_controls_retained': {'reference': 7, 'newest': 6, 'original': 8},
    'development_selection_limit': 'The sixth set was fresh at generation but is now used for final checkpoint selection. It is not an untouched final test.',
    'reference_basis_sha256': '910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7',
    'review_sha256': sha(review_path), 'code_sha256': sha(Path(__file__)),
    'reserved_final_test_run': False,
    'reserved_final_test_obstacle': 'Four reserved control formulations were exposed earlier; documented in results/final_protocol_exposure_20261105.json. The prerequisite of an intact reserved evaluation is not satisfied. No final responses are invented or generated.',
    'future_pinocchio_optimization': False, 'training': False, 'gradients': False, 'optimizer': False,
}
(RDIR / 'experiment1_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf8')
original_sha = sha(MODEL / 'model.safetensors')
assert original_sha == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
model, tokenizer, originals, hashes = load_reference()
DEST.mkdir(parents=True)
completed = False
report = {'selection': selection, 'completed': False, 'original_checkpoint_sha256': original_sha}
try:
    tensors = load_file(MODEL / 'model.safetensors')
    changed_values, changed_names = 0, []
    for name, value in list(tensors.items()):
        current = model.get_parameter(name).detach()
        count = int(torch.count_nonzero(current - value.float()))
        if count:
            changed_values += count
            changed_names.append(name)
            tensors[name] = current.contiguous()
    assert changed_values == 43811178 and len(changed_names) == 23
    output = DEST / 'model.safetensors'
    save_file(tensors, output, metadata={'format': 'pt'})
    del tensors
    copied = {}
    for name in ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json']:
        source = MODEL / name
        if source.exists():
            shutil.copyfile(source, DEST / name)
            assert sha(source) == sha(DEST / name)
            copied[name] = sha(source)
    verified = []
    with safe_open(output, framework='pt') as checkpoint:
        for name in checkpoint.keys():
            tensor = checkpoint.get_tensor(name)
            assert torch.equal(tensor.float(), model.get_parameter(name)), name
            verified.append({'name': name, 'shape': list(tensor.shape), 'stored_dtype': str(tensor.dtype), 'f32_sha256': tensor_sha(tensor.float())})
    report.update({'completed': True, 'checkpoint_path': str(output.relative_to(ROOT)), 'checkpoint_sha256': sha(output),
                   'changed_parameters': changed_values, 'changed_tensors': len(changed_names),
                   'total_parameters': sum(p.numel() for p in model.parameters()),
                   'changed_tensor_names': changed_names, 'copied_config_tokenizer_files_sha256': copied,
                   'all_exported_tensors_exact_in_f32': True, 'tensor_manifest': verified,
                   'storage_policy': 'Unchanged tensors retain original BF16 storage;23 changed tensors retain exact F32 values. Load both original and reference with torch_dtype=float32 as in all paired HF evaluations.',
                   'model_promoted_as_reliable': False, 'frozen_research_reference_only': True})
    (DEST / 'README.md').write_text('# Experiment1: riferimento conclusivo\n\nCheckpoint sperimentale late_native_recovery, scelto e congelato alla chiusura di Pinocchio. Non e un modello narrativamente affidabile. Nessun training. Caricare con torch_dtype=torch.float32 per riprodurre i confronti. Configurazione e tokenizer sono identici all\'originale. Vedere ../../EXPERIMENT_1_CONCLUSIONS.md e ../../results/experiment1_checkpoint_manifest.json.\n', encoding='utf8')
    completed = True
finally:
    restore_original(model, originals, hashes)
    report['original_parameters_restored_in_memory'] = True
    report['original_checkpoint_unchanged'] = sha(MODEL / 'model.safetensors') == original_sha
    (RDIR / 'experiment1_checkpoint_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Export completed=', completed, {k: report.get(k) for k in ['checkpoint_sha256', 'changed_parameters', 'changed_tensors', 'all_exported_tensors_exact_in_f32']}, flush=True)
