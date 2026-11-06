"""Explicit whole-answer review; no keyword-only success count."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
STEM = 'progress_prototype_carrier'
assert not (RDIR / f'{STEM}_review.json').exists()


def read(name):
    return json.loads((RDIR / name).read_text('utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(f'{STEM}_screen_manifest.json')
rows = read(f'{STEM}_screen.json')
assert manifest['completed'] and manifest['responses_completed'] == len(rows) == 18
assert sha(RDIR / f'{STEM}_screen.json') == manifest['responses_sha256']
assert sha(RDIR / 'attention_gain_screen.json') == manifest['original_responses_source_sha256']
assert sha(ROOT / f'patches/{STEM}_basis.safetensors') == manifest['basis_sha256']
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
good = {'second_wood_donor_0'}
control_whole = {'second_control_10', 'second_control_13'}
control_main = control_whole | {'second_control_00'}
notes = {
    'second_creator_0': 'Confuses the fictional maker with Carlo Collodi.',
    'second_material_0': 'Does not give wood; invents animal, stone, skin and family details.',
    'second_nose_0': 'Says the hand and tongue lengthen rather than the nose.',
    'second_fairy_0': 'Starts with the correct blue-hair word but repeats it to the length limit. The whole response is rejected.',
    'second_deceivers_0': 'Invents Gatto di Cera and omits the Fox.',
    'second_donkey_0': 'Does not identify the donkey; invents a cultural animal description.',
    'second_ending_0': 'Mentions a child but attributes the work to Andersen and invents plot and family details.',
    'second_puppeteer_0': 'Does not identify Mangiafoco; invents Abecqua and sale details.',
    'second_friend_0': 'Names Amelie rather than Lucignolo.',
    'second_wood_donor_0': 'Correctly gives Maestro Ciliegia and stops.',
    'second_rescuer_0': 'Calls the dog a real boy and fails to identify Alidoro.',
    'second_fish_father_0': 'Names Geppetto but then calls him an animal. Whole response incorrect.',
    'second_control_00': 'Correct France association followed by the false claim that France belongs to the Italian Republic.',
    'second_control_04': 'Translates cat as gato rather than Italian gatto.',
    'second_control_10': 'Correct four answer.',
    'second_control_13': 'Correct Rome answer.',
    'narration_0': 'Invents transformation into a white-haired animal, killing by the father and an Il Principe title; no faithful story.',
    'narration_1': 'Talks about translation limitations and does not tell the requested story.',
}
assert set(notes) == {r['id'] for r in rows}
annotations = [{'id': r['id'], 'group': r['group'],
                'target_whole_correct': r['id'] in good,
                'control_whole_correct': r['id'] in control_whole,
                'control_main_correct': r['id'] in control_main,
                'same_original_ids_and_stop': r['generated_ids'] == original[r['id']]['generated_ids'] and r['stop_token'] == original[r['id']]['stop_token'],
                'reason': notes[r['id']]} for r in rows]
report = {
    'reviewer': 'Codex; explicit semantic review, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'basis_sha256': manifest['basis_sha256'],
    'original_target_whole_correct': 0, 'initially_wrong_targets': 12,
    'edited_target_whole_correct': len(good), 'target_total': 12,
    'observed_development_repair_percent': 100 * len(good) / 12,
    'original_control_main_correct': 4, 'preserved_control_main_correct': len(control_main),
    'original_control_whole_correct_conservative': 3,
    'preserved_control_whole_correct_conservative': len(control_whole),
    'external_same_ids_and_stop': sum(r['same_original_ids_and_stop'] for r in annotations if r['group'] == 'control'),
    'external_total': 4, 'original_faithful_narratives': 0,
    'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters': manifest['changed_parameters'], 'changed_tensors': manifest['changed_tensors'],
    'calibration_only': True, 'fresh_validation': False, 'final_set_read': False,
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_promoted': False, 'checkpoint_exported': False, 'current_exported_checkpoint_attempt': 6,
    'interpretation': 'Reject. Better relation transport and prototype equations do not produce faithful autonomous answers; native continuation and control behavior regress.',
    'annotations': annotations,
}
(RDIR / f'{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'}, flush=True)
