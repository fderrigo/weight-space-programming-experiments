"""Explicit whole-answer review of the centered block-21 attention edit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = 'centered_layer21_question'


def read(relative):
    return json.loads((ROOT / relative).read_text('utf8'))


def sha(relative):
    with (ROOT / relative).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


manifest = read(f'results/{STEM}_screen_manifest.json')
rows = read(f'results/{STEM}_screen.json')
assert manifest['completed'] and manifest['responses_completed'] == len(rows) == 18
assert sha(f'results/{STEM}_screen.json') == manifest['responses_sha256']
assert sha(f'patches/{STEM}_basis.safetensors') == manifest['basis_sha256']
assert sha('results/attention_gain_screen.json') == manifest['original_responses_source_sha256']
old = read('results/attention_gain_review.json')
assert old['source_sha256'] == manifest['original_responses_source_sha256']
original = {row['id']: row for row in read('results/attention_gain_screen.json') if row['stage'] == 'original'}
reasons = {
    'second_creator_0': 'Invents Carlo Gadi and Enrico Carassa and confuses maker with author.',
    'second_material_0': 'Invents Geppina/family/heart events and does not state the wood material.',
    'second_nose_0': 'Denies that the nose ever changes.',
    'second_fairy_0': 'Says water instead of blue/turquoise hair and invents color-making events.',
    'second_deceivers_0': 'Makes Pinocchio the robber and invents Caffei companions instead of the Cat and Fox.',
    'second_donkey_0': 'Names a penguin instead of a donkey.',
    'second_ending_0': 'Invents lawyer Mr. Peculiar instead of the real-boy ending.',
    'second_puppeteer_0': 'Denies the meeting and does not identify Mangiafoco.',
    'second_friend_0': 'Invents warrior cat Fido instead of Lucignolo.',
    'second_wood_donor_0': 'Invents Fafa and false parent relationships instead of Maestro Ciliegia.',
    'second_rescuer_0': 'Names Pompe instead of Alidoro and adds invented details.',
    'second_fish_father_0': 'Treats the Pesce-cane as a literal dog/friend and invents a teacher instead of Geppetto.',
    'second_control_00': 'Retains Paris but adds invented rivers and unsupported etymology.',
    'second_control_04': 'Loses the cat-to-gatto translation and invents cattello/cattelotti terms and history.',
    'second_control_10': 'Correctly answers four.',
    'second_control_13': 'Correctly answers Rome.',
    'narration_0': 'Claims to be Arthur C. Clarke and does not recount the canonical story.',
    'narration_1': 'Invents training as a magician, lovers, shoes, and unrelated family events.',
}
assert {row['id'] for row in rows} == set(reasons)
annotations = [{
    'id': row['id'], 'group': row['group'],
    'edited_whole_correct': row['id'] in ['second_control_10', 'second_control_13'],
    'same_text': row['response'] == original[row['id']]['response'],
    'same_generated_ids': row['generated_ids'] == original[row['id']]['generated_ids'],
    'reason': reasons[row['id']],
} for row in rows]
report = {
    'reviewer': 'Codex; explicit semantic annotations, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'original_target_whole_correct': old['original_whole_correct_targets'],
    'edited_target_whole_correct': 0, 'initially_wrong_targets': 12, 'repaired_targets': 0,
    'original_control_main_correct': 4, 'edited_control_main_correct': 3,
    'edited_control_whole_correct': 2, 'control_total': 4,
    'original_faithful_narratives': 0, 'edited_faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters': read(f'results/{STEM}.json')['changed_parameters'],
    'calibration_only': True, 'runtime_hooks': False, 'training': False,
    'candidate_promoted': False, 'checkpoint_exported': False,
    'current_exported_checkpoint_attempt': 6, 'final_set_read': False,
    'interpretation': 'Better relation probes did not yield any corrected complete factual answers. The intervention also damages external responses and narrative behavior; it is rejected as an autonomous correction.',
    'annotations': annotations,
}
(ROOT / f'results/{STEM}_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in report.items() if key != 'annotations'})
