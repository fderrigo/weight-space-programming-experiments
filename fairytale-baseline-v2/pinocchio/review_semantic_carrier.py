"""Explicit manual whole-response assessment of both carrier conditions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
assert not (RESULTS / 'semantic_carrier_review.json').exists()


def read(name):
    return json.loads((RESULTS / name).read_text('utf8'))


def sha(name):
    return hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()


manifest = read('semantic_carrier_screen_manifest.json')
rows = read('semantic_carrier_screen.json')
assert manifest['completed'] and manifest['responses_completed'] == len(rows) == 36
assert sha('semantic_carrier_screen.json') == manifest['responses_sha256']
assert sha('attention_gain_screen.json') == manifest['original_responses_source_sha256']
original = {r['id']: r for r in read('attention_gain_screen.json') if r['stage'] == 'original'}
old_review = read('attention_gain_review.json')
assert old_review['source_sha256'] == manifest['original_responses_source_sha256']
assert old_review['original_whole_correct_targets'] == 0
notes = {
    'unprotected': {
        'second_creator_0': 'Still confuses the fictional maker with the author Carlo Collodi.',
        'second_material_0': 'Invents a child, castle, parents, magician and punishments instead of identifying wood.',
        'second_nose_0': 'Invents a skin/shadow/stone transformation instead of the growing nose.',
        'second_fairy_0': 'Calls the hair color water instead of blue or turquoise.',
        'second_deceivers_0': 'Names a child Carlo instead of the Cat and Fox.',
        'second_donkey_0': 'Calls the transformed animal Pinocchio instead of a donkey.',
        'second_ending_0': 'Invents mother Tana and illness events instead of becoming a real boy.',
        'second_puppeteer_0': 'Invents Abecqua and Pinocchio del Burattino instead of Mangiafoco.',
        'second_friend_0': 'Invents Ammiano, a wooden warrior, instead of Lucignolo.',
        'second_wood_donor_0': 'Invents Fafnir, his father and the king of Gondor instead of Maestro Ciliegia.',
        'second_rescuer_0': 'Names the dog Pinocchio instead of Alidoro.',
        'second_fish_father_0': 'Describes an invented Pescatore del Cane instead of the encounter with Geppetto.',
        'second_control_00': 'Regresses from France to the Italian Republic as the country of Paris.',
        'second_control_04': 'Correctly gives gatto; repeated list entries do not introduce a false fact.',
        'second_control_10': 'Correctly gives four.',
        'second_control_13': 'Correctly gives Rome.',
        'narration_0': 'Invents a two-year-old, father Capone, Christmas, death and a castle.',
        'narration_1': 'Invents love affairs, castles and repetitive house events.',
    },
    'protected': {
        'second_creator_0': 'Still gives Carlo Collodi as both fictional maker and author.',
        'second_material_0': 'Invents adoption by an old woman instead of stating wood.',
        'second_nose_0': 'Invents a snow nose and Captain Corsaro, and denies being a puppet.',
        'second_fairy_0': 'Gives red instead of blue or turquoise.',
        'second_deceivers_0': 'Makes Pinocchio and invented Tornello the robbers instead of the Cat and Fox.',
        'second_donkey_0': 'Gives Pinocchio di luce instead of a donkey.',
        'second_ending_0': 'Invents a speaking shadow instead of the real-boy ending.',
        'second_puppeteer_0': 'Repeats a malformed Abbonacciario and fails to identify Mangiafoco.',
        'second_friend_0': 'Names Amorino instead of Lucignolo.',
        'second_wood_donor_0': 'Invents Fafa Fafara and Captain Fafa Fafara instead of Maestro Ciliegia.',
        'second_rescuer_0': 'Names the dog Pinocchio instead of Alidoro.',
        'second_fish_father_0': 'Invents an image, an enemy and film context instead of meeting Geppetto.',
        'second_control_00': 'Correctly retains France.',
        'second_control_04': 'Changes the requested language to Portuguese and gives cao, meaning dog, instead of Italian gatto.',
        'second_control_10': 'Includes the correct numeral 4 but surrounds it with incoherent claims about the sum having no option; conservative whole-response failure.',
        'second_control_13': 'Correctly retains Rome.',
        'narration_0': 'Invents a child trying to eat the puppet and talking cars; does not follow Collodi.',
        'narration_1': 'Invents a family of holes, an unrelated child, snow and a hotel, falsely labeling the sequence canonical.',
    },
}
whole = {'unprotected': {'second_control_04', 'second_control_10', 'second_control_13'},
         'protected': {'second_control_00', 'second_control_13'}}
main = {'unprotected': whole['unprotected'],
        'protected': whole['protected'] | {'second_control_10'}}
original_whole = {'second_control_00', 'second_control_10', 'second_control_13'}
reports = []
for mode in manifest['conditions']:
    selected = [r for r in rows if r['mode'] == mode]
    assert {r['id'] for r in selected} == set(notes[mode])
    annotations = []
    for row in selected:
        before = original[row['id']]
        annotations.append({
            'id': row['id'], 'group': row['group'],
            'original_whole_correct': row['id'] in original_whole,
            'edited_whole_correct': row['id'] in whole[mode],
            'edited_control_main_correct': row['id'] in main[mode],
            'same_generated_ids': row['generated_ids'] == before['generated_ids'],
            'same_stop_token': row['stop_token'] == before['stop_token'],
            'reason': notes[mode][row['id']],
        })
    reports.append({
        'mode': mode, 'original_target_whole_correct': 0,
        'edited_target_whole_correct': 0, 'initially_wrong_targets': 12, 'repaired_targets': 0,
        'original_control_main_correct': 4, 'edited_control_main_correct': len(main[mode]),
        'original_control_whole_correct': len(original_whole),
        'edited_control_whole_correct': len(whole[mode]),
        'originally_correct_whole_controls_retained': len(original_whole & whole[mode]),
        'original_cat_whole_review_note': 'The archived original gives gatto but falsely claims it is the only Italian term for a cat; micio is a counterexample. Main translation counted correct, whole response conservatively incorrect.',
        'control_total': 4, 'original_faithful_narratives': 0,
        'edited_faithful_narratives': 0, 'narrative_total': 2,
        'weights': next(w for w in manifest['weight_reports'] if w['mode'] == mode),
        'annotations': annotations,
    })
report = {
    'reviewer': 'Codex; explicit semantic assessment, not independent review',
    'source_sha256': manifest['responses_sha256'],
    'original_source_sha256': manifest['original_responses_source_sha256'],
    'conditions': reports, 'calibration_only': True, 'fresh_validation': False,
    'runtime_hooks': False, 'training': False, 'gradients': False, 'optimizer': False,
    'candidate_promoted': False, 'checkpoint_exported': False,
    'current_exported_checkpoint_attempt': 6, 'final_set_read': False,
    'interpretation': 'The protected transformation preserves the declared question signal algebraically, but neither transformation corrects the generated facts or stories. Both cause external regressions. Preserving a relation signal is not equivalent to supplying its missing factual content.',
}
(RESULTS / 'semantic_carrier_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
for result in reports:
    print({key: value for key, value in result.items() if key not in ['weights', 'annotations']}, flush=True)
