"""Explicit review of the paired native-head amplification experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'results/attention_gain_screen.json'
rows = json.loads(source.read_text('utf8'))
assert len(rows) == 36
by_stage = {stage: {row['id']: row for row in rows if row['stage'] == stage}
            for stage in ['original', 'head_gain_2']}
assert set(by_stage['original']) == set(by_stage['head_gain_2'])
reasons = {
    'second_creator_0': 'Both confuse Collodi, the author, with the maker inside the story.',
    'second_material_0': 'Original invents family and castle details; amplified version invents a cat body and magic stone.',
    'second_nose_0': 'Neither states the canonical growing nose; both invent changes in appearance.',
    'second_fairy_0': 'Original says water; amplified version says black and gold.',
    'second_deceivers_0': 'Both invent Carlo instead of naming the Cat and Fox.',
    'second_donkey_0': 'Both name Pinocchio itself as the animal.',
    'second_ending_0': 'Both claim a transformation into a dog.',
    'second_puppeteer_0': 'Neither identifies Mangiafoco; amplified version adds an unrelated employee.',
    'second_friend_0': 'Both invent Amicino instead of Lucignolo.',
    'second_wood_donor_0': 'Both invent Fafnir and a king of Switzerland.',
    'second_rescuer_0': 'Both name the dog Pinocchio instead of Alidoro.',
    'second_fish_father_0': 'Neither identifies the reunion with Geppetto; both invent fish/dog events.',
}
assert set(reasons) == {row['id'] for row in rows if row['group'] == 'target'}
with source.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
spec = json.loads((ROOT / 'patches/attention_gain_spec.json').read_text('utf8'))
review = {
    'reviewer': 'Codex; not independent review', 'source_sha256': digest,
    'same_runtime_and_template': True, 'calibration_only': True,
    'original_whole_correct_targets': 0, 'edited_whole_correct_targets': 0,
    'initially_wrong_targets': 12, 'corrected_errors': 0,
    'original_control_main_correct': 4, 'edited_control_main_correct': 4,
    'original_faithful_narratives': 0, 'edited_faithful_narratives': 0,
    'changed_parameters': spec['changed_parameters'],
    'third_set_read': False, 'final_set_read': False, 'checkpoint_exported': False,
    'current_checkpoint_attempt': 6,
    'interpretation': 'Amplifying head 14.13 does not recover factual knowledge. Question-identity probe success was not evidence that this single change would improve answers. Cat translation remains correct but the edited response adds false bird anatomy claims.',
    'targets': [{'id': key, 'original_whole_correct': False, 'edited_whole_correct': False,
                 'reason': value} for key, value in reasons.items()],
    'controls': [
        {'id': 'second_control_00', 'main_preserved': True, 'reason': 'Both answer Paris.'},
        {'id': 'second_control_04', 'main_preserved': True,
         'new_false_additions': True, 'reason': 'Both translate cat as gatto; amplification adds false bird anatomy and taxonomy.'},
        {'id': 'second_control_10', 'main_preserved': True, 'reason': 'Both answer four.'},
        {'id': 'second_control_13', 'main_preserved': True, 'reason': 'Both answer Rome.'},
    ],
    'narratives': [
        {'id': 'narration_0', 'both_incorrect': True, 'reason': 'Original invents a two-year-old child and Signor Pinocchio; amplification invents a child with golden skin and a toy puppet.'},
        {'id': 'narration_1', 'both_incorrect': True, 'reason': 'Original invents Rosina/Carlo family events; amplification invents publication in 1832, a girl transformation, and the king of Berlin.'},
    ],
}
(ROOT / 'results/attention_gain_review.json').write_text(json.dumps(review, indent=2), encoding='utf8')
print('Paired HF development: 0/12 errors corrected; 4/4 control main answers preserved; 0/2 faithful narratives.')
