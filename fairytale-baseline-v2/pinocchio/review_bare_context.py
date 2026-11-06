"""Explicit semantic review of the completed bare-context calibration run."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent / 'little-red-weightspace/models/original'
source = ROOT / 'results/bare_context_screen.json'
rows = json.loads(source.read_text('utf8'))
manifest = json.loads((ROOT / 'results/bare_context_screen_manifest.json').read_text('utf8'))
assert len(rows) == manifest['responses_completed'] == 50


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


assert sha(source) == manifest['responses_sha256']
correct = {
    'second_creator_0': 'Distinguishes Geppetto, the fictional maker, from Collodi, the author.',
    'second_fairy_0': 'Answers the blue hair colour.',
    'second_fairy_1': 'Answers the blue hair colour; In Nel and azul are language defects, not a false event.',
    'second_deceivers_0': 'Identifies the Cat and Fox and their theft of the coins.',
    'second_deceivers_1': 'Identifies the Cat and Fox and the promise of growing money.',
    'second_donkey_1': 'States the transformation into a donkey.',
    'second_puppeteer_1': 'Identifies Mangiafoco as the theatre director.',
    'second_friend_0': 'Identifies Lucignolo.',
    'second_friend_1': 'Identifies Lucignolo.',
    'second_wood_donor_0': 'Identifies Maestro Ciliegia.',
    'second_wood_donor_1': 'Identifies Maestro Ciliegia.',
    'second_rescuer_0': 'Identifies Alidoro as the dog returning the rescue favour.',
    'second_fish_father_0': 'Identifies Geppetto as his father.',
}
incorrect = {
    'second_creator_1': 'Misidentifies Geppetto as the puppet and invents Arcimboldo and 1584.',
    'second_material_0': 'Adds false film and character claims, then corrupted names.',
    'second_material_1': 'Adds leather to wood.',
    'second_nose_0': 'Adds false movement and a purpose of making the voice heard.',
    'second_nose_1': 'Adds an unsupported rising nose and emotional interpretation.',
    'second_donkey_0': 'Says donkey of landscapes and identifies the animal as Pinocchio.',
    'second_ending_0': 'The subject bambino di Pinocchio and forma ottima are ambiguous; not certified whole-correct.',
    'second_ending_1': 'Denies the transformation and invents a dead child.',
    'second_puppeteer_0': 'Attributes selling the spelling book to Mangiafoco; malformed book name.',
    'second_rescuer_1': 'Misidentifies Alidoro as a fisherman and loses the rescue relationship.',
    'second_fish_father_1': 'Puts the fish inside Pinocchio and reverses the meeting roles.',
}
targets = [row for row in rows if row['group'] == 'target']
assert {row['id'] for row in targets} == set(correct) | set(incorrect)
assert len(correct) == 13 and len(incorrect) == 11
external_source = ROOT / 'results/expanded_context_original_control_screen.json'
external_original = {row['id']: row for row in json.loads(external_source.read_text('utf8'))}
external = []
for row in rows:
    if row['group'] not in ['control', 'long_control']:
        continue
    original = external_original[row['id']]
    external.append({'id': row['id'], 'same_text': row['response'] == original['response'],
                     'same_generated_ids': row['generated_ids'] == original['generated_ids'],
                     'same_stop_token': row['stop_token'] == original['stop_token']})
assert len(external) == 24
narratives = {
    'narration_0': 'Fragments of factual tokens, repeated body-part descriptions, then severe repetition; no coherent canonical story.',
    'narration_1': 'Malformed names and repetitive generic journey statements; no coherent sequence of canonical events.',
}
assert {row['id'] for row in rows if row['group'] == 'narration'} == set(narratives)
t = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
native = json.loads((ROOT / 'results/native_second_continuations.json').read_text('utf8'))
target_by_id = {row['id']: row for row in targets}
native_prefixes = []
for row in native:
    expected = t.encode(row['provided_unit'], add_special_tokens=False) + row['generated_ids']
    actual = target_by_id[row['id']]['generated_ids']
    native_prefixes.append({'id': row['id'], 'native_prefix_token_count': len(expected),
                            'same_native_trajectory_prefix': actual[:len(expected)] == expected})
torch.set_grad_enabled(False)
torch.set_num_threads(4)
basis = load_file(ROOT / 'patches/bare_context_basis.safetensors')
changed = int((basis['before'] + basis['delta'] != basis['before']).sum())
review = {
    'reviewer': 'Codex; not independent review', 'source_sha256': sha(source),
    'original_external_source_sha256': sha(external_source),
    'calibration_only': True, 'third_set_read': False, 'final_set_read': False,
    'full_correct': len(correct), 'target_total': len(targets),
    'calibration_full_correct_percent': 100 * len(correct) / len(targets),
    'external_same_text': sum(row['same_text'] for row in external),
    'external_same_generated_ids': sum(row['same_generated_ids'] for row in external),
    'external_total': len(external), 'faithful_narratives': 0, 'narrative_total': 2,
    'changed_parameters_in_candidate_matrix': changed,
    'exported_as_checkpoint': False, 'current_checkpoint_attempt': 6,
    'independent_fifty_percent_improvement_claimed': False,
    'paired_original_hf_target_responses_available_for_this_screen': False,
    'interpretation': '13/24 whole-correct development answers, with every calibration external trajectory preserved. The frozen fresh threshold is not tested. Narrative language collapses; this candidate is not an acceptable final solution. Do not equate the HF correct-answer count with an error-repair rate against the separately collected Ollama baseline.',
    'targets': [{'id': row['id'], 'whole_correct': row['id'] in correct,
                 'reason': (correct | incorrect)[row['id']]} for row in targets],
    'external': external,
    'narratives': [{'id': key, 'whole_correct': False, 'reason': value}
                   for key, value in narratives.items()],
    'native_trajectory_prefix_checks': native_prefixes,
}
(ROOT / 'results/bare_context_review.json').write_text(
    json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
print({key: value for key, value in review.items()
       if key not in ['targets', 'external', 'narratives', 'native_trajectory_prefix_checks']})
