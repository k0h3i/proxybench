"""Versioned source-value scoring for the historical system-message release."""

from copy import deepcopy
import re

from proxybench.evaluation.training_labels import parse_answer, quote_paths, score
from proxybench.runstate import binding

SCORER_VERSION = 'source-cells-v1'


def source_cells(meta):
    if 'source_cells' in meta:
        return meta['source_cells']
    return [dict(block_index=block_index, cell_index=cell_index, text=cell['text'])
            for block_index, block in enumerate(meta['packet']['manifest']['blocks'])
            for cell_index, cell in enumerate(block['cells'])]


def field_at(label, path):
    if not path.startswith('fields.'):
        raise ValueError('Equivalence needs a complete field path')
    node = label
    for part in re.findall(r'[^.\[\]]+|\[\d+\]', path):
        node = node[int(part[1:-1])] if part.startswith('[') else node[part]
    return node


def apply_other_equivalences(reference, prediction, answer, meta, decisions):
    adjusted = deepcopy(prediction)
    cells = source_cells(meta)
    for decision in decisions:
        if set(decision) != {'reference_field_path', 'prediction_field_path', 'block_index', 'cell_index',
                             'column', 'passage', 'reference_sha256', 'answer_sha256', 'reason'}:
            raise ValueError('OTHER equivalence decision has unexpected fields')
        if (decision['reference_sha256'] != binding(reference) or decision['answer_sha256'] != binding(answer)
                or not all(isinstance(decision[key], str) and decision[key].strip()
                           for key in ('column', 'passage', 'reason'))):
            raise ValueError('OTHER equivalence is not bound to exact answers and source evidence')
        reference_path, prediction_path = decision['reference_field_path'], decision['prediction_field_path']
        if re.sub(r'\[\d+\]', '[]', reference_path) != re.sub(r'\[\d+\]', '[]', prediction_path):
            raise ValueError('OTHER equivalence changes the field')
        cell = next((cell for cell in cells if cell['block_index'] == decision['block_index']
                     and cell['cell_index'] == decision['cell_index']), None)
        if cell is None or decision['passage'] not in cell['text']:
            raise ValueError('OTHER equivalence passage is absent from the declared source cell')
        expected = field_at(reference, reference_path)
        actual = field_at(adjusted, prediction_path)
        if (not isinstance(expected, dict) or not isinstance(actual, dict)
                or expected.get('value') != actual.get('value') or expected.get('value') != 'OTHER'
                or expected.get('availability') != actual.get('availability') or expected.get('availability') != 'PRESENT'
                or not expected.get('raw_text') or not actual.get('raw_text')
                or expected['raw_text'] not in cell['text'] or actual['raw_text'] not in cell['text']):
            raise ValueError('OTHER equivalence cannot change the category, availability, or source cell')
        actual['raw_text'] = expected['raw_text']
    return adjusted


def score_system(reference, answer, meta, *, subject_equivalent=False, quotation_errors=None,
                 other_equivalences=()):
    original = parse_answer(answer)
    if original is None and other_equivalences:
        raise ValueError('Malformed output cannot receive an OTHER equivalence')
    adjusted = apply_other_equivalences(reference, original, answer, meta, other_equivalences) if other_equivalences else original
    scored_answer = dict(answer)
    if adjusted is not None:
        from proxybench.training.labels import dumps
        scored_answer['text'] = dumps(adjusted)
    result = score(reference, scored_answer, '', subject_equivalent=subject_equivalent,
                   quotation_errors=quotation_errors)
    result['exact'] = original == reference if original is not None else False
    if original is not None:
        cells = [cell['text'] for cell in source_cells(meta)]
        result['unsupported_quotes'] = [path for path, quote in quote_paths(original['fields'])
                                        if not any(quote in cell for cell in cells)]
    result['other_equivalence_paths'] = [decision['prediction_field_path'] for decision in other_equivalences]
    return result
