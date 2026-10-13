"""Offline geometry contracts; no provider/Modal calls or corpus writes."""
import ast
from pathlib import Path
import unittest

from evaluation.geometry_leg import score_geometry
from evaluation.ground_truth import GroundTruth
from run_benchmark import score_leg

ROOT = Path(__file__).resolve().parents[1]


def box(text='total', coords=(.1, .1, .3, .2)):
    return {'text': text, 'bbox': list(coords)}


class GeometryContractTests(unittest.TestCase):
    def setUp(self):
        self.gt = GroundTruth('probe.png', 'easy', 'probe', 'total',
                              word_boxes=[box()], word_boxes_granularity='word')

    def test_all_new_block_adapters_use_declared_contract(self):
        for engine in ('mistral_ocr_4_1', 'paddleocr_vl_1_6', 'opendoc',
                       'deepseek_ocr_2', 'dots_mocr', 'surya_ocr_2',
                       'chandra_ocr_2', 'mineru_4_standard', 'teleocr'):
            with self.subTest(engine=engine):
                result = score_leg('geometry', engine, 'unrelated transcription', self.gt,
                                  {'text_regions': [box()], 'geometry_granularity': 'block',
                                   'geometry_coordinates': 'normalized_xyxy'}, False)
                self.assertEqual(result.prediction_granularity, 'layout_block')
                self.assertEqual(result.block_score, 1)
                self.assertEqual(result.score, 1)
                self.assertEqual(result.predicted_text, 'total')

    def test_line_fallback_label_survives(self):
        result = score_leg('geometry', 'rapidocr', '', self.gt,
                          {'text_regions': [box()], 'geometry_granularity': 'line',
                           'geometry_coordinates': 'normalized_xyxy'}, False)
        self.assertEqual(result.prediction_granularity, 'line')

    def test_declared_normalized_frame_never_guessed_as_thousand(self):
        invalid = box(coords=(.1, .1, 2, .2))
        result = score_leg('geometry', 'mathpix', '', self.gt,
                          {'words': [invalid], 'geometry_granularity': 'word',
                           'geometry_coordinates': 'normalized_xyxy'}, False)
        self.assertFalse(result.rescaled_from_thousand)
        self.assertEqual(result.scored_boxes[0]['bbox'], tuple(invalid['bbox']))
        self.assertEqual(result.invalid_boxes, 1)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.cer, 0)

    def test_invalid_box_cannot_earn_credit_even_when_centers_overlap(self):
        result = score_geometry([box(coords=(-.01, .1, .3, .2))],
                                reference=[box()], infer_scale=False)
        self.assertEqual(result.invalid_boxes, 1)
        self.assertEqual(result.located_words, 0)
        self.assertEqual(result.predicted_words, 1)
        self.assertEqual(len(result.scored_boxes), 1)

    def test_duplicate_and_malformed_boxes_cost_precision_and_keep_text(self):
        result = score_geometry([box(), box(), {'text': 'extra', 'bbox': None}],
                                reference=[box()], infer_scale=False)
        self.assertEqual(result.score, 1)
        self.assertEqual(result.predicted_words, 3)
        self.assertEqual(result.precision, .3333)
        self.assertEqual(result.invalid_boxes, 1)
        self.assertGreater(result.wer, 0)

    def test_coarse_box_is_not_split_into_synthetic_words(self):
        self.gt.word_boxes = [box('one', (.1,.1,.2,.2)), box('two', (.2,.1,.3,.2))]
        result = score_leg('geometry', 'dots_mocr', '', self.gt,
                          {'text_regions': [box('one two')], 'geometry_granularity': 'block'}, False)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.block_score, 1)
        self.assertEqual(len(result.scored_boxes), 1)

    def test_mineru_saved_table_region_keeps_box_and_renders_native_html(self):
        self.gt.word_boxes = [box('total 100')]
        result = score_leg('geometry', 'mineru_pipeline', 'unrelated prose', self.gt,
                          {'lines': [{'text': '<table><tr><td>total</td><td>100</td></tr></table>',
                                      'bbox': box()['bbox'], 'block_type': 'table_body'}]}, False)
        self.assertEqual(result.prediction_granularity, 'mixed_line_block')
        self.assertEqual(result.predicted_text, 'total 100')
        self.assertEqual(result.scored_boxes[0]['granularity'], 'block')
        self.assertEqual(result.scored_boxes[0]['bbox'], tuple(box()['bbox']))

    def test_withheld_rotated_geometry_is_an_adapter_error_not_prose_zero(self):
        with self.assertRaisesRegex(RuntimeError, 'coordinate frame'):
            score_leg('geometry', 'mathpix', 'total', self.gt,
                      {'words_unmapped_rotated': [box()]}, False)

    def test_adapter_pixel_normalization_does_not_clip(self):
        # Extract pure functions without importing/building the Modal app.
        for name in ('paddleocr_vl_1_6', 'opendoc', 'chandra_ocr_2'):
            with self.subTest(engine=name):
                path = ROOT / 'modal_apps' / f'{name}_app.py'
                tree = ast.parse(path.read_text())
                fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                          and n.name in ('_normalise_box', '_norm_bbox'))
                env = {}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
                self.assertEqual(env[fn.name]([-10, 10, 110, 30], 100, 100),
                                 [-.1, .1, 1.1, .3])

    def test_surya_polygon_normalization_does_not_clip(self):
        path = ROOT / 'modal_apps/surya_ocr_2_app.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == '_norm_poly')
        env = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        bbox, poly = env['_norm_poly']([[-10,10],[110,10],[110,30],[-10,30]],100,100)
        self.assertEqual(bbox, [-.1,.1,1.1,.3])
        self.assertEqual(poly[0], [-.1,.1])


if __name__ == '__main__':
    unittest.main()
