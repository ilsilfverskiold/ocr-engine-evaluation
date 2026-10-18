"""Per-document KV rulings file: what is stored, and how the matcher uses it.

Offline. Every test points the rulings folder at a temporary directory; the
judge call is replaced by a fixed reply, so nothing is paid and results/ is
never written.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluation.kv_value_judge as rulings
from evaluation import kv_value_groups as groups
from evaluation.ground_truth import GroundTruthLoader
from evaluation.shared_fields import reference_fields, reference_id, score_shared

ROOT = Path(__file__).resolve().parents[1]
INVOICE = ROOT / "documents/easy/invoice/invoice_000.png"
# Labels that no rule maps to "Seller" / "Client" / "Seller — IBAN".
OUTPUT = [("Seller Name", "Bradley-Andrade"), ("Client Name", "Castro PLC")]


def verdicts(gt, pairs, resolution):
    result = score_shared(gt, pairs, "key_value", resolution.synonyms, resolution.groups)
    return {v.path: v for v in result.verdicts}


class RulingsTest(unittest.TestCase):
    def setUp(self):
        self.gt = GroundTruthLoader().load(INVOICE)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = patch.object(rulings, "CACHE_ROOT", Path(self.tmp.name))
        self.root.start()
        os.environ.pop("GOOGLE_API_KEY", None)

    def tearDown(self):
        self.root.stop()
        self.tmp.cleanup()

    def judge(self, pairs, reply_for):
        """Run resolve once with a fake judge; reply_for(item) -> output_ids, synonym."""
        def ask(request):
            reply = {str(i["id"]): dict(zip(("output_ids", "synonym"), reply_for(i)), reason="test")
                     for i in request["review_items"]}
            return rulings.validate(request, reply), 0.0
        with patch.object(rulings, "_ask", ask), patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            return groups.resolve(self.gt, pairs, engine="engine_a")

    def stored(self):
        return json.loads(rulings._path(self.gt).read_text())

    def test_unknown_label_is_pending_without_a_ruling(self):
        res = groups.resolve(self.gt, OUTPUT, use_judge=False)
        self.assertIn("Seller", res.pending)
        self.assertEqual(verdicts(self.gt, OUTPUT, res)["Seller"].verdict, "missing")

    def test_accepted_and_rejected_rulings_are_filed_under_their_field(self):
        accept_seller = lambda item: ([1], True) if item["reference_key"] == "Seller" else ([], False)
        self.judge(OUTPUT, accept_seller)
        fields = self.stored()["fields"]
        seller, client = fields["Seller"], fields["Client"]
        self.assertEqual(seller["expected"], "Bradley-Andrade")
        self.assertEqual(seller["synonyms"][0]["key"], "Seller Name")
        accepted = seller["output_rulings"][0]
        self.assertEqual((accepted["verdict"], accepted["engine"]), ("accepted", "engine_a"))
        self.assertEqual(accepted["looked_at"], [{"output_id": 1, "key": "Seller Name", "value": "Bradley-Andrade"}])
        rejected = client["output_rulings"][0]
        self.assertEqual(rejected["verdict"], "rejected")
        self.assertEqual(rejected["looked_at"], [{"output_id": 2, "key": "Client Name", "value": "Castro PLC"}])

    def test_stored_rulings_are_reused_without_asking_again(self):
        self.judge(OUTPUT, lambda item: ([1], True) if item["reference_key"] == "Seller" else ([], False))
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}), \
             patch.object(rulings, '_ask') as ask:
            res = groups.resolve(self.gt, OUTPUT, use_judge=True)
            ask.assert_not_called()
        self.assertEqual(res.pending, [])  # the rejection settles Client too
        got = verdicts(self.gt, OUTPUT, res)
        self.assertEqual(got["Seller"].verdict, "correct")
        self.assertEqual(got["Client"].verdict, "missing")

    def test_determined_matches_and_wrong_values_need_no_judge(self):
        pairs = [('Seller', 'Bradley-Andrade'), ('Client', 'Definitely wrong company')]
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}), \
             patch.object(rulings, '_ask') as ask:
            res = groups.resolve(self.gt, pairs, use_judge=True)
            ask.assert_not_called()
        self.assertEqual(res.pending, [])
        got = verdicts(self.gt, pairs, res)
        self.assertEqual(got['Seller'].verdict, 'correct')
        self.assertEqual(got['Client'].verdict, 'wrong')

    def test_only_undetermined_fields_are_reviewed_after_exact_and_cached_matches(self):
        self.judge(OUTPUT, lambda item: ([1], True) if item['reference_key'] == 'Seller' else ([], False))
        pairs = [('Invoice no', '97159829'), ('Seller Name', 'Bradley-Andrade'),
                 ('Customer legal name', 'Castro PLC')]
        def answer(request):
            self.assertEqual([i['reference_key'] for i in request['review_items']], ['Client'])
            self.assertEqual(request['already_assigned_output_ids'], [1, 2])
            # Resolved fields may supply context but are not review items.
            self.assertEqual(len(request['returned_pairs']), 3)
            item = request['review_items'][0]
            reply = {str(item['id']): {'output_ids': [3], 'synonym': False, 'reason': 'test'}}
            return rulings.validate(request, reply), 0.0
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}), \
             patch.object(rulings, '_ask', side_effect=answer) as ask:
            res = groups.resolve(self.gt, pairs, use_judge=True)
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(res.pending, [])
            self.assertEqual(verdicts(self.gt, pairs, res)['Client'].verdict, 'correct')
            # The new decision is saved and must settle the next replay too.
            again = groups.resolve(self.gt, pairs, use_judge=True)
            self.assertEqual(again.pending, [])
            self.assertEqual(ask.call_count, 1)

    def test_saved_judge_response_is_recovered_without_another_api_call(self):
        request = groups.prepare(self.gt, OUTPUT)[-1][0]
        reply = {str(i['id']): {'output_ids': [1] if i['reference_key'] == 'Seller' else [],
                                'synonym': False, 'reason': 'test'} for i in request['review_items']}
        path = rulings.CACHE_ROOT / 'calls' / (request['id'] + '.json')
        path.parent.mkdir()
        path.write_text(json.dumps({'raw_text': json.dumps(reply)}))
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}), \
             patch('google.genai.Client') as client:
            res = groups.resolve(self.gt, OUTPUT, use_judge=True)
            client.assert_not_called()
        self.assertEqual(res.pending, [])
        self.assertEqual(res.cost_usd, 0.0)
        self.assertEqual(verdicts(self.gt, OUTPUT, res)['Seller'].verdict, 'correct')

    def test_synonym_reaches_another_engine_output(self):
        self.judge(OUTPUT, lambda item: ([1], True) if item["reference_key"] == "Seller" else ([], False))
        other = [("Invoice no", "97159829"), ("Seller Name", "Bradley-Andrade")]
        res = groups.resolve(self.gt, other, use_judge=False)
        got = verdicts(self.gt, other, res)["Seller"]
        self.assertEqual((got.verdict, got.how), ("correct", "cached judge-confirmed KV mapping"))

    def test_output_ruling_is_not_applied_to_a_different_output(self):
        # Accepted without the synonym flag: only a ruling about OUTPUT exists.
        self.judge(OUTPUT, lambda item: ([1], False) if item["reference_key"] == "Seller" else ([], False))
        other = [("Seller Name", "Bradley-Andrade"), ("Notes", "x")]
        self.assertEqual(groups.resolve(self.gt, OUTPUT, use_judge=False).pending, [])
        self.assertIn("Seller", groups.resolve(self.gt, other, use_judge=False).pending)

    def test_one_output_never_fills_two_fields(self):
        # Two reference fields could both claim output 1; the judge picks it for both.
        self.judge(OUTPUT, lambda item: ([1], False) if item["reference_key"] in ("Seller", "Client") else ([], False))
        res = groups.resolve(self.gt, OUTPUT, use_judge=False)
        used = [i for g in res.groups for i in g["indices"]]
        self.assertEqual(len(used), len(set(used)))

    def test_label_repeated_on_the_page_is_never_a_synonym(self):
        record = rulings._read(self.gt)
        rulings.add_synonym(record, self.gt, "Item 1 — Net worth", "Net worth", "889,20", "r1")
        rulings.add_synonym(record, self.gt, "Seller — IBAN", "IBAN", "GB81LZWO32519172531418", "r2")
        rulings.add_synonym(record, self.gt, "Seller — IBAN", "IBAN", "GB81LZWO32519172531418", "r3")
        self.assertNotIn("Item 1 — Net worth", record["fields"])
        self.assertEqual(len(record["fields"]["Seller — IBAN"]["synonyms"]), 1)

    def test_conflicting_synonym_is_ignored_by_label(self):
        record = rulings._read(self.gt)
        rulings.add_synonym(record, self.gt, "Seller — IBAN", "Account", "GB81LZWO32519172531418", "r1")
        rulings.add_synonym(record, self.gt, "Date of issue", "Account", "09/18/2015", "r2")
        maps = rulings._cache_maps(record)
        self.assertNotIn("Account", maps)  # the label alone no longer decides
        self.assertEqual(maps[rulings.pair_identity("Account", "09/18/2015")], "Date of issue")


class CommittedRulingsTest(unittest.TestCase):
    def test_every_page_file_is_current_and_readable(self):
        loader = GroundTruthLoader()
        # Images are not in Git; the corpus manifest names each page's path.
        images = {Path(e["file"]).stem: ROOT / e["file"]
                  for e in json.loads((ROOT / "documents/manifest.json").read_text())}
        pages = sorted(rulings.CACHE_ROOT.glob("*.json"))
        self.assertEqual(len(pages), 65)
        for page in pages:
            gt = loader.load(images[page.stem])
            data = json.loads(page.read_text())
            self.assertEqual((data["version"], data["reference_id"]), (rulings.VERSION, reference_id(gt)), page.name)
            paths = {f.path: f.value for f in reference_fields(gt)}
            for path, field in data["fields"].items():
                self.assertEqual(field["expected"], paths[path], (page.name, path))
                for ruling in field.get("output_rulings", []):
                    self.assertIn(ruling["verdict"], ("accepted", "rejected"))
                    self.assertTrue(ruling["looked_at"], (page.name, path))
                    self.assertTrue(all({"output_id", "key", "value"} <= set(p) for p in ruling["looked_at"]))


class BlankFieldTest(unittest.TestCase):
    """KV: a reference blank whose label is not returned is not counted; an invented value is."""

    def setUp(self):
        self.gt = GroundTruthLoader().load(ROOT / "documents/hard/form/funsd_003.png")
        fields = reference_fields(self.gt, "key_value")
        self.blanks = [f for f in fields if f.value in (None, "")]
        self.filled = [f for f in fields if f.value not in (None, "")]

    def test_unreturned_blank_is_not_counted(self):
        result = score_shared(self.gt, [], "key_value")
        self.assertEqual(result.total, len(self.filled))
        self.assertTrue(all(v.verdict == "blank_not_returned" for v in result.verdicts if v.is_blank))

    def test_invented_value_for_a_blank_still_counts(self):
        blank = self.blanks[0]
        result = score_shared(self.gt, [(blank.path, "made up")], "key_value")
        verdict = next(v for v in result.verdicts if v.path == blank.path)
        self.assertEqual(verdict.verdict, "invented")
        self.assertEqual(result.total, len(self.filled) + 1)

    def test_schema_leg_counts_every_field(self):
        result = score_shared(self.gt, [], "schema")
        self.assertNotIn("blank_not_returned", {v.verdict for v in result.verdicts})


class PublishedTextractTest(unittest.TestCase):
    def test_bank_balances_retain_published_matches_without_judging(self):
        from evaluation.keyvalue_leg import score_key_value
        gt = GroundTruthLoader().load(ROOT / 'documents/medium/bank_statement/bank_002.png')
        call = json.loads((ROOT / 'results/ocr_outputs/reconciled_outputs/'
                           'medium__bank_statement__bank_002/textract_structured/default.call.json').read_text())
        with patch.object(rulings, '_ask', side_effect=AssertionError('Unexpected judge request')):
            result = score_key_value(call['raw_text'], gt, 'textract_structured',
                                     call['structured_output'], use_judge=False)
        self.assertEqual(result.score, 58 / 62)
        self.assertEqual(result.pending_key_review, [])
        balances = [v for v in result.verdicts if v.path.startswith('Daily Balances — ')]
        self.assertEqual(len(balances), 22)
        self.assertTrue(all(v.verdict == 'correct' for v in balances))


if __name__ == "__main__":
    unittest.main()
