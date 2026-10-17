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
        res = groups.resolve(self.gt, OUTPUT, use_judge=False)
        self.assertEqual(res.pending, [])  # the rejection settles Client too
        got = verdicts(self.gt, OUTPUT, res)
        self.assertEqual(got["Seller"].verdict, "correct")
        self.assertEqual(got["Client"].verdict, "missing")

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


if __name__ == "__main__":
    unittest.main()
