from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from proxybench.annotation.bindings import review_binding, save_new_version, sha256, validate_input_binding
from proxybench.evaluation.projection import primary_paths
from proxybench.evaluation.references import admit_reference
from proxybench.schemas.records import normalize_record
from test_benchmark import RAW, context, evidence, field, record, span


class BindingTests(unittest.TestCase):
    def fixture(self, root):
        paths = {"source_manifest": "manifest.json", "review_view": "view.html", "model_input": "input.txt"}
        binding = {"binding_version": "input-binding-v1", "input_id": "synthetic-input", "review_tool_revision": "synthetic-v1",
                   "representation_decision": "SAME_EVIDENCE", "reviewer": "Synthetic reviewer", "reviewed_at": "2026-09-07",
                   "reason": "Synthetic representations contain the same source facts."}
        for key, path in paths.items():
            (root / path).write_bytes(b"synthetic source evidence")
            binding[key + "_path"] = path
            binding[key + "_sha256"] = sha256((root / path).read_bytes())
        save_new_version(root / "binding.json", binding)
        save_new_version(root / "original.json", {"legacy": "accepted original"})
        r = normalize_record(record())
        reference = {"reference_id": "r1", "input_id": "synthetic-input", "record": r,
                     "label_version": "synthetic-v1", "label_status": "HUMAN_ACCEPTED", "reviewer": "Synthetic reviewer",
                     "reviewed_at": "2026-09-07", "guide_version": "benchmark-v1", "original_label_path": "original.json",
                     "original_label_sha256": sha256((root / "original.json").read_bytes()), "primary_paths": primary_paths(r),
                     "conversion_log": [], "input_binding_path": "binding.json",
                     "input_binding_sha256": sha256((root / "binding.json").read_bytes())}
        return binding, reference

    def admit(self, reference, root, digest):
        return admit_reference(reference, root, context(), input_id="synthetic-input", model_input_sha256=digest, guide_version="benchmark-v1")

    def test_model_input_and_all_binding_artifacts_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, reference = self.fixture(root)
            digest = binding["model_input_sha256"]
            self.assertEqual(self.admit(reference, root, digest).reference_id, "r1")
            with self.assertRaises(ValueError):
                self.admit(reference, root, "f" * 64)
            for key in ("source_manifest", "review_view", "model_input"):
                p = root / binding[key + "_path"]
                original = p.read_bytes()
                p.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    self.admit(reference, root, digest)
                p.write_bytes(original)

    def test_enrichment_citations_obey_all_source_boundaries(self):
        for nested in (False, True):
            for mutation in ("valid", "unknown", "range", "hash", "prepared", "mapping"):
                with self.subTest(nested=nested, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    binding, reference = self.fixture(root)
                    citation = evidence()
                    ctx = context()
                    if mutation == "unknown":
                        citation["document_id"] = "unknown"
                    elif mutation == "range":
                        ctx.permitted_ranges["doc"] = [(0, len(RAW) - 1)]
                    elif mutation == "hash":
                        citation["source_sha256"] = "f" * 64
                    elif mutation in ("prepared", "mapping"):
                        citation.update(quote_basis="PREPARED_VIEW", view_id="view", block_id="B1")
                        if mutation == "mapping":
                            ctx.views[("view", "B1")] = {"span": span(0, len(RAW) - 1), "text": "Synthetic"}
                    enrichment = field("SYN", origin="INFERRED", rule_id="synthetic")
                    enrichment["evidence"] = [citation]
                    path = "/fields/ticker"
                    if nested:
                        path = "/fields/vote_components"
                        enrichment = field([{**record()["fields"]["vote_components"]["value"][0],
                                              "direction": enrichment}], origin="INFERRED", rule_id="synthetic")
                    reference["record"]["enrichments"] = [{"field_path": path, "field": enrichment}]
                    def admit():
                        return admit_reference(reference, root, ctx, input_id="synthetic-input",
                                               model_input_sha256=binding["model_input_sha256"], guide_version="benchmark-v1")
                    if mutation == "valid":
                        # The nested direction must use a valid direction enum.
                        if nested:
                            enrichment["value"][0]["direction"]["value"] = "FOR"
                        self.assertEqual(admit().reference_id, "r1")
                    else:
                        if nested:
                            enrichment["value"][0]["direction"]["value"] = "FOR"
                        with self.assertRaisesRegex(ValueError, "evidence fails mechanical"):
                            admit()

    def test_pending_conversion_cannot_replace_original_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, reference = self.fixture(root)
            original = (root / "original.json").read_bytes()
            reference["label_status"] = "CONVERSION_PENDING_REVIEW"
            save_new_version(root / "conversion-v1.json", reference)
            with self.assertRaises(ValueError):
                self.admit(reference, root, binding["model_input_sha256"])
            with self.assertRaises(FileExistsError):
                save_new_version(root / "original.json", reference)
            self.assertEqual((root / "original.json").read_bytes(), original)

    def test_reference_mask_and_representation_review_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, reference = self.fixture(root)
            reference["primary_paths"].remove("/fields/issuer_name")
            with self.assertRaises(ValueError):
                self.admit(reference, root, binding["model_input_sha256"])
            for status in ("PENDING", "CHANGED_EVIDENCE"):
                binding["representation_decision"] = status
                with self.assertRaises(ValueError):
                    validate_input_binding(binding, root, input_id="synthetic-input", model_input_sha256=binding["model_input_sha256"])

    def test_target_context_display_and_model_input_bindings(self):
        packet = {"manifest": {"source_sha256": "unchanged", "target": [10, 20], "blocks": [[0, 40]]},
                  "source_view": "original display", "model_input": "original model evidence"}
        original = review_binding(packet)
        for key, replacement in [("target", [20, 30]), ("blocks", [[0, 50]])]:
            changed = deepcopy(packet)
            changed["manifest"][key] = replacement
            self.assertNotEqual(review_binding(changed), original)
        for key in ("source_view", "model_input"):
            changed = deepcopy(packet)
            changed[key] += "changed"
            self.assertNotEqual(review_binding(changed), original)


if __name__ == "__main__":
    unittest.main()
