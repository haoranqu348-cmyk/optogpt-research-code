import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from active_learning import (
    CandidateProvenance,
    JointSpectrum,
    LabeledSample,
    Layer,
    MultilayerStructure,
    deduplicate_batch,
    dump_manifest,
    load_manifest,
    parse_legacy_tokens,
    read_jsonl_manifest,
    write_jsonl_manifest,
)


def spectrum(value=0.25):
    component = (value,) * 71
    return JointSpectrum(component, component, component, component)


def sample(structure):
    return LabeledSample(structure, spectrum(), CandidateProvenance("unit-test", 0))


class StructureTests(unittest.TestCase):
    def test_hash_is_stable_across_processes(self):
        structure = parse_legacy_tokens(["SiO2_100", "TiO2_50"])
        code = (
            "from active_learning import parse_legacy_tokens; "
            "print(parse_legacy_tokens(['SiO2_100','TiO2_50']).structure_hash)"
        )
        output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(output, structure.structure_hash)
        self.assertEqual(len(output), 64)

    def test_hash_is_order_sensitive(self):
        first = parse_legacy_tokens(["SiO2_100", "TiO2_50"])
        second = parse_legacy_tokens(["TiO2_50", "SiO2_100"])
        self.assertNotEqual(first.structure_hash, second.structure_hash)

    def test_logical_numeric_thickness_has_same_hash(self):
        first = MultilayerStructure((Layer("SiO2", 100),))
        second = MultilayerStructure.from_materials(["SiO2"], [100.0])
        self.assertEqual(first.structure_hash, second.structure_hash)

    def test_input_dictionary_order_does_not_change_hash(self):
        first = MultilayerStructure.from_dict(
            {"layers": [{"material": "SiO2", "thickness_nm": 100}]}
        )
        second = MultilayerStructure.from_dict(
            {"layers": [{"thickness_nm": 100, "material": "SiO2"}]}
        )
        self.assertEqual(first.structure_hash, second.structure_hash)

    def test_invalid_records_have_actionable_errors(self):
        cases = (
            (lambda: Layer("Ag", 100), "not allowed"),
            (lambda: Layer("SiO2", 105), "10 nm grid"),
            (lambda: MultilayerStructure(()), "layer count"),
            (lambda: JointSpectrum((0.0,) * 70, (0.0,) * 71, (0.0,) * 71, (0.0,) * 71), "expected exactly 71"),
            (lambda: JointSpectrum((float("nan"),) * 71, (0.0,) * 71, (0.0,) * 71, (0.0,) * 71), "must be finite"),
        )
        for constructor, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                constructor()

    def test_flat_spectrum_uses_documented_order(self):
        flat = [1.0] * 71 + [2.0] * 71 + [3.0] * 71 + [4.0] * 71
        label = JointSpectrum.from_flat(flat)
        self.assertEqual((label.rs[0], label.ts[0], label.rp[0], label.tp[0]), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(label.to_flat(), tuple(flat))


class DeduplicationTests(unittest.TestCase):
    def test_duplicates_and_existing_hashes_are_removed(self):
        first = parse_legacy_tokens(["SiO2_100"])
        second = parse_legacy_tokens(["TiO2_100"])
        result = deduplicate_batch([first, first, second], {second.structure_hash})
        self.assertEqual(result.unique, (first,))
        self.assertEqual(result.duplicate_hashes, (first.structure_hash,))
        self.assertEqual(result.excluded_hashes, (second.structure_hash,))
        self.assertEqual(result.removed_count, 2)

    def test_bad_batch_item_identifies_index(self):
        with self.assertRaisesRegex(ValueError, r"batch\[0\]"):
            deduplicate_batch([object()])


class ManifestTests(unittest.TestCase):
    def test_json_and_jsonl_round_trip(self):
        original = sample(parse_legacy_tokens(["SiO2_100", "TiO2_50"]))
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "samples.json"
            jsonl_path = Path(directory) / "samples.jsonl"
            dump_manifest([original], json_path)
            write_jsonl_manifest([original], jsonl_path)
            self.assertEqual(load_manifest(json_path), (original,))
            self.assertEqual(read_jsonl_manifest(jsonl_path), (original,))

    def test_tampered_hash_is_rejected(self):
        original = sample(parse_legacy_tokens(["SiO2_100"]))
        record = original.to_dict()
        record["structure_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match canonical hash"):
            LabeledSample.from_dict(record)

    def test_provenance_parameter_dict_order_is_canonical(self):
        first = CandidateProvenance.from_parameters("decoder", 2, {"temperature": 0.8, "top_k": 10})
        second = CandidateProvenance.from_parameters("decoder", 2, {"top_k": 10, "temperature": 0.8})
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first.to_dict(), sort_keys=True), json.dumps(second.to_dict(), sort_keys=True))

    def test_bad_provenance_parameter_key_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "parameter name must be a non-empty string"):
            CandidateProvenance("decoder", 0, parameters=((1, "bad"),))


if __name__ == "__main__":
    unittest.main()
