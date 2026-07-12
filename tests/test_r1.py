import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trust_geometry.r1_suite import build_suite, chat_template_ids, prompt_ids
from trust_geometry.steering import (
    cosine,
    hidden_index_to_decoder_layer,
    ordinal_direction,
    orthogonalize,
    pc1_direction,
)


class FakeTokenizer:
    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        # The suite only needs deterministic boundaries and single-token codewords.
        pieces = text.replace("<|", " <|").replace("|>", "|> ").split()
        out = []
        for piece in pieces:
            if piece not in self.ids:
                self.ids[piece] = len(self.ids) + 1
            out.append(self.ids[piece])
        return out


class ChatTemplateTokenizer:
    def __init__(self, value):
        self.value = value

    def apply_chat_template(self, *_args, **_kwargs):
        return self.value

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class FakeEncoding:
    ids = [7, 8, 9]


class SteeringTests(unittest.TestCase):
    def test_ordinal_direction_recovers_frozen_order(self):
        roles = ["system", "user", "cot", "assistant", "tool"]
        order = ["system", "assistant", "cot", "user", "tool"]
        axis = np.array([1.0, 0.0, 0.0])
        scores = {role: 2 - order.index(role) for role in roles}
        centroids = np.stack([scores[role] * axis for role in roles])
        direction = ordinal_direction(centroids, roles, order)
        projections = centroids @ direction
        recovered = [roles[i] for i in np.argsort(-projections)]
        self.assertEqual(recovered, order)

    def test_pc1_orientation_and_orthogonalization(self):
        roles = ["system", "user", "cot", "assistant", "tool"]
        centroids = np.array([[2, 0], [0, 0], [0.5, 0], [1, 0], [-2, 0]], dtype=float)
        direction = pc1_direction(centroids, roles)
        self.assertGreater((centroids[0] - centroids[-1]) @ direction, 0)
        cleaned = orthogonalize(np.array([1.0, 1.0, 1.0]), [np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0])])
        self.assertAlmostEqual(cosine(cleaned, np.array([1.0, 0, 0])), 0.0, places=7)
        self.assertAlmostEqual(cosine(cleaned, np.array([1.0, 1.0, 0])), 0.0, places=7)

    def test_hidden_index_mapping(self):
        self.assertEqual(hidden_index_to_decoder_layer(16), 15)
        with self.assertRaises(ValueError):
            hidden_index_to_decoder_layer(0)


class SuiteTests(unittest.TestCase):
    def test_balanced_conflict_and_control_arms(self):
        cases = build_suite(FakeTokenizer(), n_per_arm=4)
        self.assertEqual(len(cases), 24)
        for arm in ("system_user", "user_tool", "system_tool"):
            arm_cases = [case for case in cases if case.arm == arm]
            self.assertEqual(sum(case.conflict for case in arm_cases), 4)
            self.assertEqual(sum(not case.conflict for case in arm_cases), 4)
        for case in cases:
            start, end = case.target_span
            self.assertLess(start, end)
            self.assertLess(end, len(case.input_ids))
            self.assertNotEqual(case.target_token, case.alternative_token)

    def test_chat_template_ids_accepts_tokenizer_return_shapes(self):
        self.assertEqual(chat_template_ids(ChatTemplateTokenizer([[1, 2, 3]]).apply_chat_template()), [1, 2, 3])
        self.assertEqual(chat_template_ids(ChatTemplateTokenizer([4, 5, 6]).apply_chat_template()), [4, 5, 6])
        self.assertEqual(chat_template_ids(ChatTemplateTokenizer(FakeEncoding()).apply_chat_template()), [7, 8, 9])

    def test_prompt_ids_uses_explicit_harmony_prompt(self):
        ids = prompt_ids(ChatTemplateTokenizer("ignored"), "abc")
        self.assertIn(ord("a"), ids)
        self.assertIn(ord("b"), ids)
        self.assertIn(ord("c"), ids)
        self.assertEqual(ids[-1], ord(">"))


if __name__ == "__main__":
    unittest.main()
