"""Small behavioral checks; no game, neural network, or large allocations."""

import pickle
import unittest

import numpy as np

from prioritized_replay import PrioritizedNStepReplay


def image(value):
    return np.full((4, 2, 2), value, dtype=np.uint8)


def replay(capacity=16, **kwargs):
    return PrioritizedNStepReplay(capacity, observation_shape=(4, 2, 2), **kwargs)


class ReplayTests(unittest.TestCase):
    def test_three_step_actual_final_stack_and_terminal_tails(self):
        r = replay(gamma=0.5)
        r.add(0, image(0), 2, 1, image(1), False, False)
        r.add(0, image(1), 3, 2, image(2), False, False)
        r.add(0, image(2), 4, 4, image(3), False, False)
        self.assertEqual(len(r), 1)
        self.assertEqual(r.rewards[0], 3)
        self.assertEqual(r.discounts[0], 0.125)
        self.assertEqual(r.horizons[0], 3)
        self.assertTrue(np.array_equal(r.next_obs[0], image(3)))
        r.add(0, image(3), 5, 8, image(4), True, False)
        np.testing.assert_allclose(r.rewards[:4], [3, 6, 8, 8])
        np.testing.assert_allclose(r.discounts[:4], [0.125, 0, 0, 0])
        np.testing.assert_array_equal(r.horizons[:4], [3, 3, 2, 1])
        self.assertEqual(len(r.pending[0]), 0)

    def test_truncation_bootstraps_and_environment_queues_do_not_mix(self):
        r = replay(gamma=0.5)
        r.add(0, image(0), 1, 2, image(1), False, False)
        r.add(1, image(20), 6, 40, image(21), False, False)
        r.add(0, image(1), 2, 4, image(2), False, True)
        np.testing.assert_allclose(r.rewards[:2], [4, 4])
        np.testing.assert_allclose(r.discounts[:2], [0.25, 0.5])
        np.testing.assert_array_equal(r.horizons[:2], [2, 1])
        self.assertEqual(len(r.pending[1]), 1)
        r.add(0, image(100), 3, 10, image(101), True, False)
        self.assertEqual(r.rewards[2], 10)
        self.assertTrue(np.array_equal(r.obs[2], image(100)))
        self.assertEqual(r.discounts[2], 0)
        r.add(1, image(21), 7, 80, image(22), True, False)
        np.testing.assert_allclose(r.rewards[3:5], [80, 80])

    def test_circular_overwrite_and_new_item_global_max_priority(self):
        r = replay(capacity=3, n_step=1)
        for i in range(3):
            r.add(0, image(i), i, i, image(i + 1), False, False)
        r.update_priorities(np.array([1]), np.array([9.0]))
        for i in range(3, 5):
            r.add(0, image(i), i, i, image(i + 1), False, False)
        self.assertEqual(len(r), 3)
        np.testing.assert_array_equal(r.actions, [3, 4, 2])
        self.assertAlmostEqual(r.priorities[0], 9.000001)
        self.assertAlmostEqual(r.priorities[1], 9.000001)
        self.assertAlmostEqual(r._sum_tree[1], np.sqrt(r.priorities).sum())
        sample = r.sample(100, 0.8)
        self.assertTrue(np.all(sample["indices"] < 3))
        self.assertTrue(np.all(np.isfinite(sample["weights"])))
        self.assertTrue(np.all((sample["weights"] > 0) & (sample["weights"] <= 1)))

    def test_prioritization_duplicate_max_and_importance_weights(self):
        r = replay(capacity=4, n_step=1, alpha=1)
        for i in range(4):
            r.add(0, image(i), i, i, image(i + 1), False, False)
        r.update_priorities(np.array([0, 1, 2, 3, 3]), np.array([1, 1, 1, -9, 2]))
        self.assertAlmostEqual(r.priorities[3], 9.000001)
        sample = r.sample(12000, beta=1)
        frequencies = np.bincount(sample["indices"], minlength=4) / 12000
        np.testing.assert_allclose(frequencies, [1 / 12, 1 / 12, 1 / 12, 9 / 12], atol=0.002)
        expected = r.priorities.min() / r.priorities[sample["indices"]]
        np.testing.assert_allclose(sample["weights"], expected, rtol=1e-6)
        np.testing.assert_array_equal(r.sample(8, beta=0)["weights"], np.ones(8))
        with self.assertRaises(ValueError):
            r.update_priorities(np.array([0]), np.array([np.nan]))

    def test_state_roundtrip_preserves_rng_and_pending_sequences(self):
        r = replay(capacity=7, gamma=0.9)
        for i in range(5):
            r.add(0, image(i), i, i + 1, image(i + 1), False, False)
        r.add(2, image(90), 8, 100, image(91), False, False)
        r.update_priorities(np.array([0, 1, 2]), np.array([0.5, 2, 10]))
        restored = replay(capacity=7, gamma=0.9, seed=999)
        restored.load_state_dict(pickle.loads(pickle.dumps(r.state_dict())))
        expected = r.sample(15, 0.7)
        actual = restored.sample(15, 0.7)
        for key, value in expected.items():
            np.testing.assert_array_equal(value, actual[key])
        for obj in (r, restored):
            obj.add(0, image(5), 5, 6, image(6), False, True)
            obj.add(2, image(91), 8, 200, image(92), True, False)
        for name in ("obs", "next_obs", "actions", "rewards", "discounts", "horizons", "priorities"):
            np.testing.assert_array_equal(getattr(r, name)[:len(r)],
                                          getattr(restored, name)[:len(restored)])
        self.assertEqual(r.position, restored.position)
        empty = replay(capacity=1, n_step=1)
        empty.load_state_dict(empty.state_dict())
        empty.add(0, image(1), 0, 2, image(2), True, False)
        np.testing.assert_array_equal(empty.sample(2, 1)["indices"], [0, 0])


if __name__ == "__main__":
    unittest.main()
