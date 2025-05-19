import random
import unittest

import numpy as np

from envs.cant_stop import CantStopObservation, CantStopEnv
from envs.cant_stop_utils import ProgressAction, CantStopState, StopContinueChoice, ProgressActionSet


class CantStopObservationTests(unittest.TestCase):
    def test_observation_sample_30000(self):
        observation = CantStopObservation()
        for i in range(30000):
            sample = observation.sample()
            self.assertTrue(sample in observation)

    def test_specific_samples(self):
        observation = CantStopObservation()
        self.assertFalse(CantStopState(observation._column_sizes + 1, observation._column_sizes, StopContinueChoice()) in observation)
        self.assertTrue(CantStopState(observation._column_sizes - 1, observation._column_sizes - 1, StopContinueChoice()) in observation)

        x = observation._column_sizes - 1
        x[[0,1]] = 0
        self.assertTrue(CantStopState(x, x, StopContinueChoice()) in observation)

        x[2] = 0
        self.assertFalse(CantStopState(x, x, StopContinueChoice()) in observation)

class CantStopStateTests(unittest.TestCase):
    def test_embedding1(self):
        state = CantStopState(np.array([3,5,7,9,11,13,11,9,7,5,3]),
                              np.array([3,5,7,9,11,13,11,9,7,5,3]),
                              ProgressActionSet([ProgressAction(-1, 0),
                                                 ProgressAction(-1, 2),
                                                 ProgressAction(0, 1)]))

        a1 = np.array([3,5,7,9,11,13,11,9,7,5,3])
        a2 = [0] * 77
        a2[0] = 1
        a2[2] = 1
        a2[12] = 1
        self.assertTrue(np.all(state.construct_state() == np.concat([a1, a1, a2])))

    def test_embedding2(self):
        state = CantStopState(np.array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]),
                              np.array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]),
                              StopContinueChoice())

        a1 = np.array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3])
        self.assertTrue(np.all(state.construct_state() == np.concat([a1, a1])))

class ProgressActionTests(unittest.TestCase):
    def test_act_encode_decode(self):
        # observation = CantStopObservation(diceroll_encoder, 42)
        pairs = [ProgressAction(i, j) for i in range(-1, 11) for j in range(max(0, i), 11)]
        for pair in pairs:
            self.assertEqual(pair, ProgressAction.decode(pair.encode()))

    def test_split1(self):
        saved_steps = np.array([4,6,8,10,12,14,12,10,8,6,4], dtype=int)
        active = saved_steps.copy()
        active[0] = 1
        active[2] = 1

        act1 = ProgressAction(0, 2)
        self.assertEqual([ProgressAction(0, 2)], act1.split(saved_steps, active))

        act2 = ProgressAction(0, 0)
        self.assertEqual([ProgressAction(-1, 0)], act2.split(saved_steps, active))

        active[0] = 2
        self.assertEqual([ProgressAction(0, 0)], act2.split(saved_steps, active))

        act3 = ProgressAction(2, 2)
        self.assertEqual([ProgressAction(-1, 2)], act3.split(saved_steps, active))

        active[0] = 0
        self.assertEqual([], act2.split(saved_steps, active))
        self.assertEqual([ProgressAction(-1, 2)], act1.split(saved_steps, active))


class CantStopEnvTests(unittest.TestCase):
    def test_random_start_state(self):
        env = CantStopEnv(render_mode="rgb_array")
        for _ in range(1000):
            state, _ = env.reset()
            self.assertTrue(state.is_well_formed())
            for i in range(200):
                action = random.choice(list(state.current_action))
                next_state, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(next_state.is_well_formed(), f"Previous state: {state}\nAction taken: {action}\nNext state: {next_state}")
                state = next_state.copy()
                if terminated:
                    break

    def test_start_state_1(self):
        state = CantStopState(np.array([3,2,7,2,4,6,10,6,1,1,1]),
                              np.array([3,2,6,1,4,6,10,6,1,1,1]),
                              StopContinueChoice())
        self.assertTrue(state.is_well_formed())

        state.perform_continue()
        self.assertTrue(state.is_well_formed())

    def test_start_state_2(self):
        state = CantStopState(np.array([3, 2, 1, 0, 5, 2, 6, 3, 7, 0, 3]),
                              np.array([3, 2, 1, 0, 5, 0, 6, 3, 6, 0, 3]),
                              StopContinueChoice())
        self.assertTrue(state.is_well_formed())

        state.perform_stop()
        self.assertTrue(state.is_well_formed())

    def test_valid_state_1(self):
        state = CantStopState(np.array([3, 4, 1, 7, 10, 13, 3, 8, 3, 4, 1]),
                              np.array([3, 4, 0, 7, 10, 13, 3, 8, 2, 4, 1]),
                              ProgressActionSet([ProgressAction(-1, 0), ProgressAction(3, 3), ProgressAction(-1, 6)]))
        self.assertTrue(state.is_well_formed())

    def test_valid_state_2(self):
        state = CantStopState(np.array([1, 4, 4, 3, 3, 7, 8, 3, 2, 2, 3]),
                              np.array([1, 4, 4, 3, 2, 7, 8, 2, 2, 2, 3]),
                              ProgressActionSet({ProgressAction(-1, 9),
                                                 ProgressAction(-1, 6),
                                                 ProgressAction(-1, 2),
                                                 ProgressAction(4, 7),
                                                 ProgressAction(-1, 5)}))
        self.assertTrue(state.is_well_formed())

    def test_valid_state_3(self):
        state = CantStopState(np.array([2, 2, 0, 5, 9, 2, 6, 5, 5, 1, 0]),
                              np.array([1, 2, 0, 5, 9, 2, 6, 4, 5, 1, 0]),
                              ProgressActionSet({ProgressAction(5, 5),
                                                 ProgressAction(-1, 4),
                                                 ProgressAction(-1, 1),
                                                 ProgressAction(-1, 9),
                                                 ProgressAction(-1, 6)}))
        self.assertTrue(state.is_well_formed())

class CantStopVisualisationTests(unittest.TestCase):
    def test_simulation(self):
        env = CantStopEnv(render_mode="human")
        state, _ = env.reset(options={"start_base": True})
        for i in range(20000):
            action = random.choice(list(state.current_action))
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state.copy()
            if terminated:
                print("done")
                break

if __name__ == '__main__':
    unittest.main()
