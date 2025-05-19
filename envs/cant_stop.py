from enum import Enum
from typing import Any, SupportsFloat, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from .cant_stop_utils import StopContinueAction, CantStopState, CantStopAction, \
    StopContinueChoice, ProgressActionSet, ProgressAction


class PlayerState(Enum):
    SELECT_STOP_OR_CONTINUE = 0
    SELECT_DICE_COMBO = 1


class CantStopObservation(spaces.Space):
    def __init__(self, seed: int=None):
        super().__init__((11,), int, seed)
        self._column_sizes = np.array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]) + 1
        self.dice_rolls = spaces.MultiDiscrete([6,6,6,6])
        #self.diceroll_encoder = diceroll_encoder

    @property
    def column_sizes(self):
        return self._column_sizes

    def contains(self, x: CantStopState) -> bool:
        if not isinstance(x, CantStopState): return False
        return x.is_well_formed()

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        raise False

    def sample_saved_steps_remaining(self):
        rng = np.random.default_rng()
        all_ = rng.integers(1, self._column_sizes, size=11, dtype=np.uint8)
        k_value = rng.integers(0, 3, dtype=np.uint8)
        # row_indices = np.repeat(np.arange(1, dtype=np.int32), k_values)

        # Generate random values for sorting (using float32 for memory efficiency)
        rand_vals = rng.random(size=11, dtype=np.float32)
        sorted_indices = np.argsort(rand_vals, axis=0, kind='stable')
        mask = np.arange(11, dtype=np.uint8) < k_value
        col_indices = sorted_indices[mask]
        all_[col_indices] = 0
        return all_

    def sample_active_steps_remaining(self, saved_steps_remaining):
        """
        Choose up to 3 positions as active that are non-zero in _saved_steps_remaining.
        Advance these columns by at least one step.
        :param saved_steps_remaining:
        :return:
        """
        active_steps_remaining = saved_steps_remaining.copy()
        zero_mask = saved_steps_remaining == 0
        zero_mask_p_dist = 1 - zero_mask.astype(np.float32)
        zero_mask_p_dist /= np.sum(zero_mask_p_dist)

        # choose number of active positions
        num_active = np.random.choice(4)

        # choose active positions - must be nonzero i.e: non-complete.
        active_positions = np.random.choice(11, size=num_active, p=zero_mask_p_dist, replace=False)

        # set progresses of active positions. Can end up with extra zeros.
        active_steps_remaining[active_positions] = np.random.randint(np.zeros(len(active_positions)),
                                                                     saved_steps_remaining[active_positions])

        return active_steps_remaining

    def sample(self, mask: Any | None = None, probability: Any | None = None) -> CantStopState:
        assert mask is None, "mask should be None!"
        assert probability is None, "probability should be None!"

        saved_steps_remaining = self.sample_saved_steps_remaining()
        active_steps_remaining = self.sample_active_steps_remaining(saved_steps_remaining)
        #active_positions = _active_steps_remaining != _saved_steps_remaining

        current_action_selection = StopContinueChoice()
        return CantStopState(saved_steps_remaining, active_steps_remaining, current_action_selection)


class CantStopEnv(gym.Env):
    metadata = {"render_modes": {"human", "rgb_array"}, "render_fps": 10}
    def __init__(self, max_turns=None, render_mode=None):
        self._state: Optional[CantStopState] = None
        self.observation_space: CantStopObservation = CantStopObservation()
        self.action_space = spaces.Discrete(77)
        self.render_fps = self.metadata["render_fps"]

        if max_turns is None:
            self.max_turns = 200
        else:
            self.max_turns = max_turns

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be in {self.metadata['render_modes']}, of which {render_mode} is not.")
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.window_size = (600, 710)
        self.dice_combo_view_width_prop = 0.0
        self.play_history = None

    def reset(self,
              *args,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[CantStopState, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if isinstance(options, dict) and options.get("start_base", False):
            self._state = CantStopState(np.array([3,5,7,9,11,13,11,9,7,5,3]),
                                        np.array([3,5,7,9,11,13,11,9,7,5,3]),
                                        StopContinueChoice())
        else:
            self._state = self.observation_space.sample()
        if self.render_mode == "human":
            self._render_frame()
        return self._state, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

        colours = self._state.raw_squares()

        canvas = pygame.Surface(self.window_size)
        progress_canvas = pygame.Surface(((1-self.dice_combo_view_width_prop)*self.window_size[0], self.window_size[1]))
        #dice_canvas = pygame.Surface((self.dice_combo_view_width_prop*self.window_size[0], self.window_size[1]))
        canvas.fill((255, 255, 255))
        progress_canvas.fill((255, 255, 255))

        base_width, base_height = progress_canvas.get_width(), progress_canvas.get_height()
        square_width = min(base_width, base_height) / len(self.observation_space.column_sizes)
        max_col_size = np.max(self._state.column_limits - 1)
        for col_idx, col in enumerate(colours):
            for square_idx, square_colour in enumerate(col):
                pygame.draw.rect(progress_canvas,
                                 square_colour,
                                 pygame.Rect(col_idx * square_width,
                                             (max_col_size - square_idx) * square_width,
                                             square_width,
                                             square_width),
                                 border_radius=5)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            canvas.blit(progress_canvas, progress_canvas.get_rect())
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def step(self, action: CantStopAction) -> tuple[Optional[CantStopState], SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Apply an action in the Cant Stop game environment and return the resulting state, reward, and flags.
        :param action:
        :return: state: The updated game state.
                 reward: The reward obtained from this step.
                 terminated: Whether the game is over.
                 truncated: Whether the game was truncated due to max turns.
                 info: Additional info (empty in this case).
        """
        reward = 0.0
        terminated = False

        if isinstance(self._state.current_action, StopContinueChoice):
            if action == StopContinueAction.CONTINUE:
                # Potential penalty for busting
                bust_penalty = -3.0 * np.sum(self._state.active_advances)
                busted = self._state.perform_continue()
                if busted:
                    reward = bust_penalty

            elif action == StopContinueAction.STOP:
                # Reward for securing progress
                progress_reward = 2.0 * np.sum(self._state.active_advances)
                completion_reward = (20.0 * np.sum(self._state.full_active_cols.astype(float)) *
                                     (1 - self._state.num_turns / self.max_turns))
                reward += progress_reward + completion_reward

                completed = self._state.perform_stop()
                if completed:
                    reward += 100.0 * (1 - self._state.num_turns / self.max_turns)
                    terminated = True  # game is over after 3 columns

            else:
                raise ValueError("Invalid Stop/Continue action.")

        elif isinstance(self._state.current_action, ProgressActionSet):
            # Apply the chosen progression action
            self._state.perform_progression(action)

            reward += 1.0 / self._state.column_limits[action.larger]
            if action.smaller != -1:
                reward += 1.0 / self._state.column_limits[action.smaller]

        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        if self.render_mode == "human":
            self._render_frame()

        return self._state, reward, terminated, False, {}