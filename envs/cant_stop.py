from enum import Enum
from typing import Any, SupportsFloat, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from .cant_stop_utils import StopContinueAction, ProgressAction, CantStopActionType, CantStopState, CantStopAction, \
    StopContinueChoice, ProgressActionSet


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


class BoardColumnState(Enum):
    EMPTY = 0
    SAVED = 1
    PROGRESS = 2

    def __str__(self):
        lst = ["_", "X", "O"]
        return lst[self.value]

    def get_color(self):
        lst = ["gray", "black", "green"]
        return lst[self.value]

class BoardColumn:
    def __init__(self, size, saved_steps_left=None, new_steps_left=None):
        self.size = size
        if saved_steps_left is None:
            saved_steps_left = self.size - 1

        if new_steps_left is None:
            new_steps_left = saved_steps_left

        self.saved_steps_left = saved_steps_left
        self.new_steps_left = new_steps_left

    def advance(self):
        self.new_steps_left -= 1
        assert self.new_steps_left >= 0, "Advancing further should be impossible!"

    def stop(self):
        self.saved_steps_left = self.new_steps_left

    def __call__(self):
        num_saved = self.size - self.saved_steps_left
        num_new = self.size - self.new_steps_left - num_saved
        num_empty = self.size - num_saved - num_new
        return ([BoardColumnState.SAVED] * num_saved +
                [BoardColumnState.PROGRESS] * num_new +
                [BoardColumnState.EMPTY] * num_empty)

    def __str__(self):
        return "".join([str(x) for x in self()])

class Board:
    def __init__(self, column_sizes, saved_steps_remaining, active_steps_remaining):
        self._column_sizes = column_sizes
        self.progresses = [BoardColumn(column_size, saved, active)
                           for column_size, saved, active in zip(column_sizes,
                                                                 saved_steps_remaining,
                                                                 active_steps_remaining)]

    def advance(self, *cols):
        for col in cols:
            self.progresses[col].advance()

    def stop(self):
        for progress in self.progresses:
            progress.stop()

    def __call__(self):
        return [x() for x in self.progresses]

    def __str__(self):
        s = "-----BOARD-----\n"
        for progress in self.progresses:
            s += str(progress)
        s += "---------------"
        return s

class CantStopEnv(gym.Env):
    metadata = {"render_modes": {"human", "rgb_array"}, "render_fps": 4}
    def __init__(self, render_mode=None, render_fps=60):
        self._state = None
        self.observation_space: CantStopObservation = CantStopObservation()
        self.action_space = spaces.Discrete(77)
        self.render_fps = render_fps
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be in {self.metadata['render_modes']}, of which {render_mode} is not.")
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.board = None
        self.window_size = (600, 600)
        self.dice_combo_view_width_prop = 0.0
        self.play_history = None

    def reset(self,
              *args,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[CantStopState, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._state = self.observation_space.sample()
        if self.render_mode == "human":
            self._render_frame()
        return self._state, {}

    def render(self):
        if self.render_mode in ("rgb_array",):
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

        self.board = Board(self.observation_space.column_sizes,
                           self._state._saved_steps_remaining,
                           self._state._active_steps_remaining)

        canvas = pygame.Surface(self.window_size)
        progress_canvas = pygame.Surface(((1-self.dice_combo_view_width_prop)*self.window_size[0], self.window_size[1]))
        #dice_canvas = pygame.Surface((self.dice_combo_view_width_prop*self.window_size[0], self.window_size[1]))
        canvas.fill((255, 255, 255))
        progress_canvas.fill((255, 255, 255))

        base_width, base_height = progress_canvas.get_width(), progress_canvas.get_height()
        square_width = min(base_width, base_height) / len(self.observation_space.column_sizes)

        board_data = self.board()
        for col_idx, col in enumerate(board_data):
            for square_idx, square_data in enumerate(col):
                pygame.draw.rect(progress_canvas,
                                 square_data.get_color(),
                                 pygame.Rect(col_idx * square_width,
                                             (len(board_data) - square_idx) * square_width,
                                             square_width,
                                             square_width),
                                 border_radius=5)

        """pygame.draw.rect(progress_canvas)
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )"""

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
        if isinstance(self._state.current_action, StopContinueChoice):
            if action == StopContinueAction.CONTINUE:
                reward = self._state.perform_continue()
                return self._state, reward, False, False, {}

            elif action == StopContinueAction.STOP:
                reward, completed = self._state.perform_stop()
                return self._state, reward, completed, False, {}

        elif isinstance(self._state.current_action, ProgressActionSet):
            self._state.perform_progression(action)
            return self._state, 0, False, False, {}

        raise ValueError(f"Bad step inputted.")
