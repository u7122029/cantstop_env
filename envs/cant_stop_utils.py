import math
from enum import Enum

import numpy as np
from typing import Protocol, runtime_checkable
from typing_extensions import TypeVar, Optional

from tabulate import tabulate
from abc import ABC, abstractmethod

@runtime_checkable
class CantStopAction(Protocol):
    def encode(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def decode(idx):
        raise NotImplementedError()

class StopContinueAction(Enum):
    STOP = 0
    CONTINUE = 1

    def encode(self):
        return self.value

    @staticmethod
    def decode(idx):
        lst = [StopContinueAction.STOP, StopContinueAction.CONTINUE]
        return lst[idx]

    def __repr__(self):
        x = ["STOP", "CONTINUE"]
        return x[self.value]

    def __str__(self):
        return self.__repr__()

class ProgressAction:
    def __init__(self, a, b=-1):
        self._smaller = min(a,b)
        self._larger = max(a,b)

    @property
    def smaller(self):
        return self._smaller

    @property
    def larger(self):
        return self._larger

    def copy(self):
        return ProgressAction(self.smaller, self.larger)

    def encode(self) -> int:
        return (self.smaller + 1) * (22 - self.smaller) // 2 + self.larger

    def split(self, saved_dists: np.ndarray, active_dists: np.ndarray) -> list:
        active = active_dists < saved_dists
        num_active = active.sum()
        full: np.ndarray = active_dists == 0
        out = []
        if self.smaller == self.larger and (active[self.smaller] or num_active <= 2) and active_dists[self.smaller] >= 2:
            out.append(ProgressAction(self.smaller, self.larger))
            return out
        elif self.smaller == self.larger and (active[self.smaller] or num_active <= 2) and active_dists[self.smaller] == 1:
            out.append(ProgressAction(self.smaller, -1))
            return out
        elif self.smaller == self.larger and (active[self.smaller] or num_active <= 2) and active_dists[self.smaller] == 0:
            return out

        if num_active in (0, 1):
            if not full[self.smaller] and not full[self.larger]: out.append(ProgressAction(self.smaller, self.larger))
            elif not full[self.smaller]: out.append(ProgressAction(self.smaller, -1))
            elif not full[self.larger]: out.append(ProgressAction(self.larger, -1))
        elif num_active == 2:
            if active[self.smaller] or active[self.larger]:
                if not full[self.smaller] and not full[self.larger]: out.append(ProgressAction(self.smaller, self.larger))
                elif not full[self.smaller]: out.append(ProgressAction(self.smaller, -1))
                elif not full[self.larger]: out.append(ProgressAction(self.larger, -1))
            else:
                if not full[self.smaller]: out.append(ProgressAction(self.smaller, -1))
                if not full[self.larger]: out.append(ProgressAction(self.larger, -1))
        elif active[self.smaller] and active[self.larger]:
            if not full[self.smaller] and not full[self.larger]: out.append(ProgressAction(self.smaller, self.larger))
            elif not full[self.smaller]: out.append(ProgressAction(self.smaller, -1))
            elif not full[self.larger]: out.append(ProgressAction(self.larger, -1))

        return out

    def is_well_formed(self) -> bool:
        if self.smaller == -1:
            return 0 <= self.larger <= 10
        return 0 <= self.smaller <= self.larger <= 10

    @staticmethod
    def decode(idx: int):
        if idx < 0:
            raise ValueError("Invalid idx value")
        if idx <= 10:
            return ProgressAction(-1, idx)

        e = idx
        sqrt_val = math.sqrt(617 - 8 * e)
        s_floor = int((23 - sqrt_val) // 2)

        # Check s_floor
        if 0 <= s_floor <= 10:
            base = (s_floor + 1) * (22 - s_floor) // 2
            if base + s_floor <= e <= base + 10:
                return ProgressAction(s_floor, e - base)

        # Check s_floor - 1
        s_candidate = s_floor - 1
        if 0 <= s_candidate <= 10:
            base = (s_candidate + 1) * (22 - s_candidate) // 2
            if base + s_candidate <= e <= base + 10:
                return ProgressAction(s_candidate, e - base)

        # Check s_floor + 1
        s_candidate = s_floor + 1
        if 0 <= s_candidate <= 10:
            base = (s_candidate + 1) * (22 - s_candidate) // 2
            if base + s_candidate <= e <= base + 10:
                return ProgressAction(s_candidate, e - base)

        raise ValueError("Invalid idx value")

    @classmethod
    def from_dice_combinations(cls, d1, d2, d3, d4):
        return cls(d1 + d2, d3 + d4), cls(d1 + d3, d2 + d4), cls(d1 + d4, d2 + d3)

    def __repr__(self):
        return f"ProgressAction({self.smaller}, {self.larger})" if self.smaller != -1 else f"ProgressAction({self.larger})"

    def __eq__(self, other):
        if other is None: return False
        if not isinstance(other, ProgressAction): return False
        return self.smaller == other.smaller and self.larger == other.larger

    def __hash__(self):
        return hash((self.smaller, self.larger))

class CantStopActionChoice(ABC):
    def __init__(self, name, choices):
        self._name = name
        self._choices = frozenset(choices)

    @property
    def choices(self):
        return self._choices

    @property
    def name(self):
        return self._name

    def __contains__(self, item):
        return item in self._choices

    def __iter__(self):
        return iter(self._choices)

    @abstractmethod
    def copy(self):
        raise NotImplementedError()
        #return CantStopActionChoice(self._name, [x for x in self._choices])


class StopContinueChoice(CantStopActionChoice):
    def __init__(self):
        super().__init__("StopContinueChoice", [StopContinueAction.STOP, StopContinueAction.CONTINUE])

    def copy(self):
        return StopContinueChoice()


class ProgressActionChoice(CantStopActionChoice):
    def __init__(self, items):
        super().__init__("ProgressActionChoice", items)

    def as_encoded(self):
        advances_encoded = np.zeros(77, dtype=int)

        for pa in self:
            advances_encoded[pa.encode()] = True
        return advances_encoded

    @staticmethod
    def from_encoded(encoded: np.ndarray):
        """
        Converts an encoding into a set of actions.
        :param encoded:
        :return:
        """
        raise NotImplementedError()

    def copy(self):
        return ProgressActionChoice([x for x in self.choices])

class CantStopState:
    def __init__(self,
                 saved_steps_remaining,
                 active_steps_remaining,
                 current_action: CantStopActionChoice):
        self._num_turns = 0
        self._column_limits = np.array([3,5,7,9,11,13,11,9,7,5,3])
        self._saved_steps_remaining: np.ndarray = saved_steps_remaining.astype(int)
        self._active_steps_remaining: np.ndarray = active_steps_remaining.astype(int)
        self._current_action: Optional[CantStopActionChoice] = current_action

    @property
    def saved_steps_remaining(self):
        return self._saved_steps_remaining

    @property
    def active_steps_remaining(self):
        return self._active_steps_remaining

    @property
    def column_limits(self):
        return self._column_limits

    @property
    def current_action(self):
        return self._current_action

    @property
    def full_saved_cols(self) -> np.ndarray:
        return self._saved_steps_remaining == 0

    @property
    def full_active_cols(self):
        return (self._active_steps_remaining == 0) & self.active_cols

    @property
    def active_cols(self):
        return self._active_steps_remaining < self._saved_steps_remaining

    @property
    def is_complete(self):
        return np.sum(self._saved_steps_remaining == 0) >= 3

    @property
    def num_turns(self):
        return self._num_turns

    @property
    def active_advances(self):
        return self.saved_steps_remaining - self.active_steps_remaining

    def to_np_embedding(self):
        if self.current_action is None:
            return None
        base = np.concat([self.saved_steps_remaining, self.saved_steps_remaining])
        if isinstance(self.current_action, ProgressActionChoice):
            base = np.concat([base, self.current_action.as_encoded()])
        return base

    def compute_reward(self, action_performed: CantStopAction, busted: bool=False):
        # NOTE: DEPRECATED
        if not isinstance(action_performed, CantStopAction):
            raise ValueError(f"action_performed type should be StopContinueAction, not {type(action_performed)}.")

        reward = 0
        if isinstance(action_performed, ProgressAction):
            if action_performed.smaller != -1:
                reward += 1 / self.column_limits[action_performed.smaller]
            reward += 1 / self.column_limits[action_performed.larger]
            return reward

        completed_mask = self.full_active_cols.astype(int) * 100
        if action_performed is StopContinueAction.STOP:
            return np.sum((self._saved_steps_remaining - self._active_steps_remaining).astype(int))
        elif action_performed is StopContinueAction.CONTINUE:
            if busted: return -np.sum(completed_mask * (self._saved_steps_remaining - self._active_steps_remaining))
            return -1
        else:
            raise ValueError(f"action_performed is of incompatible form {action_performed}.")

    def perform_continue(self) -> bool:
        """

        :return:
        """
        if not isinstance(self.current_action, StopContinueChoice):
            raise Exception("You cannot roll the dice again while selecting a combination.")

        busted = False
        dice_rolls = np.random.choice(6, size=(4,)).tolist()
        all_progressions = ProgressAction.from_dice_combinations(*dice_rolls)

        possible_advances = []
        for p in all_progressions:
            possible_advances.extend(p.split(self._saved_steps_remaining, self._active_steps_remaining))

        if not possible_advances:
            self._num_turns += 1
            self._active_steps_remaining = self._saved_steps_remaining.copy()
            self._current_action = StopContinueChoice()
        else:
            self._current_action = ProgressActionChoice(possible_advances)
            busted = True
        return busted

    def perform_stop(self) -> bool:
        if not isinstance(self.current_action, StopContinueChoice):
            raise Exception("You cannot stop while selecting a combination.")

        self._num_turns += 1
        completed = False
        self._saved_steps_remaining = self._active_steps_remaining.copy()
        if self.is_complete:
            completed = True
            self._current_action = None
        return completed


    def perform_progression(self, progression: ProgressAction):
        if not isinstance(self._current_action, ProgressActionChoice):
            raise Exception("Current action selection is not a set.")

        if progression not in self._current_action:
            raise ValueError("Provided action is not in the set of possible actions.")

        if progression.smaller != -1: self._active_steps_remaining[progression.smaller] -= 1
        self._active_steps_remaining[progression.larger] -= 1

        self._current_action = StopContinueChoice()

        # Don't return reward.

    def as_tuple(self):
        return self._saved_steps_remaining, self._active_steps_remaining, self.current_action

    def set_action_choices(self, dice_rolls):
        all_progressions = ProgressAction.from_dice_combinations(*dice_rolls)

        possible_advances = []
        for p in all_progressions:
            possible_advances.extend(p.split(self._saved_steps_remaining, self._active_steps_remaining))

        self._current_action = set(possible_advances)

    def is_well_formed(self) -> bool:
        # Fast type checks with short-circuiting
        if not isinstance(self._saved_steps_remaining, np.ndarray):
            return False
        if not isinstance(self._active_steps_remaining, np.ndarray):
            return False
        if not isinstance(self._current_action, (type(None), CantStopActionChoice)):
            return False

        # Early check: bounds must hold
        saved = self._saved_steps_remaining
        active = self._active_steps_remaining
        if not np.all((0 <= active) & (active <= saved) & (saved <= self.column_limits)):
            return False

        # If 3 or more columns are full, action must be None
        if np.sum(self.full_saved_cols) >= 3 and self._current_action is not None:
            return False

        # Action-specific validation
        action = self._current_action
        if isinstance(action, ProgressActionChoice):
            active_cols = self.active_cols
            num_active_cols = np.count_nonzero(active_cols)

            for pa in action:
                # Avoid function call if already False
                if not pa.is_well_formed():
                    return False

                larger = pa.larger
                if saved[larger] == 0 or active[larger] == 0:
                    return False

                if num_active_cols == 3 and not active_cols[larger]:
                    return False

                smaller = pa.smaller
                if smaller != -1:
                    if num_active_cols == 2:
                        if smaller != larger:
                            if not (active_cols[smaller] or active_cols[larger]):
                                return False
                        elif active[smaller] <= 1:
                            return False
                    elif num_active_cols == 3:
                        if not (active_cols[smaller] and active_cols[larger]):
                            return False

                    if smaller == larger and active[smaller] < 2:
                        return False

        return True

    def raw_squares(self):
        lst = []
        fsc = self.full_saved_cols
        num_black = self.column_limits - self.saved_steps_remaining
        num_green = self.column_limits - self.active_steps_remaining - num_black
        num_gray = self.active_steps_remaining
        for i in range(len(self.column_limits)):
            if fsc[i]:
                x = ["blue"] * self.column_limits[i]
            else:
                x = ["black"] * num_black[i] + ["green"] * num_green[i] + ["gray"] * num_gray[i]
            lst.append(x)
        return lst


    def __repr__(self):
        headers = ["cols"] + list(range(0, 11))
        table = [["old"] + self._saved_steps_remaining.tolist(),
                 ["new"] + self._active_steps_remaining.tolist(),
                 ["active"] + ["O" if x else "X" for x in self.active_cols.tolist()]]
        s = f"----------CSState----------\n"
        s += str(tabulate(table, tablefmt="psql", headers=headers)) + "\n"
        s += f"Action Choices: {self._current_action},\n"
        s += "---------------------------"
        return s

    def copy(self):
        return CantStopState(self._saved_steps_remaining.copy(),
                             self._active_steps_remaining.copy(),
                             self._current_action.copy() if self._current_action is not None else None)

def process_action(action: CantStopAction):
    print(action)
    #print(f"Encoded: {action.encode()}")
    #print(f"Decoded: {action.decode(action.encode())}")

if __name__ == "__main__":
    process_action(1)#StopContinueAction.STOP)