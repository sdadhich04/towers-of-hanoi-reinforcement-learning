from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import TypeAlias


State: TypeAlias = tuple[int, ...]
Action: TypeAlias = tuple[int, int]
QTable: TypeAlias = dict[tuple[State, Action], float]
VTable: TypeAlias = dict[State, float]
Policy: TypeAlias = dict[State, Action]


@dataclass(frozen=True)
class TowersOfHanoiMdp:
    disks: int = 3
    gamma: float = 0.95
    slip_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.disks <= 0:
            raise ValueError("disks must be positive")
        if not 0.0 <= self.slip_probability < 1.0:
            raise ValueError("slip_probability must be in [0, 1)")

    @property
    def states(self) -> list[State]:
        return list(product(range(3), repeat=self.disks))

    @property
    def start_state(self) -> State:
        return tuple(0 for _ in range(self.disks))

    @property
    def goal_state(self) -> State:
        return tuple(2 for _ in range(self.disks))

    @property
    def nonterminal_states(self) -> list[State]:
        return [state for state in self.states if state != self.goal_state]

    @property
    def actions(self) -> list[Action]:
        return [(source, target) for source in range(3) for target in range(3) if source != target]

    def legal_actions(self, state: State) -> list[Action]:
        if state == self.goal_state:
            return []
        top_disk_by_peg: dict[int, int] = {}
        for disk, peg in enumerate(state):
            top_disk_by_peg.setdefault(peg, disk)

        legal: list[Action] = []
        for source, target in self.actions:
            moving_disk = top_disk_by_peg.get(source)
            target_disk = top_disk_by_peg.get(target)
            if moving_disk is None:
                continue
            if target_disk is None or moving_disk < target_disk:
                legal.append((source, target))
        return legal

    def step(self, state: State, action: Action) -> State:
        if action not in self.legal_actions(state):
            return state
        source, target = action
        moving_disk = min(disk for disk, peg in enumerate(state) if peg == source)
        data = list(state)
        data[moving_disk] = target
        return tuple(data)

    def transitions(self, state: State, action: Action) -> list[tuple[float, State, float]]:
        if state == self.goal_state:
            return [(1.0, state, 0.0)]
        intended = self.step(state, action)
        reward = 100.0 if intended == self.goal_state else -1.0
        if self.slip_probability == 0.0:
            return [(1.0, intended, reward)]

        stay_probability = self.slip_probability
        move_probability = 1.0 - stay_probability
        return [
            (move_probability, intended, reward),
            (stay_probability, state, -1.0),
        ]


def initial_v_table(mdp: TowersOfHanoiMdp) -> VTable:
    return {state: 0.0 for state in mdp.states}


def initial_q_table(mdp: TowersOfHanoiMdp) -> QTable:
    return {(state, action): 0.0 for state in mdp.nonterminal_states for action in mdp.actions}


def value_iteration(mdp: TowersOfHanoiMdp, v_table: VTable) -> tuple[VTable, QTable, float]:
    new_v_table = v_table.copy()
    q_table: QTable = initial_q_table(mdp)
    max_delta = 0.0

    for state in mdp.nonterminal_states:
        legal_actions = mdp.legal_actions(state)
        if not legal_actions:
            continue
        for action in mdp.actions:
            q_table[(state, action)] = sum(
                probability * (reward + mdp.gamma * v_table[next_state])
                for probability, next_state, reward in mdp.transitions(state, action)
            )
        best_value = max(q_table[(state, action)] for action in legal_actions)
        max_delta = max(max_delta, abs(best_value - v_table[state]))
        new_v_table[state] = best_value

    return new_v_table, q_table, max_delta


def extract_policy(mdp: TowersOfHanoiMdp, q_table: QTable) -> Policy:
    policy: Policy = {}
    for state in mdp.nonterminal_states:
        legal_actions = mdp.legal_actions(state)
        if legal_actions:
            policy[state] = max(legal_actions, key=lambda action: q_table[(state, action)])
    return policy


def q_update(
    mdp: TowersOfHanoiMdp,
    q_table: QTable,
    transition: tuple[State, Action, float, State],
    alpha: float,
) -> None:
    state, action, reward, next_state = transition
    old_q = q_table[(state, action)]
    next_actions = mdp.legal_actions(next_state)
    next_value = max((q_table[(next_state, a)] for a in next_actions), default=0.0)
    q_table[(state, action)] = old_q + alpha * (reward + mdp.gamma * next_value - old_q)


def extract_v_table(mdp: TowersOfHanoiMdp, q_table: QTable) -> VTable:
    values: VTable = {}
    for state in mdp.states:
        legal_actions = mdp.legal_actions(state)
        values[state] = max((q_table[(state, action)] for action in legal_actions), default=0.0)
    return values


def custom_epsilon(n_step: int) -> float:
    return 0.2 + 0.1 / sqrt(n_step + 1)


def custom_alpha(n_step: int) -> float:
    return max(0.05, 0.8 / sqrt(n_step + 1))
