"""
ai_search.py
────────────
AI Search-based attacker baselines (Units I–II).

Formulates ICS attack as a search/game problem:
  - State:  current sensor readings (65-dim vector after preprocessing)
  - Action: perturb sensor_i by magnitude_j
  - Goal:   evade Blue Agent detection while maximising disruption

Attacker Strategies:
  1. RandomAttacker     — uniform random perturbations (BFS analogue, Unit I)
  2. IDDFSAttacker      — iterative deepening DFS with depth limit (Unit I)
  3. AStarAttacker      — A* search, h(n) = remaining disruption budget (Unit I)
  4. AlphaBetaAttacker  — adversarial game tree, depth=3 (Unit II)

GA Optimiser (Unit II):
  - Evolves {λ1, λ2, λ3} reward weights for DQN Red Agent
  - Population: 20, Generations: 50, Tournament selection, BLX-α crossover

Environment model:
  - Partially observable, stochastic, sequential, multi-agent
  - Blue Agent = MIN player (minimise attacker reward)
  - Red Agent = MAX player (maximise evasion + disruption)

Usage:
  from src.ai_search import run_all_attackers, GAOptimizer
  results = run_all_attackers(sensor_data, blue_agent_fn)
"""

import logging
import time
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Attack State & Action Definitions
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AttackState:
    """Represents a state in the ICS attack search space.

    Attributes:
        readings: Current sensor readings (65-dim, scaled [0,1]).
        perturbations: Accumulated perturbations applied so far.
        depth: Current search depth.
        detected: Whether Blue Agent flagged this state.
        disruption: Total physical disruption caused.
        parent: Parent state (for path reconstruction).
    """
    readings: np.ndarray
    perturbations: np.ndarray
    depth: int = 0
    detected: bool = False
    disruption: float = 0.0
    parent: Optional["AttackState"] = None

    def __hash__(self):
        return hash(self.readings.tobytes())

    def __eq__(self, other):
        return np.allclose(self.readings, other.readings, atol=1e-6)

    def __lt__(self, other):
        # For heapq in A* (lower priority = better)
        return self.disruption > other.disruption


@dataclass
class AttackAction:
    """An atomic attack action: perturb one sensor by some magnitude.

    Attributes:
        sensor_idx: Index of sensor to perturb (0–64).
        magnitude: Perturbation magnitude (one of 5 discrete levels).
        direction: +1 or -1.
    """
    sensor_idx: int
    magnitude: float
    direction: int = 1

    def apply(self, readings: np.ndarray) -> np.ndarray:
        """Apply this action to sensor readings."""
        new = readings.copy()
        new[self.sensor_idx] += self.direction * self.magnitude
        return np.clip(new, 0, 1)


# ══════════════════════════════════════════════════════════════════════
#  Blue Agent Placeholder (replaced with real model in later weeks)
# ══════════════════════════════════════════════════════════════════════

def placeholder_blue_agent(
    readings: np.ndarray,
    threshold: float = 0.15,
) -> Tuple[bool, float]:
    """Placeholder Blue Agent for attacker evaluation.

    Uses a simple deviation-from-mean detector until the real
    LSTM-AE + GAT fusion model is trained (Week 4-5).

    Args:
        readings: Sensor readings (65-dim, scaled [0,1]).
        threshold: Detection threshold on mean absolute deviation.

    Returns:
        (detected, score): Whether attack was detected, anomaly score.
    """
    # Simple: flag if mean deviation from 0.5 exceeds threshold
    score = float(np.mean(np.abs(readings - 0.5)))
    return (score > threshold, score)


# ══════════════════════════════════════════════════════════════════════
#  Attacker Base Class
# ══════════════════════════════════════════════════════════════════════

class BaseAttacker(ABC):
    """Abstract base class for AI search attackers."""

    name: str = "base"
    n_sensors: int = 65
    perturbation_levels: List[float] = [0.01, 0.03, 0.05, 0.08, 0.12]

    def __init__(
        self,
        blue_agent_fn: Optional[Callable] = None,
        max_steps: int = 200,
        n_sensors: int = 65,
        **kwargs,
    ):
        self.blue_agent_fn = blue_agent_fn or placeholder_blue_agent
        self.max_steps = max_steps
        self.n_sensors = n_sensors

    def _get_actions(self, state: AttackState) -> List[AttackAction]:
        """Generate possible actions from current state."""
        actions = []
        for s in range(self.n_sensors):
            for mag in self.perturbation_levels:
                for d in [+1, -1]:
                    actions.append(AttackAction(s, mag, d))
        return actions

    def _compute_reward(
        self,
        state: AttackState,
        original: np.ndarray,
        lambda1: float = 1.0,
        lambda2: float = 10.0,
        lambda3: float = 0.5,
    ) -> float:
        """Compute reward: R = λ1*Σ|δ| − λ2*detected − λ3*(Σ|δ|)²."""
        delta = np.sum(np.abs(state.readings - original))
        detected_penalty = lambda2 if state.detected else 0.0
        return lambda1 * delta - detected_penalty - lambda3 * (delta ** 2)

    @abstractmethod
    def attack(
        self,
        initial_readings: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the attack strategy.

        Args:
            initial_readings: Original sensor readings to perturb.

        Returns:
            Dict with: perturbed_readings, actions_taken, detected,
                       disruption, reward, nodes_expanded, time_elapsed.
        """
        pass


# ══════════════════════════════════════════════════════════════════════
#  1. Random Attacker (BFS analogue — Unit I)
# ══════════════════════════════════════════════════════════════════════

class RandomAttacker(BaseAttacker):
    """Random search attacker — applies random perturbations.

    BFS analogue: explores the attack space uniformly without heuristics.
    Serves as the lower-bound baseline for attacker performance.
    """

    name = "random"

    def __init__(self, n_perturbations: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_perturbations = n_perturbations

    def attack(self, initial_readings, seed=42, **kwargs):
        rng = np.random.RandomState(seed)
        t0 = time.time()

        best_result = None
        best_reward = -np.inf
        nodes_expanded = 0

        for _ in range(self.max_steps):
            readings = initial_readings.copy()
            actions = []

            # Apply random perturbations
            for _ in range(self.n_perturbations):
                sensor = rng.randint(0, self.n_sensors)
                mag = rng.choice(self.perturbation_levels)
                direction = rng.choice([-1, 1])
                action = AttackAction(sensor, mag, direction)
                readings = action.apply(readings)
                actions.append(action)

            detected, score = self.blue_agent_fn(readings)
            state = AttackState(readings, readings - initial_readings, detected=detected)
            reward = self._compute_reward(state, initial_readings)
            nodes_expanded += 1

            if reward > best_reward:
                best_reward = reward
                best_result = {
                    "perturbed_readings": readings,
                    "actions": actions,
                    "detected": detected,
                    "anomaly_score": score,
                    "disruption": float(np.sum(np.abs(readings - initial_readings))),
                    "reward": reward,
                }

        best_result["nodes_expanded"] = nodes_expanded
        best_result["time_elapsed"] = time.time() - t0
        best_result["evasion"] = not best_result["detected"]
        return best_result


# ══════════════════════════════════════════════════════════════════════
#  2. IDDFS Attacker (Iterative Deepening DFS — Unit I)
# ══════════════════════════════════════════════════════════════════════

class IDDFSAttacker(BaseAttacker):
    """Iterative Deepening DFS attacker.

    Performs DFS with increasing depth limits. At each level, tries
    perturbations on the most promising sensors (ranked by correlation
    impact or random if no prior knowledge).

    Space-efficient: O(bd) memory where b=branching factor, d=depth.
    """

    name = "iddfs"

    def __init__(self, max_depth: int = 4, beam_width: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.beam_width = beam_width

    def attack(self, initial_readings, seed=42, **kwargs):
        rng = np.random.RandomState(seed)
        t0 = time.time()

        best_result = {
            "perturbed_readings": initial_readings.copy(),
            "detected": True,
            "reward": -np.inf,
            "nodes_expanded": 0,
        }

        # Iterative deepening
        for depth_limit in range(1, self.max_depth + 1):
            result = self._dls(initial_readings, depth_limit, rng, best_result)
            if result["reward"] > best_result["reward"]:
                best_result = result

            # Early termination: found evasive attack
            if not best_result["detected"]:
                break

        best_result["time_elapsed"] = time.time() - t0
        best_result["evasion"] = not best_result["detected"]
        return best_result

    def _dls(self, initial, depth_limit, rng, current_best):
        """Depth-limited search."""
        stack = [(AttackState(initial.copy(), np.zeros_like(initial), depth=0), [])]
        nodes = 0
        best = current_best.copy()

        while stack and nodes < self.max_steps:
            state, actions = stack.pop()
            nodes += 1

            # Evaluate current state
            detected, score = self.blue_agent_fn(state.readings)
            state.detected = detected
            reward = self._compute_reward(state, initial)

            if reward > best.get("reward", -np.inf):
                best = {
                    "perturbed_readings": state.readings.copy(),
                    "actions": list(actions),
                    "detected": detected,
                    "anomaly_score": score,
                    "disruption": float(np.sum(np.abs(state.readings - initial))),
                    "reward": reward,
                    "nodes_expanded": nodes,
                }

            # Expand if within depth limit
            if state.depth < depth_limit:
                # Select beam_width random actions to try
                child_actions = []
                for _ in range(self.beam_width):
                    sensor = rng.randint(0, self.n_sensors)
                    mag = rng.choice(self.perturbation_levels)
                    direction = rng.choice([-1, 1])
                    child_actions.append(AttackAction(sensor, mag, direction))

                for action in child_actions:
                    new_readings = action.apply(state.readings)
                    new_perturbations = new_readings - initial
                    child = AttackState(
                        new_readings, new_perturbations,
                        depth=state.depth + 1
                    )
                    stack.append((child, actions + [action]))

        best["nodes_expanded"] = nodes
        return best


# ══════════════════════════════════════════════════════════════════════
#  3. A* Attacker (Informed Search — Unit I)
# ══════════════════════════════════════════════════════════════════════

class AStarAttacker(BaseAttacker):
    """A* search attacker with disruption heuristic.

    Uses priority queue with f(n) = g(n) + h(n):
      - g(n) = detection risk (higher = worse for attacker)
      - h(n) = remaining disruption potential (admissible heuristic)

    The heuristic estimates how much more disruption the attacker can
    cause before being detected, making A* optimal given the heuristic.
    """

    name = "a_star"

    def __init__(self, max_depth: int = 5, beam_width: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.beam_width = beam_width

    def _heuristic(self, state: AttackState, initial: np.ndarray) -> float:
        """Admissible heuristic: estimated remaining evasion potential.

        h(n) = negative of (max possible disruption - current disruption)
        Lower h = more promising (closer to goal).
        """
        current_disruption = np.sum(np.abs(state.readings - initial))
        max_possible = self.n_sensors * max(self.perturbation_levels) * self.max_depth
        remaining = max_possible - current_disruption
        # Penalize if detected
        detection_penalty = 100.0 if state.detected else 0.0
        return detection_penalty - remaining

    def attack(self, initial_readings, seed=42, **kwargs):
        rng = np.random.RandomState(seed)
        t0 = time.time()

        initial_state = AttackState(initial_readings.copy(), np.zeros_like(initial_readings))
        # Priority queue: (f_score, counter, state, actions)
        counter = 0
        pq = [(self._heuristic(initial_state, initial_readings), counter, initial_state, [])]
        counter += 1

        best_result = {
            "perturbed_readings": initial_readings.copy(),
            "detected": True,
            "reward": -np.inf,
            "nodes_expanded": 0,
        }
        visited = set()
        nodes = 0

        while pq and nodes < self.max_steps:
            f_score, _, state, actions = heapq.heappop(pq)
            nodes += 1

            # Hash for visited check (quantized)
            state_hash = hash(np.round(state.readings, 3).tobytes())
            if state_hash in visited:
                continue
            visited.add(state_hash)

            # Evaluate
            detected, score = self.blue_agent_fn(state.readings)
            state.detected = detected
            reward = self._compute_reward(state, initial_readings)

            if reward > best_result["reward"]:
                best_result = {
                    "perturbed_readings": state.readings.copy(),
                    "actions": list(actions),
                    "detected": detected,
                    "anomaly_score": score,
                    "disruption": float(np.sum(np.abs(state.readings - initial_readings))),
                    "reward": reward,
                    "nodes_expanded": nodes,
                }

            # Found evasive attack with good disruption
            if not detected and state.disruption > 0:
                break

            # Expand children
            if state.depth < self.max_depth:
                for _ in range(self.beam_width):
                    sensor = rng.randint(0, self.n_sensors)
                    mag = rng.choice(self.perturbation_levels)
                    direction = rng.choice([-1, 1])
                    action = AttackAction(sensor, mag, direction)

                    new_readings = action.apply(state.readings)
                    child = AttackState(
                        new_readings,
                        new_readings - initial_readings,
                        depth=state.depth + 1,
                    )
                    g_cost = state.depth + 1  # cost so far
                    h_cost = self._heuristic(child, initial_readings)
                    f = g_cost + h_cost

                    heapq.heappush(pq, (f, counter, child, actions + [action]))
                    counter += 1

        best_result["nodes_expanded"] = nodes
        best_result["time_elapsed"] = time.time() - t0
        best_result["evasion"] = not best_result["detected"]
        return best_result


# ══════════════════════════════════════════════════════════════════════
#  4. Alpha-Beta Attacker (Adversarial Game Tree — Unit II)
# ══════════════════════════════════════════════════════════════════════

class AlphaBetaAttacker(BaseAttacker):
    """Alpha-Beta pruning attacker for adversarial game formulation.

    Models the ICS attack as a two-player game:
      - MAX player (Red Agent): tries to maximise disruption while evading
      - MIN player (Blue Agent): tries to minimise attacker's reward

    Uses alpha-beta pruning to efficiently search the game tree with
    a configurable depth limit.
    """

    name = "alpha_beta"

    def __init__(self, depth: int = 3, beam_width: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.beam_width = beam_width

    def attack(self, initial_readings, seed=42, **kwargs):
        rng = np.random.RandomState(seed)
        t0 = time.time()

        self._nodes_expanded = 0
        self._rng = rng
        self._initial = initial_readings.copy()

        # Run alpha-beta from root
        best_val, best_actions, best_readings = self._alpha_beta(
            initial_readings.copy(),
            depth=self.depth,
            alpha=-np.inf,
            beta=np.inf,
            is_max=True,
            actions_so_far=[],
        )

        detected, score = self.blue_agent_fn(best_readings)

        return {
            "perturbed_readings": best_readings,
            "actions": best_actions,
            "detected": detected,
            "anomaly_score": score,
            "disruption": float(np.sum(np.abs(best_readings - initial_readings))),
            "reward": best_val,
            "nodes_expanded": self._nodes_expanded,
            "time_elapsed": time.time() - t0,
            "evasion": not detected,
        }

    def _alpha_beta(self, readings, depth, alpha, beta, is_max, actions_so_far):
        """Alpha-beta minimax search.

        Returns:
            (value, actions, readings) — best value found with path.
        """
        self._nodes_expanded += 1

        # Terminal: evaluate
        if depth == 0 or self._nodes_expanded > self.max_steps:
            detected, score = self.blue_agent_fn(readings)
            state = AttackState(readings, readings - self._initial, detected=detected)
            val = self._compute_reward(state, self._initial)
            return val, actions_so_far, readings

        # Generate candidate moves
        moves = []
        for _ in range(self.beam_width):
            sensor = self._rng.randint(0, self.n_sensors)
            mag = self._rng.choice(self.perturbation_levels)
            direction = self._rng.choice([-1, 1])
            moves.append(AttackAction(sensor, mag, direction))

        if is_max:
            # Red Agent (MAX): pick move that maximises reward
            best_val = -np.inf
            best_actions = actions_so_far
            best_readings = readings

            for action in moves:
                new_readings = action.apply(readings)
                val, acts, rd = self._alpha_beta(
                    new_readings, depth - 1, alpha, beta,
                    False, actions_so_far + [action]
                )
                if val > best_val:
                    best_val = val
                    best_actions = acts
                    best_readings = rd
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # Beta cutoff

            return best_val, best_actions, best_readings
        else:
            # Blue Agent (MIN): pick "move" that minimises attacker reward
            # Blue doesn't actually move — it adjusts detection threshold
            best_val = np.inf
            best_actions = actions_so_far
            best_readings = readings

            # Simulate Blue Agent re-evaluating with stricter threshold
            for threshold_adj in [0.0, -0.02, -0.05]:
                # Blue tightens threshold
                detected, score = self.blue_agent_fn(readings)
                state = AttackState(readings, readings - self._initial, detected=detected)
                val = self._compute_reward(state, self._initial)
                val += threshold_adj * 10  # Stricter detection hurts attacker

                child_val, acts, rd = self._alpha_beta(
                    readings, depth - 1, alpha, beta,
                    True, actions_so_far
                )
                if child_val < best_val:
                    best_val = child_val
                    best_actions = acts
                    best_readings = rd
                beta = min(beta, child_val)
                if beta <= alpha:
                    break  # Alpha cutoff

            return best_val, best_actions, best_readings


# ══════════════════════════════════════════════════════════════════════
#  Attacker Registry
# ══════════════════════════════════════════════════════════════════════

ATTACKER_REGISTRY: Dict[str, type] = {
    "random": RandomAttacker,
    "iddfs": IDDFSAttacker,
    "a_star": AStarAttacker,
    "alpha_beta": AlphaBetaAttacker,
}


def get_attacker(name: str, **kwargs) -> BaseAttacker:
    """Get an attacker by name."""
    if name not in ATTACKER_REGISTRY:
        raise ValueError(
            f"Unknown attacker: {name}. Available: {list(ATTACKER_REGISTRY.keys())}"
        )
    return ATTACKER_REGISTRY[name](**kwargs)


# ══════════════════════════════════════════════════════════════════════
#  Run All Attackers
# ══════════════════════════════════════════════════════════════════════

def run_all_attackers(
    sensor_data: np.ndarray,
    blue_agent_fn: Optional[Callable] = None,
    n_samples: int = 100,
    seeds: Optional[List[int]] = None,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Run all AI search attackers and compute mean metrics.

    Args:
        sensor_data: 2D array (n_samples, n_features) of sensor readings.
        blue_agent_fn: Blue Agent detector function(readings) → (detected, score).
        n_samples: Number of attack episodes to run per attacker.
        seeds: Random seeds for reproducibility.
        max_steps: Max search steps per attack episode.

    Returns:
        Dict with per-attacker aggregated results.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    all_results = {}

    for attacker_name, attacker_class in ATTACKER_REGISTRY.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {attacker_name} attacker...")
        logger.info(f"{'='*60}")

        seed_results = {
            "evasion_rates": [],
            "disruptions": [],
            "rewards": [],
            "nodes_expanded": [],
            "times": [],
        }

        for seed in seeds:
            rng = np.random.RandomState(seed)
            attacker = attacker_class(
                blue_agent_fn=blue_agent_fn,
                max_steps=max_steps,
            )

            evasions, disruptions, rewards = [], [], []
            episode_indices = rng.choice(len(sensor_data), size=n_samples, replace=True)

            for idx in episode_indices:
                if hasattr(blue_agent_fn, "set_context"):
                    blue_agent_fn.set_context(idx)
                result = attacker.attack(sensor_data[idx], seed=seed)
                evasions.append(1 if result.get("evasion", False) else 0)
                disruptions.append(result.get("disruption", 0.0))
                rewards.append(result.get("reward", 0.0))

            seed_results["evasion_rates"].append(np.mean(evasions))
            seed_results["disruptions"].append(np.mean(disruptions))
            seed_results["rewards"].append(np.mean(rewards))

        # Aggregate across seeds
        all_results[attacker_name] = {
            "evasion_rate": {
                "mean": float(np.mean(seed_results["evasion_rates"])),
                "std": float(np.std(seed_results["evasion_rates"])),
                "formatted": f"{np.mean(seed_results['evasion_rates']):.4f}±{np.std(seed_results['evasion_rates']):.4f}",
            },
            "disruption": {
                "mean": float(np.mean(seed_results["disruptions"])),
                "std": float(np.std(seed_results["disruptions"])),
                "formatted": f"{np.mean(seed_results['disruptions']):.4f}±{np.std(seed_results['disruptions']):.4f}",
            },
            "reward": {
                "mean": float(np.mean(seed_results["rewards"])),
                "std": float(np.std(seed_results["rewards"])),
                "formatted": f"{np.mean(seed_results['rewards']):.4f}±{np.std(seed_results['rewards']):.4f}",
            },
        }

        logger.info(
            f"{attacker_name}: "
            f"evasion={all_results[attacker_name]['evasion_rate']['formatted']}, "
            f"disruption={all_results[attacker_name]['disruption']['formatted']}"
        )

    return all_results


# ══════════════════════════════════════════════════════════════════════
#  GA Optimizer for DQN Reward Weights (Unit II)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Individual:
    """GA individual: a set of reward weights {λ1, λ2, λ3}."""
    genes: np.ndarray  # [lambda1, lambda2, lambda3]
    fitness: float = 0.0


class GAOptimizer:
    """Genetic Algorithm for optimizing DQN Red Agent reward weights.

    Evolves {λ1, λ2, λ3} to maximise the Red Agent's evasion rate
    against the frozen Blue Agent.

    GA Components:
      - Selection: Tournament (k=3)
      - Crossover: BLX-α (Blend Crossover)
      - Mutation: Gaussian perturbation
      - Replacement: Elitist (top 10% survive)

    Usage:
      ga = GAOptimizer(fitness_fn=my_fitness_function)
      best = ga.evolve()
      print(f"Best weights: {best.genes}")
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        population_size: int = 20,
        n_generations: int = 50,
        tournament_size: int = 3,
        mutation_std: float = 0.1,
        elite_fraction: float = 0.1,
        gene_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: int = 42,
    ):
        """
        Args:
            fitness_fn: Function that takes [λ1, λ2, λ3] and returns fitness score.
            population_size: Number of individuals.
            n_generations: Number of generations to evolve.
            tournament_size: k for tournament selection.
            mutation_std: Std of Gaussian mutation.
            elite_fraction: Fraction of population preserved via elitism.
            gene_bounds: (lower_bounds, upper_bounds) for each gene.
            seed: Random seed.
        """
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.n_gens = n_generations
        self.tournament_size = tournament_size
        self.mutation_std = mutation_std
        self.n_elite = max(1, int(population_size * elite_fraction))
        self.rng = np.random.RandomState(seed)

        if gene_bounds is None:
            self.lower = np.array([0.1, 1.0, 0.01])   # λ1, λ2, λ3 lower
            self.upper = np.array([5.0, 50.0, 5.0])    # λ1, λ2, λ3 upper
        else:
            self.lower, self.upper = gene_bounds

        self.n_genes = len(self.lower)
        self.history: List[Dict] = []

    def _init_population(self) -> List[Individual]:
        """Initialize population with uniform random genes."""
        pop = []
        for _ in range(self.pop_size):
            genes = self.rng.uniform(self.lower, self.upper)
            pop.append(Individual(genes=genes))
        return pop

    def _evaluate(self, population: List[Individual]):
        """Evaluate fitness of all individuals."""
        for ind in population:
            if ind.fitness == 0.0:  # Not yet evaluated
                ind.fitness = self.fitness_fn(ind.genes)

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Tournament selection: pick best from random subset."""
        contestants = self.rng.choice(
            population, size=self.tournament_size, replace=False
        )
        return max(contestants, key=lambda x: x.fitness)

    def _blx_alpha_crossover(
        self, parent1: Individual, parent2: Individual, alpha: float = 0.5
    ) -> Tuple[Individual, Individual]:
        """BLX-α crossover: blend parent genes with exploration margin."""
        genes1, genes2 = parent1.genes, parent2.genes
        d = np.abs(genes1 - genes2)

        low = np.minimum(genes1, genes2) - alpha * d
        high = np.maximum(genes1, genes2) + alpha * d

        child1_genes = self.rng.uniform(low, high)
        child2_genes = self.rng.uniform(low, high)

        # Clip to bounds
        child1_genes = np.clip(child1_genes, self.lower, self.upper)
        child2_genes = np.clip(child2_genes, self.lower, self.upper)

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation on each gene."""
        noise = self.rng.normal(0, self.mutation_std, size=self.n_genes)
        new_genes = individual.genes + noise * (self.upper - self.lower)
        new_genes = np.clip(new_genes, self.lower, self.upper)
        return Individual(genes=new_genes)

    def evolve(self) -> Individual:
        """Run the GA evolution loop.

        Returns:
            Best individual found across all generations.
        """
        population = self._init_population()
        self._evaluate(population)

        best_ever = max(population, key=lambda x: x.fitness)
        logger.info(
            f"GA Init: best_fitness={best_ever.fitness:.4f}, "
            f"genes={best_ever.genes}"
        )

        for gen in range(self.n_gens):
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Elitism: keep top individuals
            new_pop = population[:self.n_elite]

            # Breed rest via crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                c1, c2 = self._blx_alpha_crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])

            population = new_pop[:self.pop_size]
            self._evaluate(population)

            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best

            self.history.append({
                "generation": gen,
                "best_fitness": float(gen_best.fitness),
                "best_genes": gen_best.genes.tolist(),
                "mean_fitness": float(np.mean([i.fitness for i in population])),
            })

            if (gen + 1) % 10 == 0:
                logger.info(
                    f"GA Gen {gen+1}/{self.n_gens}: "
                    f"best={gen_best.fitness:.4f}, "
                    f"mean={self.history[-1]['mean_fitness']:.4f}, "
                    f"genes={gen_best.genes}"
                )

        logger.info(
            f"GA Complete: best_fitness={best_ever.fitness:.4f}, "
            f"best_genes={best_ever.genes}"
        )
        return best_ever


# ══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RAKSHAK-ICS: Run AI Search Attackers"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/proof/",
        help="Directory with processed .npy files",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of attack episodes per attacker",
    )
    parser.add_argument(
        "--attackers", nargs="*", default=None,
        help="Attackers to run (default: all)",
    )
    args = parser.parse_args()

    # Load some test data
    import sys
    sys.path.insert(0, ".")
    from src.preprocess import load_processed_data

    data = load_processed_data(args.data_dir)
    X_test = data["X_test"]

    # Flatten windows: use last timestep as current readings
    if X_test.ndim == 3:
        sensor_data = X_test[:, -1, :]  # (N, F) — last timestep
    else:
        sensor_data = X_test

    logger.info(f"Sensor data: {sensor_data.shape}")

    results = run_all_attackers(
        sensor_data=sensor_data,
        n_samples=args.n_samples,
    )

    # Print results
    print(f"\n{'='*70}")
    print("AI Search Attacker Results (mean±std across 5 seeds)")
    print(f"{'='*70}")
    for name, r in results.items():
        print(
            f"  {name:15s}: "
            f"evasion={r['evasion_rate']['formatted']}, "
            f"disruption={r['disruption']['formatted']}, "
            f"reward={r['reward']['formatted']}"
        )
    print(f"{'='*70}")
