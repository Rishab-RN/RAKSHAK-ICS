"""
ai_search.py
────────────
AI Search-based attacker baselines (Units I–II).

Formulates ICS attack as a search/game problem:
  - State:  current sensor readings (51-dim vector)
  - Action: perturb sensor_i by magnitude_j
  - Goal:   evade Blue Agent detection while maximising disruption

Attacker Strategies:
  1. RandomAttacker     — uniform random perturbations (BFS analogue, Unit I)
  2. IDDFSAttacker      — iterative deepening DFS with depth limit (Unit I)
  3. AStarAttacker      — A* search, h(n) = remaining disruption budget (Unit I)
  4. AlphaBetaAttacker  — adversarial game tree, depth=3 (Unit II, Lab 2)

GA Optimiser (Unit II):
  - Evolves {λ1, λ2, λ3} reward weights for DQN Red Agent
  - Population: 20, Generations: 50, Tournament selection, BLX-α crossover

Environment model:
  - Partially observable, stochastic, sequential, multi-agent
  - Blue Agent = MIN player (minimise attacker reward)
  - Red Agent = MAX player (maximise evasion + disruption)
"""
