
import math
import numpy as np
import gymnasium as gym
from collections import defaultdict

class MCTSAgent:
    """
    MCTS con UCT y muestreo directo de transiciones usando la tabla P de FrozenLake.
    - Evita recrear entornos dentro del bucle de simulación.
    - Soporta dinámica estocástica (elige s', r, done ~ P(s'|s,a)).
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99, uct_c: float = 1.4,
                 num_simulations: int = 64, max_depth: int = 20, seed: int = 777):
        self.env = env
        self.gamma = gamma
        self.uct_c = uct_c
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

        # Estadísticos en tabla (no árbol explícito por nodos para mantener simplicidad)
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n, dtype=float))
        self.Nsa = defaultdict(lambda: np.zeros(self.env.action_space.n, dtype=float))
        self.Ns  = defaultdict(float)

        # Recuperar la dinámica del entorno: P[s][a] = list[(p, s', r, done)]
        # Esto es específico de entornos tabulares como FrozenLake-v1.
        if hasattr(self.env.unwrapped, "P"):
            self.P = self.env.unwrapped.P
        else:
            raise RuntimeError("El entorno no expone P; esta implementación optimizada de MCTS requiere P.")

        self.n_actions = self.env.action_space.n

    def _uct(self, s, a):
        Ns = self.Ns[s]
        Na = self.Nsa[s][a]
        if Na == 0:
            return float("inf")
        return self.Q[s][a] + self.uct_c * math.sqrt(max(1e-12, math.log(Ns + 1.0)) / Na)

    def _sample_transition(self, s: int, a: int):
        """
        Muestrea (s', r, done) ~ P(s'|s,a). P es una lista de (p, s', r, done).
        """
        outcomes = self.P[int(s)][int(a)]
        ps = [o[0] for o in outcomes]
        idx = self.rng.choice(len(outcomes), p=np.array(ps, dtype=float))
        _, s_next, r, done = outcomes[idx]
        return int(s_next), float(r), bool(done)

    def _simulate(self, root_s: int):
        """
        Una simulación: recorrer hasta max_depth o terminal, eligiendo acciones por UCT
        (aleatorio al expandir) y muestreando transiciones estocásticas desde P.
        Luego, retropropagar retorno G para actualizar Q, Nsa, Ns.
        """
        s = int(root_s)
        trajectory = []  # (s, a, r)
        for depth in range(self.max_depth):
            if self.Ns[s] == 0:
                # Nodo no expandido: seleccionar acción aleatoria (expansión)
                a = int(self.rng.integers(0, self.n_actions))
            else:
                # Selección por UCT
                scores = [self._uct(s, a) for a in range(self.n_actions)]
                a = int(np.argmax(scores))

            s_next, r, done = self._sample_transition(s, a)
            trajectory.append((s, a, r))
            s = s_next
            if done:
                break

        # Backup (Monte Carlo)
        G = 0.0
        for (s_i, a_i, r_i) in reversed(trajectory):
            G = r_i + self.gamma * G
            self.Ns[s_i] += 1.0
            self.Nsa[s_i][a_i] += 1.0
            self.Q[s_i][a_i] += (G - self.Q[s_i][a_i]) / self.Nsa[s_i][a_i]

    def plan(self, obs: int) -> int:
        # Ejecutar varias simulaciones desde el estado actual
        s0 = int(obs)
        for _ in range(self.num_simulations):
            self._simulate(s0)
        # Elegir acción por explotación
        return int(np.argmax(self.Q[s0]))

    def act(self, obs: int) -> int:
        return self.plan(obs)
