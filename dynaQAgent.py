import numpy as np
from collections import defaultdict

class DynaQPlusAgent:
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 n_planning: int = 20,
                 kappa: float = 1e-3,
                 seed: int = 777):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=float)
        # Modelo: para cada (s,a) almacenamos (s', r) promedio y contador
        self.model_next = defaultdict(lambda: np.zeros(n_actions, dtype=int))  # s -> counts por a
        self.model_r_sum = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        self.model_s_next = defaultdict(lambda: np.zeros(n_actions, dtype=int))

        # Para tiempo desde última visita por (s,a)
        self.time = 0
        self.last_visited = np.zeros((n_states, n_actions), dtype=int)

        self.seen_pairs = set()

    def choose_action(self, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        self.time += 1
        self.seen_pairs.add((s,a))
        # Q-learning update con bono Dyna-Q+ (solo en planeación)
        td_target = r + self.gamma * (0 if done else np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

        # Actualizar modelo
        self.model_next[s][a] += 1
        self.model_r_sum[s][a] += r
        self.model_s_next[s][a] = s_next
        self.last_visited[s, a] = self.time

        # Planificación n
        self._planning()

    def _planning(self):
        if len(self.seen_pairs) == 0:
            return
        for _ in range(self.n_planning):
            s, a = list(self.seen_pairs)[ self.rng.integers(0, len(self.seen_pairs)) ]

            # Obtener recompensas/estado del modelo (promedio simple)
            count = self.model_next[s][a]
            if count == 0:
                # Par no visto: usamos bono puro
                model_r = 0.0
                s_next = s
            else:
                model_r = self.model_r_sum[s][a] / max(1, count)
                s_next = int(self.model_s_next[s][a])

            # Bono de exploración temporal (Dyna-Q+): kappa * sqrt(tau)
            tau = max(1, self.time - self.last_visited[s, a])
            bonus = self.kappa * np.sqrt(tau)

            td_target = (model_r + bonus) + self.gamma * np.max(self.Q[s_next])
            self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
