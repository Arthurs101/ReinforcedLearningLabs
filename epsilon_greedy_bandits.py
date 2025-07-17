import numpy as np
import matplotlib.pyplot as plt
import random

class Bandit:
    """
    Clase que representa el entorno de multi-armed bandits.
    Inicializa 10 brazos, cada uno con una probabilidad de recompensa aleatoria.
    """
    
    def __init__(self, num_arms=10):
        """
        Inicializa el bandit con num_arms brazos.
        Cada brazo tiene una probabilidad de recompensa elegida al azar entre 0 y 1.
        """
        self.num_arms = num_arms
        # Inicializar probabilidades de recompensa aleatorias para cada brazo
        self.true_rewards = np.random.random(num_arms)
        print(f"Probabilidades reales de recompensa: {self.true_rewards}")
    
    def pull(self, arm):
        """
        Tira del brazo especificado y devuelve una recompensa.
        
        Args:
            arm (int): Índice del brazo a tirar (0 a num_arms-1)
            
        Returns:
            int: 1 con la probabilidad del brazo, 0 en caso contrario
        """
        if arm < 0 or arm >= self.num_arms:
            raise ValueError(f"Brazo {arm} fuera de rango. Debe estar entre 0 y {self.num_arms-1}")
        
        # Devolver 1 con la probabilidad del brazo, 0 en caso contrario
        return 1 if random.random() < self.true_rewards[arm] else 0

class Agent:
    """
    Agente que implementa la estrategia epsilon-greedy para multi-armed bandits.
    """
    
    def __init__(self, num_arms, epsilon=0.1):
        """
        Inicializa el agente con la estrategia epsilon-greedy.
        
        Args:
            num_arms (int): Número de brazos disponibles
            epsilon (float): Probabilidad de exploración (0 a 1)
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        
        # Matriz para almacenar las recompensas estimadas (inicializada en cero)
        self.estimated_rewards = np.zeros(num_arms)
        
        # Matriz para contar la cantidad de veces que se ha extraído cada brazo
        self.pull_counts = np.zeros(num_arms, dtype=int)
    
    def select_arm(self):
        """
        Selecciona un brazo usando la estrategia epsilon-greedy.
        
        Returns:
            int: Índice del brazo seleccionado
        """
        # Con probabilidad epsilon, selecciona un brazo aleatorio (exploración)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_arms - 1)
        
        # Con probabilidad 1-epsilon, selecciona el brazo con la recompensa estimada más alta (explotación)
        # En caso de empate, selecciona aleatoriamente entre los mejores
        best_arms = np.where(self.estimated_rewards == np.max(self.estimated_rewards))[0]
        return random.choice(best_arms)
    
    def update_estimates(self, arm, reward):
        """
        Actualiza la recompensa estimada para el brazo elegido.
        
        Args:
            arm (int): Índice del brazo
            reward (int): Recompensa obtenida (0 o 1)
        """
        if arm < 0 or arm >= self.num_arms:
            raise ValueError(f"Brazo {arm} fuera de rango")
        
        # Incrementar el contador de tiradas para este brazo
        self.pull_counts[arm] += 1
        
        # Actualizar la recompensa estimada usando la fórmula incremental
        # Q_{n+1} = Q_n + (1/n) * [R_n - Q_n]
        n = self.pull_counts[arm]
        self.estimated_rewards[arm] += (1.0 / n) * (reward - self.estimated_rewards[arm])

def run_simulation(num_iterations=1000, epsilon=0.1, num_arms=10):
    """
    Ejecuta la simulación del algoritmo epsilon-greedy.
    
    Args:
        num_iterations (int): Número de iteraciones de la simulación
        epsilon (float): Valor de epsilon para la exploración
        num_arms (int): Número de brazos del bandit
        
    Returns:
        tuple: (recompensas_acumuladas, bandit, agent)
    """
    # Inicializar el entorno y el agente
    bandit = Bandit(num_arms)
    agent = Agent(num_arms, epsilon)
    
    # Variables para seguimiento
    cumulative_rewards = []
    current_cumulative = 0
    
    print(f"Iniciando simulación con {num_iterations} iteraciones, epsilon={epsilon}")
    print(f"Número de brazos: {num_arms}")
    
    # Ejecutar la simulación
    for i in range(num_iterations):
        # Seleccionar un brazo usando la estrategia epsilon-greedy
        selected_arm = agent.select_arm()
        
        # Tirar del brazo en el entorno para obtener una recompensa
        reward = bandit.pull(selected_arm)
        
        # Actualizar las recompensas estimadas
        agent.update_estimates(selected_arm, reward)
        
        # Actualizar la recompensa acumulada
        current_cumulative += reward
        cumulative_rewards.append(current_cumulative)
        
        # Imprimir progreso cada 100 iteraciones
        if (i + 1) % 100 == 0:
            print(f"Iteración {i + 1}: Recompensa acumulada = {current_cumulative}")
    
    print(f"\nSimulación completada!")
    print(f"Recompensa acumulada final: {current_cumulative}")
    print(f"Recompensa promedio por iteración: {current_cumulative / num_iterations:.4f}")
    
    return cumulative_rewards, bandit, agent

def plot_results(cumulative_rewards, bandit, agent):
    """
    Genera gráficos para visualizar los resultados de la simulación.
    
    Args:
        cumulative_rewards (list): Lista de recompensas acumuladas
        bandit (Bandit): Instancia del entorno bandit
        agent (Agent): Instancia del agente
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Recompensa acumulada en las pruebas
    ax1.plot(cumulative_rewards, linewidth=2, color='blue')
    ax1.set_xlabel('Número de Iteraciones')
    ax1.set_ylabel('Recompensa Acumulada')
    ax1.set_title('Recompensa Acumulada vs Iteraciones')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Valores estimados vs probabilidades reales
    x_pos = np.arange(len(bandit.true_rewards))
    width = 0.35
    
    ax2.bar(x_pos - width/2, bandit.true_rewards, width, label='Probabilidades Reales', 
            alpha=0.7, color='green')
    ax2.bar(x_pos + width/2, agent.estimated_rewards, width, label='Estimaciones del Agente', 
            alpha=0.7, color='red')
    
    ax2.set_xlabel('Brazo')
    ax2.set_ylabel('Probabilidad de Recompensa')
    ax2.set_title('Probabilidades Reales vs Estimaciones del Agente')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Brazo {i}' for i in range(len(bandit.true_rewards))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas adicionales
    print("\n=== ESTADÍSTICAS FINALES ===")
    print(f"Probabilidades reales: {bandit.true_rewards}")
    print(f"Estimaciones del agente: {agent.estimated_rewards}")
    print(f"Número de tiradas por brazo: {agent.pull_counts}")
    
    # Calcular el error promedio de estimación
    estimation_error = np.mean(np.abs(bandit.true_rewards - agent.estimated_rewards))
    print(f"Error promedio de estimación: {estimation_error:.4f}")
    
    # Encontrar el mejor brazo real y el mejor brazo estimado
    best_real_arm = np.argmax(bandit.true_rewards)
    best_estimated_arm = np.argmax(agent.estimated_rewards)
    print(f"Mejor brazo real: {best_real_arm} (probabilidad: {bandit.true_rewards[best_real_arm]:.4f})")
    print(f"Mejor brazo estimado: {best_estimated_arm} (estimación: {agent.estimated_rewards[best_estimated_arm]:.4f})")

def main():
    """
    Función principal que ejecuta la simulación completa.
    """
    # Configuración de la simulación
    NUM_ITERATIONS = 1000
    EPSILON = 0.1
    NUM_ARMS = 10
    
    print("=== ALGORITMO ÉPSILON-CODICIOSO PARA MULTI-ARMED BANDITS ===\n")
    
    # Ejecutar la simulación
    cumulative_rewards, bandit, agent = run_simulation(
        num_iterations=NUM_ITERATIONS,
        epsilon=EPSILON,
        num_arms=NUM_ARMS
    )
    
    # Generar gráficos
    plot_results(cumulative_rewards, bandit, agent)

if __name__ == "__main__":
    main() 