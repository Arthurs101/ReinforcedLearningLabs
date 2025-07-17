import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class Bandit:
    """
    Clase que representa el entorno de multi-armed bandits.
    Inicializa 10 brazos, cada uno con una probabilidad de recompensa aleatoria.
    """
    
    def __init__(self, num_arms=10, seed=None):
        """
        Inicializa el bandit con num_arms brazos.
        Cada brazo tiene una probabilidad de recompensa elegida al azar entre 0 y 1.
        
        Args:
            num_arms (int): Número de brazos
            seed (int): Semilla para reproducibilidad
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
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
        
        # Historial de recompensas para análisis
        self.reward_history = []
        self.cumulative_rewards = []
        self.current_cumulative = 0
    
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
        
        # Actualizar historial
        self.reward_history.append(reward)
        self.current_cumulative += reward
        self.cumulative_rewards.append(self.current_cumulative)
    
    def reset(self):
        """Reinicia el agente para una nueva simulación."""
        self.estimated_rewards = np.zeros(self.num_arms)
        self.pull_counts = np.zeros(self.num_arms, dtype=int)
        self.reward_history = []
        self.cumulative_rewards = []
        self.current_cumulative = 0

def run_simulation(bandit, agent, num_iterations=1000):
    """
    Ejecuta la simulación del algoritmo epsilon-greedy.
    
    Args:
        bandit (Bandit): Instancia del entorno bandit
        agent (Agent): Instancia del agente
        num_iterations (int): Número de iteraciones de la simulación
        
    Returns:
        tuple: (recompensas_acumuladas, bandit, agent)
    """
    # Reiniciar el agente
    agent.reset()
    
    print(f"Ejecutando simulación con epsilon={agent.epsilon}, {num_iterations} iteraciones")
    
    # Ejecutar la simulación
    for i in range(num_iterations):
        # Seleccionar un brazo usando la estrategia epsilon-greedy
        selected_arm = agent.select_arm()
        
        # Tirar del brazo en el entorno para obtener una recompensa
        reward = bandit.pull(selected_arm)
        
        # Actualizar las recompensas estimadas
        agent.update_estimates(selected_arm, reward)
        
        # Imprimir progreso cada 200 iteraciones
        if (i + 1) % 200 == 0:
            print(f"  Iteración {i + 1}: Recompensa acumulada = {agent.current_cumulative}")
    
    print(f"  Recompensa acumulada final: {agent.current_cumulative}")
    print(f"  Recompensa promedio por iteración: {agent.current_cumulative / num_iterations:.4f}")
    
    return agent.cumulative_rewards, bandit, agent

def compare_epsilon_values(bandit, epsilon_values, num_iterations=1000):
    """
    Compara el rendimiento de diferentes valores de epsilon.
    
    Args:
        bandit (Bandit): Instancia del entorno bandit
        epsilon_values (list): Lista de valores de epsilon a probar
        num_iterations (int): Número de iteraciones por simulación
        
    Returns:
        dict: Resultados de cada valor de epsilon
    """
    results = {}
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE DIFERENTES VALORES DE ÉPSILON")
    print("="*60)
    
    for epsilon in epsilon_values:
        print(f"\n--- Probando epsilon = {epsilon} ---")
        agent = Agent(bandit.num_arms, epsilon)
        cumulative_rewards, _, final_agent = run_simulation(bandit, agent, num_iterations)
        
        results[epsilon] = {
            'cumulative_rewards': cumulative_rewards,
            'final_agent': final_agent,
            'total_reward': cumulative_rewards[-1],
            'avg_reward': cumulative_rewards[-1] / num_iterations
        }
    
    return results

def plot_comparison_results(results, bandit):
    """
    Genera gráficos comparativos para diferentes valores de epsilon.
    
    Args:
        results (dict): Resultados de las simulaciones
        bandit (Bandit): Instancia del entorno bandit
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Recompensa acumulada para diferentes valores de epsilon
    for epsilon, data in results.items():
        ax1.plot(data['cumulative_rewards'], label=f'ε = {epsilon}', linewidth=2)
    
    ax1.set_xlabel('Número de Iteraciones')
    ax1.set_ylabel('Recompensa Acumulada')
    ax1.set_title('Recompensa Acumulada vs Iteraciones\npara Diferentes Valores de Épsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Recompensa promedio por iteración
    epsilons = list(results.keys())
    avg_rewards = [results[eps]['avg_reward'] for eps in epsilons]
    
    ax2.bar(epsilons, avg_rewards, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Valor de Épsilon')
    ax2.set_ylabel('Recompensa Promedio por Iteración')
    ax2.set_title('Recompensa Promedio vs Épsilon')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Comparación de estimaciones para el mejor epsilon
    best_epsilon = max(results.keys(), key=lambda eps: results[eps]['avg_reward'])
    best_agent = results[best_epsilon]['final_agent']
    
    x_pos = np.arange(len(bandit.true_rewards))
    width = 0.35
    
    ax3.bar(x_pos - width/2, bandit.true_rewards, width, label='Probabilidades Reales', 
            alpha=0.7, color='green')
    ax3.bar(x_pos + width/2, best_agent.estimated_rewards, width, 
            label=f'Estimaciones (ε = {best_epsilon})', alpha=0.7, color='red')
    
    ax3.set_xlabel('Brazo')
    ax3.set_ylabel('Probabilidad de Recompensa')
    ax3.set_title(f'Probabilidades Reales vs Estimaciones\n(Mejor ε = {best_epsilon})')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Brazo {i}' for i in range(len(bandit.true_rewards))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Número de tiradas por brazo para el mejor epsilon
    ax4.bar(x_pos, best_agent.pull_counts, alpha=0.7, color='orange')
    ax4.set_xlabel('Brazo')
    ax4.set_ylabel('Número de Tiradas')
    ax4.set_title(f'Distribución de Tiradas por Brazo\n(ε = {best_epsilon})')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Brazo {i}' for i in range(len(bandit.true_rewards))])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_greedy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_epsilon

def print_detailed_analysis(results, bandit):
    """
    Imprime un análisis detallado de los resultados.
    
    Args:
        results (dict): Resultados de las simulaciones
        bandit (Bandit): Instancia del entorno bandit
    """
    print("\n" + "="*60)
    print("ANÁLISIS DETALLADO DE RESULTADOS")
    print("="*60)
    
    # Encontrar el mejor epsilon
    best_epsilon = max(results.keys(), key=lambda eps: results[eps]['avg_reward'])
    best_agent = results[best_epsilon]['final_agent']
    
    print(f"\nMejor valor de epsilon: {best_epsilon}")
    print(f"Recompensa total con mejor epsilon: {results[best_epsilon]['total_reward']}")
    print(f"Recompensa promedio con mejor epsilon: {results[best_epsilon]['avg_reward']:.4f}")
    
    print(f"\nProbabilidades reales: {bandit.true_rewards}")
    print(f"Estimaciones del agente (ε = {best_epsilon}): {best_agent.estimated_rewards}")
    print(f"Número de tiradas por brazo: {best_agent.pull_counts}")
    
    # Calcular el error promedio de estimación
    estimation_error = np.mean(np.abs(bandit.true_rewards - best_agent.estimated_rewards))
    print(f"Error promedio de estimación: {estimation_error:.4f}")
    
    # Encontrar el mejor brazo real y el mejor brazo estimado
    best_real_arm = np.argmax(bandit.true_rewards)
    best_estimated_arm = np.argmax(best_agent.estimated_rewards)
    print(f"Mejor brazo real: {best_real_arm} (probabilidad: {bandit.true_rewards[best_real_arm]:.4f})")
    print(f"Mejor brazo estimado: {best_estimated_arm} (estimación: {best_agent.estimated_rewards[best_estimated_arm]:.4f})")
    
    # Análisis de exploración vs explotación
    print(f"\nAnálisis de Exploración vs Explotación:")
    for epsilon in sorted(results.keys()):
        agent = results[epsilon]['final_agent']
        exploration_ratio = agent.epsilon
        exploitation_ratio = 1 - agent.epsilon
        total_pulls = sum(agent.pull_counts)
        most_pulled_arm = np.argmax(agent.pull_counts)
        concentration = agent.pull_counts[most_pulled_arm] / total_pulls
        
        print(f"  ε = {epsilon}:")
        print(f"    Exploración: {exploration_ratio:.1%}, Explotación: {exploitation_ratio:.1%}")
        print(f"    Brazo más tirado: {most_pulled_arm} ({agent.pull_counts[most_pulled_arm]} veces)")
        print(f"    Concentración en mejor brazo: {concentration:.1%}")

def main():
    """
    Función principal que ejecuta todos los experimentos.
    """
    # Configuración de la simulación
    NUM_ITERATIONS = 1000
    NUM_ARMS = 10
    EPSILON_VALUES = [0.01, 0.1, 0.5]  # Valores de epsilon a probar
    
    print("=== EXPERIMENTOS CON ALGORITMO ÉPSILON-CODICIOSO ===")
    print("Comparando diferentes valores de epsilon para multi-armed bandits\n")
    
    # Crear el entorno bandit (usando la misma semilla para reproducibilidad)
    bandit = Bandit(NUM_ARMS, seed=42)
    
    # Ejecutar comparación de diferentes valores de epsilon
    results = compare_epsilon_values(bandit, EPSILON_VALUES, NUM_ITERATIONS)
    
    # Generar gráficos comparativos
    best_epsilon = plot_comparison_results(results, bandit)
    
    # Imprimir análisis detallado
    print_detailed_analysis(results, bandit)
    
    print(f"\nGráficos guardados como 'epsilon_greedy_comparison.png'")
    print("¡Experimentos completados!")

if __name__ == "__main__":
    main() 