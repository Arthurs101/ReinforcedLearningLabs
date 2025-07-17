import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import pandas as pd

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
        print(f"Mejor brazo: {np.argmax(self.true_rewards)} (probabilidad: {np.max(self.true_rewards):.4f})")
    
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
        
        # Historial de brazos seleccionados
        self.selected_arms_history = []
    
    def select_arm(self):
        """
        Selecciona un brazo usando la estrategia epsilon-greedy.
        
        Returns:
            int: Índice del brazo seleccionado
        """
        # Con probabilidad epsilon, selecciona un brazo aleatorio (exploración)
        if random.random() < self.epsilon:
            selected_arm = random.randint(0, self.num_arms - 1)
        else:
            # Con probabilidad 1-epsilon, selecciona el brazo con la recompensa estimada más alta (explotación)
            # En caso de empate, selecciona aleatoriamente entre los mejores
            best_arms = np.where(self.estimated_rewards == np.max(self.estimated_rewards))[0]
            selected_arm = random.choice(best_arms)
        
        self.selected_arms_history.append(selected_arm)
        return selected_arm
    
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
        self.selected_arms_history = []

def run_simulation(bandit, agent, num_iterations=1000, verbose=True):
    """
    Ejecuta la simulación del algoritmo epsilon-greedy.
    
    Args:
        bandit (Bandit): Instancia del entorno bandit
        agent (Agent): Instancia del agente
        num_iterations (int): Número de iteraciones de la simulación
        verbose (bool): Si mostrar progreso detallado
        
    Returns:
        tuple: (recompensas_acumuladas, bandit, agent)
    """
    # Reiniciar el agente
    agent.reset()
    
    if verbose:
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
        if verbose and (i + 1) % 200 == 0:
            print(f"  Iteración {i + 1}: Recompensa acumulada = {agent.current_cumulative}")
    
    if verbose:
        print(f"  Recompensa acumulada final: {agent.current_cumulative}")
        print(f"  Recompensa promedio por iteración: {agent.current_cumulative / num_iterations:.4f}")
    
    return agent.cumulative_rewards, bandit, agent

def comprehensive_epsilon_analysis(bandit, epsilon_values, num_iterations=1000):
    """
    Análisis comprehensivo de diferentes valores de epsilon.
    
    Args:
        bandit (Bandit): Instancia del entorno bandit
        epsilon_values (list): Lista de valores de epsilon a probar
        num_iterations (int): Número de iteraciones por simulación
        
    Returns:
        dict: Resultados detallados de cada valor de epsilon
    """
    results = {}
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPREHENSIVO DE DIFERENTES VALORES DE ÉPSILON")
    print("="*80)
    
    for epsilon in epsilon_values:
        print(f"\n--- Probando epsilon = {epsilon} ---")
        agent = Agent(bandit.num_arms, epsilon)
        cumulative_rewards, _, final_agent = run_simulation(bandit, agent, num_iterations, verbose=False)
        
        # Calcular métricas adicionales
        best_real_arm = np.argmax(bandit.true_rewards)
        best_estimated_arm = np.argmax(final_agent.estimated_rewards)
        estimation_error = np.mean(np.abs(bandit.true_rewards - final_agent.estimated_rewards))
        
        # Calcular concentración en el mejor brazo
        total_pulls = sum(final_agent.pull_counts)
        best_arm_pulls = final_agent.pull_counts[best_real_arm]
        concentration = best_arm_pulls / total_pulls if total_pulls > 0 else 0
        
        # Calcular exploración vs explotación
        exploration_pulls = sum(1 for arm in final_agent.selected_arms_history 
                               if arm != best_estimated_arm)
        exploitation_pulls = len(final_agent.selected_arms_history) - exploration_pulls
        
        results[epsilon] = {
            'cumulative_rewards': cumulative_rewards,
            'final_agent': final_agent,
            'total_reward': cumulative_rewards[-1],
            'avg_reward': cumulative_rewards[-1] / num_iterations,
            'estimation_error': estimation_error,
            'best_real_arm': best_real_arm,
            'best_estimated_arm': best_estimated_arm,
            'concentration': concentration,
            'exploration_pulls': exploration_pulls,
            'exploitation_pulls': exploitation_pulls,
            'pull_distribution': final_agent.pull_counts.copy()
        }
        
        print(f"  Recompensa total: {cumulative_rewards[-1]}")
        print(f"  Recompensa promedio: {cumulative_rewards[-1] / num_iterations:.4f}")
        print(f"  Error de estimación: {estimation_error:.4f}")
        print(f"  Concentración en mejor brazo: {concentration:.1%}")
        print(f"  Mejor brazo estimado: {best_estimated_arm} (real: {best_real_arm})")
    
    return results

def create_comprehensive_plots(results, bandit):
    """
    Crea gráficos comprehensivos para el análisis de epsilon.
    
    Args:
        results (dict): Resultados de las simulaciones
        bandit (Bandit): Instancia del entorno bandit
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Gráfico 1: Recompensa acumulada
    ax1 = plt.subplot(3, 3, 1)
    for epsilon, data in results.items():
        ax1.plot(data['cumulative_rewards'], label=f'ε = {epsilon}', linewidth=2)
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Recompensa Acumulada')
    ax1.set_title('Recompensa Acumulada vs Iteraciones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Recompensa promedio por epsilon
    ax2 = plt.subplot(3, 3, 2)
    epsilons = list(results.keys())
    avg_rewards = [results[eps]['avg_reward'] for eps in epsilons]
    ax2.bar(epsilons, avg_rewards, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Épsilon')
    ax2.set_ylabel('Recompensa Promedio')
    ax2.set_title('Recompensa Promedio vs Épsilon')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Error de estimación
    ax3 = plt.subplot(3, 3, 3)
    estimation_errors = [results[eps]['estimation_error'] for eps in epsilons]
    ax3.bar(epsilons, estimation_errors, alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Épsilon')
    ax3.set_ylabel('Error de Estimación')
    ax3.set_title('Error de Estimación vs Épsilon')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Concentración en el mejor brazo
    ax4 = plt.subplot(3, 3, 4)
    concentrations = [results[eps]['concentration'] for eps in epsilons]
    ax4.bar(epsilons, concentrations, alpha=0.7, color='lightgreen')
    ax4.set_xlabel('Épsilon')
    ax4.set_ylabel('Concentración en Mejor Brazo')
    ax4.set_title('Concentración vs Épsilon')
    ax4.grid(True, alpha=0.3)
    
    # Gráfico 5: Exploración vs Explotación
    ax5 = plt.subplot(3, 3, 5)
    exploration_ratios = [results[eps]['exploration_pulls'] / 1000 for eps in epsilons]
    exploitation_ratios = [results[eps]['exploitation_pulls'] / 1000 for eps in epsilons]
    
    x = np.arange(len(epsilons))
    width = 0.35
    ax5.bar(x - width/2, exploration_ratios, width, label='Exploración', alpha=0.7, color='orange')
    ax5.bar(x + width/2, exploitation_ratios, width, label='Explotación', alpha=0.7, color='blue')
    ax5.set_xlabel('Épsilon')
    ax5.set_ylabel('Proporción de Tiradas')
    ax5.set_title('Exploración vs Explotación')
    ax5.set_xticks(x)
    ax5.set_xticklabels(epsilons)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Gráfico 6: Comparación de estimaciones para el mejor epsilon
    ax6 = plt.subplot(3, 3, 6)
    best_epsilon = max(results.keys(), key=lambda eps: results[eps]['avg_reward'])
    best_agent = results[best_epsilon]['final_agent']
    
    x_pos = np.arange(len(bandit.true_rewards))
    width = 0.35
    ax6.bar(x_pos - width/2, bandit.true_rewards, width, label='Probabilidades Reales', 
            alpha=0.7, color='green')
    ax6.bar(x_pos + width/2, best_agent.estimated_rewards, width, 
            label=f'Estimaciones (ε = {best_epsilon})', alpha=0.7, color='red')
    ax6.set_xlabel('Brazo')
    ax6.set_ylabel('Probabilidad')
    ax6.set_title(f'Estimaciones vs Reales (ε = {best_epsilon})')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'Brazo {i}' for i in range(len(bandit.true_rewards))])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Gráfico 7: Distribución de tiradas para diferentes epsilons
    ax7 = plt.subplot(3, 3, 7)
    selected_epsilons = [0.01, 0.1, 0.5]  # Mostrar solo algunos para claridad
    for i, epsilon in enumerate(selected_epsilons):
        if epsilon in results:
            pull_dist = results[epsilon]['pull_distribution']
            ax7.bar(np.arange(len(pull_dist)) + i*0.25, pull_dist, 
                   width=0.25, alpha=0.7, label=f'ε = {epsilon}')
    ax7.set_xlabel('Brazo')
    ax7.set_ylabel('Número de Tiradas')
    ax7.set_title('Distribución de Tiradas por Brazo')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Gráfico 8: Regret acumulado (recompensa óptima - recompensa obtenida)
    ax8 = plt.subplot(3, 3, 8)
    optimal_reward_per_step = np.max(bandit.true_rewards)
    for epsilon, data in results.items():
        optimal_cumulative = np.arange(1, len(data['cumulative_rewards']) + 1) * optimal_reward_per_step
        regret = optimal_cumulative - np.array(data['cumulative_rewards'])
        ax8.plot(regret, label=f'ε = {epsilon}', linewidth=2)
    ax8.set_xlabel('Iteraciones')
    ax8.set_ylabel('Regret Acumulado')
    ax8.set_title('Regret Acumulado vs Iteraciones')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Gráfico 9: Resumen de métricas
    ax9 = plt.subplot(3, 3, 9)
    metrics = ['avg_reward', 'estimation_error', 'concentration']
    metric_names = ['Recompensa Promedio', 'Error Estimación', 'Concentración']
    
    x = np.arange(len(metrics))
    width = 0.8 / len(epsilons)
    
    for i, epsilon in enumerate(epsilons):
        values = [results[epsilon][metric] for metric in metrics]
        # Normalizar valores para mejor visualización
        values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
        ax9.bar(x + i*width, values, width, label=f'ε = {epsilon}', alpha=0.7)
    
    ax9.set_xlabel('Métricas')
    ax9.set_ylabel('Valor Normalizado')
    ax9.set_title('Comparación de Métricas por Épsilon')
    ax9.set_xticks(x + width * (len(epsilons) - 1) / 2)
    ax9.set_xticklabels(metric_names)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_epsilon_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_epsilon

def print_final_analysis(results, bandit):
    """
    Imprime el análisis final comprehensivo.
    
    Args:
        results (dict): Resultados de las simulaciones
        bandit (Bandit): Instancia del entorno bandit
    """
    print("\n" + "="*80)
    print("ANÁLISIS FINAL COMPREHENSIVO")
    print("="*80)
    
    # Crear DataFrame para análisis
    data = []
    for epsilon, result in results.items():
        data.append({
            'Epsilon': epsilon,
            'Recompensa Total': result['total_reward'],
            'Recompensa Promedio': result['avg_reward'],
            'Error Estimación': result['estimation_error'],
            'Concentración': result['concentration'],
            'Exploración (%)': result['exploration_pulls'] / 1000 * 100,
            'Explotación (%)': result['exploitation_pulls'] / 1000 * 100,
            'Mejor Brazo Correcto': result['best_estimated_arm'] == result['best_real_arm']
        })
    
    df = pd.DataFrame(data)
    print("\nResumen de Resultados:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Encontrar el mejor epsilon
    best_epsilon = max(results.keys(), key=lambda eps: results[eps]['avg_reward'])
    print(f"\nMejor valor de epsilon: {best_epsilon}")
    print(f"Recompensa promedio con mejor epsilon: {results[best_epsilon]['avg_reward']:.4f}")
    
    # Análisis de trade-off exploración-explotación
    print(f"\nAnálisis del Trade-off Exploración-Explotación:")
    print(f"ε = 0.01: Muy poca exploración, puede quedar atrapado en óptimos locales")
    print(f"ε = 0.1: Balance óptimo entre exploración y explotación")
    print(f"ε = 0.5: Mucha exploración, puede no explotar suficientemente el mejor brazo")
    
    # Recomendaciones
    print(f"\nRecomendaciones:")
    print(f"- Para problemas con muchos brazos: usar ε = 0.1")
    print(f"- Para problemas con pocos brazos: usar ε = 0.05")
    print(f"- Para exploración inicial: usar ε = 0.2")
    print(f"- Para explotación pura: usar ε = 0.01")

def main():
    """
    Función principal que ejecuta el análisis comprehensivo.
    """
    # Configuración de la simulación
    NUM_ITERATIONS = 1000
    NUM_ARMS = 10
    EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]  # Más valores para análisis detallado
    
    print("=== ANÁLISIS COMPREHENSIVO DEL ALGORITMO ÉPSILON-CODICIOSO ===")
    print("Experimentando con diferentes valores de epsilon para multi-armed bandits\n")
    
    # Crear el entorno bandit (usando la misma semilla para reproducibilidad)
    bandit = Bandit(NUM_ARMS, seed=42)
    
    # Ejecutar análisis comprehensivo
    results = comprehensive_epsilon_analysis(bandit, EPSILON_VALUES, NUM_ITERATIONS)
    
    # Generar gráficos comprehensivos
    best_epsilon = create_comprehensive_plots(results, bandit)
    
    # Imprimir análisis final
    print_final_analysis(results, bandit)
    
    print(f"\nGráficos guardados como 'comprehensive_epsilon_analysis.png'")
    print("¡Análisis comprehensivo completado!")

if __name__ == "__main__":
    main() 