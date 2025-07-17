# Algoritmo Épsilon-Codicioso para Multi-Armed Bandits

Este proyecto implementa el algoritmo epsilon-greedy para resolver el problema de multi-armed bandits, maximizando la recompensa acumulativa en una serie de pruebas.

## Descripción del Problema

El problema de multi-armed bandits es un problema clásico de aprendizaje por refuerzo donde un agente debe elegir entre múltiples acciones (brazos) en cada paso, cada una con una distribución de recompensa diferente y desconocida. El objetivo es maximizar la recompensa total acumulada a lo largo del tiempo.

## Implementación

### Clase Bandit
- Representa el entorno con 10 brazos
- Cada brazo tiene una probabilidad de recompensa aleatoria entre 0 y 1
- Método `pull(arm)` que devuelve 1 con la probabilidad del brazo, 0 en caso contrario

### Clase Agent
- Implementa la estrategia epsilon-greedy
- Mantiene estimaciones de recompensa para cada brazo
- Cuenta el número de veces que se ha tirado cada brazo
- Método `select_arm()` que balancea exploración y explotación
- Método `update_estimates()` que actualiza las estimaciones usando la fórmula incremental

### Estrategia Épsilon-Greedy
- Con probabilidad ε (epsilon): selecciona un brazo aleatorio (exploración)
- Con probabilidad 1-ε: selecciona el brazo con la mejor estimación (explotación)

## Características del Código

1. **Configuración**: 10 brazos, epsilon = 0.1, 1000 iteraciones
2. **Seguimiento**: Recompensa acumulada en cada iteración
3. **Visualización**: Gráficos de recompensa acumulada y comparación de estimaciones
4. **Estadísticas**: Error de estimación, mejor brazo real vs estimado

## Cómo Ejecutar

### Requisitos
```bash
pip install numpy matplotlib
```

### Ejecución
```bash
python epsilon_greedy_bandits.py
```

## Salida Esperada

El programa mostrará:
1. Las probabilidades reales de recompensa de cada brazo
2. Progreso de la simulación cada 100 iteraciones
3. Recompensa acumulada final y promedio
4. Dos gráficos:
   - Recompensa acumulada vs iteraciones
   - Comparación entre probabilidades reales y estimaciones del agente
5. Estadísticas finales detalladas

## Interpretación de Resultados

- **Gráfico de recompensa acumulada**: Muestra cómo el agente mejora su rendimiento a lo largo del tiempo
- **Gráfico de estimaciones**: Compara qué tan bien el agente aprendió las verdaderas probabilidades
- **Error de estimación**: Mide la precisión de las estimaciones del agente
- **Mejor brazo**: Compara si el agente identificó correctamente el brazo óptimo

## Parámetros Ajustables

Puedes modificar los siguientes parámetros en la función `main()`:
- `NUM_ITERATIONS`: Número de iteraciones de la simulación
- `EPSILON`: Valor de epsilon para el balance exploración/explotación
- `NUM_ARMS`: Número de brazos del bandit

## Fórmula de Actualización

La estimación de recompensa se actualiza usando la fórmula incremental:
```
Q_{n+1} = Q_n + (1/n) * [R_n - Q_n]
```

Donde:
- Q_n es la estimación actual
- R_n es la recompensa observada
- n es el número de veces que se ha tirado el brazo 