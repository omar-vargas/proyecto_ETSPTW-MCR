# main.py

from preprocess import read_etsp_file, filter_arcs_by_time_window
from hybrid_sa_ts import HybridSATS
import matplotlib.pyplot as plt

def main():
    # 1. Cargar datos
    data = read_etsp_file('n20w120s10_1.txt')

    # 2. Filtrar arcos no factibles según ventanas de tiempo (opcional si quieres probarlo)
    filtered_matrix, last_node = filter_arcs_by_time_window(data['nodes'], data['distance_matrix'])

    # 3. Configurar y correr algoritmo Hybrid SA/TS
    optimizer = HybridSATS(
        nodes=data['nodes'],
        distance_matrix=data['distance_matrix'],
        num_customers=data['num_customers'],
        battery_capacity=data['battery_capacity'],
        energy_rate=data['energy_rate'],
        max_iterations=15000,         # Según paper
        T0=10000,                     # Temperatura inicial
        alpha=0.95,                   # Factor de enfriamiento
        cooling_interval=100,         # Cada cuántas iteraciones se enfría
        perturbation_interval=500,    # Cada cuántas iteraciones se perturba
        R=30,                         # Número de nodos a perturbar
        tabu_tenure_moves=400,         # Tamaño de la lista tabu de movimientos
        tabu_tenure_stations=75        # Tamaño de la lista tabu de estaciones
    )

    best_solution, best_cost, cost_progression, best_cost_progression = optimizer.run()

    # 4. Mostrar resultados
    print("========= Resultados ============")
    print("Mejor ruta encontrada:", best_solution)
    print("Costo total:", best_cost)
    print("==================================")

    # 5. Graficar evolución de costos
    plt.figure(figsize=(12, 6))
    plt.plot(cost_progression, label='Costo actual', alpha=0.7, color='blue')
    plt.plot(best_cost_progression, label='Mejor costo histórico', linestyle='--', color='red')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Evolución del costo durante Hybrid SA/TS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
