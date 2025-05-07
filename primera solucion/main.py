# main.py

from experiment import read_etsp_file, filter_arcs_by_time_window
from hybrid_sa_ts import HybridSATS
import matplotlib.pyplot as plt

def run_single():
    # 1. Cargar datos
    data = read_etsp_file('n20w120s10_1.txt')

    # (Opcional) 2. Filtrar arcos no factibles según ventanas de tiempo
    filtered_matrix, last_node = filter_arcs_by_time_window(
        data['nodes'],
        data['distance_matrix']
    )
    # si no vas a usar filtered_matrix, puedes omitir este paso

    # 3. Configurar y correr algoritmo Hybrid SA/TS
    optimizer = HybridSATS(
        nodes=data['nodes'],
        distance_matrix=data['distance_matrix'],
        num_customers=data['num_customers'],
        battery_capacity=data['battery_capacity'],
        energy_rate=data['energy_rate'],
        max_iterations=45000,         # Según paper
        T0=10000,                     # Temperatura inicial
        alpha=0.95,                   # Factor de enfriamiento
        cooling_interval=100,         # Cada cuántas iteraciones se enfría
        perturbation_interval=500,    # Cada cuántas iteraciones se perturba
        R=30,                         # Número de nodos a perturbar
        tabu_tenure_moves=400,        # Tamaño de la lista tabu de movimientos
        tabu_tenure_stations=75,      # Tamaño de la lista tabu de estaciones
                          # Para reproducibilidad (si tu clase lo admite)
    )

    return optimizer.run()


def main():
    n_runs = 10
    all_costs = []
    all_solutions = []

    for run_id in range(1, n_runs+1):
        print(f"\n▶▶▶ Corrida {run_id}/{n_runs}")
        sol, cost, cost_prog, best_prog = run_single()
        print(f"  → Mejor coste en ésta corrida: {cost:.2f}")
        all_costs.append(cost)
        all_solutions.append(sol)

    # Mostrar resumen
    print("\n========== Resumen de las 10 ejecuciones ==========")
    for i, c in enumerate(all_costs, start=1):
        print(f"Corrida {i:2d}: coste = {c:.2f}")
    best_overall = min(all_costs)
    best_run     = all_costs.index(best_overall) + 1
    best_sol     = all_solutions[best_run-1]
    print("-------------------------------------------------")
    print(f"🏆 Mejor de todas: Corrida {best_run} con coste {best_overall:.2f}")
    print("Ruta:", best_sol)
    print("=================================================\n")

    # (Opcional) si quieres, puedes graficar la progresión del mejor run:
    plt.figure(figsize=(10,5))
    plt.plot(best_prog, label=f'Best progression run {best_run}', linestyle='--', color='red')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor coste acumulado')
    plt.title('Evolución del mejor coste en la mejor corrida')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
