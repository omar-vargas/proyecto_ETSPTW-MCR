import random
import math
import matplotlib.pyplot as plt


def read_etsp_file(filename):
    with open(filename, 'r') as file:
        num_customers = int(file.readline())
        num_stations = int(file.readline())
        battery_capacity = float(file.readline())
        energy_rate = float(file.readline())

        total_nodes = 1 + num_customers + num_stations + 1

        nodes = []
        for _ in range(total_nodes):
            line = file.readline().strip()
            if line:
                parts = line.split('\t')
                node = {
                    'id': int(parts[0]), 'x': float(parts[1]), 'y': float(parts[2]),
                    'e': float(parts[3]), 'l': float(parts[4]), 'is_station': int(parts[5]),
                    'recharge_rate': float(parts[6])
                }
                nodes.append(node)

        distance_matrix = [list(map(float, line.strip().split('\t'))) for line in file if line.strip()]

    return {
        'num_customers': num_customers, 'num_stations': num_stations,
        'battery_capacity': battery_capacity, 'energy_rate': energy_rate,
        'nodes': nodes, 'distance_matrix': distance_matrix
    }


def generate_initial_solution(nodes, num_customers):
    customer_nodes = [node for node in nodes if node['is_station'] == 0 and node['id'] != 0]
    sorted_customers = sorted(customer_nodes, key=lambda x: x['l'])
    return [0] + [node['id'] for node in sorted_customers] + [0]


def apply_local_search(solution):
    if len(solution) <= 4:
        return solution
    move_type = random.choice(['1-shift', '2-opt', 'swap'])
    i = random.randint(1, len(solution) - 2)
    j = random.randint(1, len(solution) - 2)
    while i == j:
        j = random.randint(1, len(solution) - 2)
    new_solution = solution[:]
    if move_type == '1-shift':
        node = new_solution.pop(i)
        new_solution.insert(j, node)
    elif move_type == '2-opt':
        i, j = sorted([i, j])
        new_solution = new_solution[:i] + new_solution[i:j + 1][::-1] + new_solution[j + 1:]
    else:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def apply_perturbation(solution, R):
    middle_nodes = solution[1:-1]
    if len(middle_nodes) <= R:
        return solution
    nodes_to_remove = random.sample(middle_nodes, R)
    new_solution = [node for node in solution if node not in nodes_to_remove]
    for node in nodes_to_remove:
        insert_position = random.randint(1, len(new_solution) - 1)
        new_solution.insert(insert_position, node)
    if new_solution[0] != 0:
        new_solution.insert(0, 0)
    if new_solution[-1] != 0:
        new_solution.append(0)
    return new_solution


def station_insertion(route, nodes, dist_matrix, battery_capacity, energy_rate):
    """
    Inserta estaciones en una ruta para que sea factible en energía.
    route: lista de IDs de nodos (clientes y depot)
    nodes: lista de diccionarios con info de cada nodo
    dist_matrix: matriz de distancias
    battery_capacity: capacidad máxima de batería
    energy_rate: tasa de consumo de energía por distancia
    """
    labels = {route[0]: [(0, battery_capacity, 0, [])]}  # nodo_id: [(tiempo, batería, costo_acumulado, camino)]
    
    for i in range(1, len(route)):
        prev = route[i-1]
        curr = route[i]
        labels[curr] = []

        for (time_prev, batt_prev, cost_prev, path_prev) in labels[prev]:
            d = dist_matrix[prev][curr]
            energy_needed = d * energy_rate

            if batt_prev >= energy_needed:
                travel_time = d
                arrival_time = time_prev + travel_time

                node_e = nodes[curr]['e']
                node_l = nodes[curr]['l']
                wait_time = max(0, node_e - arrival_time)
                arrival_time = max(arrival_time, node_e)

                if arrival_time <= node_l:
                    new_batt = batt_prev - energy_needed
                    new_cost = cost_prev + d
                    new_path = path_prev + [curr]
                    labels[curr].append((arrival_time, new_batt, new_cost, new_path))

            # Opción con recarga en el nodo anterior
            if nodes[prev]['is_station'] == 1:
                recharge_rate = nodes[prev]['recharge_rate']
                recharge_time = (battery_capacity - batt_prev) / recharge_rate
                time_with_recharge = time_prev + recharge_time
                full_batt = battery_capacity

                if full_batt >= energy_needed:
                    arrival_time = time_with_recharge + d
                    arrival_time = max(arrival_time, nodes[curr]['e'])

                    if arrival_time <= nodes[curr]['l']:
                        new_cost = cost_prev + d
                        new_path = path_prev + [f"RECARGA@{prev}", curr]
                        labels[curr].append((arrival_time, full_batt - energy_needed, new_cost, new_path))

    # Elegir mejor etiqueta en el último nodo
    final_labels = labels[route[-1]]
    if not final_labels:
        return None, float('inf')  # Inviable

    best = min(final_labels, key=lambda x: x[2])
    return best[3], best[2]

def is_time_feasible(solution, nodes, distance_matrix):
    current_time = 0
    for i in range(len(solution) - 1):
        from_node = solution[i]
        to_node = solution[i + 1]
        travel_time = distance_matrix[from_node][to_node]
        current_time += travel_time
        if current_time < nodes[to_node]['e']:
            current_time = nodes[to_node]['e']
        if current_time > nodes[to_node]['l'] + 1e-6:
            return False, current_time
    return True, current_time


def detect_move(before, after):
    for i in range(1, len(before)):
        if before[i] != after[i]:
            return (before[i], after[i])
    return (None, None)


def hybrid_sa_ts(data, max_iter=15000, T0=10000, alpha=0.95, cooling_interval=100,
                 perturb_interval=500, R=3, tabu_moves=100, tabu_stations=50):
    nodes = data['nodes']
    dist = data['distance_matrix']
    Q = data['battery_capacity']
    energy_rate = data['energy_rate']
    num_customers = data['num_customers']

    solution = generate_initial_solution(nodes, num_customers)
    while not is_time_feasible(solution, nodes, dist)[0]:
        solution = apply_local_search(solution)



    best_solution = solution[:]
    best_cost = sum(dist[solution[i]][solution[i + 1]] for i in range(len(solution) - 1))
    T = T0
    tabu_move_list, tabu_station_list = [], []
    cost_progression, best_cost_progression = [], []

    for it in range(max_iter):
        new_solution = apply_local_search(solution)
        feasible, _ = is_time_feasible(new_solution, nodes, dist)
        if not feasible:
            continue

        new_cost = sum(dist[new_solution[i]][new_solution[i + 1]] for i in range(len(new_solution) - 1))
        move = detect_move(solution, new_solution)
        if move in tabu_move_list and new_cost >= best_cost:
            continue
        delta = new_cost - best_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            solution = new_solution[:]
            if new_cost < best_cost:
                best_solution = new_solution[:]
                best_cost = new_cost

        cost_progression.append(new_cost)
        best_cost_progression.append(best_cost)

        tabu_move_list.append(move)
        if len(tabu_move_list) > tabu_moves:
            tabu_move_list.pop(0)

        cost_progression.append(new_cost)
        best_cost_progression.append(best_cost)

        if it % cooling_interval == 0:
            T *= alpha
        if it % perturb_interval == 0:
            solution = apply_perturbation(solution, R)

    return best_solution, best_cost, cost_progression, best_cost_progression


import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
data = read_etsp_file('./n20w120s10_1.txt')

# Parámetros
n_runs = 100
costs = []
solutions = []

for i in range(n_runs):
    best_sol, best_cost, _, _ = hybrid_sa_ts(data)
    costs.append(best_cost)
    solutions.append(best_sol)
    if i % 100 == 0:
        print(f"Iteración {i}: costo = {best_cost:.2f}")

# Análisis
min_cost = min(costs)
max_cost = max(costs)
mean_cost = np.mean(costs)
std_cost = np.std(costs)
best_index = costs.index(min_cost)
best_route = solutions[best_index]

# Resultados
print("\n========== Estadísticas ===========")
print(f"Mejor costo encontrado: {min_cost}")
print(f"Peor costo encontrado: {max_cost}")
print(f"Promedio de costos: {mean_cost:.2f}")
print(f"Desviación estándar: {std_cost:.2f}")
print(f"Mejor ruta encontrada: {best_route}")
print("====================================")

# Histograma
plt.figure(figsize=(10, 5))
plt.hist(costs, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución de costos en 1000 ejecuciones')
plt.xlabel('Costo total')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.tight_layout()
plt.show()