import random
def read_etsp_file(filename):
    with open(filename, 'r') as file:
        # Leer parámetros generales
        num_customers = int(file.readline())
        num_stations = int(file.readline())
        battery_capacity = float(file.readline())
        energy_rate = float(file.readline())

        total_nodes = 1 + num_customers + num_stations + 1  # 1 depot, V customers, F stations, 1 station at depot

        # Leer nodos (esperamos total_nodes líneas)
        nodes = []
        for _ in range(total_nodes):
            line = file.readline().strip()
            if line == '':
                continue  # en caso de líneas vacías
            parts = line.split('\t')
            node = {
                'id': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'e': float(parts[3]),
                'l': float(parts[4]),
                'is_station': int(parts[5]),
                'recharge_rate': float(parts[6])
            }
            nodes.append(node)

        # Leer matriz de distancias
        distance_matrix = []
        for line in file:
            if line.strip() == '':
                continue
            row = list(map(float, line.strip().split('\t')))
            distance_matrix.append(row)

    return {
        'num_customers': num_customers,
        'num_stations': num_stations,
        'battery_capacity': battery_capacity,
        'energy_rate': energy_rate,
        'nodes': nodes,
        'distance_matrix': distance_matrix
    }


def filter_arcs_by_time_window(nodes, distance_matrix, unreachable_value=99999):
    n = len(nodes)
    filtered_matrix = [[unreachable_value for _ in range(n)] for _ in range(n)]

    # Asumimos que el último nodo es el depósito o destino final
    last_node = n - 1  # Este es el último nodo en la lista
    l_last = nodes[last_node]['l']

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # evitamos bucles
            ei = nodes[i]['e']
            lj = nodes[j]['l']
            dij = distance_matrix[i][j]
            dju = distance_matrix[j][last_node]

            # Condición original + la condición para llegar al último nodo
            if ei + dij <= lj and ei + dij + dju <= l_last:
                filtered_matrix[i][j] = dij

    return filtered_matrix, last_node  # devolvemos también el ID del último nodo


def generate_initial_solution(nodes,num_customers):
    """
    Genera una solución inicial ordenando los clientes según su ventana de tiempo final (l_i).
    Luego, coloca al depósito al inicio y las estaciones de carga en sus posiciones adecuadas.
    
    Args:
        nodes (list): Lista de nodos, cada uno con id, coordenadas, ventanas de tiempo, etc.
    
    Returns:
        list: Ruta inicial con solo los clientes ordenados por su ventana de tiempo l_i.
    """
    # Filtrar solo los nodos de clientes (nodos con id de 1 a |V|)
    customer_nodes = nodes[0:num_customers-1]

    customer_nodes=[node for node in customer_nodes if node['id'] > 0]

    
    # Ordenar los clientes según el límite más tardío (l_i)
    sorted_customers = sorted(customer_nodes, key=lambda x: x['l'])
    
    # Crear la solución inicial: el depósito es el primer nodo (id 0), seguido de los clientes ordenados
    initial_solution = [0] + [node['id'] for node in sorted_customers]+[0] # Agregar depósito al inicio

    
    return initial_solution



def apply_local_search(solution, nodes, distance_matrix):
    """
    Aplica un operador de búsqueda local para mejorar la solución.

    Args:
        solution (list): Ruta actual (lista de IDs de nodos).
        nodes (list): Lista de nodos con información como los tiempos de llegada e_i y salida l_i.
        distance_matrix (list): Matriz de distancias entre nodos.
    
    Returns:
        list: Nueva ruta después de aplicar un operador de búsqueda local.
    """
    # Escoger un operador aleatorio (1-shift, 2-opt, swap)
    move_type = random.choice(['1-shift', '2-opt', 'swap'])
    
    if move_type == '1-shift':
        return apply_1shift(solution, nodes, distance_matrix)
    elif move_type == '2-opt':
        return apply_2opt(solution, nodes, distance_matrix)
    elif move_type == 'swap':
        return apply_swap(solution, nodes, distance_matrix)


def apply_1shift(solution, nodes, distance_matrix):
    if len(solution) <= 4:
        return solution

    i = random.randint(1, len(solution) - 2)
    j = random.randint(1, len(solution) - 2)
    
    while i == j:
        j = random.randint(1, len(solution) - 2)

    new_solution = solution[:]
    node = new_solution.pop(i)
    new_solution.insert(j, node)

    return new_solution


def apply_2opt(solution, nodes, distance_matrix):
    if len(solution) <= 4:
        return solution  # No se puede aplicar bien

    i = random.randint(1, len(solution) - 3)  # Excluyendo el primer y penúltimo
    j = random.randint(i + 1, len(solution) - 2)  # Excluyendo el último (que debe ser 0)

    new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]

    return new_solution



def apply_swap(solution, nodes, distance_matrix):
    if len(solution) <= 4:
        return solution

    i = random.randint(1, len(solution) - 2)
    j = random.randint(1, len(solution) - 2)
    
    while i == j:
        j = random.randint(1, len(solution) - 2)

    new_solution = solution[:]
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

    return new_solution



def apply_perturbation(solution, nodes, distance_matrix, R):
    """
    Aplica el operador de perturbación: elimina un número aleatorio de nodos y los inserta aleatoriamente.

    Args:
        solution (list): Ruta actual.
        nodes (list): Lista de nodos con información sobre cada uno.
        distance_matrix (list): Matriz de distancias entre nodos.
        R (int): Número de nodos a eliminar y reinsertar.
    
    Returns:
        list: Ruta modificada después de aplicar la perturbación.
    """
    # Eliminar R nodos aleatorios de la ruta
    nodes_to_remove = random.sample(solution[1:], R)  # Excluyendo el nodo 0 (depósito)
    new_solution = [node for node in solution if node not in nodes_to_remove]
    
    # Insertar los nodos eliminados aleatoriamente
    for node in nodes_to_remove:
        insert_position = random.randint(1, len(new_solution))  # No insertamos en la primera posición (depósito)
        new_solution.insert(insert_position, node)
    
    return new_solution

def is_time_feasible(solution, nodes, distance_matrix):
    """
    Verifica si una ruta es factible en cuanto a las restricciones de ventana de tiempo.

    Returns:
        (bool, float): Factibilidad y tiempo de llegada.
    """
    current_time = 0  

    for i in range(len(solution) - 1):
        from_node = solution[i]
        to_node = solution[i + 1]

        travel_time = distance_matrix[from_node][to_node]
        current_time += travel_time
        
        if current_time < nodes[to_node]['e']:
            current_time = nodes[to_node]['e']
        elif current_time > nodes[to_node]['l']:
            return False, current_time  

    return True, current_time



class StationInsertionProcedure:
    def __init__(self, solution, nodes, distance_matrix, Q, min_battery_level, f_Y_star, f_X, energy_rate=1.0):
        self.solution = solution
        self.nodes = nodes
        self.distance_matrix = distance_matrix
        self.Q = Q
        self.min_battery_level = min_battery_level
        self.UB = f_Y_star - f_X  # Upper Bound para el costo
        self.RD = f_X  # Remaining Distance
        self.energy_rate = energy_rate  # Tasa de consumo de energía por unidad de distancia
        self.labels = {0: [{'t': 0, 'q': Q, 'c': 0, 'predecessor': None, 'node': 0}]}  # Lista de etiquetas por nodo

    def generate_new_label(self, current_label, path_info, next_node):
        t, q, c, pred, node_idx = current_label['t'], current_label['q'], current_label['c'], current_label['predecessor'], current_label['node']
        t_new, energy_consumed, additional_cost, intermediates = path_info
        
        # Nueva etiqueta
        new_label = {
            't': t + t_new,
            'q': self.Q - 0,  # recargamos completamente en estaciones si hay
            'c': c + additional_cost,
            'predecessor': (node_idx, intermediates),
            'node': next_node
        }
        return new_label

    def is_feasible(self, label):
        return label['q'] >= self.min_battery_level and label['t'] >= 0

    def remove_redundant_labels(self, labels):
        non_dominated = []
        for label in labels:
            dominated = False
            for other in labels:
                if other == label:
                    continue
                if (other['t'] <= label['t']) and (other['q'] >= label['q']) and (other['c'] <= label['c']):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(label)
        return non_dominated

    def run(self):
        for i in range(1, len(self.solution)):
            prev_node = self.solution[i-1]
            current_node = self.solution[i]
            new_labels = []

            for label in self.labels[i-1]:
                for path_info in self.get_paths(prev_node, current_node):
                    t_new, energy_used, cost_additional, intermediates = path_info

                    # Batería restante después del trayecto
                    remaining_battery = label['q'] - energy_used
                    if remaining_battery < 0:
                        continue  # No factible por batería

                    if label['c'] + cost_additional < self.UB:
                        new_label = self.generate_new_label(label, path_info, i)
                        new_label['q'] = remaining_battery if intermediates is None else self.Q  # Si pasa estación, recarga
                        if self.is_feasible(new_label):
                            new_labels.append(new_label)

            if not new_labels:
                return -1  # No se pudieron generar etiquetas factibles (infactibilidad de batería)

            self.labels[i] = self.remove_redundant_labels(new_labels)

        # Seleccionar la mejor etiqueta final
        best_label = min(self.labels[len(self.solution) - 1], key=lambda x: x['c'])
        return self.generate_solution(best_label)

    def get_paths(self, node_i, node_j):
        paths = []

        # Directo
        dij = self.distance_matrix[node_i][node_j]
        energy_consumption_direct = dij * self.energy_rate
        if dij < 99999:
            paths.append((dij, energy_consumption_direct, dij, None))

        # 1 estación
        stations = [station for station in self.nodes if station['is_station'] == 1 and station['id'] != 0]
        for k in stations:
            i_to_k = self.distance_matrix[node_i][k['id']]
            k_to_j = self.distance_matrix[k['id']][node_j]
            if i_to_k < 99999 and k_to_j < 99999:
                energy_consumed_i_to_k = i_to_k * self.energy_rate
                recharge_time = (self.Q - (self.Q - energy_consumed_i_to_k)) / k['recharge_rate']
                total_time = i_to_k + recharge_time + k_to_j
                total_cost = i_to_k + k_to_j + recharge_time
                paths.append((total_time, energy_consumed_i_to_k, total_cost, (k['id'],)))

        # 2 estaciones
        for k1 in stations:
            for k2 in stations:
                i_to_k1 = self.distance_matrix[node_i][k1['id']]
                k1_to_k2 = self.distance_matrix[k1['id']][k2['id']]
                k2_to_j = self.distance_matrix[k2['id']][node_j]
                if i_to_k1 < 99999 and k1_to_k2 < 99999 and k2_to_j < 99999:
                    energy_consumed_i_to_k1 = i_to_k1 * self.energy_rate
                    energy_consumed_k1_to_k2 = k1_to_k2 * self.energy_rate
                    recharge_time_k1 = (self.Q - (self.Q - energy_consumed_i_to_k1)) / k1['recharge_rate']
                    recharge_time_k2 = (self.Q - (self.Q - energy_consumed_k1_to_k2)) / k2['recharge_rate']
                    total_time = i_to_k1 + recharge_time_k1 + k1_to_k2 + recharge_time_k2 + k2_to_j
                    total_cost = i_to_k1 + k1_to_k2 + k2_to_j + recharge_time_k1 + recharge_time_k2
                    paths.append((total_time, energy_consumed_i_to_k1, total_cost, (k1['id'], k2['id'])))

        return paths

    def generate_solution(self, best_label):
        # Reconstrucción de la ruta completa
        full_route = []
        label = best_label
        labels_dict = {}

        for idx in self.labels:
            for l in self.labels[idx]:
                labels_dict[l['node']] = l

        current_node = len(self.solution) - 1
        while True:
            pred_info = label['predecessor']
            full_route.append(self.solution[current_node])
            if pred_info is not None:
                pred_node, intermediates = pred_info
                if intermediates:
                    for station_id in reversed(intermediates):
                        full_route.append(station_id)
                label = labels_dict[pred_node]
                current_node = pred_node
            else:
                break

        full_route = full_route[::-1]  # Invertir la ruta para orden correcto

        # 🔥 Corregir si no termina en 0
        if full_route[-1] != 0:
            full_route.append(0)

        return full_route, best_label['c']


def detect_move(solution_before, solution_after):
    """
    Detecta el movimiento aplicado entre dos soluciones.
    Devuelve el par de nodos intercambiados/movidos.
    """
    for i in range(1, len(solution_before)):
        if solution_before[i] != solution_after[i]:
            return (solution_before[i], solution_after[i])
    return (None, None)


import math
import random

import math
import random

def hybrid_sa_ts(
    nodes, distance_matrix, num_customers, battery_capacity, energy_rate,
    max_iterations=1000, T0=100, alpha=0.95, cooling_interval=50,
    perturbation_interval=100, R=2,
    tabu_tenure_moves=10, tabu_tenure_stations=5
):
    """
    Algoritmo 4: Hybrid Simulated Annealing - Tabu Search
    Guarda el costo en cada iteración para graficar evolución.
    """

    # Inicializar solución
    solution_X = generate_initial_solution(nodes, num_customers)

    # Búsqueda local hasta encontrar una solución factible en ventanas de tiempo
    while not is_time_feasible(solution_X, nodes, distance_matrix)[0]:
        solution_X = apply_local_search(solution_X, nodes, distance_matrix)

    total_time = is_time_feasible(solution_X, nodes, distance_matrix)[1]

    # Aplicar procedimiento de inserción de estaciones
    sip = StationInsertionProcedure(
        solution_X, nodes, distance_matrix, battery_capacity, 0, float('inf'), total_time, energy_rate
    )
    result = sip.run()
    if result == -1:
        print("No se encontró solución factible inicial")
        return None

    solution_Y, cost_Y = result

    # Inicializar mejor solución encontrada
    best_solution = solution_Y
    best_cost = cost_Y

    # Inicializar temperatura
    T = T0

    # Inicializar listas tabú
    tabu_move_list = []
    tabu_station_insertion_list = []

    # Inicializar listas para evolución de costos
    cost_progression = []
    best_cost_progression = []

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Aplicar búsqueda local
        solution_X_prime = apply_local_search(solution_X, nodes, distance_matrix)

        # Detectar movimiento (para lista tabú)
        move = detect_move(solution_X, solution_X_prime)

        if move in tabu_move_list:
            continue  # Saltar si es tabú

        # Verificar factibilidad temporal
        feasible, total_time_prime = is_time_feasible(solution_X_prime, nodes, distance_matrix)
        if not feasible:
            continue

        # Aplicar procedimiento de inserción de estaciones a la nueva solución
        sip_prime = StationInsertionProcedure(
            solution_X_prime, nodes, distance_matrix, battery_capacity, 0, float('inf'), total_time_prime, energy_rate
        )
        result_prime = sip_prime.run()
        if result_prime == -1:
            continue  # No factible

        solution_Y_prime, cost_Y_prime = result_prime

        # Detectar estaciones usadas
        stations_used = tuple(sorted(set(node for node in solution_Y_prime if nodes[node]['is_station'] == 1)))

        if stations_used in tabu_station_insertion_list:
            continue  # Saltar si repite estaciones

        # Evaluar diferencia de costo
        delta = cost_Y_prime - cost_Y

        if delta < 0:
            # Mejoró: aceptar
            solution_X = solution_X_prime
            solution_Y = solution_Y_prime
            cost_Y = cost_Y_prime
        else:
            # No mejoró: aceptar con probabilidad SA
            prob = random.random()
            if prob < math.exp(-delta / T):
                solution_X = solution_X_prime
                solution_Y = solution_Y_prime
                cost_Y = cost_Y_prime

        # Actualizar mejor solución global
        if cost_Y < best_cost:
            best_solution = solution_Y
            best_cost = cost_Y

        # Actualizar listas tabú
        tabu_move_list.append(move)
        if len(tabu_move_list) > tabu_tenure_moves:
            tabu_move_list.pop(0)

        tabu_station_insertion_list.append(stations_used)
        if len(tabu_station_insertion_list) > tabu_tenure_stations:
            tabu_station_insertion_list.pop(0)

        # 🔥 Guardar evolución del costo en cada iteración
        cost_progression.append(cost_Y)
        best_cost_progression.append(best_cost)

        # Cooling (enfriamiento)
        if iteration % cooling_interval == 0:
            T = alpha * T

        # Perturbación
        if iteration % perturbation_interval == 0:
            solution_X = apply_perturbation(solution_X, nodes, distance_matrix, R)

    return best_solution, best_cost, cost_progression, best_cost_progression


if __name__ == '__main__':

    data = read_etsp_file('n20w120s10_1.txt')
    filtered_matrix, last_node = filter_arcs_by_time_window(data['nodes'], data['distance_matrix'])
    best_solution, best_cost, cost_progression, best_cost_progression = hybrid_sa_ts(
        nodes=data['nodes'],
        distance_matrix=data['distance_matrix'],
        num_customers=data['num_customers'],
        battery_capacity=data['battery_capacity'],
        energy_rate=data['energy_rate'],
        max_iterations=15000,     
        T0=10000,                 
        alpha=0.95,               
        cooling_interval=100,     
        perturbation_interval=500, 
        R=30,                     
        tabu_tenure_moves=400,     
        tabu_tenure_stations=75    
    )

    print("========= Resultados ============")
    print("Mejor ruta encontrada:", best_solution)
    print("Costo total:", best_cost)
    print("==================================")


    import matplotlib.pyplot as plt

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