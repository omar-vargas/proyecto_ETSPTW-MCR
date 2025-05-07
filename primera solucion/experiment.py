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

