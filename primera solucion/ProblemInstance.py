
class ProblemInstance:
    def __init__(self, filename):
        self.filename = filename
        self.nodes = []
        self.distance_matrix = []
        self.num_customers = 0
        self.num_stations = 0
        self.battery_capacity = 0
        self.energy_rate = 1.0
        self.filtered_matrix = None
        self.last_node = None

    def load(self):
        with open(self.filename, 'r') as file:
            self.num_customers = int(file.readline())
            self.num_stations = int(file.readline())
            self.battery_capacity = float(file.readline())
            self.energy_rate = float(file.readline())

            total_nodes = 1 + self.num_customers + self.num_stations + 1  # depot + customers + stations + depot-station

            for _ in range(total_nodes):
                line = file.readline().strip()
                if line:
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
                    self.nodes.append(node)

            # Read distance matrix
            for line in file:
                if line.strip():
                    row = list(map(float, line.strip().split('\t')))
                    self.distance_matrix.append(row)

    def filter_arcs_by_time_window(self, unreachable_value=99999):
        n = len(self.nodes)
        self.filtered_matrix = [[unreachable_value for _ in range(n)] for _ in range(n)]

        last_node = n - 1
        self.last_node = last_node
        l_last = self.nodes[last_node]['l']

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ei = self.nodes[i]['e']
                lj = self.nodes[j]['l']
                dij = self.distance_matrix[i][j]
                dju = self.distance_matrix[j][last_node]

                if ei + dij <= lj and ei + dij + dju <= l_last:
                    self.filtered_matrix[i][j] = dij
