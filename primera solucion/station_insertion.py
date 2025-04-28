# station_insertion.py

class StationInsertionProcedure:
    def __init__(self, solution, nodes, distance_matrix, Q, min_battery_level, f_Y_star, f_X, energy_rate=1.0):
        self.solution = solution  # lista de nodos
        self.nodes = nodes
        self.distance_matrix = distance_matrix
        self.Q = Q
        self.min_battery_level = min_battery_level
        self.UB = f_Y_star - f_X
        self.RD = f_X
        self.energy_rate = energy_rate
        self.labels = {0: [{'t': 0, 'q': Q, 'c': 0, 'predecessor': None, 'node': 0}]}

    def generate_new_label(self, current_label, path_info, next_node):
        t, q, c, pred, node_idx = current_label['t'], current_label['q'], current_label['c'], current_label['predecessor'], current_label['node']
        t_new, energy_consumed, additional_cost, intermediates = path_info

        new_label = {
            't': t + t_new,
            'q': self.Q - 0,
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

                    remaining_battery = label['q'] - energy_used
                    if remaining_battery < 0:
                        continue

                    if label['c'] + cost_additional < self.UB:
                        new_label = self.generate_new_label(label, path_info, i)
                        new_label['q'] = remaining_battery if intermediates is None else self.Q
                        if self.is_feasible(new_label):
                            new_labels.append(new_label)

            if not new_labels:
                return -1

            self.labels[i] = self.remove_redundant_labels(new_labels)

        best_label = min(self.labels[len(self.solution) - 1], key=lambda x: x['c'])
        return self.generate_solution(best_label)

    def get_paths(self, node_i, node_j):
        paths = []

        dij = self.distance_matrix[node_i][node_j]
        energy_consumption_direct = dij * self.energy_rate
        if dij < 99999:
            paths.append((dij, energy_consumption_direct, dij, None))

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

        full_route = full_route[::-1]
        if full_route[-1] != 0:
            full_route.append(0)

        return full_route, best_label['c']
