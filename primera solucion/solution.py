# solution.py

import random

class Solution:
    def __init__(self, nodes, distance_matrix, num_customers):
        self.nodes = nodes
        self.distance_matrix = distance_matrix
        self.num_customers = num_customers
        self.route = self.generate_initial_solution()

    def generate_initial_solution(self):
        customer_nodes = self.nodes[0:self.num_customers-1]
        customer_nodes = [node for node in customer_nodes if node['id'] > 0]
        sorted_customers = sorted(customer_nodes, key=lambda x: x['l'])
        return [0] + [node['id'] for node in sorted_customers] + [0]

    def is_time_feasible(self):
        current_time = 0
        for i in range(len(self.route) - 1):
            from_node = self.route[i]
            to_node = self.route[i + 1]

            travel_time = self.distance_matrix[from_node][to_node]
            current_time += travel_time

            if current_time < self.nodes[to_node]['e']:
                current_time = self.nodes[to_node]['e']
            elif current_time > self.nodes[to_node]['l']:
                return False, current_time

        return True, current_time

    def apply_local_search(self):
        move_type = random.choice(['1-shift', '2-opt', 'swap'])
        if move_type == '1-shift':
            return self.apply_1shift()
        elif move_type == '2-opt':
            return self.apply_2opt()
        elif move_type == 'swap':
            return self.apply_swap()

    def apply_1shift(self):
        if len(self.route) <= 4:
            return self.route

        i = random.randint(1, len(self.route) - 2)
        j = random.randint(1, len(self.route) - 2)
        while i == j:
            j = random.randint(1, len(self.route) - 2)

        new_route = self.route[:]
        node = new_route.pop(i)
        new_route.insert(j, node)
        self.route = new_route
        return self.route

    def apply_2opt(self):
        if len(self.route) <= 4:
            return self.route

        i = random.randint(1, len(self.route) - 3)
        j = random.randint(i + 1, len(self.route) - 2)

        new_route = self.route[:i] + self.route[i:j+1][::-1] + self.route[j+1:]
        self.route = new_route
        return self.route

    def apply_swap(self):
        if len(self.route) <= 4:
            return self.route

        i = random.randint(1, len(self.route) - 2)
        j = random.randint(1, len(self.route) - 2)
        while i == j:
            j = random.randint(1, len(self.route) - 2)

        new_route = self.route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        self.route = new_route
        return self.route

    def apply_perturbation(self, R):
        if len(self.route) <= R + 2:
            return self.route

        nodes_to_remove = random.sample(self.route[1:-1], R)
        new_route = [node for node in self.route if node not in nodes_to_remove]

        for node in nodes_to_remove:
            insert_position = random.randint(1, len(new_route) - 1)
            new_route.insert(insert_position, node)

        self.route = new_route
        return self.route
