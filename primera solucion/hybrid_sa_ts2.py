# hybrid_sa_ts.py

import random
import math
from station_insertion import StationInsertionProcedure

def detect_move(solution_before, solution_after):
    """
    Detecta el movimiento aplicado entre dos soluciones.
    """
    for i in range(1, len(solution_before)):
        if solution_before[i] != solution_after[i]:
            return (solution_before[i], solution_after[i])
    return (None, None)

class HybridSATS:
    def __init__(
        self, nodes, distance_matrix, num_customers, battery_capacity, energy_rate,
        max_iterations=1000, T0=100, alpha=0.95,
        cooling_interval=50, perturbation_interval=100, R=2,
        tabu_tenure_moves=10, tabu_tenure_stations=5
    ):
        self.nodes = nodes
        self.distance_matrix = distance_matrix
        self.num_customers = num_customers
        self.battery_capacity = battery_capacity
        self.energy_rate = energy_rate

        self.max_iterations = max_iterations
        self.T0 = T0
        self.alpha = alpha
        self.cooling_interval = cooling_interval
        self.perturbation_interval = perturbation_interval
        self.R = R
        self.tabu_tenure_moves = tabu_tenure_moves
        self.tabu_tenure_stations = tabu_tenure_stations

        self.cost_progression = []
        self.best_cost_progression = []


    def apply_simulated_annealing(self, current_solution, evaluation_criterion , alpha:int = 0.9995, temp_init:int = 100):

        best_solution = current_solution
        best_feasible, best_energy = evaluation_criterion(current_solution)
        temperature = temp_init

        while temperature > 1e-8 or not best_feasible:

            operator = random.choice(['shift', '2opt', 'swap', 'shuffle'])  

            if operator == 'shift':
                new_solution = self.apply_1shift(current_solution)
            elif operator == '2opt':
                new_solution = self.apply_2opt(current_solution)
            elif operator == 'swap':
                new_solution = self.apply_swap(current_solution)
            elif operator == 'shuffle':
                new_solution = self.apply_shuffle(current_solution)

            
            feasible, new_energy = evaluation_criterion(new_solution)

            if feasible:
                current_energy = evaluation_criterion(current_solution)[1]
                delta_energy = new_energy - current_energy
                if delta_energy < 0:
                    current_solution = new_solution
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_solution = current_solution
                        best_energy = current_energy
                        best_feasible = feasible
                else:
                    acceptance_probability = math.exp(-delta_energy/temperature)
                    if random.random() < acceptance_probability:
                        current_solution = new_solution
                        current_energy = new_energy
            
            if best_feasible:
                temperature = temperature*alpha


        return best_solution

    def generate_initial_solution(self):
        
        customer_nodes = self.nodes[0:self.num_customers-1]
        customer_nodes = [node for node in customer_nodes if node['id'] > 0]
        sorted_customers = sorted(customer_nodes, key=lambda x: x['l'])
        initial_solution = [0] + [node['id'] for node in sorted_customers] + [0]

        return initial_solution

    def is_time_feasible(self, solution):
        current_time = 0
        for i in range(len(solution) - 1):
            from_node = solution[i]
            to_node = solution[i + 1]
            travel_time = self.distance_matrix[from_node][to_node]
            current_time += travel_time
            if current_time < self.nodes[to_node]['e']:
                current_time = self.nodes[to_node]['e']
            elif current_time > self.nodes[to_node]['l']:
                return False, current_time
        return True, current_time

    def apply_local_search(self, solution):
        move_type = random.choice(['1-shift', '2-opt', 'swap'])
        if move_type == '1-shift':
            return self.apply_1shift(solution)
        elif move_type == '2-opt':
            return self.apply_2opt(solution)
        elif move_type == 'swap':
            return self.apply_swap(solution)

    def apply_1shift(self, solution):
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

    def apply_2opt(self, solution):
        if len(solution) <= 4:
            return solution
        i = random.randint(1, len(solution) - 3)
        j = random.randint(i + 1, len(solution) - 2)
        new_solution = solution[:i] + solution[i:j+1][::-1] + solution[j+1:]
        return new_solution

    def apply_swap(self, solution):
        if len(solution) <= 4:
            return solution
        i = random.randint(1, len(solution) - 2)
        j = random.randint(1, len(solution) - 2)
        while i == j:
            j = random.randint(1, len(solution) - 2)
        new_solution = solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
    
    def apply_shuffle(self, solution):
        solution = solution.copy()
        random.shuffle(solution)
        return solution

    def apply_perturbation(self, solution):
        nodes_to_remove = random.sample(solution[1:-1], self.R)
        new_solution = [node for node in solution if node not in nodes_to_remove]
        for node in nodes_to_remove:
            insert_position = random.randint(1, len(new_solution) - 1)
            new_solution.insert(insert_position, node)
        return new_solution

    def run(self):
        # Inicializar
        solution_X = self.generate_initial_solution()

        # Hacer que la inicial sea factible
        
        solution_X = self.apply_simulated_annealing(solution_X, self.is_time_feasible)

        feasible, total_time = self.is_time_feasible(solution_X)

        sip = StationInsertionProcedure(
            solution_X, self.nodes, self.distance_matrix,
            self.battery_capacity, 0, float('inf'), total_time, self.energy_rate
        )
        result = sip.run()
        if result == -1:
            print("No se encontró solución inicial factible")
            return None

        solution_Y, cost_Y = result
        best_solution = solution_Y
        best_cost = cost_Y

        # Inicializar listas tabu
        tabu_move_list = []
        tabu_station_insertion_list = []

        T = self.T0
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            solution_X_prime = self.apply_simulated_annealing(solution_X)
            move = detect_move(solution_X, solution_X_prime)

            if move in tabu_move_list:
                continue

            feasible, total_time_prime = self.is_time_feasible(solution_X_prime)
            if not feasible:
                continue

            sip_prime = StationInsertionProcedure(
                solution_X_prime, self.nodes, self.distance_matrix,
                self.battery_capacity, 0, float('inf'), total_time_prime, self.energy_rate
            )
            result_prime = sip_prime.run()
            if result_prime == -1:
                continue

            solution_Y_prime, cost_Y_prime = result_prime

            stations_used = tuple(sorted(node for node in solution_Y_prime if self.nodes[node]['is_station'] == 1))
            if stations_used in tabu_station_insertion_list:
                continue

            delta = cost_Y_prime - cost_Y

            if delta < 0:
                solution_X = solution_X_prime
                solution_Y = solution_Y_prime
                cost_Y = cost_Y_prime
            else:
                if random.random() < math.exp(-delta / T):
                    solution_X = solution_X_prime
                    solution_Y = solution_Y_prime
                    cost_Y = cost_Y_prime

            if cost_Y < best_cost:
                best_solution = solution_Y
                best_cost = cost_Y

            tabu_move_list.append(move)
            if len(tabu_move_list) > self.tabu_tenure_moves:
                tabu_move_list.pop(0)

            tabu_station_insertion_list.append(stations_used)
            if len(tabu_station_insertion_list) > self.tabu_tenure_stations:
                tabu_station_insertion_list.pop(0)

            self.cost_progression.append(cost_Y)
            self.best_cost_progression.append(best_cost)

            if iteration % self.cooling_interval == 0:
                T = self.alpha * T

            if iteration % self.perturbation_interval == 0:
                solution_X = self.apply_perturbation(solution_X)

        return best_solution, best_cost, self.cost_progression, self.best_cost_progression
