import random
import math
from typing import List, Dict, Tuple
import re
import sys
from collections import defaultdict


class Node:
    def __init__(self, id: int, x: float, y: float, earliest: float, latest: float, service_time: float = 0, has_private_charger: bool = False):
        self.id = id
        self.x = x
        self.y = y
        self.earliest = earliest
        self.latest = latest
        self.service_time = service_time
        self.has_private_charger = has_private_charger

    def distance_to(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class ETSPTW_MCR:
    def __init__(
            self,
            nodes: List[Node], depot: Node, charging_stations: List[Node],
            battery_capacity: float, consumption_rate: float,
            private_charging_rate: float, public_charging_rate: float):
        """
        Args:
            nodes: List of customer nodes (excluding depot)
            depot: The depot node (start and end point)
            charging_stations: List of public charging station nodes
            battery_capacity: attery capacity of the vehicle
            consumption_rate: energy consumed per unit of distance
            private_charging_rate: charging rate at private stations (customer locations)
            public_charging_rate: charging rate at public charging stations
        """
        self.nodes = nodes
        self.depot = depot
        self.charging_stations = charging_stations
        self.battery_capacity = battery_capacity
        self.consumption_rate = consumption_rate
        self.private_charging_rate = private_charging_rate
        self.public_charging_rate = public_charging_rate

        # Compute distance matrix
        self.all_nodes = [depot] + nodes + charging_stations
        self.distance_matrix = self._compute_distance_matrix()

        # Compute time matrix (assuming constant speed = 1 for simplicity. Distance = travel time)
        self.time_matrix = self.distance_matrix.copy()

        # Preprocessing
        self._preprocess_arcs()

        # Parameters from the paper
        self.tabu_tenure1 = 400  # For move operations
        self.tabu_tenure2 = 75   # For station insertion
        self.perturbation_freq = 500
        self.perturbation_size = 30
        self.init_temp = 10000
        self.cooling_rate = 0.95
        self.cooling_length = 100
        self.omega = 0.7  # Probability to continue from best ETSPTW-MCR solution

        # Tabu lists
        self.tabu_list1 = []  # For move operations
        self.tabu_list2 = []  # For station insertion

        # Best solutions found
        self.best_tsp_solution = None
        self.best_etsptw_solution = None

    def _preprocess_arcs(self):
        # Create node sets as defined in the paper
        V = self.nodes
        V0 = V + [self.depot]
        V_N1 = V + [self.depot]  # Using depot as the end node (N+1)

        # Remove arcs that violate time window constraints
        self.valid_arcs = defaultdict(list)

        for i in V0:
            for j in V_N1:
                if i == j:
                    continue

                # Condition 1: e_i + t_ij > l_j
                if i.earliest + self.time_matrix[(i.id, j.id)] > j.latest:
                    continue

                # Condition 2: e_i + t_ij + t_j,depot > depot.latest
                # We use the depot as the end node (N+1)
                if i.earliest + self.time_matrix[(i.id, j.id)] + self.time_matrix[(j.id, self.depot.id)] > self.depot.latest:
                    continue

                # If we get here, the arc is feasible
                self.valid_arcs[i.id].append(j.id)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            content = f.read()

        # Instance parameters
        problem_section = re.search(
            r'\[PROBLEM\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if not problem_section:
            raise ValueError("Missing [PROBLEM] section in input file")

        params = {}
        for line in problem_section.group(1).split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = float(value.strip())

        required_params = [
            'BATTERY_CAPACITY', 'CONSUMPTION_RATE', 'PRIVATE_CHARGE_RATE', 'PUBLIC_CHARGE_RATE'
        ]
        for param in required_params:
            if param not in params:
                raise ValueError(
                    f"Missing required parameter {param} in [PROBLEM] section")

        # Nodes
        nodes_section = re.search(
            r'\[NODES\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if not nodes_section:
            raise ValueError("Missing [NODES] section in input file")

        nodes = []
        depot = None
        for line in nodes_section.group(1).split('\n'):
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            earliest = float(parts[3])
            latest = float(parts[4])
            service_time = float(parts[5])
            has_private_charger = bool(
                int(parts[6])) if len(parts) > 6 else False

            node = Node(node_id, x, y, earliest, latest,
                        service_time, has_private_charger)

            # Assuming depot has ID 0
            if node_id == 0:
                depot = node
            else:
                nodes.append(node)

        if not depot:
            raise ValueError(
                "Depot (node with ID 0) not found in [NODES] section")

        # Charging stations
        charging_stations = []
        stations_section = re.search(
            r'\[CHARGING_STATIONS\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if stations_section:
            for line in stations_section.group(1).split('\n'):
                if not line.strip() or line.strip().startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue

                station_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                # Charging stations have no time windows or service times
                charging_stations.append(
                    Node(station_id, x, y, 0, float('inf'), 0, False))

        return cls(
            nodes=nodes,
            depot=depot,
            charging_stations=charging_stations,
            battery_capacity=params['BATTERY_CAPACITY'],
            consumption_rate=params['CONSUMPTION_RATE'],
            private_charging_rate=params['PRIVATE_CHARGE_RATE'],
            public_charging_rate=params['PUBLIC_CHARGE_RATE']
        )

    def save_solution(self, solution: Dict, file_path: str):
        with open(file_path, 'w') as f:
            f.write(f"Total distance: {solution['total_distance']}\n")
            f.write(f"Total time: {solution['total_time']}\n")
            f.write(f"Feasible: {solution['feasible']}\n\n")

            f.write("Route:\n")
            for node in solution['route']:
                f.write(f"{node.id} ({node.x}, {node.y})")
                if node.id in solution['charging_ops']:
                    charge_amount, charge_time = solution['charging_ops'][node.id]
                    charger_type = "private" if node.has_private_charger else "public"
                    f.write(
                        f" [Charge: {charge_amount:.2f} units, {charge_time:.2f} time, {charger_type}]")
                f.write("\n")

            f.write("\nCharging operations:\n")
            for node_id, (amount, time) in solution['charging_ops'].items():
                f.write(
                    f"Node {node_id}: Charge {amount:.2f} units in {time:.2f} time\n")

    # Compute the distance matrix between all nodes
    def _compute_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        distance_matrix = {}
        for n1 in self.all_nodes:
            for n2 in self.all_nodes:
                distance_matrix[(n1.id, n2.id)] = n1.distance_to(n2)
        return distance_matrix



    def create_starting_population(self, population_size:int  = 100):
        population = [{
            'route': [self.depot] + sorted(self.nodes, key=lambda x: x.latest) + [self.depot],
            'charging_ops': {},
            'feasible': False,
            'total_distance': 0,
            'total_time': 0}]
            
        for i in range(1, population_size):
            individual_route = self.nodes.copy()
            random.shuffle(individual_route)
            individual_route = [self.depot] + individual_route + [self.depot]

            individual = {
            'route': individual_route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': 0,
            'total_time': 0
        }
            population.append(individual) 

        return population
    
    def natural_selection(self, old_generation):
        survivors = []
        random.shuffle(old_generation)
        mid = len(old_generation)//2 

        if len(old_generation)%2 == 1:
            old_generation.pop()
        
        for i in range(mid):
            if self.calculate_total_delay(old_generation[i]) < self.calculate_total_delay(old_generation[mid + i]):
                survivors.append(old_generation[i])
            else:
                survivors.append(old_generation[mid + i])

        return survivors

    def create_offspring(self, sol_A, sol_B):
        offspring = []
        start = random.randint(1, len(sol_A['route'])-1)
        end = random.randint(1, len(sol_A['route'])-1)

        sol_A_section = sol_A['route'][start:end]
        sol_A_section_ids = [node.id for node in sol_A_section]
        sol_B_remeaning_nodes = [node for node in sol_B['route'] if node.id not in sol_A_section_ids]

        for i in range(len(sol_A['route'])):
            if start <= i < end:
                offspring.append(sol_A_section.pop(0))
            else:
                offspring.append(sol_B_remeaning_nodes.pop(0))
        
        return {
            'route': offspring,
            'charging_ops': {},
            'feasible': False,
            'total_distance': 0,
            'total_time': 0
        }
    
    def apply_crossover(self, survivors):
        offsprings = []

        mid = len(survivors) //2 
        if len(survivors)%2 == 1:
            survivors.pop()

        for i in range(mid):
            sol_A = survivors[i]
            sol_B = survivors[mid+i]

            for j in range(2):
                offsprings.append(self.create_offspring(sol_A, sol_B))
                offsprings.append(self.create_offspring(sol_B, sol_A))
        
        return offsprings
    
    def apply_mutations(self, generation, probability:float = 0.1):
        new_generation = []
        for sol in generation:
            if random.random() < probability:
                sol = self.apply_local_search(sol)
            new_generation.append(sol)
        return new_generation

    def elitism(self, survivors, generation):

        keyFunc = lambda node: self.calculate_total_delay(node)
        survivors.sort(key=keyFunc)
        generation.sort(key=keyFunc)

        best_offsprings = survivors[:10] + generation[:-10]
        best_offsprings.sort(key=keyFunc)

        return best_offsprings



    def initialize_solution_genetic(self, num_generations: int = 1000) -> Dict:
        # Initial solution (sorted by l_i)

        x_solution = {
            'route': [self.depot] + sorted(self.nodes, key=lambda x: x.latest) + [self.depot],
            'charging_ops': {},
            'feasible': False,
            'total_distance': 0,
            'total_time': 0
        }

        delay = self.calculate_total_delay(x_solution)

        if delay > 0:
            old_generation = self.create_starting_population()
            
            for i in range(num_generations):
                survivors = self.natural_selection(old_generation)
                crossovers = self.apply_crossover(survivors)
                mutants = self.apply_mutations(crossovers)
                new_generation = self.elitism(survivors, mutants)

                best_individual = new_generation[0]
                print(self.calculate_total_delay(best_individual))  #Eliminar este print
                if self.calculate_total_delay(best_individual) <= 0:
                    x_solution = best_individual
                    break

                old_generation = new_generation


  
        delay = self.calculate_total_delay(x_solution)
  
        
        x_solution['feasible'] = (delay <= 0)
        x_solution['total_distance'] = self._calculate_route_distance(x_solution['route'])

        y_solution = self.station_insertion(x_solution)

        if self.is_etsptwmcr_feasible(y_solution):
            return x_solution, y_solution
        else:
            for _ in range(0, 100):
                x_solution = self.apply_local_search(x_solution)
                y_solution = self.station_insertion(x_solution)
                if self.is_etsptwmcr_feasible(y_solution):
                    return x_solution, y_solution

        return x_solution, y_solution


    def initialize_solution(self) -> Dict:
        # Initial solution (sorted by l_i)
        x_solution = {
            'route': [self.depot] + sorted(self.nodes, key=lambda x: x.latest) + [self.depot],
            'charging_ops': {},
            'feasible': False,
            'total_distance': 0,
            'total_time': 0
        }




        delay = self.calculate_total_delay(x_solution)
        i = 0
        while delay > 0 and i < 50000:
            x_solution = self.apply_local_search(x_solution, consider_delay = True)
            newDelay = self.calculate_total_delay(x_solution)
            if newDelay < delay:
                delay = newDelay
                print(f"Dealy: {delay}")
            i+=1
            

        
        x_solution['feasible'] = (delay <= 0)
        x_solution['total_distance'] = self._calculate_route_distance(
            x_solution['route'])

        y_solution = self.station_insertion(x_solution)

        if self.is_etsptwmcr_feasible(y_solution):
            return x_solution, y_solution
        else:
            for _ in range(0, 100):
                x_solution = self.apply_local_search(x_solution)
                y_solution = self.station_insertion(x_solution)
                if self.is_etsptwmcr_feasible(y_solution):
                    return x_solution, y_solution

        return x_solution, y_solution

    def _calculate_route_distance(self, route: List[Node]):
        total = 0
        for i in range(len(route)-1):
            total += self.distance_matrix[(route[i].id, route[i+1].id)]
        return total

    # Checks if a solution is time feasible (without considering battery)
    def is_time_feasible(self, solution: Dict) -> bool:
        current_time = self.depot.earliest
        route = solution['route']

        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]

            travel_time = self.time_matrix[(from_node.id, to_node.id)]
            arrival_time = current_time + travel_time

            # Check time window
            if arrival_time > to_node.latest:
                return False

            # Update current time (wait if early)
            current_time = max(arrival_time, to_node.earliest)

            # Add service time (if not depot)
            if to_node != self.depot:
                current_time += to_node.service_time

        return True

    def is_etsptwmcr_feasible(self, solution: Dict) -> bool:
        """
        Check if a solution is feasible for the ETSPTW-MCR problem.

        Args:
            solution: The solution to check, containing 'route' and 'charging_ops'

        Returns:
            bool: True if the solution is feasible, False otherwise
        """
        if not solution or not solution.get('route'):
            return False

        route = solution['route']
        charging_ops = solution.get('charging_ops', {})

        # 1. Route starts and ends at depot
        if route[0] != self.depot or route[-1] != self.depot:
            return False

        # 2. Time window feasibility
        current_time = self.depot.earliest
        current_battery = self.battery_capacity

        for i in range(1, len(route)):
            prev_node = route[i-1]
            current_node = route[i]

            travel_time = self.time_matrix[(prev_node.id, current_node.id)]
            arrival_time = current_time + travel_time

            # Time window
            if arrival_time > current_node.latest:
                return False

            # Update current time (wait if early)
            current_time = max(arrival_time, current_node.earliest)

            charge_amount = charging_ops.get(current_node.id, (0, 0))[0]
            charge_time = charging_ops.get(current_node.id, (0, 0))[1]

            # Charging only done at valid locations
            if charge_amount > 0:
                if current_node == self.depot:
                    if current_node not in self.charging_stations:
                        return False
                elif current_node in self.charging_stations:
                    pass
                elif not current_node.has_private_charger:
                    return False

            current_time += charge_time

            distance = self.distance_matrix[(prev_node.id, current_node.id)]
            energy_consumed = distance * self.consumption_rate

            if current_battery < energy_consumed:
                return False

            current_battery -= energy_consumed

            # Apply charging
            if charge_amount > 0:
                if current_node in self.charging_stations:
                    # Public charging - we assume full charge
                    current_battery = self.battery_capacity
                else:
                    # Private charging
                    current_battery = min(
                        self.battery_capacity, current_battery + charge_amount)

            # Add service time (if not depot)
            if current_node != self.depot:
                current_time += current_node.service_time

        # 3. Returns to depot with non-negative battery
        if current_battery < 0:
            return False

        # 4. Total time does not exceed depot latest time
        if current_time > self.depot.latest:
            return False

        # 5. All customers are visited exactly once (excluding depot and charging stations)
        customer_ids = {node.id for node in self.nodes}
        visited_customers = {node.id for node in route if node in self.nodes}

        if customer_ids != visited_customers:
            return False

        return True

    def calculate_total_delay(self, solution: Dict) -> float:
        total_delay = 0
        current_time = self.depot.earliest

        for i in range(1, len(solution['route'])):
            prev_node = solution['route'][i-1]
            node = solution['route'][i]

            # Travel time
            current_time += self.time_matrix[(prev_node.id, node.id)]

            # Calculate delay
            delay = max(0, current_time - node.latest)
            total_delay += delay

            # Service time (if not depot)
            if node != self.depot:
                current_time += node.service_time

        return total_delay

    # Apply local search operators to improve the solution
    def apply_local_search(self, current_solution: Dict, consider_delay: bool = False) -> Dict:
        operator = random.choice(['shift', '2opt', 'swap', 'shuffle'])
        
        if operator == 'shift':
            new_solution = self.one_shift_move(current_solution)
        elif operator == '2opt':
            new_solution = self.two_opt_move(current_solution)
        elif operator == 'swap':
            new_solution = self.swap_move(current_solution)
        elif operator == 'shuffle':
            new_solution = self.shuffle_move(current_solution)

        # During initialization, use delay-based acceptance
        if consider_delay:
            current_delay = self.calculate_total_delay(current_solution)
            new_delay = self.calculate_total_delay(new_solution)
            return new_solution if new_delay <= current_delay else current_solution
        else:
            return new_solution

    # Perform a 1-shift neighborhood move
    def one_shift_move(self, solution: Dict) -> Dict:
        route = solution['route'].copy()

        # Exclude depot from shifting
        if len(route) <= 3:  # Only depot and one customer
            return solution

        # Select a customer to move (not depot)
        customer_pos = random.randint(1, len(route)-2)
        customer = route.pop(customer_pos)

        # Select new position to insert (not before first depot or after last)
        new_pos = random.randint(1, len(route)-1)
        while new_pos == customer_pos:
            new_pos = random.randint(1, len(route)-1)

        route.insert(new_pos, customer)

        return {
            'route': route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': self._calculate_route_distance(route),
            'total_time': 0
        }

    # Perform a 2-opt neighborhood move
    def two_opt_move(self, solution: Dict) -> Dict:
        route = solution['route'].copy()

        if len(route) <= 4:  # Need at least 2 customers for 2-opt
            return solution

        # Select two different edges (i, i+1) and (j, j+1)
        i = random.randint(1, len(route)-3)
        j = random.randint(i+1, len(route)-2)

        # Reverse the subroute between i+1 and j
        new_route = route[:i+1] + route[j:i:-1] + route[j+1:]

        return {
            'route': new_route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': self._calculate_route_distance(new_route),
            'total_time': 0
        }

    # Perform a swap neighborhood move
    def swap_move(self, solution: Dict) -> Dict:
        route = solution['route'].copy()

        if len(route) <= 3:  # Need at least 2 customers to swap
            return solution

        # Select two different customer positions
        i = random.randint(1, len(route)-2)
        j = random.randint(1, len(route)-2)
        while i == j:
            j = random.randint(1, len(route)-2)

        # Swap the customers
        route[i], route[j] = route[j], route[i]

        return {
            'route': route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': self._calculate_route_distance(route),
            'total_time': 0
        }

    def shuffle_move(self, solution: Dict) -> Dict:
        route = solution["route"].copy()
        customers = route[1:-1]
        random.shuffle(customers)
        route = [route[0]] + customers + [route[-1]]

        return {
            'route': route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': self._calculate_route_distance(route),
            'total_time': 0
        }



    """
    Insert charging stations into a TSPTW solution using dynamic programming
    """
    def station_insertion(self, tsp_solution: Dict) -> Dict:
        X_prime = tsp_solution['route']
        # number of customers (excluding depots)
        N = len(X_prime) - 2

        UB = float('inf')
        if self.best_etsptw_solution and self.best_etsptw_solution['feasible']:
            UB = self.best_etsptw_solution['total_distance']
        else:
            UB = 2 * self._calculate_route_distance(X_prime)

        # Calculate remaining distance (RD) for each node in the route
        RD = [0] * (N + 2)
        for i in range(N + 1, 0, -1):
            RD[i-1] = RD[i] + \
                self.distance_matrix[(X_prime[i-1].id, X_prime[i].id)]

        # Initialize labels
        L = [[] for _ in range(N + 2)]
        L[0].append({
            't': self.depot.earliest,
            'q': self.battery_capacity,
            'c': 0,
            'pred': None,
            'charge_at': None,
            'charge_amount': 0,
            'path': []
        })

        for i in range(1, N + 2):
            current_node = X_prime[i]
            prev_node = X_prime[i-1]
            current_RD = RD[i]

            for label in L[i-1]:
                t = label['t']
                q = label['q']
                c = label['c']
                path = label['path']

                # Option 1: Direct path without charging
                distance = self.distance_matrix[(
                    prev_node.id, current_node.id)]
                energy_needed = distance * self.consumption_rate

                if q >= energy_needed:
                    arrival_time = t + \
                        self.time_matrix[(prev_node.id, current_node.id)]
                    if arrival_time > current_node.latest:
                        continue  # Time window violation

                    start_time = max(arrival_time, current_node.earliest)
                    new_cost = c + distance

                    if new_cost < UB:
                        new_label = {
                            't': start_time + (current_node.service_time if i != N + 1 else 0),
                            'q': q - energy_needed,
                            'c': new_cost,
                            'pred': (i-1, len(L[i-1]) - 1),
                            'charge_at': None,
                            'charge_amount': 0,
                            'path': path + [(prev_node.id, current_node.id)]
                        }

                        # Update UB if this is a complete solution with enough charge
                        if i == N + 1 and new_label['q'] >= 0:
                            UB = min(UB, new_cost)

                        L[i].append(new_label)

                # Option 2: Charge at previous node if possible
                if prev_node.has_private_charger or prev_node in self.charging_stations:
                    if prev_node in self.charging_stations:
                        # Public charging - full charge
                        charge_amount = self.battery_capacity - \
                            (q - energy_needed)
                        charge_time = charge_amount / self.public_charging_rate
                    else:
                        # Private charging - limited by waiting time
                        arrival_at_prev = t
                        wait_time = max(
                            0, prev_node.earliest - arrival_at_prev)
                        max_charge = wait_time * self.private_charging_rate
                        charge_amount = min(
                            max_charge, self.battery_capacity - (q - energy_needed))
                        charge_time = charge_amount / self.private_charging_rate

                    if charge_amount > 0 and q >= energy_needed:
                        total_cost = c + distance
                        if total_cost < UB:
                            new_q = min(self.battery_capacity, q -
                                        energy_needed + charge_amount)

                            if prev_node in self.charging_stations:
                                departure_time = t + charge_time
                                arrival_time = departure_time + \
                                    self.time_matrix[(
                                        prev_node.id, current_node.id)]
                            else:
                                departure_time = max(t, prev_node.earliest)
                                arrival_time = departure_time + \
                                    self.time_matrix[(
                                        prev_node.id, current_node.id)]

                            if arrival_time > current_node.latest:
                                continue  # Time window violation

                            start_time = max(
                                arrival_time, current_node.earliest)

                            new_label = {
                                't': start_time + (current_node.service_time if i != N + 1 else 0),
                                'q': new_q,
                                'c': total_cost,
                                'pred': (i-1, len(L[i-1]) - 1),
                                'charge_at': prev_node.id,
                                'charge_amount': charge_amount,
                                'path': path + [(prev_node.id, current_node.id)]
                            }

                            # Update UB if this is a complete solution with enough charge
                            if i == N + 1 and new_label['q'] >= 0:
                                UB = min(UB, total_cost)

                            L[i].append(new_label)

                # Option 3: Insert charging stations along the path
                if energy_needed > q or (q - energy_needed) < current_RD:
                    # Find candidate charging stations between prev_node and current_node
                    candidate_stations = []
                    for cs in self.charging_stations:
                        if cs == prev_node or cs == current_node:
                            continue

                        # Check if station is roughly between the nodes
                        dist_to_cs = self.distance_matrix[(
                            prev_node.id, cs.id)]
                        dist_from_cs = self.distance_matrix[(
                            cs.id, current_node.id)]

                        if dist_to_cs + dist_from_cs < 1.5 * distance:  # Within reasonable detour
                            candidate_stations.append(
                                (cs, dist_to_cs, dist_from_cs))

                    # Sort by total detour distance
                    candidate_stations.sort(key=lambda x: x[1] + x[2])

                    # Consider top 3 closest stations to limit computation
                    for cs, dist_to_cs, dist_from_cs in candidate_stations[:3]:
                        energy_to_cs = dist_to_cs * self.consumption_rate
                        energy_from_cs = dist_from_cs * self.consumption_rate

                        if q >= energy_to_cs:
                            # Need to charge enough to reach next node with some buffer
                            required_charge = max(
                                0, energy_from_cs - (self.battery_capacity - (q - energy_to_cs)))
                            charge_amount = required_charge + 0.1 * self.battery_capacity  # Small buffer
                            charge_time = charge_amount / self.public_charging_rate

                            total_cost = c + dist_to_cs + dist_from_cs
                            if total_cost >= UB:
                                continue

                            # Time calculations
                            arrival_at_cs = t + \
                                self.time_matrix[(prev_node.id, cs.id)]
                            departure_from_cs = arrival_at_cs + charge_time
                            arrival_at_current = departure_from_cs + \
                                self.time_matrix[(cs.id, current_node.id)]

                            if arrival_at_current > current_node.latest:
                                continue  # Time window violation

                            start_time = max(
                                arrival_at_current, current_node.earliest)
                            new_q = self.battery_capacity - energy_from_cs

                            new_label = {
                                't': start_time + (current_node.service_time if i != N + 1 else 0),
                                'q': new_q,
                                'c': total_cost,
                                'pred': (i-1, len(L[i-1]) - 1),
                                'charge_at': cs.id,
                                'charge_amount': charge_amount,
                                'path': path + [(prev_node.id, cs.id), (cs.id, current_node.id)]
                            }

                            # Update UB if this is a complete solution with enough charge
                            if i == N + 1 and new_label['q'] >= 0:
                                UB = min(UB, total_cost)

                            L[i].append(new_label)

            # If no labels at this node, solution is infeasible
            if not L[i]:
                return {
                    'route': X_prime,
                    'charging_ops': {},
                    'feasible': False,
                    'total_distance': float('inf'),
                    'total_time': float('inf')
                }

            # Apply dominance rules to prune labels
            L[i] = self._prune_labels(L[i])

        # Find the best feasible label at the end of the route
        feasible_labels = [label for label in L[N+1] if label['q'] >= 0]
        if not feasible_labels:
            return {
                'route': X_prime,
                'charging_ops': {},
                'feasible': False,
                'total_distance': float('inf'),
                'total_time': float('inf')
            }

        best_label = min(feasible_labels, key=lambda x: x['c'])

        # Reconstruct the solution with charging operations
        charging_ops = {}
        route_with_charging = [self.depot]
        current_label = best_label

        # We need to reconstruct the path including any inserted charging stations
        full_path = []
        while current_label['pred'] is not None:
            full_path.extend(reversed(current_label['path']))
            if current_label['charge_at'] is not None:
                charging_node_id = current_label['charge_at']
                charging_ops[charging_node_id] = (
                    current_label['charge_amount'],
                    current_label['charge_amount'] / self.public_charging_rate
                )
            current_label = L[current_label['pred'][0]][current_label['pred'][1]]

        # Build the complete route with charging stations
        full_path = list(reversed(full_path))
        node_sequence = [self.depot.id]
        for arc in full_path:
            if arc[0] == node_sequence[-1]:
                node_sequence.append(arc[1])

        # Convert node IDs back to Node objects
        id_to_node = {node.id: node for node in self.all_nodes}
        final_route = [id_to_node[node_id] for node_id in node_sequence]

        return {
            'route': final_route,
            'charging_ops': charging_ops,
            'feasible': True,
            'total_distance': best_label['c'],
            'total_time': best_label['t']
        }

    # Prune dominated labels (those that are worse in both time and battery)
    def _prune_labels(self, labels: List[Dict]) -> List[Dict]:
        if not labels:
            return []

        # Sort labels by time
        labels.sort(key=lambda x: x['t'])

        pruned = [labels[0]]
        for label in labels[1:]:
            # Check if this label is dominated by any in pruned
            dominated = False
            for p in pruned:
                if (p['t'] <= label['t'] and
                    p['q'] >= label['q'] and
                        p['c'] <= label['c']):
                    dominated = True
                    break

            if not dominated:
                # Also remove any labels in pruned that this one dominates
                pruned = [p for p in pruned if not (label['t'] <= p['t'] and
                                                    label['q'] >= p['q'] and
                                                    label['c'] <= p['c'])]
                pruned.append(label)

        return pruned

    # Determine if we should evaluate this TSPTW solution for ETSPTW-MCR
    def should_evaluate(self, tsp_solution: Dict, best_etsptw_solution: Dict, tabu_lists) -> bool:
        # Check if better than best TSP solution
        if not self.best_tsp_solution or tsp_solution['total_distance'] < self.best_tsp_solution['total_distance']:
            return True

        # Check if not in tabu list
        solution_signature = self.get_solution_signature(tsp_solution)
        if solution_signature not in tabu_lists.tabu_list2:
            return True

        # Check if has more slack time than best ETSPTW solution
        if best_etsptw_solution and self.calculate_slack_time(tsp_solution) > self.calculate_slack_time(best_etsptw_solution):
            return True

        return False

    # Create a signature for a solution to use in tabu list
    def get_solution_signature(self, solution: Dict) -> Tuple:
        # Use first and last customer and their arrival times as signature
        if len(solution['route']) <= 2:
            return (0, 0, 0, 0)

        first_customer = solution['route'][1]
        last_customer = solution['route'][-2]

        # Calculate arrival times (simplified)
        arrival_first = self.depot.earliest + \
            self.time_matrix[(self.depot.id, first_customer.id)]
        # This would need proper calculation in full implementation
        arrival_last = arrival_first

        return (first_customer.id, last_customer.id, arrival_first, arrival_last)

    # Calculate total slack time in the solution
    def calculate_slack_time(self, solution: Dict) -> float:
        slack = 0
        current_time = self.depot.earliest

        for i in range(len(solution['route'])-1):
            from_node = solution['route'][i]
            to_node = solution['route'][i+1]

            arrival_time = current_time + \
                self.time_matrix[(from_node.id, to_node.id)]
            slack += max(0, to_node.latest - arrival_time)

            current_time = max(arrival_time, to_node.earliest)
            if to_node != self.depot:
                current_time += to_node.service_time

        return slack

    # Perturb the solution by removing and reinserting R customers.
    def perturb_solution(self, solution: Dict) -> Dict:
        route = solution['route'].copy()

        if len(route) <= 3:  # Need at least 2 customers to perturb
            return solution

        R = min(self.perturbation_size, len(route)-2)
        removed_positions = random.sample(range(1, len(route)-1), R)
        removed_positions.sort(reverse=True)  # Remove from back to front

        removed_customers = []
        for pos in removed_positions:
            removed_customers.append(route.pop(pos))

        # Reinsert randomly
        for customer in removed_customers:
            insert_pos = random.randint(1, len(route)-1)
            route.insert(insert_pos, customer)

        # Make time-feasible again
        new_solution = {
            'route': route,
            'charging_ops': {},
            'feasible': False,
            'total_distance': self._calculate_route_distance(route),
            'total_time': 0
        }

        # Add a maximum number of attempts to avoid infinite loops
        max_attempts = 1000
        attempts = 0

        while not self.is_time_feasible(new_solution) and attempts < max_attempts:
            new_solution = self.apply_local_search(
                new_solution, consider_delay=True)
            attempts += 1

        # If still not feasible after max attempts, return the original solution
        if not self.is_time_feasible(new_solution):
            return solution

        return new_solution

    # Main hybrid SA/TS algorithm.
    def hybrid_sa_ts(self, max_iter: int = 15000) -> Dict:
        # Initialization
        current_tsp, current_etsptw = self.initialize_solution_genetic()
        self.best_tsp_solution, self.best_etsptw_solution = current_tsp.copy(), current_etsptw.copy()

        temp = self.init_temp
        perturbation_counter = 0

        for iteration in range(max_iter):
            # Local search for TSPTW
            new_tsp = self.apply_local_search(current_tsp)

            # Update best TSP solution
            if new_tsp['total_distance'] < self.best_tsp_solution['total_distance']:
                self.best_tsp_solution = new_tsp.copy()

            # Evaluate for ETSPTW-MCR if conditions met
            if self.should_evaluate(new_tsp, self.best_etsptw_solution, self):
                new_etsptw = self.station_insertion(new_tsp)

                # Update best ETSPTW-MCR solution
                if new_etsptw['feasible'] and new_etsptw['total_distance'] < self.best_etsptw_solution['total_distance']:
                    self.best_etsptw_solution = new_etsptw.copy()

                # Add to tabu list
                solution_signature = self.get_solution_signature(new_tsp)
                self.tabu_list2.append(solution_signature)
                if len(self.tabu_list2) > self.tabu_tenure2:
                    self.tabu_list2.pop(0)

            # SA acceptance criterion
            delta = new_tsp['total_distance'] - current_tsp['total_distance']
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                current_tsp = new_tsp.copy()
            else:
                # With probability (1-omega), continue from best ETSPTW solution
                if random.random() > self.omega and self.best_etsptw_solution:
                    # Convert best ETSPTW solution back to TSPTW by removing charging stations
                    tsp_route = [
                        node for node in self.best_etsptw_solution['route'] if node == self.depot or node in self.nodes
                    ]
                    current_tsp = {
                        'route': tsp_route,
                        'charging_ops': {},
                        'feasible': False,
                        'total_distance': self._calculate_route_distance(tsp_route),
                        'total_time': 0
                    }

            # Perturbation
            perturbation_counter += 1
            if perturbation_counter >= self.perturbation_freq:
                current_tsp = self.perturb_solution(current_tsp)
                perturbation_counter = 0

            # Cooling schedule
            if iteration % self.cooling_length == 0:
                temp *= self.cooling_rate

        return self.best_etsptw_solution


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python etsptwmcr <filename>")
        sys.exit(1)

    input_filename = sys.argv[1]

    # Generate output filename by inserting '_solution' before the extension
    if '.' in input_filename:
        name_parts = input_filename.rsplit('.', 1)
        output_filename = f"{name_parts[0]}_solution.{name_parts[1]}"
    else:
        output_filename = f"{input_filename}_solution"

    problem = ETSPTW_MCR.from_file(input_filename)

    solution = problem.hybrid_sa_ts(max_iter=1000)
    problem.save_solution(solution, output_filename)
    print(f"Solution saved to file '{output_filename}'")
