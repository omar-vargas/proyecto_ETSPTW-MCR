#!/usr/bin/env python3
import sys
import random
import math
import re

# ------------------------------------
# 1) Definición de nodos y lectura
# ------------------------------------
class Node:
    def __init__(self, node_id, x, y, e, l, s, g):
        self.id         = node_id      # 0 = depósito; 1..|V| = cliente; |V|+1.. = estación pública
        self.x          = x
        self.y          = y
        self.e          = e            # ventana de tiempo inicio
        self.l          = l            # ventana de tiempo fin
        self.is_station = bool(s)      # True si hay estación (pública o privada)
        self.g          = g            # tasa de recarga

class ETSPTWProblem:
    def __init__(self, filename):
        self._read_problem(filename)

    def _read_problem(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        it = iter(lines)

        self.num_customers = int(next(it))    # |V|
        self.num_stations  = int(next(it))    # |F|
        self.Q             = float(next(it))  # batería Q
        self.h             = float(next(it))  # consumo h

        total = 1 + self.num_customers + self.num_stations
        self.nodes = []
        for _ in range(total):
            row = next(it)
            parts = re.split(r'\s+', row)
            if len(parts) != 7:
                raise ValueError(f"Esperaba 7 cols, encontré {len(parts)}: {parts}")
            nid,x,y,e,l,s,g = parts
            self.nodes.append(Node(int(nid), float(x), float(y),
                                   float(e), float(l), int(s), float(g)))

        # matriz distancia/tiempo
        self.dist = []
        for _ in range(total):
            row = next(it)
            parts = re.split(r'\s+', row)
            if len(parts) != total:
                raise ValueError(f"Fila de matriz con {len(parts)} cols, esperaba {total}")
            self.dist.append([float(d) for d in parts])

    def write_solution_file(self, route, cost, filename):
        """
        Escribe en:
        ----------------------------------------------------------------
        Total Distance
        <coste>
        Route
        D -> C# -> S# -> CS# -> ... -> D
        ----------------------------------------------------------------
        """
        V = self.num_customers
        labels = []
        for p in route:
            node = self.nodes[p]
            if node.id == 0:
                labels.append('D')
            elif node.is_station:
                if node.id > V:
                    labels.append(f'S{node.id - V}')
                else:
                    labels.append(f'CS{node.id}')
            else:
                labels.append(f'C{node.id}')

        with open(filename, 'w') as f:
            f.write('----------------------------------------------------------------\n')
            f.write('Total Distance\n')
            f.write(f'{cost}\n')
            f.write('Route\n')
            f.write(' -> '.join(labels) + '\n')
            f.write('----------------------------------------------------------------\n')


def hybrid_sa_ts(data,
                 max_iter=15000, T0=10000, alpha=0.95,
                 cooling_interval=100, perturb_interval=500,
                 R=3, tabu_moves=100):
    """
    data debe llevar:
      data['nodes'], data['distance_matrix'], data['battery_capacity'],
      data['energy_rate'], data['num_customers']
    Devuelve: best_route (lista de índices), best_cost, cost_prog, best_cost_prog
    """
    nodes = data['nodes']
    dist  = data['distance_matrix']
    Q     = data['battery_capacity']
    h     = data['energy_rate']
    N     = data['num_customers']

    # Genera solución inicial trivial (orden por l_i) y la hace factible en TW
    def initial_sol():
        cust = [n for n in nodes if not n.is_station and n.id!=0]
        perm = [0] + [n.id for n in sorted(cust, key=lambda x: x.l)] + [0]
        return perm

    def is_time_feasible(sol):
        t = 0
        for i in range(len(sol)-1):
            u,v = sol[i], sol[i+1]
            t += dist[u][v]
            t = max(t, nodes[v].e)
            if t>nodes[v].l+1e-6:
                return False
        return True

    def local_search(sol):
        if len(sol)<=4: return sol[:]
        a,b = random.sample(range(1,len(sol)-1),2)
        m = random.choice(['shift','2opt','swap'])
        s = sol[:]
        if m=='shift':
            x=s.pop(a); s.insert(b,x)
        elif m=='2opt':
            i,j = sorted([a,b])
            s = s[:i]+s[i:j+1][::-1]+s[j+1:]
        else:
            s[a],s[b]=s[b],s[a]
        return s

    # inicial
    sol = initial_sol()
    while not is_time_feasible(sol):
        sol = local_search(sol)

    best = sol[:]
    best_cost = sum(dist[sol[i]][sol[i+1]] for i in range(len(sol)-1))
    T = T0
    tabu = []
    cost_prog, best_prog = [], []

    for it in range(max_iter):
        cand = local_search(sol)
        if not is_time_feasible(cand): continue
        c = sum(dist[cand[i]][cand[i+1]] for i in range(len(cand)-1))
        move = tuple(sorted((cand[1],cand[-2])))
        if move in tabu and c>=best_cost:
            continue
        Δ = c-best_cost
        if Δ<0 or random.random()<math.exp(-Δ/T):
            sol = cand[:]
            if c<best_cost:
                best_cost=c
                best=cand[:]
        tabu.append(move)
        if len(tabu)>tabu_moves: tabu.pop(0)

        cost_prog.append(c)
        best_prog.append(best_cost)
        if it%cooling_interval==0:
            T*=alpha
        if it%perturb_interval==0:
            sol = local_search(sol)  # o tu perturbación

    return best, best_cost, cost_prog, best_prog


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("Uso: python et_sptw_solver.py <problem_file.txt> [n_runs]")
        sys.exit(1)

    prob_file = sys.argv[1]
    n_runs    = int(sys.argv[2]) if len(sys.argv)>2 else 10  # por defecto 10 ejecuciones

    problem = ETSPTWProblem(prob_file)
    data = {
        'nodes':           problem.nodes,
        'distance_matrix': problem.dist,
        'battery_capacity':problem.Q,
        'energy_rate':     problem.h,
        'num_customers':   problem.num_customers
    }

    best_overall = None
    best_cost    = float('inf')

    for run in range(1, n_runs+1):
        route, cost, _, _ = hybrid_sa_ts(data)
        print(f"[Run {run:>2}] coste = {cost:.2f}")
        if cost < best_cost:
            best_cost    = cost
            best_overall = route[:]

    print("\n>>> MEJOR DE TODAS LAS EJECUCIONES:")
    print("Coste:", best_cost)
    print("Ruta:", best_overall)

    problem.write_solution_file(best_overall, best_cost, 'solution.txt')
    print("Solución final guardada en solution.txt")
