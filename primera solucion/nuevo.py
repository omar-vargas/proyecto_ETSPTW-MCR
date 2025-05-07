import math
import networkx as nx
from networkx.algorithms import approximation as approx

class ETSPTW_MCR:
    def __init__(self, filename):
        # Lee el fichero
        with open(filename) as f:
            nc = int(f.readline())
            ns = int(f.readline())
            self.Q = float(f.readline())
            self.energy_rate = float(f.readline())
            total = 1 + nc + ns + 1
            self.nodes = []
            for _ in range(total):
                parts = f.readline().split()
                self.nodes.append({
                    'id':int(parts[0]),
                    'x':float(parts[1]),
                    'y':float(parts[2]),
                    'e':float(parts[3]),
                    'l':float(parts[4]),
                    'is_station':int(parts[5]),
                    'r':float(parts[6])
                })
            # lee matriz distancia
            self.d = [list(map(float,f.readline().split())) for _ in range(total)]
        # indexación de nodos
        self.depot = 0
        self.customers = [n['id'] for n in self.nodes if n['id'] not in (0, total-1) and n['is_station']==0]
        self.stations  = [n['id'] for n in self.nodes if n['is_station']==1]
        # shortcuts
        self.e = {n['id']:n['e'] for n in self.nodes}
        self.l = {n['id']:n['l'] for n in self.nodes}

    def _penalty(self,i,j,lam):
        # tardanza si llego a j justo saliendo a earliest desde i
        tard = max(0.0, self.e[i] + self.d[i][j] - self.l[j])
        return lam[j] * tard

    def christofides_relaxed(self, lam):
        # construye grafo métrico completo
        G = nx.Graph()
        all_nodes = [self.depot] + self.customers
        for u in all_nodes:
            G.add_node(u)
        for i in all_nodes:
            for j in all_nodes:
                if i<j:
                    w = self.d[i][j] + self._penalty(i,j,lam)
                    G.add_edge(i,j, weight=w)
        # Christofides approximation
        tour = approx.traveling_salesman_problem(G, weight='weight', cycle=True, method='christofides')
        # aseguramos que empiece/termine en depot
        if tour[0]!=self.depot:
            idx = tour.index(self.depot)
            tour = tour[idx:] + tour[1:idx+1]
        return tour

    def eval_tour(self, tour):
        """Simula horarios, calcula distancia total real y tardanzas por cliente"""
        t = 0.0
        dist = 0.0
        tard = {i:0 for i in tour}
        for u,v in zip(tour, tour[1:]):
            dist += self.d[u][v]
            t += self.d[u][v]
            if t < self.e[v]: t = self.e[v]
            if t > self.l[v]:
                tard[v] = t - self.l[v]
        return dist, tard

    def update_lambdas(self, lam, tard, step):
        for i in lam:
            lam[i] = max(0.0, lam[i] + step * tard.get(i,0.0))

    def greedy_insert_stations(self, tour):
        """Repara el tour base con recargas mínimas: 
           inserta la estación más cercana cuando la batería no llega."""
        newt = [tour[0]]
        battery = self.Q
        for nxt in tour[1:]:
            cur = newt[-1]
            dcur = self.d[cur][nxt]
            if battery >= dcur:
                newt.append(nxt)
                battery -= dcur
            else:
                # elige la estación que minimiza desvío cur->k->nxt
                best = None
                best_cost = float('inf')
                for k in self.stations:
                    c = self.d[cur][k] + self.d[k][nxt]
                    if c<best_cost:
                        best_cost, best = c, k
                # inserta k
                newt.append(best)
                # recarga full
                battery = self.Q
                # avanza a nxt
                newt.append(nxt)
                battery -= self.d[best][nxt]
        # cerrar en depot
        if newt[-1]!=self.depot:
            # recarga si hace falta
            last = newt[-1]
            if battery < self.d[last][self.depot]:
                # en vez de buscar, recargo en la última estación disponible
                newt.append(self.stations[0])
            newt.append(self.depot)
        return newt

    def solve(self, max_iter=50):
        # inicializa λᵢ=0 para cada cliente
        lam = {i:0.0 for i in self.customers}
        best_feasible = None
        best_dist = float('inf')

        for it in range(1, max_iter+1):
            # 1) Christofides sobre grafo penalizado
            tour0 = self.christofides_relaxed(lam)
            # 2) Eval real (sin recargas)
            dist0, tard = self.eval_tour(tour0)
            # 3) Si es factible (sin tardanzas) lo guardo
            if all(t<=1e-6 for t in tard.values()) and dist0<best_dist:
                best_dist, best_feasible = dist0, tour0[:]

            # 4) paso subgradiente simple
            s2 = sum(v*v for v in tard.values())
            if s2>0:
                # step = α * (dist0) / s2
                step = 1.0 * dist0 / s2
                self.update_lambdas(lam, tard, step)

        # si encontré solución factible (sin tardanzas), uso esa; si no, uso última
        base = best_feasible or tour0
        # 5) inserto estaciones
        full_route = self.greedy_insert_stations(base)
        # 6) recalculo distancia total con recargas
        total = sum(self.d[u][v] for u,v in zip(full_route, full_route[1:]))
        return full_route, total


if __name__ == "__main__":
    prob = ETSPTW_MCR("n20w120s10_1.txt")
    route, cost = prob.solve(max_iter=100)
    print("Ruta final con recargas:", route)
    print("Costo total (distancia):", cost)
