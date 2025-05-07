from __future__ import annotations

import math
import random
import sys
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

###############################################################################
# Data structures
###############################################################################

@dataclass(slots=True)
class Node:
    id: int
    x: float
    y: float
    e: float  # earliest
    l: float  # latest
    service: float  # service time (0 for stations)
    private_charger: bool = False
    is_station: bool = False

    def dist(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Problem:
    battery_capacity: float
    consumption_rate: float
    private_rate: float
    public_rate: float
    depot: Node
    customers: List[Node]
    stations: List[Node]

    @property
    def all_nodes(self) -> List[Node]:
        return [self.depot, *self.customers, *self.stations]


###############################################################################
# Parsing
###############################################################################

def parse_instance(path: str | Path) -> Problem:
    """Parse an instance file in the custom format."""
    section = None
    params: Dict[str, float] = {}
    depot: Optional[Node] = None
    customers: List[Node] = []
    stations: List[Node] = []

    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line
                continue
            if section == "[PROBLEM]" and "=" in line:
                k, v = [t.strip() for t in line.split("=", 1)]
                params[k] = float(v)
            elif section == "[NODES]":
                parts = line.split()
                if len(parts) < 7:
                    raise ValueError("Invalid node line: " + line)
                nid, x, y, e, l, srv, priv = parts[:7]
                n = Node(int(nid), float(x), float(y), float(e), float(l), float(srv), bool(int(priv)))
                if n.id == 0:
                    depot = n
                else:
                    customers.append(n)
            elif section == "[CHARGING_STATIONS]":
                parts = line.split()
                if len(parts) < 3:
                    continue
                sid, x, y = parts[:3]
                s = Node(int(sid), float(x), float(y), 0, float("inf"), 0, False, True)
                stations.append(s)

    if depot is None:
        raise ValueError("Depot not defined (node ID 0)")

    return Problem(
        battery_capacity=params["BATTERY_CAPACITY"],
        consumption_rate=params["CONSUMPTION_RATE"],
        private_rate=params["PRIVATE_CHARGE_RATE"],
        public_rate=params["PUBLIC_CHARGE_RATE"],
        depot=depot,
        customers=customers,
        stations=stations,
    )


###############################################################################
# Utilities
###############################################################################

def travel_time(p: Problem, i: Node, j: Node) -> float:
    return i.dist(j)


def energy_required(p: Problem, i: Node, j: Node) -> float:
    return p.consumption_rate * i.dist(j)


###############################################################################
# Solution representations
###############################################################################

@dataclass
class Route:
    order: List[Node]
    distance: float = math.inf
    slack: float = 0.0


@dataclass
class ETSolution:
    route: List[Node]
    total_distance: float
    total_time: float
    feasible: bool
    charging_ops: List[Tuple[int, float, float, str]]


###############################################################################
# Hybrid SA/TS solver
###############################################################################

class HybridSATS:
    C = 0.95
    CL = 100
    TL1 = 400
    TL2 = 75
    MAX_ITER = 1500
    PERTURB = 500
    R = 30
    OMEGA = 0.7
    INIT_ITER = 100
    T0 = 10000.0

    def __init__(self, problem: Problem):
        self.p = problem
        self.rng = random.Random(int(os.getenv("ETSPTWMCR_SEED", 42)))
        self.move_tabu: deque[Tuple[str, int, int]] = deque()
        self.route_tabu: deque[Tuple[int, int, int, int]] = deque()

    # Initial solution: customers sorted by latest deadline
    def _initial_solution(self) -> Route:
        cust_sorted = sorted(self.p.customers, key=lambda n: n.l)
        r = Route(order=cust_sorted)
        self._make_tw_feasible(r)
        return r

    # Greedy repair for time‑window feasibility
    def _make_tw_feasible(self, r: Route) -> None:
        changed = True
        while changed:
            times = self._arrival_times(r.order)
            violations = [i for i, t in enumerate(times) if t > r.order[i].l]
            if not violations:
                return
            changed = False
            for idx in violations:
                if idx > 0:
                    r.order[idx - 1], r.order[idx] = r.order[idx], r.order[idx - 1]
                    changed = True
                    break

    # Local search (shift, swap, 2‑opt)
    def _local_search(self, r: Route, iter_no: int) -> Route:
        best = list(r.order)
        best_dist = self._route_distance(best)
        improved = True
        while improved:
            improved = False
            move_type = self.rng.choice(["shift", "two_opt", "swap"])
            if move_type == "shift":
                for i in range(len(best)):
                    for j in range(len(best)):
                        if i == j:
                            continue
                        new = best.copy()
                        node = new.pop(i)
                        new.insert(j, node)
                        if self._feasible_tw(new):
                            dist = self._route_distance(new)
                            if dist < best_dist:
                                best, best_dist, improved = new, dist, True
            elif move_type == "swap":
                for i in range(len(best) - 1):
                    for j in range(i + 1, len(best)):
                        new = best.copy()
                        new[i], new[j] = new[j], new[i]
                        if self._feasible_tw(new):
                            dist = self._route_distance(new)
                            if dist < best_dist:
                                best, best_dist, improved = new, dist, True
            else:  # 2‑opt
                n = len(best)
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        new = best[:i] + best[i:j][::-1] + best[j:]
                        if self._feasible_tw(new):
                            dist = self._route_distance(new)
                            if dist < best_dist:
                                best, best_dist, improved = new, dist, True
        return Route(order=best, distance=best_dist, slack=self._slack(best))

    def _station_insertion(self, r: Route) -> ETSolution:
        return StationInsertion(self.p, self.rng).run(r)

    # Helpers
    def _route_distance(self, cust_order: List[Node]) -> float:
        dist = self.p.depot.dist(cust_order[0])
        for a, b in zip(cust_order, cust_order[1:]):
            dist += a.dist(b)
        dist += cust_order[-1].dist(self.p.depot)
        return dist

    def _arrival_times(self, cust_order: List[Node]) -> List[float]:
        t = self.p.depot.e
        times: List[float] = []
        prev = self.p.depot
        for n in cust_order:
            t += travel_time(self.p, prev, n)
            t = max(t, n.e)
            times.append(t)
            t += n.service
            prev = n
        return times

    def _feasible_tw(self, cust_order: List[Node]) -> bool:
        prev = self.p.depot
        t = self.p.depot.e
        for n in cust_order:
            t += travel_time(self.p, prev, n)
            if t > n.l:
                return False
            t = max(t, n.e) + n.service
            prev = n
        t += travel_time(self.p, prev, self.p.depot)
        return t <= self.p.depot.l

    def _slack(self, cust_order: List[Node]) -> float:
        times = self._arrival_times(cust_order)
        return sum(n.l - t for n, t in zip(cust_order, times))

    # Main loop
    def solve(self) -> ETSolution:
        current = self._initial_solution()
        best_et: ETSolution = self._station_insertion(current)
        T = self.T0
        stagnant = 0
        for it in range(1, self.MAX_ITER + 1):
            candidate = self._local_search(current, it)
            et_candidate = None
            if candidate.distance < best_et.total_distance:
                sig = self._route_signature(candidate.order)
                if sig not in self.route_tabu:
                    et_candidate = self._station_insertion(candidate)
                    self.route_tabu.append(sig)
                    if len(self.route_tabu) > self.TL2:
                        self.route_tabu.popleft()
                    if et_candidate.feasible and et_candidate.total_distance < best_et.total_distance:
                        best_et = et_candidate
                        stagnant = 0
            accept = candidate.distance < current.distance or math.exp((current.distance - candidate.distance) / T) > self.rng.random()
            if accept:
                current = candidate
            else:
                if self.rng.random() < self.OMEGA:
                    current = Route(order=[n for n in best_et.route if not n.is_station and n.id != 0],
                                     distance=best_et.total_distance,
                                     slack=0)
            stagnant += 1
            if stagnant >= self.PERTURB:
                self.rng.shuffle(current.order)
                self._make_tw_feasible(current)
                stagnant = 0
            if it % self.CL == 0:
                T *= self.C
        return best_et

    def _route_signature(self, cust_order: List[Node]) -> Tuple[int, int, int, int]:
        first, last = cust_order[0].id, cust_order[-1].id
        times = self._arrival_times(cust_order)
        return (first, last, int(times[0]), int(times[-1]))


###############################################################################
# Station insertion (DP)
###############################################################################

class StationInsertion:
    def __init__(self, p: Problem, rng: random.Random):
        self.p = p
        self.rng = rng
        self.Q = p.battery_capacity

    def run(self, r: Route) -> ETSolution:
        custs = r.order
        sequence = [self.p.depot, *custs, self.p.depot]
        labels: Dict[int, List[Tuple[float, float, float, List[Node], List[Tuple[int, float, float, str]]]]] = defaultdict(list)
        labels[0].append((self.p.depot.e, self.Q, 0.0, [self.p.depot], []))
        UB = float("inf")
        for i in range(1, len(sequence)):
            prev = sequence[i - 1]
            curr = sequence[i]
            prev_lbls = labels[i - 1]
            for t, q, c, path, ops in prev_lbls:
                need = energy_required(self.p, prev, curr)
                if q >= need:
                    nt = max(t + travel_time(self.p, prev, curr), curr.e)
                    if nt <= curr.l:
                        nq = q - need
                        nc = c + prev.dist(curr)
                        if nc < UB:
                            labels[i].append((nt + curr.service, self.Q if curr.is_station or curr.private_charger else nq, nc, path + [curr], ops + self._maybe_charge(curr, nq, nt)))
                            if i == len(sequence) - 1 and nc < UB:
                                UB = nc
                for s in self.p.stations:
                    need1 = energy_required(self.p, prev, s)
                    need2 = energy_required(self.p, s, curr)
                    if q >= need1 and self.Q >= need2:
                        arrival_s = t + travel_time(self.p, prev, s)
                        charge = self.Q - (q - need1)
                        time_charge = charge / self.p.public_rate
                        depart_s = arrival_s + time_charge
                        arrival_c = max(depart_s + travel_time(self.p, s, curr), curr.e)
                        if arrival_c <= curr.l:
                            nc = c + prev.dist(s) + s.dist(curr)
                            if nc < UB:
                                labels[i].append((arrival_c + curr.service, self.Q if curr.is_station or curr.private_charger else self.Q - need2, nc, path + [s, curr], ops + [(s.id, charge, time_charge, "public")] + self._maybe_charge(curr, self.Q - need2, arrival_c)))
                                if i == len(sequence) - 1 and nc < UB:
                                    UB = nc
            pruned: List[Tuple] = []
            for lbl in labels[i]:
                dominated = False
                for ol in labels[i]:
                    if ol is lbl:
                        continue
                    if ol[0] <= lbl[0] and ol[2] <= lbl[2]:
                        dominated = True
                        break
                if not dominated:
                    pruned.append(lbl)
            labels[i] = pruned
            if not labels[i]:
                return ETSolution(route=[], total_distance=math.inf, total_time=math.inf, feasible=False, charging_ops=[])
        best = min(labels[len(sequence) - 1], key=lambda x: x[2])
        return ETSolution(route=best[3], total_distance=best[2], total_time=best[0], feasible=True, charging_ops=best[4])

    def _maybe_charge(self, node: Node, q_after: float, arrival: float):
        if node.private_charger:
            charge = self.Q - q_after
            time_needed = charge / self.p.private_rate
            return [(node.id, charge, time_needed, "private")]
        elif node.is_station:
            return []
        else:
            return []


###############################################################################
# CLI
###############################################################################

import os

def _format_solution(sol: ETSolution) -> str:
    lines = [
        f"Total distance: {sol.total_distance}",
        f"Total time: {sol.total_time}",
        f"Feasible: {sol.feasible}",
        "",
        "Route:",
    ]
    lines.extend(f"{n.id} ({n.x}, {n.y})" for n in sol.route)
    lines.append("\nCharging operations:")
    for nid, chg, t, typ in sol.charging_ops:
        lines.append(f"Node {nid}: Charge {chg:.2f} units in {t:.2f} time, {typ}")
    return "\n".join(lines)


def main(argv: List[str]):
    if len(argv) != 3:
        print("Usage: python etsptwmcr.py instance.txt solution.txt")
        return
    prob = parse_instance(argv[1])
    solver = HybridSATS(prob)
    start = time.perf_counter()
    sol = solver.solve()
    elapsed = time.perf_counter() - start
    print(f"Solved in {elapsed:.2f}s – distance {sol.total_distance}")
    Path(argv[2]).write_text(_format_solution(sol))


if __name__ == "__main__":
    main(sys.argv)
