Format of the problem files for the ETSPTW and ETSPTW-MCR:
----------------------------------------------------------------------------------------------
<number of customers (|V|)>
<number of charging stations (|F|)>
<battery capacity of the electric vehicle (Q)>
<energy consumption rate of the electric vehicle (h)>

<id>	<x-coordinate>	<y-coordinate>	<time windows start (e)>	<time windows end (l)>	<presence of charging station (s)>	<recharging rate (g)>

<Travelling distance/time matrix>
----------------------------------------------------------------------------------------------
The nodes are classified as follows:
0 represents the depot location.
1,...,|V| represent the customer locations.
|V|+1,...,|V|+|F| represent the public charging stations.

*Node |V|+|F| represents the charging station which is constructed at the depot node location.