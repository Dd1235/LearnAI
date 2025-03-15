Greedy BFS:

heuristic function: h(n)
this gives you an estimate of how far each node is "away" from the goal. So if you have a maze, you can assign each cell its manhattan distance as the 'estimate', but this might not account for the walls blocking the way. If at a dead end, trace back to the last choice, and continue on.

Now what if there are two cells, one is say 13 and the other 12, but the 13 is 'easier' to get to, then it might be better. how to incorporate this? Think of the cost to reach where you are. 
A* algorithm.

The key idea behind A* is to combine Dijkstra(that explores the whole world) with Greedy BFS, and get better performance than Dikjstra. 

`f(n) = g(n) + h(n)`

So at every point, you look at all the ways you can expand the set of vertices we have right now, look at edges that go from {S} to V\{S} and add in the node with smallest value for f(n)

A* gives you the optimal solution if:
- `h(n)` is admissible. That is h(n) should never overestimate the actual cost of getting to the goal. It can be lower, but not higher.
- `h(n)` is consistent. That is `h(n) <= h(n') + c` where n' is the successor and c is the cost of the step from n to n'

Look into other implementations based on A* that are optimized for specific usecase, A* uses a lot of memory so there are some that work on that and so on.

### 8 puzzle problem


Note: expections in N-puzzle, may not be solvable

Linear Search, Binary Search, DFS, BFS -> uninformed 

h-score -> number of misplaced tiles
update g-score on traversing

Move the empty space in all possible directions, calculate f-score for each state. Hence you are 'expanding' the current state. Push it into the closed list, and newly generated states are pushed into open list. Select one with least f-score and expand again.