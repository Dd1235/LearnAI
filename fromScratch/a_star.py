import heapq

GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


# manhattan distance heuristic, not the number of misplaced tiles
def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x = (val - 1) // 3
                goal_y = (val - 1) % 3
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance


def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


def neighbors(state):
    moves = []
    r, c = find_zero(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = [row[:] for row in state]
            # replace zero with its neighbour
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            moves.append(new_state)
    # return all possible moves
    return moves


def is_goal(state):
    return state == GOAL


def astar(start):
    visited = set()
    pq = []
    heapq.heappush(pq, (manhattan_distance(start), 0, start, []))

    while pq:
        f, g, state, path = heapq.heappop(pq)
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        if is_goal(state):
            return path + [state]

        for neighbor in neighbors(state):
            heapq.heappush(
                pq,
                (g + 1 + manhattan_distance(neighbor), g + 1, neighbor, path + [state]),
            )

    return None


# Demo
start_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]

path = astar(start_state)

if path:
    for step in path:
        for row in step:
            print(row)
        print("------")
else:
    print("No solution.")
