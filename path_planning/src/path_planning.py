import heapq

def a_star(grid, start, goal):
    """Find shortest path from start to goal using A*."""
    rows, cols = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            print("Visited cells:", list(came_from.keys()))
            return path[::-1]

        for dx, dy in neighbors:
            neighbor_x, neighbor_y = current[0] + dx, current[1] + dy
            neighbor = (neighbor_x, neighbor_y)
            if (0 <= neighbor_x < rows and 0 <= neighbor_y < cols and 
                grid[neighbor_x, neighbor_y] == 0):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None