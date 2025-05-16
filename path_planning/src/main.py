import numpy as np
import cv2
from grid_processing import process_grid
from path_planning import a_star

def main():
    mock_grid = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    start = (0, 0)
    goal = (9, 9)

    processed_grid = process_grid(mock_grid)
    np.save('data/test_grid.npy', processed_grid)

    path = a_star(processed_grid, start, goal)
    if path:
        print("Path:", path)
        vis_grid = cv2.cvtColor((processed_grid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for x, y in path:
            vis_grid[x, y] = [0, 255, 0]  # Green path
        cv2.imwrite('data/path_output.png', vis_grid)
    else:
        print("No path found")

if __name__ == "__main__":
    main()