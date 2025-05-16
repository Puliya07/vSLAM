import numpy as np
import cv2

def process_grid(grid):
    """Turn the grid into 0s (free) and 1s (obstacles)."""
    grid = np.array(grid, dtype=np.uint8)
    binary_grid = (grid > 0).astype(np.uint8)  # Manual binarization
    print("Processed grid:\n", binary_grid)
    return binary_grid