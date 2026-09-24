# =============================
# core/spatial.py
# =============================
import random
from config import GRID_SIZE, GRID_DEPTH

class SpatialGrid:
    def __init__(self, n):
        self.positions = [
            (random.randint(0, GRID_SIZE-1),
             random.randint(0, GRID_SIZE-1),
             random.randint(0, GRID_DEPTH-1))
            for _ in range(n)
        ]

    def distance(self, i, j):
        x1,y1,z1 = self.positions[i]
        x2,y2,z2 = self.positions[j]
        return abs(x1-x2) + abs(y1-y2) + abs(z1-z2)