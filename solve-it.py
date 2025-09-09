import numpy as np
row = 11
col = 11
layer = 13
layer_size = row * col

maze = [
    [ # 1
        "00010000010",
        "01111111110",
        "01000000010",
        "01111111010",
        "01010000010",
        "11010000010",
        "00010000010",
        "01111111111",
        "01000101000",
        "01000101110",
        "00000100000",
    ],
    [ # 1b
        "11111111010",
        "11111111111",
        "01111111110",
        "11111111111",
        "01010111011",
        "11111111111",
        "11111111110",
        "11111111111",
        "11110101011",
        "11111111111",
        "11111101111",
    ],
    [ #2
        "00000101000",
        "11011101111",
        "01000101010",
        "11111101010",
        "00010001010",
        "11111111011",
        "00000001010",
        "00000001110",
        "00000001010",
        "11111111010",
        "00000001000",
    ],
    [
        "01111101011",
        "11111111111",
        "01111111011",
        "11111111111",
        "11110111111",
        "11111111111",
        "11011111110",
        "11111111111",
        "11111111111",
        "11111111111",
        "01011111111",
    ],
    [ # 3
        "00010101010",
        "11110111110",
        "01000101010",
        "01011111110",
        "01010001000",
        "01111111011",
        "01010100010",
        "01110101111",
        "01010101000",
        "01010101011",
        "01010000000",
    ],
    [
        "01010101010",
        "11111111111",
        "11111111111",
        "11111111111",
        "11011111111",
        "11111111111",
        "11111111111",
        "11111111111",
        "11011111110",
        "11111111111",
        "11111111111",
    ],
    [ # 4
        "01010001000",
        "01011111111",
        "01010100000",
        "01110100000",
        "00010100000",
        "11110100000",
        "01010100000",
        "11111111111",
        "00010100010",
        "01110101010",
        "00000101010",
    ],
    [
        "11111101110",
        "11111111111",
        "11010100000",
        "11111100000",
        "11111100000",
        "11111100000",
        "01010100000",
        "11111111111",
        "11110111111",
        "11111111111",
        "11111101010",
    ],
    [ # 5
        "01000101000",
        "01111111111",
        "01000100000",
        "01111101110",
        "01010001010",
        "11010101110",
        "01010100000",
        "01011111111",
        "01010000010",
        "01111111010",
        "00010001010",
    ],
    [
        "01010111110",
        "11111111111",
        "11111111111",
        "11111111111",
        "01111111011",
        "11111111111",
        "11110111111",
        "11111111111",
        "11111111110",
        "11111111111",
        "11110111011",
    ],
    [ # 6
        "00010001010",
        "01111101010",
        "01000101000",
        "11011111111",
        "01010101000",
        "01010101110",
        "01010101000",
        "11011101011",
        "01011101000",
        "01111101110",
        "00000101010",
    ],
    [
        "11111111111",
        "11111111111",
        "01110101111",
        "11111111111",
        "11011101011",
        "11111111111",
        "01111101111",
        "11111111111",
        "01111111111",
        "11111111111",
        "11111111111",
    ],
    [ # 7
        "11111111111",
        "11111111111",
        "01010100000",
        "11010111110",
        "01010001010",
        "01011111110",
        "01010000000",
        "01011111111",
        "01000000010",
        "11111111111",
        "11111111111",
    ],
]

way = np.zeros(row * col * layer, dtype=int)

maze_flat = []
for l in maze:
    for r in l:
        maze_flat.extend([int(c) if c == '1' else 0 for c in r])
maze = np.array(maze_flat, dtype=int)

def update_cell(n , value, map):
    if(way[n] != value - 1):
        return False
    update = False
    current_layer = n // layer_size
    current_row = (n % layer_size) // col
    current_col = n % col
    # Left
    if(current_col - 1 >= 0 and way[n - 1] == 0 and map[n-1] == 0):
        way[n - 1] = value
        update = True
    # Right
    if(current_col + 1 < col and way[n + 1] == 0 and map[n+1] == 0):
        way[n + 1] = value
        update = True
    # Up
    if(current_row - 1 >= 0 and way[n - col] == 0 and map[n-col] == 0):
        way[n - col] = value
        update = True
    # Down
    if(current_row + 1 < row and way[n + col] == 0 and map[n+col] == 0):
        way[n + col] = value
        update = True
    # Layer Up
    if(current_layer - 1 >= 0 and way[n - layer_size] == 0 and map[n - layer_size] == 0):
        way[n - layer_size] = value
        update = True
    # Layer Down
    if(current_layer + 1 < layer and way[n + layer_size] == 0 and map[n + layer_size] == 0):
        way[n + layer_size] = value
        update = True
    return update


# Gravity-based solver with combined directions
from collections import deque

# Custom gravity directions: all permutations and sign flips of (0,0,1), (2,1,0), (3,2,1)
from itertools import permutations, product

def all_signs(vec):
    # For a vector, return all sign combinations (excluding all zeros)
    signs = product(*[(-1,1) if v != 0 else (0,) for v in vec])
    return [tuple(s*v for s,v in zip(sign, vec)) for sign in signs if any(s*v != 0 for s,v in zip(sign, vec))]

base_dirs = [ (0,0,1), (2,1,0), (3,2,1) ]
gravity_dirs = set()
for base in base_dirs:
    for perm in set(permutations(base)):
        for v in all_signs(perm):
            gravity_dirs.add(v)
gravity_dirs = list(gravity_dirs)
print("Gravity directions:", gravity_dirs)

def in_bounds(x, y, z):
    return 0 <= x < col and 0 <= y < row and 0 <= z < layer and maze[idx(x, y, z)] == 0

def idx(x, y, z):
    return z * layer_size + y * col + x

def roll(x, y, z, dx, dy, dz, maze):
    # Move in (dx,dy,dz) until hitting a wall, edge, or a T/cliff
    while True:
        if (dx == 3 or dx == -3) and in_bounds(x + (1 if dx > 0 else -1), y, z):
            x += (1 if dx > 0 else -1)
        elif (dy == 3 or dy == -3) and in_bounds(x, y + (1 if dy > 0 else -1), z):
            y += (1 if dy > 0 else -1)
        elif (dz == 3 or dz == -3) and in_bounds(x, y, z + (1 if dz > 0 else -1)):
            z += (1 if dz > 0 else -1)
        elif (dx == 2 or dx == -2) and in_bounds(x + (1 if dx > 0 else -1), y, z):
            x += (1 if dx > 0 else -1)
        elif (dy == 2 or dy == -2) and in_bounds(x, y + (1 if dy > 0 else -1), z):
            y += (1 if dy > 0 else -1)
        elif (dz == 2 or dz == -2) and in_bounds(x, y, z + (1 if dz > 0 else -1)):
            z += (1 if dz > 0 else -1)
        elif (dx == 1 or dx == -1) and in_bounds(x + (1 if dx > 0 else -1), y, z):
            x += (1 if dx > 0 else -1)
        elif (dy == 1 or dy == -1) and in_bounds(x, y + (1 if dy > 0 else -1), z):
            y += (1 if dy > 0 else -1)
        elif (dz == 1 or dz == -1) and in_bounds(x, y, z + (1 if dz > 0 else -1)):
            z += (1 if dz > 0 else -1)
        else:
            break
    return x, y, z

def solve_with_gravity():
    start = (2, 2, 0)  # (x, y, z) from your way[0 * layer_size + 2 * col + 4]
    end = (8, 8, 12)   # (x, y, z) from way[12 * layer_size + 8 * col + 8]
    visited = set()
    queue = deque()
    queue.append( (start, []) )
    visited.add(start)
    while queue:
        (x, y, z), path = queue.popleft()
        if (x, y, z) == end:
            return path + [(x, y, z)]
        for dx, dy, dz in gravity_dirs:
            rx, ry, rz = roll(x, y, z, dx, dy, dz, maze)
            if (rx, ry, rz) != (x, y, z) and (rx, ry, rz) not in visited:
                print("touched:", (rx, ry, rz), len(path) + 1)
                visited.add((rx, ry, rz))
                queue.append( ((rx, ry, rz), path + [(x, y, z)]) )
                way[idx(rx, ry, rz)] = len(path) + 1
            # else:
            #     print("negtive", (rx, ry, rz), (x, y, z), (dx, dy, dz))
    return None

def print_gravity_solution(sol):
    if not sol:
        print("No solution found with gravity!")
        return
    print(f"Gravity solution path (length {len(sol)}):")
    for step in sol:
        print(f"  z={step[2]+1}, y={step[1]+1}, x={step[0]+1}")

gravity_solution = solve_with_gravity()
print_gravity_solution(gravity_solution)


def print_way():
    for l in range(layer):
        print(f"Layer {l+1}")
        for r in range(row):
            line = ""
            for c in range(col):
                v = way[idx(c, r, l)]
                line += f"{v:03} " if v != 0 else "--- " if maze[l*layer_size + r*col + c] == 1 else "XXX "
            print(line.rstrip())
        print()

print_way()
