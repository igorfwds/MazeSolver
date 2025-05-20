import time
import json
from collections import deque
from typing import List, Tuple, Optional

def parse_maze(labyrinth: str) -> Tuple[List[List[str]], Tuple[int, int], Tuple[int, int]]:
    grid = [list(line) for line in labyrinth.splitlines()]
    start = end = (-1, -1)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'S':
                start = (i, j)
            elif cell == 'E':
                end = (i, j)
    return grid, start, end

def bfs(grid: List[List[str]], start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    rows, cols = len(grid), len(grid[0])
    visited = set()
    prev = {}

    queue = deque([start])
    visited.add(start)
    prev[start] = None

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                if grid[nx][ny] != '#':  # aceita ' ', 'S', 'E', '.' etc.
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    prev[(nx, ny)] = (x, y)

    if end not in prev:
        return None

    # Reconstruir o caminho
    path = []
    at = end
    while at:
        path.append(at)
        at = prev.get(at)
    path.reverse()
    return path

def mark_path(grid: List[List[str]], path: List[Tuple[int, int]]) -> None:
    for x, y in path[1:-1]:  # não marcar S nem E
        grid[x][y] = '·'

def write_outputs(grid: List[List[str]], path: List[Tuple[int, int]]) -> None:
    with open('output.txt', 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')
    with open('output.json', 'w') as f:
        json.dump(path, f)

def solve_maze(labyrinth: str) -> float:
    start_time = time.time()

    grid, start, end = parse_maze(labyrinth)
    path = bfs(grid, start, end)

    if path is None:
        raise ValueError("Sem caminho possível de S até E.")

    mark_path(grid, path)
    write_outputs(grid, path)

    end_time = time.time()
    return (end_time - start_time) * 1000  # milissegundos

# Exemplo de uso (remova isso ao entregar)
if __name__ == "__main__":
    maze = (
        "#######\n"
        "#S #  #\n"
        "#  #E #\n"
        "#     #\n"
        "#######"
    )

    maze_2 = (
    "##########\n"
    "#S     #E#\n"
    "### #### #\n"
    "#   #    #\n"
    "# ###### #\n"
    "#        #\n"
    "##########"
    )

    complex_maze = (
    "########################################\n"
    "#S   #     #       #         #         #\n"
    "#### # ### # ##### # ####### # ####### #\n"
    "#    # #   #     # # #       # #     # #\n"
    "# #### # ### ### # # # ####### # ### # #\n"
    "#      #     #   # #   #           # # #\n"
    "####### ##### # # # ### ####### ### # # #\n"
    "#     #   #   # # #   # #           # #\n"
    "# ### ### # ### # ### # ########### ###\n"
    "#   #   # # #     #   # #         #   #\n"
    "### ### # # # ##### ### # ####### ### #\n"
    "#   # # # # #     #   # # #     #     #\n"
    "# ### # # # ##### ### # # # ### #######\n"
    "#     # # #     # #   # # # #         #\n"
    "##### # # ##### # ### # # # ######### #\n"
    "#     # #     # #   # # # #         # #\n"
    "# ##### ##### # ### # # # ######### # #\n"
    "#       #   # #     # # #         # # #\n"
    "####### # # # ##### # # ######### # # #\n"
    "#       # # #     # # # #         # # #\n"
    "# ####### # ##### # # # # ####### # # #\n"
    "#         #       #       #     #   #E#\n"
    "########################################"
    )

    

    

    duration = solve_maze(complex_maze)
    print(f"Resolvido em {duration:.2f}ms")
