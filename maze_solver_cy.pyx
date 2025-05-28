# distutils: language=c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import collections
import numpy as np
cimport numpy as cnp

ctypedef cnp.int32_t grid_cell_type_t
ctypedef cnp.uint8_t visited_cell_type_t
ctypedef cnp.int32_t coord_type_t # Para os arrays de predecessores

DEF PATH_CELL = 0
DEF WALL_CELL = 1
# START_CELL e END_CELL não são usados diretamente no int_grid do BFS
# se S e E são apenas coordenadas e o grid os marca como PATH_CELL.

cpdef list find_shortest_path_cython(cnp.int32_t[:, ::1] int_grid, tuple start_coords, tuple end_coords):
    """
    Encontra o caminho mais curto no labirinto (grid de inteiros) usando BFS
    otimizado com rastreamento de predecessores para labirintos grandes.
    """
    cdef int rows = int_grid.shape[0]
    cdef int cols = int_grid.shape[1]
    
    cdef coord_type_t start_r = start_coords[0]
    cdef coord_type_t start_c = start_coords[1]
    cdef coord_type_t end_r = end_coords[0]
    cdef coord_type_t end_c = end_coords[1]

    # Fila para o BFS: armazena apenas tuplas (linha, coluna)
    queue = collections.deque()
    queue.append((start_r, start_c))
    
    # Array 'visited'
    visited_np_array = np.zeros((rows, cols), dtype=np.uint8) # np.bool_ é uint8
    visited_np_array[start_r, start_c] = 1 # Marca o início como visitado
    cdef visited_cell_type_t[:, ::1] visited = visited_np_array

    # Arrays para armazenar predecessores (um para linhas, outro para colunas)
    # Inicializados com -1 para indicar que não há predecessor atribuído.
    # Usar np.full para inicialização clara.
    pred_r_np = np.full((rows, cols), -1, dtype=np.int32)
    pred_c_np = np.full((rows, cols), -1, dtype=np.int32)
    cdef coord_type_t[:, ::1] pred_r = pred_r_np
    cdef coord_type_t[:, ::1] pred_c = pred_c_np

    # Direções: Cima, Baixo, Esquerda, Direita
    cdef int directions[4][2]
    directions[0][0] = -1; directions[0][1] = 0  # Cima
    directions[1][0] = 1;  directions[1][1] = 0  # Baixo
    directions[2][0] = 0;  directions[2][1] = -1 # Esquerda
    directions[3][0] = 0;  directions[3][1] = 1  # Direita
    
    cdef coord_type_t r, c, nr, nc # Tipagem para coordenadas
    cdef tuple current_pos_tuple
    cdef int i # Contador de direções
    cdef bint path_found = False # Flag Cython booleana

    while queue:
        current_pos_tuple = queue.popleft()
        r = current_pos_tuple[0]
        c = current_pos_tuple[1]
        
        if r == end_r and c == end_c:
            path_found = True
            break # Destino alcançado

        for i in range(4):
            dr = directions[i][0]
            dc = directions[i][1]
            
            nr = r + dr
            nc = c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols: # Checagem de limites
                if int_grid[nr, nc] != WALL_CELL and not visited[nr, nc]:
                    visited[nr, nc] = 1 # Marca como visitado
                    pred_r[nr, nc] = r  # Armazena predecessor
                    pred_c[nr, nc] = c
                    queue.append((nr, nc))
                    
    if not path_found:
        return None # Nenhum caminho encontrado

    # Reconstruir o caminho de trás para frente usando predecessores
    # Usar collections.deque para appendleft eficiente
    path_deque = collections.deque()
    cdef coord_type_t curr_r = end_r
    cdef coord_type_t curr_c = end_c
    cdef coord_type_t prev_r, prev_c

    # Loop enquanto não chegamos ao início (que não tem predecessor marcado ou é -1)
    # ou se a célula atual não tem predecessor (indicando erro ou início)
    while True:
        path_deque.appendleft((curr_r, curr_c))
        if curr_r == start_r and curr_c == start_c:
            break # Chegamos ao início

        prev_r = pred_r[curr_r, curr_c]
        prev_c = pred_c[curr_r, curr_c]

        if prev_r == -1 and prev_c == -1: 
            # Se chegamos aqui e não é o start_node, algo está errado (caminho quebrado)
            # Isso não deveria acontecer se 'S' é alcançável e a lógica está correta.
            # Para um 'S' válido, o loop quebraria na condição anterior.
            # Retornar None indica que a reconstrução falhou, embora o BFS tenha encontrado 'E'.
            return None 

        curr_r = prev_r
        curr_c = prev_c
            
    return list(path_deque) # Converte o deque para uma lista Python