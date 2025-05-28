# distutils: language=c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: profile=False
# cython: embedsignature=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef cnp.int32_t grid_cell_type_t
ctypedef cnp.uint8_t visited_cell_type_t
ctypedef cnp.int32_t coord_type_t
ctypedef cnp.uint32_t queue_index_t

DEF PATH_CELL = 0
DEF WALL_CELL = 1

# Estrutura para queue circular otimizada
cdef struct CircularQueue:
    coord_type_t* data_r
    coord_type_t* data_c
    queue_index_t head
    queue_index_t tail
    queue_index_t size
    queue_index_t capacity

cdef inline void init_queue(CircularQueue* q, queue_index_t capacity) nogil:
    """Inicializa a queue circular com capacidade especificada"""
    q.data_r = <coord_type_t*>malloc(capacity * sizeof(coord_type_t))
    q.data_c = <coord_type_t*>malloc(capacity * sizeof(coord_type_t))
    q.head = 0
    q.tail = 0
    q.size = 0
    q.capacity = capacity

cdef inline void free_queue(CircularQueue* q) nogil:
    """Libera a memória da queue"""
    if q.data_r != NULL:
        free(q.data_r)
    if q.data_c != NULL:
        free(q.data_c)

cdef inline bint is_queue_empty(CircularQueue* q) nogil:
    """Verifica se a queue está vazia"""
    return q.size == 0

cdef inline void enqueue(CircularQueue* q, coord_type_t r, coord_type_t c) nogil:
    """Adiciona elemento na queue (assume que há espaço)"""
    q.data_r[q.tail] = r
    q.data_c[q.tail] = c
    q.tail = (q.tail + 1) % q.capacity
    q.size += 1

cdef inline void dequeue(CircularQueue* q, coord_type_t* r, coord_type_t* c) nogil:
    """Remove elemento da queue (assume que não está vazia)"""
    r[0] = q.data_r[q.head]
    c[0] = q.data_c[q.head]
    q.head = (q.head + 1) % q.capacity
    q.size -= 1

# Direções pré-compiladas como constantes
DEF DR_UP = -1
DEF DC_UP = 0
DEF DR_DOWN = 1
DEF DC_DOWN = 0
DEF DR_LEFT = 0
DEF DC_LEFT = -1
DEF DR_RIGHT = 0
DEF DC_RIGHT = 1

cpdef list find_shortest_path_cython_optimized(cnp.int32_t[:, ::1] int_grid, tuple start_coords, tuple end_coords):
    """
    Versão otimizada do BFS para encontrar o caminho mais curto no labirinto.
    
    Otimizações implementadas:
    - Queue circular customizada em C para eliminar overhead do Python
    - Eliminação de alocações desnecessárias
    - Uso de nogil para paralelismo potencial
    - Estruturas de dados mais eficientes
    - Eliminação de checagens redundantes
    """
    cdef int rows = int_grid.shape[0]
    cdef int cols = int_grid.shape[1]
    
    cdef coord_type_t start_r = start_coords[0]
    cdef coord_type_t start_c = start_coords[1]
    cdef coord_type_t end_r = end_coords[0]
    cdef coord_type_t end_c = end_coords[1]

    # Validação rápida das coordenadas
    if (start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols or
        end_r < 0 or end_r >= rows or end_c < 0 or end_c >= cols):
        return None

    if int_grid[start_r, start_c] == WALL_CELL or int_grid[end_r, end_c] == WALL_CELL:
        return None

    # Caso especial: início igual ao fim
    if start_r == end_r and start_c == end_c:
        return [(start_r, start_c)]

    # Inicializar estruturas de dados
    cdef queue_index_t max_queue_size = rows * cols
    cdef CircularQueue queue
    init_queue(&queue, max_queue_size)

    # Arrays visitados e predecessores
    visited_np_array = np.zeros((rows, cols), dtype=np.uint8)
    cdef visited_cell_type_t[:, ::1] visited = visited_np_array

    pred_r_np = np.full((rows, cols), -1, dtype=np.int32)
    pred_c_np = np.full((rows, cols), -1, dtype=np.int32)
    cdef coord_type_t[:, ::1] pred_r = pred_r_np
    cdef coord_type_t[:, ::1] pred_c = pred_c_np

    # Inicializar BFS
    enqueue(&queue, start_r, start_c)
    visited[start_r, start_c] = 1

    cdef coord_type_t r, c, nr, nc
    cdef bint path_found = False

    # BFS principal com nogil para máxima performance
    with nogil:
        while not is_queue_empty(&queue):
            dequeue(&queue, &r, &c)
            
            # Verificar se chegamos ao destino
            if r == end_r and c == end_c:
                path_found = True
                break

            # Explorar direções (desenrolado para performance)
            # Direção UP
            nr = r + DR_UP
            nc = c + DC_UP
            if (nr >= 0 and nr < rows and nc >= 0 and nc < cols and
                int_grid[nr, nc] != WALL_CELL and not visited[nr, nc]):
                visited[nr, nc] = 1
                pred_r[nr, nc] = r
                pred_c[nr, nc] = c
                enqueue(&queue, nr, nc)

            # Direção DOWN
            nr = r + DR_DOWN
            nc = c + DC_DOWN
            if (nr >= 0 and nr < rows and nc >= 0 and nc < cols and
                int_grid[nr, nc] != WALL_CELL and not visited[nr, nc]):
                visited[nr, nc] = 1
                pred_r[nr, nc] = r
                pred_c[nr, nc] = c
                enqueue(&queue, nr, nc)

            # Direção LEFT
            nr = r + DR_LEFT
            nc = c + DC_LEFT
            if (nr >= 0 and nr < rows and nc >= 0 and nc < cols and
                int_grid[nr, nc] != WALL_CELL and not visited[nr, nc]):
                visited[nr, nc] = 1
                pred_r[nr, nc] = r
                pred_c[nr, nc] = c
                enqueue(&queue, nr, nc)

            # Direção RIGHT
            nr = r + DR_RIGHT
            nc = c + DC_RIGHT
            if (nr >= 0 and nr < rows and nc >= 0 and nc < cols and
                int_grid[nr, nc] != WALL_CELL and not visited[nr, nc]):
                visited[nr, nc] = 1
                pred_r[nr, nc] = r
                pred_c[nr, nc] = c
                enqueue(&queue, nr, nc)

    # Liberar memória da queue
    free_queue(&queue)

    if not path_found:
        return None

    # Reconstruir caminho otimizado
    cdef list path = []
    cdef coord_type_t curr_r = end_r
    cdef coord_type_t curr_c = end_c
    cdef coord_type_t prev_r, prev_c

    # Construir caminho de trás para frente
    while True:
        path.append((curr_r, curr_c))
        
        if curr_r == start_r and curr_c == start_c:
            break

        prev_r = pred_r[curr_r, curr_c]
        prev_c = pred_c[curr_r, curr_c]

        if prev_r == -1 and prev_c == -1:
            return None  # Erro na reconstrução

        curr_r = prev_r
        curr_c = prev_c

    # Reverter o caminho para ordem correta
    path.reverse()
    return path


def parse_maze_text(maze_text):
    """
    Converte texto do labirinto para formato de grid inteiro.
    
    Args:
        maze_text: string com o labirinto, linhas separadas por \n
        
    Returns:
        tuple: (int_grid, start_coords, end_coords) ou None se inválido
    """
    lines = maze_text.strip().split('\n')
    if not lines:
        return None
    
    rows = len(lines)
    cols = len(lines[0]) if lines else 0
    
    # Verificar se todas as linhas têm o mesmo tamanho
    for line in lines:
        if len(line) != cols:
            return None
    
    # Criar grid e encontrar S e E
    int_grid = np.zeros((rows, cols), dtype=np.int32)
    start_coords = None
    end_coords = None
    
    for r in range(rows):
        for c in range(cols):
            char = lines[r][c]
            if char == '#':
                int_grid[r, c] = WALL_CELL
            elif char == ' ' or char == '·':
                int_grid[r, c] = PATH_CELL
            elif char == 'S':
                int_grid[r, c] = PATH_CELL
                start_coords = (r, c)
            elif char == 'E':
                int_grid[r, c] = PATH_CELL
                end_coords = (r, c)
            else:
                int_grid[r, c] = PATH_CELL  # Assume qualquer outro char como caminho
    
    if start_coords is None or end_coords is None:
        return None
     