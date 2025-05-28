import numpy as np
import maze_solver_cy # Importa o módulo Cython compilado
import time
import traceback # Para imprimir a pilha de erros em exceções inesperadas

# Mapeamento de caracteres para inteiros (consistente com .pyx e suas informações salvas)
CHAR_TO_INT = {
    ' ': 0,  # Caminho livre
    '#': 1,  # Parede
    'S': 2,  # Início (tratado como caminho livre no int_grid para BFS)
    'E': 3   # Fim (tratado como caminho livre no int_grid para BFS)
}
PATH_CELL_INT = CHAR_TO_INT[' ']
WALL_CELL_INT = CHAR_TO_INT['#']

# --- Funções Auxiliares (mantidas da versão anterior) ---

def parse_maze_for_cython(maze_str: str):
    """
    Analisa a string do labirinto e retorna:
    1. char_grid (list de list de str): O labirinto original para desenho.
    2. int_grid_np (np.ndarray int32): O labirinto como inteiros para Cython.
    3. start_coords (tuple): Coordenadas (linha, coluna) de 'S'.
    4. end_coords (tuple): Coordenadas (linha, coluna) de 'E'.
    Levanta ValueError se o labirinto for malformado.
    """
    if not maze_str.strip():
        raise ValueError("O labirinto fornecido (string) está vazio ou contém apenas espaços em branco.")
    lines = maze_str.strip().split('\n')
    if not lines:
        raise ValueError("O labirinto (string) não contém linhas.")
    rows = len(lines)
    cols = len(lines[0])
    char_grid = [['' for _ in range(cols)] for _ in range(rows)]
    int_grid_np = np.empty((rows, cols), dtype=np.int32)
    start_coords = None
    end_coords = None
    for r, line in enumerate(lines):
        if len(line) != cols:
            raise ValueError(f"Linha {r+1} tem comprimento inconsistente. Esperado: {cols}, Obtido: {len(line)}.")
        for c, char_val in enumerate(line):
            char_grid[r][c] = char_val
            if char_val == 'S':
                start_coords = (r, c)
                int_grid_np[r, c] = PATH_CELL_INT
            elif char_val == 'E':
                end_coords = (r, c)
                int_grid_np[r, c] = PATH_CELL_INT
            elif char_val == '#':
                int_grid_np[r, c] = WALL_CELL_INT
            elif char_val == ' ':
                int_grid_np[r, c] = PATH_CELL_INT
            else:
                raise ValueError(f"Caractere inválido '{char_val}' no labirinto em ({r},{c}). Use apenas 'S', 'E', '#', ' '.")
    if start_coords is None:
        raise ValueError("Ponto de início 'S' não encontrado no labirinto.")
    if end_coords is None:
        raise ValueError("Ponto de chegada 'E' não encontrado no labirinto.")
    return char_grid, int_grid_np, start_coords, end_coords

def draw_path_on_char_grid(char_grid: list, path: list) -> list:
    grid_with_path = [row[:] for row in char_grid]
    if path:
        for r, c in path:
            if grid_with_path[r][c] != 'S' and grid_with_path[r][c] != 'E':
                grid_with_path[r][c] = '·' # Seu caractere para caminho percorrido
    return grid_with_path

def maze_to_string(grid: list) -> str:
    return "\n".join("".join(row) for row in grid)

# --- Nova Função Principal Conforme Interface Esperada ---
def solve_maze(labyrinth: str) -> float:
    """
    Recebe um labirinto como string (com \n separando as linhas)
    e retorna o tempo total em milissegundos para resolvê-lo.
    Você deve gerar um arquivo chamado output.txt para que ele possa ser
    auditado.
    """
    output_filename_fixed = "output_capivaras.txt" # Nome fixo conforme a interface
    
    # Inicia a contagem de tempo ANTES de qualquer processamento do labirinto
    overall_start_time = time.perf_counter()
    execution_time_ms = 0.0 # Valor padrão para tempo em caso de erro muito inicial

    try:
        # 1. Parse o labirinto (pode levantar ValueError)
        char_grid, int_grid_np, start_coords, end_coords = parse_maze_for_cython(labyrinth)
        
        # Garante que o array NumPy é C-contíguo para o Cython
        if not int_grid_np.flags['C_CONTIGUOUS']:
            int_grid_np = np.ascontiguousarray(int_grid_np, dtype=np.int32)

        # 2. Resolva usando Cython (esta é a parte principal da "resolução")
        path = maze_solver_cy.find_shortest_path_cython_optimized(int_grid_np, start_coords, end_coords)
        
        # Finaliza a contagem de tempo APÓS a parte principal da resolução
        overall_end_time = time.perf_counter()
        execution_time_ms = (overall_end_time - overall_start_time) * 1000
        
        # 3. Prepare a string da solução para o arquivo de auditoria
        solution_str_for_file = ""
        if path:
            grid_with_solution = draw_path_on_char_grid(char_grid, path)
            solution_str_for_file = maze_to_string(grid_with_solution)
            # Adiciona o tempo ao final do arquivo de solução, se desejado (opcional, não na interface)
            # solution_str_for_file += f"\n\nTempo de resolução: {execution_time_ms:.4f} ms"
        else:
            solution_str_for_file = "Nenhum caminho encontrado no labirinto (Cython).\n"
            solution_str_for_file += f"(Tempo de processamento: {execution_time_ms:.4f} ms)\n\n"
            solution_str_for_file += labyrinth # Inclui o labirinto original no output para auditoria

        # 4. Escreva no arquivo output.txt
        with open(output_filename_fixed, "w", encoding="utf-8") as f:
            f.write(solution_str_for_file)
            
        return execution_time_ms

    except ValueError as e: # Erros de parsing do labirinto
        overall_end_time = time.perf_counter() # Tempo até o erro
        execution_time_ms = (overall_end_time - overall_start_time) * 1000
        error_message = f"Erro ao processar o labirinto: {e}\n"
        error_message += f"Tempo decorrido até o erro: {execution_time_ms:.4f} ms\n\n"
        error_message += "Labirinto fornecido:\n" + labyrinth
        try:
            with open(output_filename_fixed, "w", encoding="utf-8") as f:
                f.write(error_message)
        except Exception as fe:
            print(f"Erro crítico: Não foi possível escrever a mensagem de erro em '{output_filename_fixed}': {fe}")
        return execution_time_ms # Retorna o tempo gasto até o erro

    except ImportError: # Módulo Cython não encontrado
        overall_end_time = time.perf_counter()
        execution_time_ms = (overall_end_time - overall_start_time) * 1000
        error_message = "Erro: O módulo Cython 'maze_solver_cy' não foi encontrado.\n"
        error_message += "Certifique-se de que compilou o arquivo .pyx (python setup.py build_ext --inplace).\n"
        error_message += f"Tempo decorrido até o erro: {execution_time_ms:.4f} ms"
        try:
            with open(output_filename_fixed, "w", encoding="utf-8") as f:
                f.write(error_message)
        except Exception as fe:
            print(f"Erro crítico: Não foi possível escrever a mensagem de erro em '{output_filename_fixed}': {fe}")
        return execution_time_ms

    except Exception as e: # Outras exceções inesperadas
        overall_end_time = time.perf_counter()
        execution_time_ms = (overall_end_time - overall_start_time) * 1000
        print("Ocorreu um erro inesperado. Detalhes abaixo e no 'output.txt'.")
        traceback.print_exc() # Imprime o traceback completo no console para depuração
        
        error_message = f"Um erro inesperado ocorreu: {type(e).__name__} - {e}\n"
        error_message += f"Tempo decorrido até o erro: {execution_time_ms:.4f} ms\n\n"
        error_message += "Traceback (resumo):\n" + traceback.format_exc(limit=5) + "\n\n"
        error_message += "Labirinto fornecido:\n" + labyrinth
        try:
            with open(output_filename_fixed, "w", encoding="utf-8") as f:
                f.write(error_message)
        except Exception as fe:
            print(f"Erro crítico: Não foi possível escrever a mensagem de erro em '{output_filename_fixed}': {fe}")
        return execution_time_ms


# --- Bloco Principal para Execução e Teste ---
if __name__ == "__main__":
    # Lembre-se de compilar o Cython primeiro!
    # No terminal, na pasta dos arquivos: python setup.py build_ext --inplace

    # Exemplos de conteúdo para os arquivos .txt (crie-os na mesma pasta do script):
    labirinto1 = """
###############
#S     #     E#
# ### ##### ###
# #       #   #
# ##### # ### #
#     # #   # #
### # ##### # #
#   #     #   #
# ### ### ### #
# #   #   #   #
# # ### ##### #
# #   #     # #
# ### # ### # #
#     #     # #
###############
""".strip()
    
    labirinto2 = """
##########################################################################################################################################################################################################################################################
#      # ###    #       # # ####   ##  ##            #     ##  ##    ###  # # # # ### #        # #    #  ## #   # #  # #            #   # #   #  # ###   ## ##   ## #    #  ##   ##  ## #         ### #         ##   ##    # # # #   ###    #     ###    #
## ###    #   # # ## #         # #      #   #   # ##  ##   ####    # # #  # ##    #     #     # # ##    #  # #  #  #  # #    #  ##  #  # ###     #   # # # #  #        # ##    #  # #   # # # ###      #    # #    #    #  #  #   ###   #   ####  #   ####
# ##    #  # ## ### #              # #   ##   ##    #  #      # ###  #  #  # #   ##    #    # #        ##      #   #         ###  # # # # # #  #  # #     # #          # #   ### ##  ##      #  #  #  ##  #  #    #       ## #####  #      #  ###   #  # #
#          #    ## ##  # ##       ## #  #   # #  #       #   # # # #   #         ##  # #  # #               ###      #   #  #      #      # ##  ## ##       # #  ##  ###  #         #   #          ##   #      #    #     #  # # # #        #    E    #  #
#    # #  #    ####      #  #  # #     # ##          #  #      # #     ###    #  #   #  # # ###      #####    ###            #   ###  ## #   #  ## #   #  ####   #      ##         #       #  #     #  #  # ##   #  ##           ##  #       ##  # #     #
#    #      #    #   ##            #  ##  ###    ###          ## # #   ## #   #    #  # #  ####     #                    ## #  #     ###      ##  # ##    # #  #        # ##        ###          #    ###   #       # # #  # ##   ### ##        ##    ## #
# ###  ##  ##  ####  #  #   # #           # #     #  # #   # #   ## #  ## #  ###  #       #  # ### #      # ## # # # #  ### #  #  ##    ## #     #           #       #             # #    #   #    #  # #  #   # #  #     #      ##  #      #  # ## #    #
# # ##  #              # ## #   ## # #     ##   ##   #    #     ####      #  #       #                      # # #        #    #    ### # ##    ### ##     ##     # # # #  ##   ###  # #   #  #  ##    #  ## ## ##        # #            #   #  # ##  #  ##
##   # # #       #        #  # ## #  #  #  #  ##  # #     #                  #    #  #   #     #####   #### ## ##  #    #          #    #   ##  #    #   #    ##  #  # # # #          #    #    # ####   #  # # #    #  #        ##        # #  #   ##   #
###      # # # # #   # #   # #      # #  #     #        #  #  ## #  ##    ## #   ##       #    #     ##  #####     #    #      # #  ## #   #         #     #   #  # #     # #   #  # #       ##          #   #     # #  #       #   # ##  #   # # ##    ##
# #         #    #  ## ##    #    #  #   #    #   #  #   #       # ###  # ##   #   # #### # #   #  ####        # #       #   # #   # #    # #     #   ####   #  ###      # ## #     ## #  #     ##      #  #   # ## ## ##   #      ### ## # #    # #     #
#  #    #       ##   #  ##    #    # #        #     #   #  ##   #  #    ###        #    ##     ##  ### # # #  #                # #  #  ##   ## #   #   # ###    ##  ##  ###      #     # ## ##   # ##     #    #     # #    #  #      ##   #######     # #
## ##    ###    #   #  #              ## #  ### #    #        ##       ##  #     #  # #     #   #   ##   ##   # #  #    ##       #   ### #     #   #  #   ##   ####          #  #     #   #   # ##       #     #  #   # ## #         ###     ##    ## # ##
# #  #  ## # #          #       #       # #     ##  #     # ##  #####  # #      #     #    #  # #           #    #     #####   ##    #  ##  #          #  #        #      #    #  # #              #     #   ## #  #     #   #   # ### ##   #      #   # #
#     #       #  ##     #   ####     ##    #    #   #   ##      ##        ## #    #   ## #  #   ###   ##  ##   #        #   #  # #   #     ###    ##   #     ##   ###    ##  # ##     # #  #       #   ## # # ###        #    ####  ##   ##   ##     #   #
# ### #  ####       ####      #       ###       ###      ### ##    ## ##   # # #     #      ##      #         ##      #    # #    #   # #  #   #     # ##  ### #   #   #    ##  #   #      ###       #     #  #      ##       # ##     #               # #
#   ###  ## #     ##    #        #    # #        #    ######     #  # #       #  ##  # #  #  ##            #   #    #                      #  # ##    ###   #  #        ###  ##   # ## ##    #   ##   #   #       #  #            #    # #  ##   ##      #
#            # ###  #  ####     #    #    #   # # #  #    # #####    #### # #   #  ##       ### #         #  #  #        #    #   ##  ##   ##            #         #  ## #      ###   #  # # # ## #         #  #   #    #   ## #   #          ### # ##   #
#  #    # ###  ##         # # ##   #  # #    # # #         #     ##  # # #       #   #        ##  # #          #   #   #   #      #     # ###   #     #                # # ###       #  ###   ## # #      #      ## #  # #  ####         ###   #   ##  ###
# ## # #   #    #    #       ##   #    #    ####     ##   ###  # #   # #                           ##    #   #     #   ###    #        ## ##  ##     #  #       #      # ## #     #              #   # #    ##   #  #  ##   #  #   #  ## ##   # #  #     #
#      # #             #    #     # ##   # #  #  #  # ###          #   #      #    #       #    ###     ## #  # #    # # # ##      #      #        ###               #   #   # # ##  # #   # ###  ##     #   # #### # #      # # ##  #   #          ##   #
# # #     #  ##          #    #     ###     #  # # # ##  #     ##     # #       #  #  #   ##            #  ## ###    #       #   #  #    ## #      #     ##  #   #### #  #         ## ## #  ###   # ##         ## # #   #  # #   #     ## #          #   #
# #  ## #   #  ####  #   #  #  ## #  #     # #         #   #   ## #        #  #  ## ##  #  ###      # ###   #  ##   #  #  #      ###   ## #  ##       #    #       #  #     #    ## #   #       ### #       # # # #   ###         # ##       # #  #     ##
# ### #   #     ## #             #     ##     #  ##     #          ###  #    #  # #   ##     #   # ##### ## #   # ##  #   #### #  ##       #  ## ###    # #     # ##       #   # #             #    #    #   #  #  ## #       #     # #     ##     #     #
#  #  #     #  #   #                  #   ##       #  #  ##     ##  #   #       ##  # # #  ##  #       #    ##   #  #   #   #  # #    #      #      #   # ##    # #  ## # #     ## ####  ## ###  # #     #    # #   ##  #            ##    ####   #   ## #
## #  #       #      #     # # #  ## #     #      # # #   #    #  #  #     #  #     #     # ## #    #  ### #      ##   #   #  # # # #    ##  # ###   # #  ### ### # #  #   # ######        ##### #  #  #  ##  ###       #   #  #    #    # # # #  #    ###
#  ## #   #     #                # #     #   #    ## ##  ##   #    # # # #    ##    # #   #        # ##    ### ###  # #     #     #          #       ##   #     #       # #   # #    #     ## ## #     ##   #   # # ##           #    #  ## ## #         #
## # #   ## # ##  ###   #     # #    # # ##   ##  #        #       #   # #  # #    ## ##      #  #       #  # # #  #  # ##   ## #   #    #              ## #  #    ## # #  ##   # ##  #    #  #  #    # ###  # #      #  #  #       #     # #       #    #
###     # #  #  #  # #    #  #    ##   #   #      # #  #   #  #   # ##  #   #      #    #     #   ##   #  # #    ##   #    # #   #       # #  # #  #    #    #     #     ##    ##    #       #      #     ##     #    ##    #     # ## #      # #        #
#   ###    #          ###  # #        ### #   #  # #  #  #   ##  #  # ##    ## # #     #      ###   # ###### #    #  #    #      #   #     #  #     ## ####   #  #    #  # ##                 #  #   #   #     ##  ##      ## #  #  ##   ##   ##     ### #
#  #  # ##  #      #         #   ##       #  ##   ##   # ###   ##        ####   # #                # ##    # ##            #       ###       #         #          #   ##    #    #    # #   #   #  # ## #     #  #             #      # # #  ###        ##
# #  # ##      ##   #      ### #        #         ##  #  #  #  # #      #   ##       ## #              ##    ##    #### # #  #  #   ##        #  ### # # # # #      #     #     #        #  # ## #           ##   #    ##   #  #    #  #    # #   #   # ##
#  ##    #  #  # #  # #   #     #    #       #         # ##    #    ##     ###   #  ###     # ##    # #   # # ##   # ## #    ##   #     # ##    #  # ####                  ## #  ###  # ##   ##    #    #  #  ###      #      # # #         #      # #   #
##       #   #   #      #  # #   #       #     #  ##   #       #      ### #  ##        #          #   #   #  ###    # # #   #   #  #   # #    ## #  ##    #       #  # # #    #    ##  #    #   #      #          #   #   ###     #       ##   ###     # #
#  # # #  #    # ##   #    #  #   #   ## #   ## ## ##   ##        #    #  #          #### #  ##        ## # # ###    # #    # ##    ##      ###   #   #   # #       # # #   ##    # ## # #   ##     #  #    # #    #  ## # # # #  # ###     ####     # ###
#  #     #   #   #  #  #     ##         ##  # #    #   ## #          ##   #    #      #    #        ####         #       #       ##   #   ##     ##   # ##      #     # #   #    # #      #  #       # ## #    ##     #      ##   #  #    ###  ## ## # # #
# # #  ##   #   # #    #  # #   #   #    # # #   ##      #   # #           ## #    ##  # ###    # #    #  # # ##         #      ##      #  #   # #   #### #  ##  # # #        ##     #         #         ####       # ## # #    ##     #  #  #  # ###   ##
##  #  # #        #   ##   ##          ###  #           ##    #      ## #   # # #  ##    ##  ## #   #   # # #             ###       #    #           ##     ##  #  # #  # #     # #   ##  #####  ##  #       ##   #    #   #       # #     ##   #### ##  #
#   #    ## #   #   #          #    ##          # ## #####   #  ##     #       ##    ## # #     #  #       #     ##   #    #  # ##    ### #     ####  #  ### #  #    #  # #    # #     #    #    #       ##                     #     #  #   ##          #
#    # #  #  #  ##    #     ## #####    #         #  # #   # # #  ##  #     #    ##    # #     # ##     # #  # ##  #  ##       #  # #  # #    #                       #### #  #  #  # #### # # #  ###     # #   # #      #   ###   ###  ##  #    ###   ###
##  #  #     ##  ## #   #    ##   #   #     # # ##   #  ##                        # #  #####      # # ####   #   ## #      #   #    ##         # #       #         #  # #   # ###### ### #       # #  #  #    #  #   #      # ##         #  #        #   #
#  ## ## ###  # #  ###            # # ###    #  #  ##   ##  #  #     #     ##  #        #    #    #  # #     #### #   #   ## ## #### ### #   ##         # #   #         #         # ##  #    #   #  # #  #   # ###  #     ## ##  #   ## #  # #### #   ## #
#       #       #         ##  # ##   #   #   ### #       # # ### #    #   #    ##      #  #  # ####### #   #    #     ###  # #    # ##  #  ##   ##  #      #  #   #           # #  ##    #### # #    #    #     #    #     #         ##      #      #  # #
#  ##        ## ##   #      # ## # ##    ##   #     #      #  # # ##       ##  #   ##          #                #    #  ## ##        # # #  #           #    # #   # #    ##   ##  ###     ##    #   # ###    #   ####    ## # #          # # #     #   ##
#   #   # #   ##  #      #     # # #     #####        ##       ##     ###     ##             #     #  #  #   ##   #          ### #      ###    ###  #  ##          ##  #              #      ##     ##    ##         #                        #   #      #
#          # #  #   #          # ##   # ##### #  # # #      #     #  #  #  # # #      # ##    #  ##   #       # #  #   #     #  #  #  #   ## # ##  ### ## #    ####  #     ## ###  #  #    #  ##  ###  #    #   ## #  # ##    #  ##   ###  # #       #   #
# #  #    #     #          #     #    #     # ##        #  ## #    #  ###         ##   #            #    #    #  ###    #  ## #  ##         ### ##    ##  #  # # # #     # #  # #      #    #  #    # ### # ###    #  #   ##   #       #      # #  # #  ##
# ##### # ## ### # #   ## #   ####  #   ## #       # #    #    ##    #   # # #    ##      #   # ##  #   ##      #  #   #   ##    ##   # ##  #    #      # # #   ##   #   ##      #  ##  #### #  # # ##     ### # #  #  #       #     ## ## ###  #  ##  # #
#  ##    ## #  ## ##  #  #   #### #  ##  #    #    ##    ##  # ##    # #   #    #  #   #     #      #       #      #  #           ## #  # #           #     #     ### ## ##    #  # ##   ## # #   # #    # #           ##  #   #        #     ## ##   ## #
# #  #  # #  ##  #      ###      #   #     #  #   #     #     #       #  #  #           #      #         #      #    #  # # #    #   #  ###  ##      ##   ## #    #    #    #  #  ## #  ##        #  # # #   ##   # #       #   # ##     #  #     #   ## #
##  #  #  ##           #    #      #   #  #  ##      #          #  # #         #     #          # #    #  #            # ## # #    #    # ##      #     # #   #       #   ##     #  #          ##  # ##  #   #   ## ###   #   ## ## ##   #  # ## #   #####
#     ##   #    ## #     #      ###   ##   ##          #       #        #  #  #         # ###    # #         # #      ##        #    #         #   # #        #   #               #  # ###   #  #   #     #  #     ## ###  #  #          # #   #####  # ##
###  #   #  #      # # ###  # # ##     ## ##   #   #  #   # ##     # #    # ## #    #  #    #      #   #   #     #  ##  # ##        #   #  #              #   #    #    #   # #    # #    #           # #     #  #    #  #        # # ##  #   #    # #   #
##    #    # ##   #       ##  #     ## #   #          #    ##   #   ###    #        # #   #  #    #   ####      ## #  #     #   #  #    #       #  ##         ##   #    ###  ##   ## #    ##     #  #   ##   #      #       ##  # # ##   ###  #     #  # #
##        #     ##       ## ##    #    #          # #   ##  #     # ### #  # #        # ###    #   ##  ##  # #                  # # #       #          #   #    ### ## #     ###      # ## # ###   #        #  #  #         #### # #               #    ##
##   #  ## # #     ##         #   # #  ## #    ##  ###     #   # # ##      ##      #  #   ###  #  ##  #     #### #   # # ##  ##       ### #     #            #                # #   # ##               #  # ##    ##  #      #   #   #      ###      #   #
#   #                 ##          #   #       #   # #     # #  #####   #    #             # #    # #  #   #  # #  # ## ##     # #          #   ## # ##     ## #  #         # # #  #  #          ####              ##     #  #   #        #      #  #    ##
# # ###      #      # #   # #    # #      # ##  #   #       #     #   #      #  ##     ##   #     ###   #  #   #   #  #      ###     #  ##         #       ##    #   #      ##  #  ##     #  #  # ###                ### #       #   #         # ##      #
#    # #    #     #     # #     ## #  #     ###    # # # #    #  #      #         ##  #  ###        #  #        #  #       # ##   ##    #  ##    #  ##       #   # # ##    #          #    #   #      # #  ###   #  #         #    ###     ##     #     ##
#    #  #  # ## #         # #        #    #  #     ###   #     #     #   #  #    ##  #     # ##          #   #   #   ## #    # ##     # #### ###    # # ####           #  # ##  #       #  # # # #   # ## ####     ## ## #          #           #   #    #
#       ###         #     # #    #      ##    #          ##   # #             #    # #  #  # #   #  ### # ##       #  # ##    # # #  ##  ##    # #  # ## # #           #  #   #     #  ## # #     #   ###        ##  # ## # #  ####     #    #    ##   # #
# #    ##  #  #              #   # #  #  # ##      #     ##   # #   # ##  #### #     # # #  # #  # #    # #      ##        ##    #   #  #  #      ##     #       ##       ####     ##    #     #       ### #    #  #               #    #       #   # #  #
#          #                #   ## #   ## #  ##   #  # # ### #     # ###    # #  ##        ##     # # #  #        #      ##   #    #  ####   # ###    ###             #  # ##  #    ##        #            # #  #   ##          #      ##      # #      ##
#     ##       #    ##   #  #     ##   ## #  # #     ##       #     ###  # # # ##  #   # # #  # # #          ##    #   ##  ## #        ## #  # ##  #     #      #  #       #  #   ##        # # ####   #   ##  ## #     #        ###    ##        #   #  #
# #   #  ### ##   #  #  ## #  #  #       # #  #             ### #  #   #   #   ####### ### # # # #          # ## #   ####  #   ## # ###  # #             #     # # #      ##   #      #      ##   #       # #               ##    #   #   # #       #    #
#      #         ### ##  # #   ##       ######       ### #    #        #  #  #    #  # #   #            # # #    #   #  ## #   #   ###   ##    #     #      #  # #       #   # #  # #     #      ##     ## ##       #  # #  #    #  # #### #     #   #   #
#  #  #         ### # # ##  #   ##  #   ## # ###        ## #     #  #  #     #       #   #   ##     # # # #   #      ##  ##  ## ###  #    #     #            ##       #   ##     #         ##      #          #   ## #   #  ### #            #  #   #  ###
##    #  ##  #  ##  #  #  ## #  #    #  # #### #  #  ####       #   ###  # #  ##  ####      #   #  #    ##  #       #  ##  ##  #   #  #   ##  #  #      # #    #        ##    # #       #  #  #   # #       # #  #   #  # #   #    #      ##  #  #     # #
#   #  #####  # #### #  #    # #  #     #     ###  #      #   ##    #       #  #      #  ##   ##  #   ###  # #  # # #    #         ##      #       # # #    ##  #    #  #  # #    #      ## #     # #  #  #  #      #   #    #    # #                # # #
# #       #       #           #      ##  ##        # # ##   #  ##    ##  #   # #  ## ##     #   # ##         # #  ### # #      #   #       #   ## #     #  #  ###            #  #       #    # # #    #       #     #    ##   #        # #       #   #   #
#           #    #   #   ####   ##    ###   #  #  #  ## # #   #              # #  #      #    #       #   # ##  #         #  #                     ####  ####  # #              #  #   #  #  # ##     ### #     #  # # #   #   #  #         #  # #   #   #
#  ##   ### ##   #  #    # #  #    #   ##        #    #   #      ## #       #                 #  # #  #  ##      #   # # # ##      ##  #    #      ###  #   # #       #  ## # ##  #    ## #   ##    ###         #  #   #   # #   #  ##  #  #   #        ##
#    # #      #  ###   #           # #  #             #    ## #  #      # #     ###  # # #            ##    #      #    ##        # #           # ##    #   ## #      ##  #  #   #             #   #   #    ##       #        #  ## ##  # # #  #        ##
###  # #  # #    #  #  ##       #                 # ####    #           #   ##    #  # # # #    ##        # #    ##    #  ##  #  #        # # # #    ##    #     #      #    #    # #    #    # #    ####   #    ### ##      ##  #        #  ##   #      #
# ##  #    ###  ##      #   #    #          ##      #   #      ##      #   #    #       ###  ##  # #  #              #   # #  ###         #  ### ###       # #  #  # # #    #    ##   ##       #  # #  #   ### # #    # #     ###    #  ###   #   #  ##  #
##    #    ## #   #  ###   ## ##                         ##         #   #  ##   ##  #   #    # ###  ##   ##   #    # #### #    ## ###       # ## ## # ##   ## ##     #      # ##   #    # # ##     #  ## ###          # #      ## ##  #       #  # # #   #
# ###   ##  # # #     ##  ##  #     ## ##  #   # ##  #        #  #   # # #   # ### #  # #  #  ## #  ##   ##   #               ### # #  #  #  #    #   #  #  # # ##   #   #   ## #    #  # # #       #      #  #        ##        ##  #  #### #  ###     ##
#     ###                      #            ###  # #       #    # #  #     ##  #    #  #         # #    #   #    #  # # #    #     #       #  #  #   ##          #       #  #    ##### #  # #   # #  ##  # #   #       #   #   #  # #   # #     ##  # #  #
#  #### #     #    ##  ##   #       ##   ##  # #   ##    ##   #    #  # ## ### #  ####      #     #         #        # #     ##   ###   ## ##                 # # #  #     #        # ##          ## # #          #        ##   # ##  # #  # #     # #   #
##    # ## ## # #  #  #  #  ##   ##   ##   #   # #  ### ## ####   ##   #      #   #  # #     # #     #       #   #   #     #    #      #    #     # #       # #        ###                #  ##  ## #   #             #   # #  #  #       #   #      #####
# ### #    #   ### #    #  ##   #  #  # #            # #    ##     # # #     #   #     ### #      ##   # ##     #     ##  #  #  #   #  ## ##   #  ## ##   #        #            #   ##    ##                         #         ## #  #  # # # #  #   #  ##
#   # #    #  #         # ## #  #    # #   # # # ##         #   ### #     #### #####   ##         #    ##       #     #     #   #     #  #      # ###  # #  #    #     ##          #### #  #  #  #  ## ##    #      ## # #     #       #        #    #   #
#  ##     ## ####     ##   #     #  # # #    ## # # # ##  ####     ### #     # #     #   # ##     #   #   #  # # #     #    ##  #                #     #  ##   ###         #     #   #      #  #   #   ##      #   #  ##    #     ##### # # #    #  ### ##
# # ###   ##  ##   # #  ### ###   # ###       #     #  ### #       ###  #  #  #          #   #    ##  #   ##    #    ## #    #       #       #         #   ## #         # #    #     # ##     #  ##             ##   #  # ## #              #         #  #
##    #    # #  #     #   #        #       #   ##  # ##    #     ##    #   #  #     #    # ##   ## # #     ## #   # # ##  ##  ## #     #       #    #   # #        #  #    #    #        #   # #   ##    ###     #   #  # ##     #   # ##        #  #    #
# #       # #  ## #          # #  #    ##  #  # #   # #          #     # #  # ##      #     #  # # #   ## #     #   # ####   #     ##  #   #   # ##   ## #       #      ## #  #       # ####   #   # #     #  ### # #### #      #         #    # ### # # #
# # # # #  ###  #      # #    #        #  #          ###     # #       ## #      # #         ##   #   ##   ###    ##     #         #  #  ## ###      #       #  # #  # # ##     # # #    # ##             ## #  # #          ###        ###  #   #   ### #
#       #  # #### # #        #    ##         #   ##    # #       #  #     #              # #  ##   ###   #        ##  # #  #            ## # #   #    ##      #    #    #       ###  # ##  #         ###   #   ##  # ##             # ##       # ## # #  #
#  ##    #  #    #         #  ##  #    ###  #                      #      # # #   #       ##      #     ##  #     #     #   ##   # ##  # #          # #  # ##  #    # ##  #   # # # #    ##     # ###  #      #   ###    #     #        ##        # #    #
#   #        #  #  # #   #  ###    #  #    #  #             # ### ###  #     # #   #    # #      ####  #  #  #     # #    #     #        #      ###   # #   # ###  # # # # ##   #  # ###      ## ###   #   #   #     # # ## # #      ##        #   # #   #
#          # #    # #  ##    #  ##     # ##    #####  ###      #   # #  #  #   ## # ## #  # #    ### #     ##       # #  #    ## # ###    ##     #      #             #######   # #  # # #        #    #    ###      ##  #  ## #       #    ###  #  ##   #
###   ##           #     #    ## #    # #  ##     #       ###  #  ###  # #      #   #   #   # ##    #  ### #        ### ###     ###      #      # ##     #         ## # ##  # ####   ####   #  #  ##  #  #  #     # ##     ##         #  # # ###### #  ###
## # #      ##    #     #    #   #    ##        # #  #     #     #     #    #    ###       #    #        # #     #   # #  #   ##  #  #   # #    #   #   # #     ##  # ## #     ## # #         ## ##          #    # # #   #  ## ##  #  # #         #    ##
## #    ## ##     # # #  #  #      ##     ##      # #  ###  # #  # ###  #   ## ####         #   # ##    #   ##     # #   ##  # ##   #     #        #   #  # ##  ## ##    ##      #  #  #     ###      # # #       # #   # #          #     ####  ####  # #
# #  #   #  # ####     # #  ##  # # # #             ###  # #   #  # # # #   ## #   ##      ##   ##      #        #   #             #  ## ## #  #          ###### #    #   ## #  #   ##   #     #   #### ###    #     #   ## ### ##   ## # #   # #  ##### #
#     #   ##    # #    #   ##    #     ##  # ### #   #    ### #    ##  #  #   ##  #  #  #     ## ##        #  ### #  ##           ### # #     # #   ##    #       # #  ##    ##  #      ### #        #  #        ## # ##  ##    #  # ##  ## #    # ## ## #
##        #   #      # # #       ## ##  ##   ####           #         #   # ## ##       #   # #       # ### ###    ##  # #   #         ## #       # ##   #   #   ## #   ### # #  #    # #   #       ##    #         #     # ##         # # #     #       #
#    # # #       ## # #  # ##  #    ## # # #### ##  #     # #      # #  #     #  #         ## # ###  ##         #    ##  #  #  #  ##  # #  ###  ##      # # ###  ####    #    ## ###   # ##     # #      # #   ##  # #       #     #   ##     #    ## #  #
#      ## ### #  #   #     #  # #  #   ## #        # #  # ##   # ##   # ## # #  # # # #  #  ## # ##            # # # # ## # # #      # # ##    #           #  #   #   #                         # ##      # #    #     #   # #          ## ## #   #   #  #
#            ##    ###           #   #      # #####    ### ###   # #   ###   #  ##   # #  #  ## #    #       # #         #  #   # # # # ##  #  #          ##    #      #######  ##    ##  ###  #       ####  # #  ####  ##  #   #   #   ##    #   #  # # #
##  # ####     #   #         ##       ##     ##           #    ##  # ##  #     # #    ##  #   ##    ##  # #       ##         #  ##  #        ### #      #   #  # #    # ###   #     #     ## # #  #      ####      #   # #    ###  #         #     #   ###
# ##### #   ## #    # #    ##       #           # ## ##   #         ##    #      # #  #  # ## #         # ### #   #  # # #   ###   #          # #  #   ###   #  ###  #        #   ##    ####   #     ##   # ###          ###    #     ##   #   ###    #  #
#  ####       #   # #  ##  ###     ##  #         #  ##       #      ## ##    ###   # ##   #     #          ## # #       ## #   # # #  # #    ##  #   #  #    # ## ## ####            #           #  #     ### # #    #  ##  #    # #  #   # # ###   #   ##
#  ##       ####  ###  #    #          #  ##   ##   ### #       ##   #    # ##   #    # #      #   #    # #          ## #      # #      # ##  #   ##  # #    #   #     ###      #  #   # # ###    ### #  #    ## ## # ### #     # ##     #  #  # # ## #  #
#      # ##   #  #    # ##      ## #    #  # #   #   #  ##  ##   ## ##  #   #   #  #    #  # # ##  ###         #  ##     # #       # #  ##        #  #  #   # #      ##          # #        # # ##       # ##       ##   ###   #  ## #   # #  ####     ###
# #  # ###   ##    #   #   # #   # ###     # # ###  #    # #    #  #  #  #        #      #      #      # #   #  #     ## ##    ## #  #  #  ## # #  ##    #        #  # # #   #   ## #    #   #  #     #    ##     #        #   # #### ##   #      ## #   #
#          #                     ###  ####        ###  #  # #    ## # ###    ##   ## # ##   #        ##  # # ##         #     # #   # # #  #  #      ##       #         ###      ##  #        #    # #    #  ##  ####  ###   ##      #  # # ##  ### #  # #
#        #     # #     #   ##   #   #  #    ##     ### # #     # #  ## #       ## #   #         #    ##         #      #       #       # #  ###   # ##   ##  ##  #       #    #  #   #  #       ##     ##    #    ## # #     #     # ##     #### # ##   ##
#     #   ### ### #    ##  ## #         ##   ##  #    ##  ###     #             ### #     # #    # #     #     ###  #    #     # #        # #   #  # ## #   # ##    #           ###  #          # #       ##    #   # ## #  ## # ### ##   #          # # #
###   #  ##   #    ##   #   #            ## #  # # # #      #   ##   #   ###### # ##### #    # # ##      ##    #  #        ##  ##   #  ##   #    ## ###### ##   # #  ###       ### #  #  ##  #  ## # # #         #    # #   ## #  #   #     # #  #  #    #
#  # #  #  #   #       #            # ## ##         #     ##       #   ## ##      ##       # #   ## #     ####  # # ####  # ##                 #  ### #   # ##     ####     ##     #   # ##          #  # ##  ###    #     #      #     #  #  # #      # #
# #  #  # ##  #      # # ##  ## #  #   # ## # # ##            # ## # #  #   #  #     # #  ##    #  #       ####      ###  # #        # #    # #    ## # ##  ## ###    #       #   #   #  #        #      ##      ## #    # ##              # ## ##     # #
#  # # #   ### #         #          #    #   # ##   #   #       #       #   #  #  #    #      #   #  # #  ##     #             #   # #           #      #    # #   #          #  ###   #  ## ##    ######  ## #     ##### #    # ### ##   #      ## # ## #
#              #       #    #   #    #      # # ## # ## ##  #  #       ## ##         #    # #   # # #####     # #  ##   #### #  # #    # #  ##   ##   ##    #  #   #   #  #   ###   #      ##    #   ####     #   # #####           ##       #  #    #  ##
### # #   #  ##   #    ##  #        # # #   #    #  #          #         #  #      ## #      ##  #   #     #           #  #       ##     ### #    #      #  ## ##  #  #   ##     #     #  # #  ## #    # #  ##          #     # #     # # # #   #     #  #
# # #        ##  #        ##  # ###    ###     # #   #   # ##  #   # ### #  #  #    #          # #     #     ###  #  #      #  #   ##      #### ##        # #    #      ###  # #   #      # #  #         #         #    ## #  ### #    # # #     #  #  # #
## #    #        #    # # # #    #   # ####   ###       ### ## #  #      #              # #     #       # #   ##### ##       #  #    #     #    ##   # # #    # # #  # #  ## #    #           ### #  #   ##  #   #     #     ##       #  #     #      #  #
##       #  ## #  ##           #     #  # #      #     #    #          # #   #     ##  #  #    # # #   #   # #       ## #   # # ## # #   ##        #     ##           #  ###  # ###  ##   # # ####  #  #  #   #            ## #     ## #   # ##     # #  #
###  ## ##  ##  ###  ##### #    ##     #    #                    # #     #  # ## ##   ###   # ### #    #      #   # # ## #    #### # #     ##    ##     #    #   #       # #  # ##  #   #  ### #      #   #     #     # ## #           ####   #  ## ##   #
####  ## ##  #       #       # #  ##   #   # #    ###  # #  # ##      #    #  ##  ### #       #      #  # #  #  #####    ## # # ###  #         #         #       #  #             #  # #          # ##   #     ##          ###        ###          # # ###
##    #    ##  #    #      #        # # #        #   # ### #      #  ##    #  # #       ###  #   ##   # # # ##   #  ##   #    ## ## ##### #   #  #  #   #       #  #   # # #   # # #   #  #        #  ### ## #   #  #      # #   #      #  ##   ##  #  # #
#      ##   #     #     ####     ## #   #  # # ## #### #  ##   #  ##      ### # ##   # ##     ##       #    #  ##    #         ###        #    # ##    ###  ### #   ###      #  #              ##    # #   ##   ###     #  #    # ## #### #       #    # #
#   #    #   #     ##   ## # ## ##   #  ##  #     # # ##          #   #   ## #  #   #         ## # # # #      #     ##  # #  #  # #  #    #  #   ## ##   ###   #   #### #  #  #  ##  #  #         #      #   ###    #        ##      ##           # # #  #
#     #  ##  #  ## ###     #      ### #  # #   #   #  ##   #    ##  #  ###       # #       # #####     #  # #  ### ## #   ##     ##   #  # #   #    #        #      #   # #    #  #     #    ##     #    ## # ### #  # ##                #               #
#           #  #   # #    ## ##   ##  ## ##   # #      # #             ### ####   ##   #  ### # #    ##   #       # #   #    #  #  # ###     ###   ##  #  #   #    ##       ## #    # #    #    #  # #  ## ##     #### ##   #     #   #  #  # #    ## #  #
###      #   ###    # #        ##     # ## ##   #  #       ## #       #   #     #    #   # #   #    ##          #   #     #    ## #   #     #        ##  # #  # #  ##  #  ##                   #  ####  #          #  ### #  #  #        #       ##  ##  #
#  ###     #        #  ###   # ##     #          #  #  #      #  #      #### ##        #  # #   #   ##  ##  #  #### #    # #    #  # #### #  #       #    #  #  #    #   #       #####     #    #           # #   ## # #     # #      ##       #  ##    ##
#      # #    ###  #        #    #    # #  #  #  #   #     # #### # ##   #   #   #   #     #    #     # #  #    #     # #     ##    # #    #  # ##  ##    #    #              ##       #   #  ##         #         #    #       #   #   #  # # #  ###  # #
##  #  # ##       # ##  #####           #  #       #  # # #      #     #     #   # ###  ##      ##  # # #    ##      #    # ## # ## ##  #     #   #        ###    ##   #      # #    #    ## ## #    # ####    #     #  #  #  #   #    # #     # #       #
###    #    #  # ##           #    #        # ##   ##   ##    #    #   ##  ## #           #    ## #      ####     ##  #     #  #    # #  #   #   #    #    #    #  ##   #  #  #        ## ### # #   # ### ## # # #  ##   #  ##          #  # # #### #    #
#      ## #         #        #         ######   # ## # ## #    ##     ##   ####  # #    ###    #####      # ##   ## #      # #  ##  #  #      # #  ##  ###  ##         # #  ##   ##        #  #  ##  ##    # # # #  #   #   #   #  ###   # # # #   # ### #
### ##     #   #      # #  # ## #  #     #      ##        #   #    ###        #     #    #   ##        #       #    # ## ###   #  #   # ##     ##  # #  ###       ##       #    ##  #      #  ##   #      # #    #    #       # ## ##    # # ## # # #    #
#     ##   # #    #  #  #  #      #    ##     #        #    #      ##     ##   #        #   # #   ## #  #  #       ##   #  ##  ##     #    ##       #  #  ##  ##   # #      # #  ## #  #  ##  # #  #     # #       # # #### #  #         ##    ##    ##  #
###   ### # ##    #     # ## ##  #    ###   #   #  # # #   #       #          # # ##   ##          # #    #    ## #  #  #       ##     #    ##      ##   #       #   ##  #        #   #    #        #  #       # ##   # #       #    ##  #  ##  #     #  #
# #      #                 # #   # #   #  #   #   ##   #  #     #   ## # # # #               ## # ##      #          #   # #      #  #  #   #   # #     # #   # # ##  #  ## #      ###  #  ### # ##  # #   # # #         ##  ## #     #    # # #    #    #
#  # #  #          ##  # ## # #       # #       # #          ##          # ###    #    # #  #   # #          #                 #  #  ##   #   #  # # #        #  #  #  # ###   #  ###   ## # ##  #  ###### ##    ###        # # #  ##        # ###   # # #
# # # #   ## ##  ## # #  ##    # #         #  # ###    # ##  # # # # ## # # ## #  #           ####          #                 #  #   ##       ##  ##  #  #   #  #  # #   #            # #     #  ##  #  ###   #   ##  ###      # #     # #    # ##       #
# # #   # ### #  ## #     ####  #    #####    # ###  #    #  ### #  ## ## ### #  #    ##        ####   # #  #       ### #   #    # ####  #  #   # #  #         # #    ###  #              ##  ## #   #   #   # ### #         # ## ##  # #     # ##      ##
###   #     #   ##    # # # # #  #            #     # #  # #       # #  ##  ##     ##   # #  #  #   ## ##       #  #               #    #    ## # #   #        #    #     #     ##  ####       # #     #    #  # ## # #     ## #    #    # #     #   ### #
#  #         ## ##  #  #  #  ## ### # #   #  ##         ##    #    # #         ###  #  ##        ##    #      ##     ##      # #   #   #  #               ##    #   ###    # # # ## ##         #  #  # #    #     ##     #   # #     #    #  # ## ##  # ##
##  #  #  ## #       #  ##       #      ###   #   # #      ## # # #  ###   #         #   ## ##   ##    ### ##  ##   #   #  # # ##  ## # #   # #     #           #     ##    # ####  # ##           #    # # ##   # #    # ##   # #  ## # ## ##   # #  ## #
###  #       # ## ##             ##   ### ##  #    #    ##  #       #  #   # ##    # # ###        ###      #  #     #       ##  #   #      #   # # #     ## #         ##   #        #     ##             #  #  ## ##            #  #     # # # #  ##     #
##       ##  # #  #   # ##      # ##   # # # #  #   # #  # ##        #         ##  ##  #       #   #  ##      # ##     # ## #  #        ###  ##  ## # #   ## # #   ## #      # # ##         #     # #      ### # #   #     # #    # #   ## #    #   #    #
# #  #    #  # #  #     ## #  #  #   ##  # #       #    #   # ##  ##  ##   #      #    # #  ### #  ##     #  #   #     # #       #   # # #  ### # ## #   #  ## ### # # ##   ##  # #      #  #    #  #    #  # ##   ##      ###     #   ##  ##            #
#   ###  ##   ##             ##  #         # S#   #   #    ##     #  #                 # #   ## #   ## # #  ## #  #   # # #  ##     ##   #     ##     ###  #  # #      ##  ## #    #    # #  #      # # #  ## #   ##    #  #  #   # ##  ##      #  ##    #
##  #  #  # # # #  #  # ##     ###  ## # ##  #   ## # ## #      # ##   # #  #    # #    #  #   #    #   #   # #   #  ##      #   #  #  #  ###        ##    ###  ##  #        ##  # #     ## # ### #                #      #        #  #   #   ##  #   ## #
# #      #    #    ##  #    #  #   ##         #     ####          #        ##    # # #       #      ##        #  #   #     #       #    ## #   #    #   #          ####  #  #  #     #     #     #   #          ##  ##  #   ##         #   ###          ##
#  #    #   ##     #   #       #     #  #  ##   #     ####   # ##  ##                ###    #        ### #     ## # #       ### #  ###### #   #     #  #  #   #  #   # # ###  #  # #    #    #   ## #   ##   # ##   #   # # #   # #  #       # #     #   #
##  #  #   ##  ##  ## # #         #     #      #     # ###  # #   #     # #  ## ##  ###           ##  ## # ##  #  #   ###     #      #      #   # #    ## # ## ### #  # ##   ##  #  ##    #   #   #        # #   ####       ####   # ##          ##      #
#### # #  #  ##  #  #         #  #          ##         ## ##  #      ##    #   #   #  ##          #  #  ###  #   ### #   #     #  #    #    ##   ## #   ##    #   # ##    ##   ### #     # ##   #   #  ##       # #   #  ##  #       # # #        #      #
## #  #           # ##  #  #  # #   #      ##        ## #        #   # # #   ##   # #      # ##       # ###        ##  #       #    ##    #  ##            ###    #     #  # ###  # ##     #   ##       # #    ## ##   ##    # # ##      ##  #       ##  #
#   #              ###     # #      #           # # #  # #### # ##  # ## # #   ##           # #     ###  #    #          #  ##        ##           #  #    # #   #   ## ##  ## #  #  ##           ####    #   # #    # #  # ####  ## #  #   #            #
####  ##   #       #       # #   #     #  ## ##  #           #      # #### ###    #   #  #   # #   ##      #   #      ##  ##         # #  ##   #   # #  ###         # #   # #    # # ### ###  #    #   #     #      #### ##  #     #      #  #      #    #
#  #            ### # #   #  #   ##   ### #  #     #      #   #         #  ##  ###        ##  #  #  ##          # #  #    # #       # ## #    ##  #   # ##  ##  ### # ##       #  ## #      #          #     ##    # # ##  ######          ##   ####    ##
##   ###  #  #####    #   #    #  # # ##    ##      #      # #     ##   #  #    #     #    #  #  # #     #  #  #   ##  # ##   # # # # # ## ## ##      #       # #      ###     #    ##     ##      #  ##  # #  #      ##      #    #   #  ###     #      #
# # #   #   ##    ##       #  ## #   #   #      # # # ##      #   #   # # ### #   #      #      #  ##    ### # # ## ###           #        # # ##     ##  # # ## ###  #    #  ##                      ### # ## # #             #    #      #    ######   #
#      #   ## #  ##  #    ### #    ##    #       # #    #   #   # ###      #     # ##  ###     # #       # #     ###    # # ## # #    # #      ##       ### #     ###   ## #  ## ###    ####    #     ### #   #        # #  ### # #  #   ## ###### #  #  #
#  ##        #     ##     ####    ##      #  ##      #     #  ##    ##     # # #   #      ##  # #  ## #  #    # #       #   #      #      #   ##  #      #          #    #            # ###    ##  #          ##   # #   #    ##    # ##      #    #  ## #
##   ##### #      ##    #  #   #      #       #      #  #    #    #   ##  #     #  #   #    ##  #     ##  #   # #     ##     #    #  ### ###  # ##  #   #          ## #  ##   ## #   ##     #    #  ##   #  ##   # # #  # #  # # ##      ###        #    #
#      # #  ##   #   ##  # #    #     # ###  # #     #    # #   #  #   ##   # ##    # ###     ##      #    # #   #    #     #  #  #### #  ####  #  #   ##       #     # #  #      ####     ###        ###   # #### # # #    # #   # #  # #   #      #  # #
####   #          ## #   #  # # #          #    #  ##  #  ###      #    #   #  ##       ###    #     #       ##         #  #  #     #   #### #       # # #  #   ## ##   #     #   #       ##    #  #       #     # # ##      #  #    # # ## ##     # #   #
# #   ##   ##    #   # #    # ##   #      ##     # ##  #     # ##  # #         ###  # ## # ##   ##    #    #      #          #   #   #  # ##       ## ### #   #       ### ###  ## ####       # #          ##  #  #      ##   #       #   #          #    #
#     #  #   #      # #   ###  #       #  ###       #   #      #         #  #   ##  ## #              # #   ####  # ##        #       #  #     # # # ##     #  #  #     #      # ## ##      # #   #     # #   # #  #   ## #     #  ##  #   #      #     ##
##  #    #      ##  # #  ###  #    #  # # ###  ## ##    #                   ## #      #   ##  ##         #  #   #  #        # #  # #       # #     #      ## # ### # #          # ##    ####         #        # ##    #          #  #    #  # # #        #
# # #    #  ###     ##         #  ## ###  #   #     #  # #   #     #   #      # ##  # ### #   # ##     #  # ###        #    ######   # ###  ## ###  # #       #    # #### ### # #   ## # #            ## #    ### # # ## ###      # #  ##      #         #
# #      #      ##    ##         ## # # #    # ###  #   ## ##       # #  ##      #  ##   #  # #      #   ##    ## # #             #     #   #    #  #   # #####   ##    ##  ##     ##  # # ## # ##     #  #  # #          #     ##   #  ### #           ##
# ###   # # ##  ## ### #     #      ##        #  # #  #    #    # # ##    ##   #        # #   #     ##             #   #  #    #   #### ### # ### #   #        ## ## #   #       # #  # # ## #   #  # #  # #            ## #  #     ## ##     ##       ###
## #  #  #         ##    ###         #     # ##  # # # # #  #     #   # #  # #  #  #    #   # #      #  # ###  # #   # #   #   # # ##          ## # ## #    #    # #  # #            #      #    #            #    #  #    # ## #    #       # #       # #
##  #  # # ##    #   #    #   #  # ### ##      # # ##  #     #    #             ##    #  ###  ##       ## ##  # #    # # ##    ##    #####     #   ##     #  #  #  # # #  ###   #                #      # #  ###    #    #       #  # # #    ##     #### #
#    #            #   #   #    #        ##    # ##   ##     #  # #  # # #   # ## #    #   # ####       ##  #  #  #    ####  # #    #  ## #       ###      #  # #     #  # #         # #   #   #            #        #  ###     ##     #   ### #  #      ##
#    #      #  #  ###  ##       ### ##       #  # #        #   ##     # # ### #    #       ##  ## ##  ##             ###   ## ####   ##       ##   #  # ##   # ######     ###       #   #   ### #  #           # ## # #  ##    # #   # #  ## ## ##     # #
# #    #  #   ###  # # ####  # # #    ##    ###    #       ## #   #     ## # #####    #  #       ##        ### #     #    ##    # ## #    #    #   ####           ###       #####  # #   # ####  #  #   # #    #        #    #  # #   #      # #  ### #  #
##      ##     # ##     ## #  #   ####  # ### # #     #  ##       ## #  #   ###   #  ##   # ##   #     #  #  #    # #  #         # # #      #    #   ##   # #     ##            ##   #         ### ## #     #    #   ##    # # #  #   ##     #      # ## #
# # ##   #    ## #                  #    # ## #     #    ## # # ## ###### # ###    #       #  # ####  ## #  #       ## #  #         #  ## ##             ###    ###      # #  ##        #     # #  # #  # ##   # #   ####                       ###   ## #
## ##       #   #       ###    # #   #    ### #      # #     #     #       # #   #      #####        #       #    # ### #       # ####        #  ## # ##  ##    #   ## #     #  # ##    #   #    ## #    ### ##  #   #     #  #   # ##  #    # ###  #   ##
#  ##    #  #  #     # ## #   # #     ##  ##  # #   #  ###   #    #   # ##  ##    # #   #      # #   # #  #     #####         ##    ### #  # #       ##     #       #      #   ## ##  #     # # # ###  #  #  ##  # #           #  ##  ### #     #      # #
## #  #                  ##   ###   #   ##   # #        #  #  #       # #  #  #      #    #      ###   # # ##    #     #####  # ##     #         # # #     #    #     ## ##            ### #    #        # #  #   ##    # #  # #  #  #    #              #
# #  ####  #      #       ## #    ## ## ##      #   ##  # ###  ##      #          #    # #  #  #  #   # #  #      #    # #  #  # #          #      #   ##    #  #       #        # # ## # ##  # #                 #    # #      #   # ## ##    #  ##   ###
###   # ##    ## #   ##   #       # # ####   # # ##    #    ###  #   # #          #      #  #   #  #  #   ###       #  #    #### ##  ##    #  # ##     #  ## #   ###   #        #  #         ## # # #### ###  ##  # ## ###                ## ##  # #  #  #
#          # ##   #   ###  # # #     ###  #  #  ##  #   # # #     ### # # ## ##  #     #   #   # # ###   #  #     #    #      # #  #         ## ###    ##      #    #       #####  #   #      #     # #  # #     #   ### #  # ##   # ## #  #  ##  # #  # #
#   #  #  #     #  #   #    #            ##     # #  ##     ##          ##  #  #      # # # # # #     # #  #         #   #      ###   #  #  # #   #      #  #    # ###   # #            ##           #       #   #  #  ####  #    #      #   ##          #
#    #    ###      #         # ### # # #   # #  #     #        #       #    ##            # # #  #     #               ## #  # ##         ### # #      # ##     #   #  # #   #  #         #          # #  # #  #  #   ## #      #     # #  #  ##   ##### #
#   ##  ##      #     #  ## # # #   ##       #  ##  ## # #   #     #    #       #     # #    # #  ##       #     ##     # #  ##     # #   ## # #  #  #    ## #  ##   #  # # #   #       #       ## #  # # #    # ##    # # # # # #        #     #     #  #
#  #   #   #     #  #    #    #  # #   # #   #  #  #    #   # # #   # #   #    ##  ###      #    #  #     #       # # #       # #### ##     #   #   #  ##     #   ##      #  #         ###  #####    #   #### ## #         ## #  ## # # #  ##   #   #  # #
###      # ##     ## #         ### #   #  #    ##  # #    #   #  #   #         #  #       # #      #     ## #     # # # #     ##       ###  #      #  # ###   #        #       # #      ###   #   #  #     #  # #####     #  #  #  #   ###  #     ## # # #
##      #  # # #       # # ###    # ##   # #  #  ## #     ### ##     #  #   #    ## #    #       #  ### ##     #    #         # #          #  #   #       #        ##  #  #    ##     #          #     # #      ##      #   ##   #       ##            # #
# ### ##### #  #          #     #    # # ## #      #      # #      #  #  #  #   ##  # ## ##   ###       #          # #  ##  ## ##    #  #      # # ### #  #  # #    #        ### #   ## #    #    #    #  # ###    #     #       #    #   #       #      #
#  ##   #   #  ##    #    ###     ### ##       #   ##           ## #      ###         ##      #  ###    #  #             #        ##                   #    # # #       #     ### # # ## ##    ##    ###   # #  # #  ##     ## # # ###   ###   ##  #   # #
#   ## ## #   ## #    # # #   # ## ###   # ##    #  #  ## #### #  ##  # ##  #   # ##    # #  # ##  # #  #    #    #   #  #    #       # #   #  #   #   ### ##     ##           #  # ##    ####      ##    # #    ##    # #  # ###  ##  ##     #  ##    # #
#   #  # ## ##    #       # #  ##    #   ###  # #         ###  # #     # # # #   ### ##    #   #     # #          #      #    #   # #       #       #  #           ###    #     #         #   #        #   # #     #  ## #      # #       # #  #  # ###  #
#   #  #                    #       #  #      # # #  # ####    ### ###  # #        #         #    # #  # ## ## #        #      ##        #       ## ### ##      #   #    # #  ###   #  # #   #  #    #   #  #    #      # #  ## ##    #      # ##  # # # #
###              #    #    #      ##  # # # #  # ##  #     ##  #   #  # # # #    ##   #        #  #           #    #        ##   #        #     ##  #      #  #   ##    #####      #          #  #  # #  # #        #    #   ##  #   # # #   # #  # # # ##
#      #   ##        #    ## #                    # #   #    #  ###         ### #     #   # # # ## #   # #  #     # ## #      # #    ## # #     #    #  #     #  #  #     #  #   ##        # #####    # ##    ##       ##  # #    #### ##   # # #        #
#          #   #          # # #  #  #  #  #   #       #       #          ##  ##      #  #  #  ## # # # # ##  #        #  # #       #        #   # # # #  #  #    #        ## #    # #    ##  #     ## #          #  ##     ##   ###       # #    # ###  ##
#   #     #  ### #  # # #  #     #        #  #  #   ##       #  ###       #     #  #      # # #      #        # #       #   #     ###  # # ##  #  #  #    # #   # # #  ##     #    #    #     ### #  #           #    # ##   #     ## #     #           ##
#  #         #       ##     #      #    #  # #   ##### #  #  # #  ###      ##  ###  # #   ## ### #   #  #   #    #     ##       # #       ### ###       #  ##    ##       ##       #  #  #      ###         # #  #         # # #        #  ##      ##    #
##  #   #         # # #       ## # # ###  #           #  ##    # ###  ##   #   ##  ##        ##  #   ##    # #    # #  #  ## #     #    ##      #   ### ##        ###  #          #                ### # #      #  # # ##     #    #           #    ###  #
#          ## ##         #      #   ## #    #  # #   # #  ###### #    ##  ##  ##  #   #  ####               #  # # #   #  #        #  ### ## #   #    ####  # # # #        #     # ##         ###  ##  # #     #  ##  # # #  #        #    #   #    #  # #
#  ##      ####   ####  ###   #   ##   ###    #       # # ##   ##         #      ###  # #           # #     # #  #    #   # # ## #    # ## #     #   #  #  #    #    ##         # ##  # #  # #                #  ## ###  #  # #    # ###    #  #   #     #
##   # ##    #       #       #  ## ##  # #          #     # #   #    # # ###     ###  ## # # #     #   # #    #       # # #  #    # #   #    ##      #    # #  #     #     ## #   #           #    # ##  # #                 ##   #      #       # ## ####
# #        # # #     # #   # #      #  ## # ##   #   ## #           ## #   # #    #     #           ##   #  #  ## #   #   # #       #      ##  ##  #      #  #   #     # #    #  # #  #   # #  #    ###    # #  ## #       ##        #  # # ###  ##      #
##  # #    ## #     #    #     #  #  # # #       #    #    ###    #### #  ##  #    ##    #####    #  ##   #      ##    # # #  #       #    ## # ## #    ##  ## ###         #    ## # #                        #       # # ##   ## # #        # #     #   #
#   #   #    #    #     #  ####  #  #  #       # #  #   #####  #     #    #     #  ##    ## # ####   # #         # #  #  ##     ###   # #     #     # ##    #  #     #    # #     #    ##  # ##         #     #       # ##       #   #  # #      ## ##   #
###   #     #    #  ##     #  ## ##    #   #  # ##  ##         #  #         #   ## #  #   #     #   ## #  #    #      #    # #  #   # ##         #  #       #    #   # # # ##              ###  #### # ####    ###   #    # #    # ## #         # #    ###
##   #  #     # ##  ##   #  # #  #        #   # #           #    #           # ##      #   # # ###       #         #  ### #   #    # #      ### ##   #      #         ## #        #  #                 ### ##   #       # #  #             # #        ## #
#         #   #  #                  ## #   # #  #     #    # ##   #       #   # #  ##  #    ##     ## # #        # #   #  ##  #    ## #   ## ### #     # #     ####     #   # #     ##          ##    #    #   #         ##   # #                     ## #
#           # #            #####     # # ###         ##       ##  #  #  #  ##       # #  ## #  #         # #  #    #   ##  #  # #     #       ##  ####   #      ## ##  ##   #     #         ## #  ### #       # #   # # # # # #     # #####    #  # #    #
# ## #  ##          #  #    ##   #      ## #  # #    #   #     #     #  #  ##    # #   # #     # # ##      ##    # ####      # #     #  #                # #      # ##    # #       ##  ## #  #  #  ## #    ##    ##    ##              #  #  #       # ##
#  # #     ##   #  ##         ##          #    #  #     # ###  #           # ### # #  #    #  #   #         # # ##      ##   ##   ## #  #           ##   #    # #   #    #  #     #           #     # ##    #   #      #  #       #   ## # ## #  # # #   #
####     #  #      ##     #### ##     ## #      # #   # ### #    ##      #       # #  ## #  #   ##  # ### # #               # # # ##              #  #   #       #   #            # #     # #  ####     # ##        #         #   # #         #  ##      #
#  ##           #  ## #     # #  ##    #   #                       ## ## #    #   #  #  #    ##      # ##  ##   ##### ## ##    ###   #     ###       ####### # #         #   ###  #     ##       # #  # #  # ##     # # ##   ###    ##         ##  #     #
##  ## ##  # ##  # #     ##   #   #      ##    #      #   # #     ###     #### #   #   #         ##    #      ##   #   ### # # #   # # #  #            #  ##   #  #         ## ## # # # ##      #    #   ##  # #    # #  #   ##     #  #   #    #     #  #
# #  ## #     #   # # #    # ## ## #     ##   ##    ##     # # #####  ## ###   #####      #  #####  #         ##    #     #   ##       ##  ### # #      ##     #         # ##   #   #  # #   #     #   #       #    #    #      ###   ##   # ##   # ###  #
#     # # #  ## # #      # #         ###    #    #   #   # #  ### #  #  # ###      #       ####  #        # #  # # #  #           # # #       # # ###    # # #       #    ## ##  ##  ###    #         #  #####   ##  #    # #  ##   #  # ##  #     #     #
# #      #   #       #   #  #####  ##     # # #  ##   #  ## ###            ###         #  #  #  #     #   #  #   ##     #     # #            #  #   #        ##  # #  ### #  # ###   #    ##      #   # ## ## #    #  ##      #  ####  # #    ###    #   #
# ####  # ### #    #####   #   ### ##     ##    #  #  ##         #   # ### #   ##  #    #               #   ##  ##   #   #       #  # #    ##      #        #  #      #  ###         #        ## ####  ###           #    #      #            #      #   #
#        #           #   #      #  #             # #        #   ## # #   ## #   #   #  ####     ##  #             ##      ### ##         ## #      #  ## ## ### #   # #    # ##  ##    #    #         ## #     #      #          # #  #   # #  # ##  #   #
##  #       # #     # #         # #  #        #  ## ###  ## ###   #          #    ## #     #  #      #          # # #      #      ##  #  ## ##   #  # # #      #                    #  ##     #      ###  ##                 #  #   #  #   #   #         #
## #   #  ##       #    ##      #       #   #    #       # ##        ##  # #     # # #   #   ###          ##  ##  #   #  #     #  # #        #  ##    #  #  ##   ##          ##    #       ### ##  #  #       #   #        ##         #    #    ###      #
###      # #  #  #    # #      # ## #     ## #   # #           #  #         #          # ###  #   #  # #     #    #       ## ##  #     #  #          #    ###  #        # #           # #    #  # ##         #    #  #   # #     #       #     # #    #  #
#    # #   #  # #  #   #         ##### ##  #   #  # # ##   # # #     ##    ## ##   ## #        ###              #                  #         #       #    #  #   ##     # ## # #    ##    #    #      #   ## ##    ### #   #   #    #      # #   ##   #  #
# #      #      # ##   #  #   #    # #     # #     ###  ##    #  #  ##     #  ##        #   ##  #     #   # #              #     #   #    # ###   ## # ##       # #   # # ##  #   ### #  ## ## #  #  #    #  #   #   ## #  #  #    ###   #  ## #     ### #
#  ###  # #        # #  #   # ###   # # # #  #   # ####   #  #  # ##   ## #  #  #     ##   ## ## #   ##  #  ##          # # #   # #           #     ###  #  ##     #   ## # #   ## # # #                         #        #                #  #   # #  ###
##   ##  ##  #  #       ##   #  # #     #  ## #     # #   #     ##            #  #      ##  #   #      #  #   # ##     #     # # # # # #  ### #     #             # #        #  # ##       ## #  #  ####     #            ##       # #       # #    #    #
#  #      ##           ### #        #         # #  #         # # ##  # # #     ##          # #   #      ###  #  #      # # #         #   #  #  ## #    ###       #    # ##  ##   # # #          #   # # #   ##  #       # ##   ##             #   #   ####
#       #      ##  ##   #  #  ## #       ## #      # ##  # #              ##   #      #  #      ###  #      #    # #      ##      # #    ##   #   ##     #   #        #       ##    #  # #    ##  # ##       #      #   #  # ##    #      #  ###   #    ##
#    #    ## ## #    #       #     ##  #     ##  #  #     # ##      #    # # # #    #      ## # #  #   #  # ##      ##    #   ## ##   ## #  ## # #   # ##  #  ##   #  # #  #  ##   ##         ## #  ## ##  #  ##   #            #   ###     #          # #
#####                 ##   ## #    # #       #       #  #     #     ##   #                # # # #     #     #        # #### ## # #   # ###  #                    #    ### #        ###        #  #  #### ## # #  #  #  #  # ## # #   #    #  ####  ###   #
### #    # #       #   ## #   #        # #  # #  #  #            # ##     ## #    ##  #                 ##   #       ## #  # ### #     #   # #    #  #     #          #    #   #     #  # # #####     ###  #  #    #    #  # #  #### # ##   ######    #  #
# # #    # ## ###  # # ##  #   #     #              #   #     ##    # # #    #      #   #     # #   ## ##    #####  #   #### # # #     #      #  #  ###  ## # #      # #          #  #   #   ## #  # #     ##   ###   #              #  ##   # ##        #
# ##  #      #      ###  #   #    # #         # ## # #    #     #        ##  #         #     #  # ##  #    #  # #   #    # #  # #        ###       ## ##     ##                 #    # #  ## # #   #   #  #   #### ##   ##        # #  # ##   ## # ###   #
##      # ##  ##     #####  #      #  #    # #   ##  ##  #  # #    #    ##  ###      #      ## ##      ###         # ##    #   ## ####  ##         ##      # ###   ##     #         # #    #  ###   #       ##  ## ##    #   #  ###   ##      #   #  ### #
###    # #     #   ##  ##    ##    ##    # ##    #   #   # #     ####   # #  ###  # #  #   # ## #####  #        #     #  #       #          # #     #    ## #       # #            ##    #  #     #    #  # #  # ### ######    #      # ##  ###   #      #
#   #     #    ## #  ###      ## #       #     # #       # #        # #    ## ##    #  ###   ##   # #   #     # #     # ###  #   ### #       #      # #  ## #    #   #   #    #    # ##   #  ###  #    # #        # ##   # # #   ####    #   ##  #  ###  #
###  #            #     #    ##   # #   # #       #  # #     #   # ####    # # #   ##    # #     #            #   ###   #     #   #         #     ##      #    #  #   ##   #  ##  #   ## #       # # ## # ####               #       #    #            # #
#      #      #        #     #  #      # # ##    #   #  ###   ##   ##          #   ##        # # #    #   # #    # ####  ###  #    # #  ##    #      #       #   # #  ##   ##       #          #   # #            #  ## #  #  # #     # #  # # ##  #  #  #
#   #  #   ###  # # ##   #     ##        ##  #  #  # #    ##     # #  ##  #  #      ##       ###    # ##  # # # #   #   #   #  #        #                  #    # # #          #          ##   #       #   #          ##     #        ## ###     # #    ##
#       ###            ### # ##  # ##     ## # #  ##          ## #     ##  #  ##     #  ### #     ##      #      ##  # # # # #  ##  #   # # #    #   #           #  # #  ##      #     ####   ##             #   #   #   #          # ##    # ##  #  # # #
#          #     #  ##    ##  #          # #     #       ## #    #    #     ##   ##      # # ##      # ##  # ##   ## #    #  # #  #  #   ##  #   ##  #      #       ## ##  #     ### #  ##       # #    #    ## ##  ## #  #   # # #  # #   #### # ##  #  #
##          #     #  # #          ##  ## #   #  ## ##   #   #    #  #    ##      # #       #   ##  #   ###    #    #    #    # #    ##  #       ## #   #   ##    # ## #    #    #   #  ## #      #     ####  #         # #  # #    #     #  # # #        #
##  #    #   #  #  #      ## # #    # #       #   #  #     ##      #       #      #  ## #    #    ##   ##          ##   ## ##   # #     #  #     #      # #    # #   #                #    #       #   #    #   #  #### ##      ##   # #     # #      ## #
##### ##  # #            #  #  ### #  #  #    #    # # #      #       ##  ##    ###     #            # #      #        #   #     #          ## #           # #             #      # ##  #  # #  ##   #      #      #  ##    ##  #   #   #     #    ##   ##
###    ##  #   ##   # #      #### # #       # # # # # # #         #  #   #                   ##                #    #   #   ##   # ##     ##    #   ##  #    #     #  #  # # #  #       #               #        # #           # ##   ##   ## #  #       #
#     ## #  ##    #  ####     #   #   ##     ##    #  #  # ##            #  #   #  # ###   #     ##   # #                 #  #   #  #    ##  #    # #     #   #      ####       #     #   #  ### #  #        # ##      #   ####            #    #        #
##  #  #   #   # ##   # #  ##  #  ###  #  #   # #               #   #  #  #   ###  #     #  #         #  ##           #   #      #     ##  #   ## #       ## #      #   ##   #  #  #    ###  ## #             #   # # # ##  #  ###   # ## ## #  #   #   ##
#   # #   #   #  # ##  ##  ##        ## #   ## #      ## ##   #  #  #    # ##   #### ##          #       #  #        ##   # ## #    #  #    #    #       # #  ##   ##    #  #    #   #  #   # ####      #      #  ##  #  # #        #   #   ##  #  # #  ##
###  #   ##     ##      #   # #        # #    #     # ##  ##  ## # #  #  #       ##   #  # # ##  #   ##  #    #  # #      ##   #  ###   #    # #    ##       #       #  ##  #   # #          #   #     #             ##### #   #   #  #          #  #   ##
#   ###    # #  #   ##  #   #  #         # #         #####     ##           #      ##      # ## ##   #      #    # # # #   #     #  # #  # #    #  #     ## #       #     #                  #    ## #  #  #     #   # #  # ## #   # # # ##             ##
##########################################################################################################################################################################################################################################################
""".strip()


print(f"Processando labirinto :")
maze_content_str = labirinto2

if maze_content_str is not None: # Verifica se a leitura do arquivo foi bem sucedida
    returned_time_ms = solve_maze(maze_content_str) # Chama a nova função de interface
    print(f"-> Chamada a solve_maze para o labirinto concluída.")
    print(f"   Tempo retornado: {returned_time_ms:.4f} ms.")
    print(f"   Arquivo gerado: 'output_capivaras.txt' (verifique seu conteúdo).")
else:
    # read_maze_from_file já imprimiu uma mensagem de erro
    print(f"   Não foi possível ler ou processar o labirinto. Pulando.")

print("\n" + "="*60 + "\n")