# Maze Solver com Cython

Este projeto implementa um solucionador de labirintos utilizando o algoritmo de busca em largura (BFS), com otimizações em Cython para alto desempenho.

## Estrutura do Projeto

* `maze_solver_cy.pyx`: implementa o algoritmo BFS em Cython.
* `setup.py`: script de build para compilar o Cython.
* `team_capivaras.py`: script principal que contém a função `solve_maze()`.
* `requirements.txt`: lista de dependências do projeto.

## Requisitos

* Python 3.9+
* Compilador C (ex: `clang`, `gcc`)
* Ambiente virtual Python recomendado

## Instalação

1. Clone o repositório e acesse a pasta do projeto:

   ```bash
   git clone https://github.com/igorfwds/MazeSolver.git
   cd MazeSolver
   ```

2. Crie e ative um ambiente virtual (opcional):

   * **macOS/Linux**:

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * **Windows (CMD)**:

     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Windows (PowerShell)**:

     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Compile o módulo Cython:

   ```bash
   python setup.py build_ext --inplace
   ```

## Como Usar

Execute o script principal:

```bash
python team_capivaras.py
```

Dentro do script, a função `solve_maze()` é chamada com uma string representando o labirinto.

### Formato da String do Labirinto

* `' '` (espaço): caminho livre
* `'#'`: parede
* `'S'`: início
* `'E'`: fim

#### Exemplo:

```python
maze = '''
###############
#     #       #
# ### # ##### #
# #   #     # #
# # ##### ### #
# #     #   # #
# ##### ### # #
#     #   # #E#
### # ### # ###
#S# #     #   #
###############
'''

solve_maze(maze)
```

## Licença

MIT
