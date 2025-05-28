from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy # Para obter o include_dir do NumPy

extensions = [
    Extension(
        "maze_solver_cy", # Nome do módulo compilado
        ["maze_solver_cy.pyx"],
        include_dirs=[numpy.get_include()], # Necessário para typed memoryviews do NumPy
        # descomente as linhas abaixo para otimizações extras em alguns compiladores
        # extra_compile_args=["-O3", "-march=native"], 
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"})
)