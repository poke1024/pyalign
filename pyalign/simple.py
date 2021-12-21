"""
A simple high level API, which should be sufficient for many simple use cases.
"""


from pyalign.gaps import LinearGapCost
from pyalign.solve import Alignment, GlobalSolver, SemiglobalSolver, LocalSolver
from pyalign.problems import binary, alphabetic
from typing import List
from functools import lru_cache


_codomain = {
    False: Alignment,
    True: List[Alignment]
}


@lru_cache(maxsize=8)
def _make_binary(eq, ne, direction):
    return binary(eq=eq, ne=ne, direction=direction)


@lru_cache(maxsize=8)
def _make_solver(solver_class, gap_cost=1, return_all=False):
    return solver_class(
        gap_cost=LinearGapCost(gap_cost),
        codomain=_codomain[return_all])


def _alignment(solver_class, a, b, score=None, gap_cost=1, return_all=False, **kwargs):
    if score is not None:
        if 'eq' in kwargs or 'ne' in kwargs:
            raise ValueError("argument 'score' cannot be used with 'ne', 'eq'")
        alphabet = list(set([*a, *b]))
        pf = alphabetic(
            alphabet, score, direction='maximize')
    else:
        pf = _make_binary(
            eq=kwargs.get('eq', 1),
            ne=kwargs.get('ne', 0),
            direction='maximize')

    solver = _make_solver(
        solver_class=solver_class,
        gap_cost=gap_cost,
        return_all=return_all)

    return solver.solve(pf.new_problem(a, b))


def global_alignment(*args, **kwargs):
    return _alignment(GlobalSolver, *args, **kwargs)


def semiglobal_alignment(*args, **kwargs):
    return _alignment(SemiglobalSolver, *args, **kwargs)


def local_alignment(*args, **kwargs):
    return _alignment(LocalSolver, *args, **kwargs)
