"""
A simple high level API, which should be sufficient for many simple use cases.
"""


from pyalign.gaps import LinearGapCost
from pyalign.solve import Alignment, GlobalSolver, SemiglobalSolver, LocalSolver
from pyalign.problems import binary
from typing import List


_codomain = {
    False: Alignment,
    True: List[Alignment]
}


def _alignment(mk_solver, a, b, eq=1, ne=-1, gap_cost=1, return_all=False):
    pf = binary(eq=1, ne=-1, direction="maximize")

    solver = mk_solver(
        gap_cost=LinearGapCost(gap_cost),
        codomain=_codomain[return_all])

    return solver.solve(pf.new_problem(a, b))


def global_alignment(*args, **kwargs):
    return _alignment(GlobalSolver, *args, **kwargs)


def semiglobal_alignment(*args, **kwargs):
    return _alignment(SemiglobalSolver, *args, **kwargs)


def local_alignment(*args, **kwargs):
    return _alignment(LocalSolver, *args, **kwargs)
