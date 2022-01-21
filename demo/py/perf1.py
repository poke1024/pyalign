import pyalign
import time


def run(seq):
    pf = pyalign.problems.alphabetic(
        "ACGT",
        pyalign.problems.Equality(eq=1, ne=0),
        direction="maximize")

    solver = pyalign.solve.GlobalSolver(
        gap_cost=pyalign.gaps.LinearGapCost(0),
        codomain=pyalign.solve.Alignment)

    t0 = time.time()
    solver.solve(
        pf.new_problem(seq, seq))
    t1 = time.time()

    print("time (ms): ", (t1 - t0) * 1000)


run("ACGT" * 1000)
