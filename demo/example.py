import pyalign.utils
import pyalign.solve
import pyalign.gaps

pf = pyalign.utils.SimpleProblemFactory(pyalign.utils.BinarySimilarity(eq=1, ne=-1))
solver = pyalign.solve.Solver(gap_cost=pyalign.gaps.AffineGapCost(0))
alignment = solver.solve(pf.new_problem("INDUSTRY", "INTEREST"))
print(alignment.score)
alignment.print()
