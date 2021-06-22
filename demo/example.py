import pyalign.utils
import pyalign.solve
import pyalign.gaps

pf = pyalign.utils.SimpleProblemFactory(pyalign.utils.SimilarityOperator(neq=-1))
solver = pyalign.solve.Solver(gap_cost=pyalign.gaps.AffineGapCost(0))
alignment = solver.solve(pf.new_problem("INDUSTRY", "INTEREST"))
print(alignment.score)
alignment.print()
