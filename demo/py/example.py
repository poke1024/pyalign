import pyalign.utils
import pyalign.gaps

pf = pyalign.utils.SimilarityProblemFactory(pyalign.utils.Binary(eq=1, ne=-1))
solver = pyalign.solve.LocalSolver(gap_cost=None)
alignment = solver.solve(pf.new_problem("INDUSTRY", "INTEREST"))
print(alignment.score)
alignment.print()
