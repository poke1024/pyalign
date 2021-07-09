import pyalign.problem
import pyalign.gaps

pf = pyalign.problem.ProblemFactory(
	pyalign.problem.Binary(eq=1, ne=-1), direction="maximize")
solver = pyalign.solve.LocalSolver(gap_cost=None)
alignment = solver.solve(pf.new_problem("INDUSTRY", "INTEREST"))
print(alignment.score)
alignment.print()
