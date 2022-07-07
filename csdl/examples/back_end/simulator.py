from csdl import Model, GraphRepresentation, SimulatorBase
from csdl.rep.graph_representation import GraphRepresentation
from csdl.opt.combine_operations import combine_operations
from csdl.opt.remove_dead_code import remove_dead_code
from csdl.opt.run_all_optimizations import run_all_optimizations


class Simulator(SimulatorBase):
    pass


model = Model()
rep = GraphRepresentation(model)
rep = remove_dead_code(rep)
rep = combine_operations(rep)
rep = apply_uq_reduction(rep)
sim = Simulator(rep)

# alternatively (short version)...
sim = Simulator(run_all_optimizations(GraphRepresentation(Model())))

sim.run()
print(sim['x'])
