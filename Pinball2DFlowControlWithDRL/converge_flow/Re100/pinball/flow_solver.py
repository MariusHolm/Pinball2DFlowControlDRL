from pinball.utils import as_mpi4py_comm, unique_points
from mpi4py import MPI as pyMPI
from dolfin import *
import numpy as np


class FlowSolver(object):
    '''
    Solve Navier-Stokes in the box domain below. Following arXiv:1812.08529
    domains 1, 3, 4, have (Uinfty, 0) prescribed on them while 2 is outflow
    boundary. Boundary conditions on the cylinders (C) are rotations.
                4
    ul(x)----------------ur(x)
    |     C                |
    | 1  C                 | 2
    |     C                |
    ll(x)----------------lr(x)
                3
    '''
    def __init__(self, comm, flow_params, geometry_params, solver_params):
        '''IPCS solver'''
        mu = Constant(flow_params['mu'])              # dynamic viscosity
        rho = Constant(flow_params['rho'])            # density

        mesh_file = geometry_params['mesh']
        # Load mesh with markers
        mesh = Mesh(comm)
        h5 = HDF5File(comm, mesh_file, 'r')
        h5.read(mesh, 'mesh', False)

        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        h5.read(surfaces, 'facet')

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        u_n, p_n = Function(V), Function(Q)
        # Starting from rest or are we given the initial state
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):
            if path in flow_params:
                comm = mesh.mpi_comm()
                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)

        u_, p_ = Function(V), Function(Q)  # Solve into these

        dt = Constant(solver_params['dt'])
        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        # Define variational problem for step 1
        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        U_infty = Constant((flow_params['U_infty'], 0))  # Scaled e_x
        # Define boundary conditions for non-cylinder boundaries
        bcu_inlet = DirichletBC(V, U_infty, surfaces, 1)
        bcu_top = DirichletBC(V, U_infty, surfaces, 4)
        bcu_bot = DirichletBC(V, U_infty, surfaces, 3)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, 2)

        # Finally we have rotations on the cylinders
        tags, exprs, info = FlowSolver.setup_cylinder_bcs(surfaces, 4)
        bcu_cylinder = [DirichletBC(V, expr, surfaces, tag) for tag, expr in zip(tags, exprs)]

        # All bcs objects togets
        bcu = [bcu_inlet, bcu_top, bcu_bot] + bcu_cylinder
        bcp = [bcp_outflow]

        As = [Matrix(comm) for i in range(3)]
        bs = [Vector(comm) for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solvers = [LUSolver(comm, A, 'mumps') for A in As]
        # Set matrices for once, likewise solver don't change in time

        gtime = 0.  # External clock

        # Things to remember for evolution
        self.cylinder_bc_exprs = exprs
        self.cylinder_bc_tags = tags
        self.cylinder_info = info
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        tags = tuple(map(int, tags))
        # Also expose measure for assembly of outputs outside
        self.ext_surface_measures = [Measure('ds', domain=mesh, subdomain_data=surfaces, subdomain_id=tag)
                                     for tag in tags]

        self.viscosity = mu
        self.density = rho
        self.normal = n
        # Finally the communicator
        self.comm = as_mpi4py_comm(comm)

    def evolve(self, bc_values):
        '''Make one time step with the given rotation magnitudes'''
        assert len(bc_values) == len(self.cylinder_bc_tags), (bc_values, self.cylinder_bc_tags)
        # Set rotation
        for expr, value in zip(self.cylinder_bc_exprs, bc_values):
            expr.A = value

        # Make a step
        self.gtime += self.dt(0)

        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        solution_okay = True
        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            try:
                solver.solve(uh.vector(), b)
            except:
                solution_okay = False

        solution_okay = solution_okay and not np.any(np.isnan(u_.vector().get_local()))
        solution_okay = solution_okay and not np.any(np.isnan(p_.vector().get_local()))
        # Reduce accross CPUs
        solution_okay = self.comm.allreduce(solution_okay, op=pyMPI.PROD)

        if not solution_okay: warning('Simulation gone wrong')

        u_n.assign(u_)
        p_n.assign(p_)

        # Share with the world
        return u_, p_, solution_okay

    @staticmethod
    def setup_cylinder_bcs(surfaces, tag):
        '''Discover cylinders and make rotation expression for them'''
        # By convention cylinders are labels after tag; local
        tags = [t for t in set(surfaces.array()) if t > tag]

        mesh = surfaces.mesh()
        comm = as_mpi4py_comm(mesh.mpi_comm())
        # Global tags are
        tags = list(set(sum(comm.allgather(list(tags)), [])))

        x = mesh.coordinates()

        # On each surface the value is given by customizing the following template
        rot_expr = lambda: Expression(('A*(x[1]-CY)/sqrt((x[0]-CX)*(x[0]-CX) + (x[1]-CY)*(x[1]-CY))',
                                       '-A*(x[0]-CX)/sqrt((x[0]-CX)*(x[0]-CX) + (x[1]-CY)*(x[1]-CY))'),
                                      degree=1, CX=0, CY=0, A=0)

        mesh.init(1, 0)
        values, cylinder_info = [], []
        for tag in tags:
            # Discover center points as center of mass of vertices lying
            # on the cylinder
            v_idx = sum((list(f.entities(0)) for f in SubsetIterator(surfaces, tag)), [])
            # Send the points to root
            global_circle_points = comm.gather(x[v_idx], 0)

            center_x, center_y, radius = (None, )*3
            # Let it compute the info
            if comm.rank == 0:
                global_circle_points = unique_points(np.row_stack(global_circle_points))
                center_x, center_y = np.mean(global_circle_points, axis=0)
                radius = np.linalg.norm(global_circle_points[0] - np.array([center_x, center_y]))

                comm.bcast((center_x, center_y, radius), 0)
            # Just listen
            else:
                center_x, center_y, radius = comm.bcast((center_x, center_y, radius), 0)

            expr = rot_expr()
            expr.CX = center_x
            expr.CY = center_y  # Leaving magnitude to the controller

            values.append(expr)
            cylinder_info.append((center_x, center_y, radius))
        return tags, values, cylinder_info


def steady_ns(surfaces, mu, rho, U_infty, f=Constant((0, 0)), w0=None):
    '''
    For the transient problem in FlowSolver a suitable intial condition
    is obtained by solving the stationary problem.
    '''
    # NOTE: the problem is solved fully coupled (as opposed to segragated
    # solve above). This leads to larger linear systems in Newton step-
    # it might be needed to solve on coarser mesh and then interpolate
    # initial conditions
    mesh = surfaces.mesh()

    Vel= VectorElement('Lagrange', triangle, 2)
    Pel = FiniteElement('Lagrange', triangle, 1)
    W = FunctionSpace(mesh, MixedElement([Vel, Pel]))

    w = Function(W)

    if w0 is not None:
        w.assign(interpolate(w0, W))

    u, p = split(w)
    v, q = TestFunctions(W)

    sigma = lambda u, p, mu=Constant(mu): -p*Identity(2) + 2*mu*sym(grad(u))

    F = inner(Constant(rho)*dot(grad(u), u), v)*dx - inner(sigma(u, p), sym(grad(v)))*dx
    F += inner(div(u), q)*dx
    F -= inner(f, v)*dx

    U_infty = Constant((U_infty, 0))
    # Do-nothing on outflow is enforced waekly
    bcs = [DirichletBC(W.sub(0), U_infty, surfaces, 1),
           DirichletBC(W.sub(0), U_infty, surfaces, 3),
           DirichletBC(W.sub(0), U_infty, surfaces, 4)]

    # Exprs here are 0 (no-slip) by default
    tags, exprs, info = FlowSolver.setup_cylinder_bcs(surfaces, 4)
    bcs.extend([DirichletBC(W.sub(0), expr, surfaces, tag)
                for tag, expr in zip(tags, exprs)])

    # Define the solver parameters
    snes_solver_parameters = {'nonlinear_solver': 'snes',
                              'snes_solver': {'linear_solver': 'lu',
                                              'maximum_iterations': 50,
                                              'report': True,
                                              'error_on_nonconvergence': False}}

    dw = TrialFunction(W)

    problem = NonlinearVariationalProblem(F, w, bcs, J=derivative(F, w, dw))
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters)
    solver.parameters['newton_solver']['relaxation_parameter'] = 0.2

    # Solve the problem
    niters, converged = solver.solve()

    # Components
    return w, converged
