from __future__ import print_function
from pinball.utils import as_mpi4py_comm
from mpi4py import MPI as py_mpi
import dolfin as df
import numpy as np
from dolfin import *


class DragProbe(object):
    '''Integral proble of drag over the tagged mesh oriented exterior surface n.ds'''
    def __init__(self, mu, n, ds, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.n = n
        self.ds = ds
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        # Stress
        sigma = 2*Constant(self.mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = dot(dot(sigma, self.n), self.flow_dir)*self.ds

        return assemble(form)


class PenetratedDragProbe(object):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, rho, mu, n, ds, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.rho = rho
        self.n = n
        self.ds = ds
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        mu, rho, n = self.mu, self.rho, self.n
        # Stress
        sigma = 2*Constant(mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = dot(-rho*dot(outer(u, u), n) + dot(sigma, n), self.flow_dir)*self.ds

        return assemble(form)

    
class PointProbe2017(object):
    '''Perform efficient evaluation of function u at fixed points'''
    def __init__(self, u, locations):
        # The idea here is that u(x) means: search for cell containing x,
        # evaluate the basis functions of that element at x, restrict
        # the coef vector of u to the cell. Of these 3 steps the first
        # two don't change. So we cache them

        # Locate each point
        mesh = u.function_space().mesh()
        limit = mesh.num_entities(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()
        # In parallel x might not be on process, the cell is None then
        cells_for_x = [-1]*len(locations)
        for i, x in enumerate(locations):
            cell = bbox_tree.compute_first_entity_collision(Point(*x))
            if -1 < cell < limit:
                cells_for_x[i] = int(cell)

        V = u.function_space()

        found_cells = np.repeat(-1, len(locations))
        # We insist that all the points are inside!
        comm = as_mpi4py_comm(V.mesh().mpi_comm())
        comm.Allreduce(np.array(cells_for_x), found_cells, op=py_mpi.MAX)
        outliers = list(np.where(found_cells < 0)[0])
        assert not outliers, 'Probes outside mesh %r' % locations[outliers]

        element = V.dolfin_element()
        size = V.ufl_element().value_size()
        # Build the sampling matrix
        evals = []
        for x, cell in zip(locations, cells_for_x):
            # If we own the cell we alloc stuff and precompute basis matrix
            if cell > -1:
                basis_matrix = np.zeros(size*element.space_dimension())
                coefficients = np.zeros(element.space_dimension())

                cell = Cell(mesh, cell)
                vertex_coords, orientation = cell.get_vertex_coordinates(), cell.orientation()
                # Eval the basis once
                element.evaluate_basis_all(basis_matrix, x, vertex_coords, orientation)

                basis_matrix = basis_matrix.reshape((element.space_dimension(), size)).T
                # Make sure foo is bound to right objections
                def foo(u, c=coefficients, A=basis_matrix, elm=cell, vc=vertex_coords):
                    # Restrict for each call using the bound cell, vc ...
                    u.restrict(c, element, elm, vc, elm)
                    # A here is bound to the right basis_matrix
                    return np.dot(A, c)
            # Otherwise we use the value which plays nicely with MIN reduction
            else:
                foo = lambda u, size=size: (np.finfo(float).max)*np.ones(size)

            evals.append(foo)

        self.probes = evals
        # To get the correct sampling on all cpus we reduce the local samples across
        # cpus
        self.comm = comm
        self.readings = np.zeros(size*len(locations), dtype=float)
        self.readings_local = np.zeros_like(self.readings)
        # Return the value in the shape of vector/matrix
        self.nprobes = len(locations)

    def sample(self, u):
        '''Evaluate the probes listing the time as t'''
        self.readings_local[:] = np.hstack([f(u) for f in self.probes])    # Get local
        self.comm.Allreduce(self.readings_local, self.readings, op=py_mpi.MIN)  # Sync

        return self.readings.reshape((self.nprobes, -1))

    
class PointProbe2018(object):
    '''Perform efficient evaluation of function u at fixed points'''
    def __init__(self, u, locations):
        # The idea here is that u(x) means: search for cell containing x,
        # evaluate the basis functions of that element at x, restrict
        # the coef vector of u to the cell. Of these 3 steps the first
        # two don't change. So we cache them

        # Locate each point
        mesh = u.function_space().mesh()
        limit = mesh.num_entities(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()
        # In parallel x might not be on process, the cell is None then
        cells_for_x = [-1]*len(locations)
        for i, x in enumerate(locations):
            cell = bbox_tree.compute_first_entity_collision(Point(*x))
            if -1 < cell < limit:
                cells_for_x[i] = cell

        V = u.function_space()

        found_cells = np.repeat(-1, len(locations))
        # We insist that all the points are inside!
        comm = as_mpi4py_comm(V.mesh().mpi_comm())
        comm.Allreduce(np.array(cells_for_x), found_cells, op=py_mpi.MAX)
        outliers = list(np.where(found_cells < 0)[0])
        assert not outliers, 'Probes outside mesh %r' % locations[outliers]

        element = V.dolfin_element()
        size = V.ufl_element().value_size()
        # Build the sampling matrix
        evals = []
        dm = V.dofmap()
        for x, cell in zip(locations, cells_for_x):
            # If we own the cell we alloc stuff and precompute basis matrix
            if cell > -1:
                basis_matrix = np.zeros(size*element.space_dimension())
                coefficients = np.zeros(element.space_dimension())
                # NOTE: avoid using DOLFIN's restric; instead reach into
                # function's vector
                cell_dofs = dm.cell_dofs(cell) + dm.ownership_range()[0]

                cell = Cell(mesh, cell)
                vertex_coords, orientation = cell.get_vertex_coordinates(), cell.orientation()
                # Eval the basis once
                basis_matrix.ravel()[:] = element.evaluate_basis_all(x, vertex_coords, orientation)
                basis_matrix = basis_matrix.reshape((element.space_dimension(), size)).T
                
                # Make sure foo is bound to right objections
                def foo(u_vec, c=coefficients, A=basis_matrix, dofs=cell_dofs):
                    # Restrict for each call using the bound cell, vc ...
                    c[:] = u_vec.getValues(dofs)
                    # A here is bound to the right basis_matrix
                    return np.dot(A, c)
            # Otherwise we use the value which plays nicely with MIN reduction
            else:
                foo = lambda u, size=size: (np.finfo(float).max)*np.ones(size)

            evals.append(foo)

        self.probes = evals
        # To get the correct sampling on all cpus we reduce the local samples across
        # cpus
        self.comm = as_mpi4py_comm(V.mesh().mpi_comm())
        self.readings = np.zeros(size*len(locations), dtype=float)
        self.readings_local = np.zeros_like(self.readings)
        # Return the value in the shape of vector/matrix
        self.nprobes = len(locations)

    def sample(self, u):
        '''Evaluate the probes listing the time as t'''
        u_vec = as_backend_type(u.vector()).vec()  # This is PETSc
        self.readings_local[:] = np.hstack([f(u_vec) for f in self.probes])    # Get local
        self.comm.Reduce(self.readings_local, self.readings, op=py_mpi.MIN)  # Sync

        return self.readings.reshape((self.nprobes, -1))


PointProbe = PointProbe2017 if df.__version__ == '2017.2.0' else PointProbe2018

# To make life easier we subclass each of the probes above to be able to init by
# a FlowSolver instance and also unify the sample method to be called with both
# velocity and pressure
class PressureProbeANN(PointProbe):
    '''Point value of pressure at locations'''
    def __init__(self, flow, locations):
        PointProbe.__init__(self, flow.p_, locations)

    def sample(self, u, p): return PointProbe.sample(self, p)


class VelocityProbeANN(PointProbe):
    '''Point value of velocity vector at locations'''
    def __init__(self, flow, locations):
        PointProbe.__init__(self, flow.u_, locations)

    def sample(self, u, p): return PointProbe.sample(self, u)

    
class DragProbeANN(DragProbe):
    '''Drag on ith-cylinder'''
    def __init__(self, i, flow, flow_dir=Constant((1, 0))):
        DragProbe.__init__(self,
                           mu=flow.viscosity,
                           n=flow.normal,
                           ds=flow.ext_surface_measures[i],
                           flow_dir=flow_dir)


class LiftProbeANN(DragProbeANN):
    '''Lift on ith cylinder'''
    def __init__(self, i, flow, flow_dir=Constant((0, 1))):
        DragProbeANN.__init__(self, i, flow, flow_dir)


class PenetratedDragProbeANN(PenetratedDragProbe):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, i, flow, flow_dir=Constant((1, 0))):
        PenetratedDragProbe.__init__(self,
                                     rho=flow.density,
                                     mu=flow.viscosity,
                                     n=flow.normal,
                                     ds=flow.ext_surface_measures[i],
                                     flow_dir=flow_dir)

class PenetratedLiftProbeANN(PenetratedDragProbe):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, i, flow, flow_dir=Constant((0, 1))):
        PenetratedDragProbe.__init__(self,
                                     rho=flow.density,
                                     mu=flow.viscosity,
                                     n=flow.normal,
                                     ds=flow.ext_surface_measures[i],
                                     flow_dir=flow_dir)
