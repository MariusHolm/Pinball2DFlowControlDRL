from __future__ import print_function

from pinball.msh_convert import convert
from pinball.flow_solver import FlowSolver
from pinball.utils import mpi_comm_self, mpi_comm_world
from pinball.probes import (PenetratedDragProbeANN as DragProbe,
                            PenetratedLiftProbeANN as LiftProbe,
                            PressureProbeANN as PressureProbe)

import matplotlib.pyplot as plt
from collections import deque
from mpi4py import MPI
import subprocess, os, sys
import dolfin as df
import numpy as np
import csv
import math
import time
import shutil

# Converged fields are provided in folders named re100, re150, re200.
converged_flow_folder = 're200_new' # renamed to avoid overwrite

if not os.path.exists(converged_flow_folder):
    os.makedirs(converged_flow_folder)

# Mesh generation
msh_file = '/'.join([converged_flow_folder, 'geometry.msh'])
h5_file = '/'.join([converged_flow_folder, 'geometry.h5'])
xml_file = '/'.join([converged_flow_folder, 'geometry.xml'])



if not os.path.exists(h5_file):
    # NOTE: Mesh generation must be done in serial / single threaded
    assert MPI.COMM_WORLD.size == 1
    subprocess.call(['gmsh -2 %s -o %s -format msh2' % ('pinball/geometry.geo', msh_file)], shell=True)
    assert os.path.exists(msh_file)

    mesh = convert(msh_file, h5_file)
    assert os.path.exists(h5_file)
    sys.exit("Mesh and .h5 created in serial. Restart simulation with MPI:\n mpirun -np number_threads python3 create_init_fields.py\n")


dt = 0.0003125
U_infty = 1.

# Solver setup
solver = FlowSolver(
    # Suppose you exec this as `mpirun -np 3 python test_flow_solver.py`
    # then
    #
    # if comm is mpi_comm_self() then you have 3 copies of the same
    # simulation running on differents procs (below there will be 3 prints
    # as withing its own communicator the (single) process is always root
    # with rank 0.
    #
    # if comm is mpi_comm_world() then there is once simulation where the
    # workload of assembly/solving is shared among 3 proces (there will
    # be one print below)
    comm=mpi_comm_world(),
    flow_params={'mu': 0.005,
                 'rho': 1,
                 'U_infty': U_infty},
    solver_params={'dt': dt},
    geometry_params={'mesh': h5_file}
)

ncyls = len(solver.cylinder_bc_tags)
# Probe setup for reward
drag_probes = [DragProbe(i, solver) for i in range(ncyls)]
lift_probes = [LiftProbe(i, solver) for i in range(ncyls)]

# ------------------ PRESSURE PROBE SETUP FOR STATES -------------------- #
number_cylinders = 3
# Cylinder placements
radius = 0.5
x_center = [-np.cos(np.pi/6)*3*radius, 0, 0] # Cylinder centers form equilateral triangel with 3R sides
y_center = [0, -1.5*radius, 1.5*radius] # 3R apart

# MAIN GRID AFTER CYLINDERS
list_position_probes = []

positions_probes_for_grid_x = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]
positions_probes_for_grid_y = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]

for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))

# SQUARE PROBE GRID BEFORE AND PARALLEL TO CYLINDERS
positions_probes_for_grid_x = [-0.5, -0.25, 0, 0.25, 0.50]
positions_probes_for_grid_y = [-2.5, -2.0, -1.5, 1.5, 2.0, 2.5]

for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))

# SQUARE PROBE GRID PARALLEL TO FIRST CYLINDER BEFORE THE OTHER 2 CYLINDERS
positions_probes_for_grid_x = [-1.5, -1.25, -1.0, -0.75]
positions_probes_for_grid_y = [-2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5]

for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))

# CIRCULAR PROBES AROUND CYLINDERS
list_radius_around = [radius + 0.1, radius + 0.3]
list_angles_around = np.arange(0, 360, 10)

for i in range(number_cylinders):
    for crrt_radius in list_radius_around:
        for crrt_angle in list_angles_around:
            angle_rad = np.pi * crrt_angle / 180.0
            list_position_probes.append(np.array([x_center[i] + crrt_radius * math.cos(angle_rad), y_center[i] + crrt_radius * math.sin(angle_rad)]))

pressure_probes = PressureProbe(solver, list_position_probes)

# Safe guard io by giving file unique name based on rank in world process
# if this is simulation with comm self
if solver.comm.size < MPI.COMM_WORLD.size:
    world_rank = MPI.COMM_WORLD.rank
    u_out, p_out = df.File('./results/u_%drank.pvd' % world_rank), df.File('./results/p_%drank.pvd' % world_rank)
else:
    u_out, p_out = df.File('./results/u.pvd'), df.File('./results/p.pvd')

# Initial state
u_out << solver.u_
p_out << solver.p_

is_root = MPI.COMM_WORLD.size == 1 or solver.comm.rank == 0

# Want to remap up rotation over a few steps of the solver (Only for non-zero rotation)
rot = lambda A, t: A*t/(100*dt) if t < 100*dt else A

counter = 0 # numerical timestep counter

start = time.time() # Start timing for timing comparison between number of threads used

# Save drag/lift values to .csv file for possible debug and visualization.
with open('results/drag_lift.csv', 'w+', newline='\n') as csvfile:
    fieldnames = ['time', 'drag0', 'drag1', 'drag2', 'lift0', 'lift1', 'lift2']
    writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # gtime = 1200 for initialization fields used in Master project
    while solver.gtime < 1200:
        counter += 1
        t = solver.gtime
        # Input the rotation magnitude into the lambda function "rot".
        # We want to create a baseline where no rotation of the cylinders take place
        u, p, status = solver.evolve([0, 0, 0])

        # choose frequency of .pvd/.vtu dumps
        if counter % 100 == 0:
            u_out << u
            p_out << p
            # Instantaneous
            i_drags = [dp.sample(u, p) for dp in drag_probes]
            i_lifts = [lp.sample(u, p) for lp in lift_probes]
            i_pressures = pressure_probes.sample(u, p)[[2, 3, 4, 5]].T

            # Only root process prints drag/lift/pressure values
            if is_root:
                df.info('t %g' % solver.gtime)
                df.info('\tDrag: %r' % i_drags)
                df.info('\tLift: %r' % i_lifts)
                csvfile.write('%g,%r,%r,%r,%r,%r,%r\n' % (solver.gtime, i_drags[0], i_drags[1], i_drags[2], i_lifts[0], i_lifts[1], i_lifts[2]) )

# After flow initialization is finished we save the flow state to use as initial flow of controlled flow simulation.
encoding = df.XDMFFile.Encoding.HDF5
mesh = df.Mesh(xml_file)
comm = mesh.mpi_comm()

u_init = '/'.join([converged_flow_folder, 'u_init.xdmf'])
p_init = '/'.join([converged_flow_folder, 'p_init.xdmf'])

df.XDMFFile(comm, u_init).write_checkpoint(u, 'u0', 0, encoding)
df.XDMFFile(comm, p_init).write_checkpoint(p, 'p0', 0, encoding)

end = time.time()
print(end - start)

if is_root:
    # When finished, copy the inital fields to the folders required for parallel DRL training.
    shutil.copytree(converged_flow_folder, '../../mesh/' + converged_flow_folder)
    shutil.copytree(converged_flow_folder, '../../simulation_base/mesh/' + converged_flow_folder)
