"""Resume and use the environment in configuration"""

import sys
import os
import shutil
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Env2DPinball import Env2DPinball
import numpy as np
from dolfin import Expression
import math

import os
cwd = os.getcwd()

# Default simulation is Re = 100 with 80 actuations per episode.

# Define which Reynolds number to simulate.
Re = 100

# Define nb_actuations, False = 160 actuations for chosen Re, True = 80 actuations
actuations_80 = True

if Re == 100 and actuations_80:
    nb_actuations = 80

elif Re == 100 and not actuations_80:
    nb_actuations = 160

elif Re == 150 and actuations_80:
    nb_actuations = 80

elif Re == 150 and not actuations_80:
    nb_actuations = 160

elif Re == 200:
    nb_actuations = 160

elif Re == 'custom':
    nb_actuations = 80 # placeholder, standard 80 actuations per episode

else:
    sys.exit("\nChoose which Reynolds number to simulate!\n")


def resume_env(plot=False,
                dump=100,
                make_converge=False, # True if creating initial fields by running in serial. This will overwrite the provided initialization fields.
                random_start=False,
                single_run=False):

    U_infty = 1.
    number_cylinders = 3

    if Re == 100:
        simulation_duration = 70.0 #
        dt = 0.005
        mu = 0.01
        root_folder = 'mesh/re100'
        root = 'mesh/re100/geometry'

        # Baseline mean(mean_drag) and mean(sum_drag) calculated on the last 100 timeunits of convergence simulation
        mean_drag = -0.640088
        sum_drag = -1.920265

    elif Re == 150:
        simulation_duration = 70.0
        dt = 0.0025
        mu = 0.00667
        root_folder = 'mesh/re150'
        root = 'mesh/re150/geometry'
        mean_drag = -0.598862
        sum_drag = -1.796587

    elif Re == 200:
        simulation_duration = 70.0
        dt = 0.0003125
        mu = 0.005
        root_folder = 'mesh/re200'
        root = 'mesh/re200/geometry'
        mean_drag = -0.579101
        sum_drag = -1.737302

    elif Re == 'custom':
        simulation_duration = 70.0
        dt = 0.005
        mu = 0.01
        root_folder = 'mesh/custom' # same folder as Env2DPinball.py will save initialization fields.
        root = 'mesh/re100/geometry' # If custom .geo file used for mesh, change root to  path to .geo file, e.g. 'mesh/custom/geometry'

        # Re 100 results as placeholder
        mean_drag = -0.640088
        sum_drag = -1.920265

    else:
        sys.exit("Choose a Reynolds number for the simulation.")

    # DEBUG: Make sure correct Reynolds number and nb_actuations is used
    print("\nReynolds number=", Re)
    print("\nnb_actuations =", nb_actuations)

    if (not os.path.exists('mesh')):
        sys.exit("Missing mesh folder with initial fields. \nPlease initialize a flow from folder 'converge_flow/.")

    # Cylinder placements
    radius = 0.5
    x_center = [-np.cos(np.pi/6)*3*radius, 0, 0] # Cylinder centers form equilateral triangel with 3R sides
    y_center = [0, -1.5*radius, 1.5*radius] # 3R apart

    cylinder_center = []
    for i in range(number_cylinders):
        cylinder_center.append(np.array([x_center[i], y_center[i]]))

    geometry_params = {'cylinder_center': cylinder_center,
                       'cylinder_radius': radius}

    flow_params = {'mu': mu,
                  'rho': 1,
                  'U_infty': U_infty}

    solver_params = {'dt': dt}

    # DEBUG: Make sure correct mu and dt are used. Calculate how many solver steps to expect in debug.csv file
    print("\ndt =", solver_params['dt'])
    print("\nmu =", flow_params['mu'])
    print("\nnumber of solver steps =", simulation_duration/solver_params['dt'])


    # --------------------- PROBE POSITIONING ------------------------------- #

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


    # SAVE PROBE LOCATIONS
    output_params = {'locations': list_position_probes,
                    'probe_type': 'pressure'}

    optimization_params = {"num_steps_in_pressure_history": 1,
                        "min_rotation_cyl": -1.,
                        "max_rotation_cyl": 1.,
                        "smooth_control": (nb_actuations/dt)*(0.1*dt/nb_actuations),
                        "random_start": random_start}

    inspection_params = {"plot": plot,
                        "dump": dump,
                        "mean_drag": mean_drag,
                        "sum_drag": sum_drag,
                        "show_all_at_reset": False,
                        "single_run":single_run
                        }

    reward_function = 'plain_drag_lift'
    # Reward functions implemented in Env2Pinball.py:
    # reduce_drag: 'plain_drag_lift'
    # increase_drag: 'more_drag_simple_actuation'

    verbose = 0

    # Time spent per new action.
    duration_execute = simulation_duration / nb_actuations
    print("\nDuration per execute =", duration_execute)

    # Numerical solver steps per episode
    simulation_steps = simulation_duration / dt
    print("Steps per episode =", simulation_steps)

    # Create initialization fields in serial. (Can also be done with MPI in folder 'converge_flow/')
    if (make_converge):
        n_iter = int(1200.0 / dt)
        print("Make converge initial state for {} iterations.".format(n_iter))
    else:
        n_iter = None

    simu_name = 'Simu'

    env_2d_pinball = Env2DPinball(path_root=root,
                                    geometry_params=geometry_params,
                                    flow_params=flow_params,
                                    solver_params=solver_params,
                                    output_params=output_params,
                                    optimization_params=optimization_params,
                                    inspection_params=inspection_params,
                                    n_iter_make_ready=n_iter, # Calculate if necessary
                                    verbose=verbose,
                                    reward_function=reward_function,
                                    duration_execute=duration_execute,
                                    simu_name=simu_name,
                                    root_folder=root_folder)

    return(env_2d_pinball)
