# Creating initialization fields

Each folder contains a pre-made script utilizing MPI to create a converged initialization field for the given Reynolds number and copy the necessary files to the necessary folders in the DRL code. These also include pre-determined mesh configuration files (`.geo`) , with the same meshes as used in the thesis. The **custom** folder is added to be ready for new Reynolds numbers, new mesh refinement levels, and more.

## Running the script: Create_init_fields.py

We first give a brief instruction on how to create initial fields, and then give a more in-depth explanation of the different parts of the script, and what the main configurable parameters of the script are.  

1. First run the script in serial as: `python3 create_init_fields.py` 
2. This will create the necessary folders to store mesh files, and calls `gmsh` (http://gmsh.info/) to create a mesh (`.msh`) file from the `.geo` file given in the `ReXXX/pinball/` folder. To create a different mesh, edit the `.geo` file and create converged initialization fields by running the script. 
3. The simulation is then stopped and can now be restarted with MPI activated to speed up the fluid mechanical simulation for creating initial velocity and pressure fields. 
4. Run the script with MPI as: `mpirun -np num_procs python3 create_init_fields.py`

The script is built up as:

* Lines 22-30: Define folder names to store mesh and dump files. 
* Lines 45-46: Define the inflow velocity and numerical timestep `dt`. (Can also be done explicitly in the FlowSolver setup)
* Lines 49-67: Setup the flow solver, here we can also edit the value of `mu` to increase or reduce the Reynolds number. (Lower `mu `= higher Reynolds, `mu=0.01-> Re=100`, `mu=0.005 -> Re=200`)
* Lines 71-72: Introduce drag and lift probes for the cylinders.
* Lines 75-117: Calculate positions of pressure probes around the cylinders and in the wake behind them.
* Lines 121-129: Some dump routines for the velocity and pressure fields.
* Lines 141-168: Main part of the script where the actual simulation is done. We also dump values to a `.csv` file and to the terminal. `gtime` specifies how long the simulation should run for, corresponding to `simulation_duration` in `simulation_base/env.py` of the DRL code.
* Lines 171-187: Write the final velocity and pressure field to `.xdmf` files which are used as initialization fields for DRL simulations. We also add a print statement for comparing timing results using different number of MPI processes, and copy the initialization fields and mesh files to the correct folders, ready to be implemented in the DRL code.