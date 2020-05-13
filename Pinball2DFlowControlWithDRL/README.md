## Getting started with simulations

The main code is located in **Pinball2DFlowControlWithDRL**. The simulation parameters are chosen in the **simulation_base** folder, in the file **env.py**.

The main script for launching training simulations is the **script_launch_parallel.sh** script. The script will take care of launching servers for simulations in addition to launching parallel trainings. The script will take 10 seconds per server to launch, e.g. 14 parallel trainings will take 140 seconds to launch.

It is recommended to run the code in the Docker container. The procedure for setting up a Docker container is described in the README in the `/Docker/` folder. The container will make sure all packages and dependencies are of the correct versions.

Explanations of how to launch the Docker container can be found in the **Docker** folder. 

It is recommended to run the code on Virtual Machines (VM) as the simulations take up to multiple days.

The **converge_flow/** folder contains MPI scripts for creating converged initialization fields. See the provided README in the folder for more details.

## Main commands

* **script_launch_parallel.sh**: Automatically launch training in parallel. The scripts is used as `bash script_launch_parallel.sh session-name first-port num-servers` which will start `num-servers` parallel DRL trainings. The simulations will run "independently" of each other, i.e. the agent will take different actions for each parallel simulation, and gather experience from each one. 
* **python3 single_runner.py**: Evaluate the latest saved DRL policy. After the training of the DRL agent has finished we evaluate the agent by simulating without exploration noise. The trained agent is saved in the folder `saver_data/` and is used to control the actions taken during the `single_runner.py` simulation. The trained agent can then be moved by copying the folder `saver_data/` to a new machine for evaluation, if that is desirable. 
*	**Note**: The evaluation must be done in such a way that the folder path is the same during evaluation, as during training. E.g. if the training was done in `/home/fenics/Pinball/` we cannot evaluate in `home/fenics/` even on the same machine, but the evaluation can be done on another machine if the path is the same `home/fenics/Pinball/`.
* **Note about pre-training**: As for **single_runner.py** evaluation having to be done in the same folder path as training, the same applies to pre-training. In addition, pre-training (e.g. training an agent at lower Reynolds number for 100-150 episodes), must be done with as many parallel environments as the training at higher Reynolds will be done with. 


## Static scripts - same for every simulation
* **Env2DPinball.py**: Custom TensorForce environment seen by the Reinforcement Learning agent. This is the main class of the repository with .csv dumping routines, reward functions for the agent, how the actions determined by the agent are applied, how an episode is reset, and more. The script is also able to create initialization fields of a simulation, but for more computationally intensive simulations it is recommended to use the MPI version in **converge_flow/**.
* **launch_parallel_training.py**: This script is used to configure the DRL agent and **how many episodes the training should run for**. 
  * Note: 2 small changes must be done for the code to run in the Docker container available for download. 
    * Line 67: In the dictionary specifying agent saver. Swap `frequency` with `seconds`.
    * Line 79: Comment out "`save_best_agent`" (moved to line 108)
    * Line 108: Uncomment `save_best_agent`.
* **launch_servers.py**: Checks that the ports, defined in the bash script by `[first-port, first-port + num-servers]`, are free and creates subfolders for each of the parallel training simulations. 
* **echo_server.py**: Echo server class sending data and instructions through a socket. 
* **RemoteEnvironmentClient.py**: Used to communicate with RemoteEnvironmentServer. A pair of RemoteEnvironmentClient (REC) and RemoteEnvironmentServer allows transmission of information through a socket. 
* **RemoteEnvironmentServer.py**: Takes a valid TensorForce environment and adds socketing so communication can take place.
* **flow_solver.py**: Numerical solver computing Navier-Stokes in the fluidic pinball system. 
* **msh_convert.py**: Converts a `.msh` file created by the *gmsh* software to an `.h5` file which is used to solve the fluid mechanics by `flow_solver.py`.
* **pinball_utils.py**: Redefines a few MPI commands for the Dolfin package of FEniCS..
* **probes.py**: Contains different types of Drag, Lift and Recirculation Area Probes. The utilized ones are imported to **Env2DPinball.py**.
* **utils.py**: Utilities used in `launch_servers.py` to check that ports are free.
* **simulation_base/start_one_server.py**: Is called by `launch_servers.py` and checks for a single port/host that it is free.


##  Simulation scripts - changes for every simulation
* **simulation_base/env.py**: This is the script that will change most between different simulations. The most important possible changes are described in the same order as they appear in the code.
	* First we can change which Reynolds number we want to simulate at and how many actions per episode a simulation an agent will apply. Pre-made options of simulating at Reynolds numbers 100, 150, and 200 (experimental and computationally intensive).
	* For each Reynolds number there are a few pre-defined simulation setups, but these can be changed according to new configurations. Note that `dt` is dependent on the mesh and the Reynolds number of the simulation. Changing it too much might break the solver. `mu` is varied to change the Reynolds number and complexity of the simulation.
	* Then comes the probe placements, which are completely customizable. Our setup uses a few sets of probes around each cylinder, in addition to probes before, beside, and in the wake behind the cylinders. 
	* `min/max_rotation` sets the magnitude limits of the rotations. `smooth_control` calculates a parameter which stops new actions from deviating very much from previous actions during the first steps, which might break the physics of the simulation. 
	* Finally we choose the reward function, two are implemented and correspond to all results in the thesis, but more reward functions can easily be defined in `Env2DPinball.py`. 
	* `make_converge` calculates the number of numerical solver steps to calculate if flow initialization is chosen to be done in serial, directly in the DRL code. (Flow initialization is recommended to be done in Parallel with MPI in `converge_flow/`).


## Visualizing results

To visualize learning, actions, drag, and lift curves we have created a few simple python scripts.

* **concat_results.py**: concatenates the average drag and lift values from each training episode and presents the learning curve of an agent. 
* **plot_drag_lift.py**: plots drag and lift values as they develop during the evaluation using **single_runner.py**.
* **action_plot.py**: plots actions taken during the **single_runner.py** evaluation, each cylinder separately and all actions together in the same plot. 

## Important folders

* **saver_data**: Folder where the agent is saved during training. 
* **results**: Will contain the velocity and pressure `.vtk/.pvd` dumps of the **single_runner.py** evaluation. The flow can then easily be visualized and observed using ParaView.
* **saved_models**: Contains `.csv` files of data dumps with drag, lift, reward, actions, etc. from **single_runner.py** simulations. These ` .csv` files are used by **plot_drag_lift.py** and **action_plot.py**.
* **env_xx**: One folder per parallel environment used during training. Each folder is essentially a copy of **simulation_base/env.py** with the addition of two folders.
  * **saved_models**: `.csv` dump files with same contents as described above, but now contains results of each training episode of the environment. (**concat_results.py** reads these `.csv` files and combines the results.)
  * **best_model**: `.csv` dump files only of the best episode/model of the environment.  
* **pre_trained_models**: `saver_data/` folders for pre-trained agents. 4 agents, both for increasing and reducing drag, have been pre-trained for 150 episodes using 14 parallel environments, and one agent has been trained using 60 parallel environments to increase drag for 150 episodes.