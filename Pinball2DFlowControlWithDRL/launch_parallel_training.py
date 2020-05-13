import argparse
import os
import sys
import csv
import socket
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.env import resume_env, nb_actuations
from RemoteEnvironmentClient import RemoteEnvironmentClient


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

# Change dump=*** here if more or less frequent dumps are desired.
example_environment = resume_env(plot=False, dump=100)

use_best_model = True

environments = []
for crrt_simu in range(number_servers):
    environments.append(RemoteEnvironmentClient(
        example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
        timing_print=(crrt_simu == 0)
    ))

if use_best_model:
    evaluation_environment = environments.pop()
else:
    evaluation_environment = None

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=example_environment, max_episode_timesteps=nb_actuations,
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, estimate_terminal=True,
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=number_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=72000), # USE seconds INSTEAD OF frequency IF USING DOWNLOADED DOCKER CONTAINER
    # the high value of the frequency parameter here is so that no erase of best_model
)

runner = ParallelRunner(
    agent=agent, environments=environments, evaluation_environment=evaluation_environment,
    save_best_agent=use_best_model, # COMMENT OUT 'save_best_agent' IF USING DOWNLOADED DOCKER CONTAINER
)

cwd = os.getcwd()
evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)

def evaluation_callback_1(r):
    if(not os.path.exists(evaluation_folder + "/saved_models/output.csv")):
        print("no output.csv file, check path\n")
        sys.exit()
    else:
        with open(evaluation_folder + "/saved_models/output.csv", 'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ';')
            for row in data:
                lastrow = row
            avg_drag = float(lastrow[1])

    return avg_drag

def evaluation_callback_2(r):
    if(not os.path.exists(evaluation_folder + "/saved_models/debug.csv")):
        print("no debug.csv file, check path\n")
        sys.exit()
    else:
        debug_data = np.genfromtxt(evaluation_folder + "/saved_models/debug.csv", delimiter=";")
        debug_data = debug_data[1:,1:]
        avg_data = np.average(debug_data[-20:], axis=0)
        avg_drag = np.average([avg_data[2], avg_data[3], avg_data[4]])

    return avg_drag

runner.run(
    num_episodes=800, max_episode_timesteps=nb_actuations, sync_episodes=True,
    evaluation_callback=evaluation_callback_2
    # save_best_agent=use_best_model # Use here if using DOWNLOADED DOCKER CONTAINER
)
runner.close()
