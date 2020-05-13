import os
import socket
import numpy as np
import csv

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from simulation_base.env import resume_env, nb_actuations

example_environment = resume_env(plot=False, dump=100, single_run=True)

deterministic = True

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

saver_restore = dict(directory=os.getcwd() + "/saver_data/")

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
    parallel_interactions=1,
    saver=saver_restore,
)

agent.initialize()


if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

def one_run():
    print("start simulation")
    state = example_environment.reset()
    example_environment.render = True

    for k in range(3*nb_actuations):
        action = agent.act(state, deterministic=deterministic, independent=True)
        state, terminal, reward = example_environment.execute(action)

    data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
    data = data[1:,1:]
    m_data = np.average(data[len(data)//2:], axis=0)
    # Print statistics
    print("Single Run finished. AvgDrag0: {}, AvgDrag1: {}, AvgDrag2: {}, AvgLift0: {}, \
        AvgLift1: {}, AvgLift2: {}".format(m_data[1], m_data[2], m_data[3], m_data[4], m_data[5], m_data[6]))

    name = "test_strategy_avg.csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+name)):
        with open("saved_models/"+name, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Name", "Drag0", "Drag1", "Drag2", "Lift0", "Lift1", "Lift2",
                                "Qs0", "Qs1", "Qs2", "Reward", "mean_drag", "mean_lift"])
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())
    else:
        with open("saved_models/"+name, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow([example_environment.simu_name] + m_data[1:].tolist())



if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()
