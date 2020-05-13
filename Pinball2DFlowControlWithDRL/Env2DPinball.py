# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.environments import Environment
import tensorforce
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pinball_utils import mpi_comm_self, mpi_comm_world
from msh_convert import convert

from threading import Thread
from tensorforce import TensorforceError

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from dolfin import Expression, File, plot

from probes import (PenetratedDragProbeANN as DragProbe,
                    PenetratedLiftProbeANN as LiftProbe,
                    PressureProbeANN as PressureProbe,
                    VelocityProbeANN as VelocityProbe,
                    RecirculationAreaProbe)
from flow_solver import FlowSolver
from dolfin import *

import numpy as np
import os

import pickle

import time
import math
import csv
import random as random

import shutil


# Ring buffer of fixed size. When filled the oldest data will be overwritten.
class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


class Env2DPinball(Environment):
    """Environment Class for 2D flow simulation of the fluidic pinball."""
    #  __init__(path_root)
    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params,
                optimization_params, inspection_params, n_iter_make_ready=None, verbose=0, size_history=2000,
                reward_function='plain_drag_lift', size_time_state=50, duration_execute=0.5, simu_name="Simu", root_folder='mesh/re100'):
        """

        """
        self.observation = None
        self.thread = None

        self.path_root = path_root
        self.root_folder = root_folder

        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params

        self.comm = mpi_comm_world()

        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params

        self.verbose = verbose
        self.n_iter_make_ready = n_iter_make_ready

        self.size_history = size_history
        self.reward_function = reward_function
        self.size_time_state = size_time_state
        self.duration_execute = duration_execute

        self.simu_name = simu_name


        # Create csv file to save DRL model.
        name="output.csv"
        last_row = None
        if (os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, 'r') as f:
                for row in reversed(list(csv.reader(f, delimiter=";", lineterminator="\n"))):
                    last_row = row
                    break
        if (not last_row is None):
            self.episode_number = int(last_row[0])
            self.last_episode_number = int(last_row[0])
        else:
            self.last_episode_number = 0
            self.episode_number = 0

        # Initialize drag, lift and recirc area for each cylinder.
        self.episode_drags0 = np.array([])
        self.episode_drags1 = np.array([])
        self.episode_drags2 = np.array([])

        self.episode_areas = np.array([])

        self.episode_lifts0 = np.array([])
        self.episode_lifts1 = np.array([])
        self.episode_lifts2 = np.array([])

        self.episode_mean_drag = np.array([])
        self.episode_mean_lift = np.array([])

        self.episode_reward = np.array([])

        self.flag_need_reset = False

        self.initialized_visualization = False

        self.start_class(complete_reset=True)


    def start_class(self, complete_reset=True):
        if complete_reset == False:
            self.solver_step = 0
        else:
            self.solver_step = 0
            self.accumulated_drag = 0
            self.accumulated_lift = 0

            self.reward = 0

            self.initialized_output = False
            self.resetted_number_probes = False
            self.area_probe = None
            self.history_parameters = {}
            # history_parameters dict to save probe, drag, and lift history data.

            # number of cylinders
            for current_cyl in range(len(self.geometry_params["cylinder_center"])):
                self.history_parameters["cylinder_{}".format(current_cyl)] = RingBuffer(self.size_history)

            self.history_parameters["number_of_cylinders"] = len(self.geometry_params["cylinder_center"])

            for current_probe in range(len(self.output_params["locations"])):
                if self.output_params["probe_type"] == 'pressure':
                    self.history_parameters["probe_{}".format(current_probe)] = RingBuffer(self.size_history)

                elif self.output_params["probe_type"] == 'velocity':
                    self.history_parameters["probe_{}_u".format(current_probe)] = RingBuffer(self.size_history)
                    self.history_parameters["probe_{}_v".format(current_probe)] = RingBuffer(self.size_history)


            self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

            # Allocate memory for history paramaters
            self.history_parameters["drag0"] = RingBuffer(self.size_history)
            self.history_parameters["drag1"] = RingBuffer(self.size_history)
            self.history_parameters["drag2"] = RingBuffer(self.size_history)

            self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

            self.history_parameters["lift0"] = RingBuffer(self.size_history)
            self.history_parameters["lift1"] = RingBuffer(self.size_history)
            self.history_parameters["lift2"] = RingBuffer(self.size_history)

            self.history_parameters["mean_drag"] = RingBuffer(self.size_history)
            self.history_parameters["mean_lift"] = RingBuffer(self.size_history)

            self.history_parameters["reward"] = RingBuffer(self.size_history)

            #-----------------------------------------------------------------------
            # Set Path to numerical .msh and .h5 file.
            msh_file = '.'.join([self.path_root, 'msh'])
            h5_file = '.'.join([self.path_root, 'h5'])

            if not os.path.exists(h5_file):
                # if no .h5 file of mesh, convert .msh to .h5
                mesh = convert(msh_file, h5_file)

            self.geometry_params['mesh'] = h5_file

            # ----------------------------------------------------------------------
            # if necessary, load initialization fields
            if self.n_iter_make_ready is None:
                if self.verbose > 0:
                    print("Load initial flow")

                # Load initialization fields
                self.flow_params['u_init'] = '/'.join([self.root_folder, 'u_init.xdmf'])
                self.flow_params['p_init'] = '/'.join([self.root_folder, 'p_init.xdmf'])

                if self.verbose > 0:
                    print("Load buffer history")

                if not "number_of_probes" in self.history_parameters:
                    self.history_parameters["number_of_probes"] = 0

                if not "number_of_cylinders" in self.history_parameters:
                    self.history_parameters["number_of_cylinders"] = len(self.geometry_params["cylinder_center"])

                if not "lift" in self.history_parameters:
                    self.history_parameters["lift"] = RingBuffer(self.size_history)

                if not "recirc_area" in self.history_parameters:
                   self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

                # if not the same number of probes, reset
                if not self.history_parameters["number_of_probes"] == len(self.output_params["locations"]):
                    for current_probe in range(len(self.output_params["locations"])):
                        if self.output_params["probe_type"] == 'pressure':
                            self.history_parameters["probe_{}".format(current_probe)] = RingBuffer(self.size_history)
                        elif self.output_params["probe_type"] == 'velocity':
                            self.history_parameters["probe_{}_u".format(current_probe)] = RingBuffer(self.size_history)
                            self.history_parameters["probe_{}_v".format(current_probe)] = RingBuffer(self.size_history)

                    self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

                    self.resetted_number_probes = True


            # ----------------------------------------------------------------------
            # Create flow simulation object
            self.flow = FlowSolver(self.comm, self.flow_params, self.geometry_params, self.solver_params)

            # ----------------------------------------------------------------------
            # Setup probes
            if self.output_params["probe_type"] == 'pressure':
                self.ann_probes = PressureProbe(self.flow, self.output_params['locations'])

            elif self.output_params["probe_type"] == 'velocity':
                self.ann_probes = VelocityProbe(self.flow, self.output_params['locations'])

            else:
                raise RuntimeError("unknown probe type")

            # ----------------------------------------------------------------------
            # probe setup for Pinball solver
            self.drag_probes = [DragProbe(i, self.flow) for i in range(len(self.geometry_params['cylinder_center']))]
            self.lift_probes = [LiftProbe(i, self.flow) for i in range(len(self.geometry_params['cylinder_center']))]

            # ----------------------------------------------------------------------
            # Initialize rotation and action as zeros
            self.Qs = np.zeros(len(self.geometry_params['cylinder_center']))
            self.action = np.zeros(len(self.geometry_params['cylinder_center']))

            # ----------------------------------------------------------------------
            # Compute probe positions?
            self.compute_positions_for_plotting()

            # ----------------------------------------------------------------------
            # if necessary, make converge
            if self.n_iter_make_ready is not None:
                self.u_, self.p_, self.status = self.flow.evolve(self.Qs)

                path=''
                if "dump" in self.inspection_params:
                    path = 'results/area_out.pvd'

                self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)
                if self.verbose > 0:
                    print("Compute initial flow")

                for _ in range(self.n_iter_make_ready):
                    self.u_, self.p_, self.status = self.flow.evolve(self.Qs)

                    self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                    self.drag = [dp.sample(self.u_, self.p_) for dp in self.drag_probes]
                    self.lift = [lp.sample(self.u_, self.p_) for lp in self.lift_probes]
                    self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                    self.write_history_parameters()
                    self.dump_values()
                    self.output_data()

                    self.solver_step += 1

            if self.n_iter_make_ready is not None:
                encoding = XDMFFile.Encoding.HDF5
                mesh = convert(msh_file, h5_file)
                comm = mesh.mpi_comm()

                u_init = '/'.join([self.root_folder, 'u_init.xdmf'])
                p_init = '/'.join([self.root_folder, 'p_init.xdmf'])

                # save field data
                XDMFFile(comm, u_init).write_checkpoint(self.u_, 'u0', 0, encoding)
                XDMFFile(comm, p_init).write_checkpoint(self.p_, 'p0', 0, encoding)

                shutil.copytree(self.root_folder, 'simulation_base/mesh/serial/')

                sys.exit("\nInitialization fields have been created!\nReset simulation using make_converge=False\n")

            # ----------------------------------------------------------------------
            # if reading from disk, show to check everything ok
            if self.n_iter_make_ready is None:
                #Let's start in a random position of the vortex shedding
                #
                if self.optimization_params["random_start"]:
                    rd_advancement = np.random.randint(650)
                    for j in range(rd_advancement):
                        self.flow.evolve(self.Qs)
                    print("Simulated {} iterations before starting the control".format(rd_advancement))

                self.u_, self.p_, self.status = self.flow.evolve(self.Qs)
                path=''
                if "dump" in self.inspection_params and "single_run" in self.inspection_params and self.inspection_params['single_run'] == True:
                    path = 'results/area_out.pvd'
                self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                self.drag = [dp.sample(self.u_, self.p_) for dp in self.drag_probes]
                self.lift = [lp.sample(self.u_, self.p_) for lp in self.lift_probes]
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                self.write_history_parameters()

            # ----------------------------------------------------------------------
            if self.resetted_number_probes:
                for _ in range(self.size_history):
                    self.execute()

            self.ready_to_use = True


    def write_history_parameters(self):
        """Save values to dict"""
        for current_cyl in range(len(self.geometry_params["cylinder_center"])):
            self.history_parameters["cylinder_{}".format(current_cyl)].extend(self.Qs[current_cyl])

        if self.output_params["probe_type"] == 'pressure':
            for current_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}".format(current_probe)].extend(self.probes_values[current_probe])

        elif self.output_params["probe_type"] == 'velocity':
            for current_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}_u".format(current_probe)].extend(self.probes_values[2 * current_probe])
                self.history_parameters["probe_{}_v".format(current_probe)].extend(self.probes_values[2 * current_probe + 1])

        self.history_parameters["drag0"].extend(np.array(self.drag[0]))
        self.history_parameters["drag1"].extend(np.array(self.drag[1]))
        self.history_parameters["drag2"].extend(np.array(self.drag[2]))

        self.history_parameters["recirc_area"].extend(np.array(self.recirc_area))


        self.history_parameters["lift0"].extend(np.array(self.lift[0]))
        self.history_parameters["lift1"].extend(np.array(self.lift[1]))
        self.history_parameters["lift2"].extend(np.array(self.lift[2]))

        self.history_parameters["mean_drag"].extend(np.array(np.mean(self.drag)))
        self.history_parameters["mean_lift"].extend(np.array(np.mean(self.lift)))

        self.history_parameters["reward"].extend(np.array(self.reward))


    def compute_positions_for_plotting(self):
        """Compute grid locations of pressure probes"""
        # where the pressure probes are
        self.list_positions_probes_x = []
        self.list_positions_probes_y = []

        # get the positions
        for crrt_probe in self.output_params['locations']:
            if self.verbose > 2:
                print(crrt_probe)

            self.list_positions_probes_x.append(crrt_probe[0])
            self.list_positions_probes_y.append(crrt_probe[1])


    def dump_values(self):
        """Dump drag, lift, actions, reward, etc. to terminal and .csv file during simulation."""
        if self.solver_step % self.inspection_params["dump"] == 0 and self.inspection_params["dump"] < 10000:
            print("%s | Ep N: %4d, step: %4d, drag0: %.4f, drag1: %.4f, drag2: %.4f, Rec Area: %.4f, lift0: %.4f, lift1: %.4f, lift2: %.4f, Qs0: %.4f, Qs1: %.4f, Qs2: %.4f, Reward: %.4f"%(self.simu_name,
            self.episode_number,
            self.solver_step,
            self.history_parameters["drag0"].get()[-1],
            self.history_parameters["drag1"].get()[-1],
            self.history_parameters["drag2"].get()[-1],
            self.history_parameters["recirc_area"].get()[-1],
            self.history_parameters["lift0"].get()[-1],
            self.history_parameters["lift1"].get()[-1],
            self.history_parameters["lift2"].get()[-1],
            self.Qs[0],
            self.Qs[1],
            self.Qs[2],
            self.history_parameters["reward"].get()[-1]))

            name = "debug.csv"
            if(not os.path.exists("saved_models")):
                os.mkdir("saved_models")
            if(not os.path.exists("saved_models/"+name)):
                with open("saved_models/"+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Name", "Episode", "Step", "Drag0", "Drag1", "Drag2", "RecircArea", "Lift0", "Lift1", "Lift2", "Qs0", "Qs1", "Qs2", "Reward", "mean_drag", "mean_lift"])
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["drag0"].get()[-1],
                                          self.history_parameters["drag1"].get()[-1],
                                          self.history_parameters["drag2"].get()[-1],
                                          self.history_parameters["recirc_area"].get()[-1],
                                          self.history_parameters["lift0"].get()[-1],
                                          self.history_parameters["lift1"].get()[-1],
                                          self.history_parameters["lift2"].get()[-1],
                                          self.Qs[0],
                                          self.Qs[1],
                                          self.Qs[2],
                                          self.history_parameters["reward"].get()[-1],
                                          self.history_parameters["mean_drag"].get()[-1],
                                          self.history_parameters["mean_lift"].get()[-1]])
            else:
                with open("saved_models/"+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["drag0"].get()[-1],
                                          self.history_parameters["drag1"].get()[-1],
                                          self.history_parameters["drag2"].get()[-1],
                                          self.history_parameters["recirc_area"].get()[-1],
                                          self.history_parameters["lift0"].get()[-1],
                                          self.history_parameters["lift1"].get()[-1],
                                          self.history_parameters["lift2"].get()[-1],
                                          self.Qs[0],
                                          self.Qs[1],
                                          self.Qs[2],
                                          self.history_parameters["reward"].get()[-1],
                                          self.history_parameters["mean_drag"].get()[-1],
                                          self.history_parameters["mean_lift"].get()[-1]])

        if("single_run" in self.inspection_params and self.inspection_params["single_run"] == True):
            """Dump to different .csv file if deterministic single run simulation."""
            self.sing_run_output()

    def sing_run_output(self):
        name = "test_strategy.csv"
        if(not os.path.exists("saved_models")):
            os.mkdir("saved_models")
        if(not os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Name", "Step", "Drag0", "Drag1", "Drag2", "RecircArea",
                                    "Lift0", "Lift1", "Lift2", "Qs0", "Qs1", "Qs2", "Reward", "mean_drag", "mean_lift"])
                spam_writer.writerow([self.simu_name,
                                      self.solver_step,
                                      self.history_parameters["drag0"].get()[-1],
                                      self.history_parameters["drag1"].get()[-1],
                                      self.history_parameters["drag2"].get()[-1],
                                      self.history_parameters["recirc_area"].get()[-1],
                                      self.history_parameters["lift0"].get()[-1],
                                      self.history_parameters["lift1"].get()[-1],
                                      self.history_parameters["lift2"].get()[-1],
                                      self.Qs[0],
                                      self.Qs[1],
                                      self.Qs[2],
                                      self.history_parameters["reward"].get()[-1],
                                      self.history_parameters["mean_drag"].get()[-1],
                                      self.history_parameters["mean_lift"].get()[-1]])

        else:
            with open("saved_models/"+name, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.simu_name,
                                      self.solver_step,
                                      self.history_parameters["drag0"].get()[-1],
                                      self.history_parameters["drag1"].get()[-1],
                                      self.history_parameters["drag2"].get()[-1],
                                      self.history_parameters["recirc_area"].get()[-1],
                                      self.history_parameters["lift0"].get()[-1],
                                      self.history_parameters["lift1"].get()[-1],
                                      self.history_parameters["lift2"].get()[-1],
                                      self.Qs[0],
                                      self.Qs[1],
                                      self.Qs[2],
                                      self.history_parameters["reward"].get()[-1],
                                      self.history_parameters["mean_drag"].get()[-1],
                                      self.history_parameters["mean_lift"].get()[-1]])
        return


    def output_data(self):
        """Function to output drag, lift and area data."""
        if "dump" in self.inspection_params and self.inspection_params["dump"] < 10000:

            # Defines dump frequency
            modulo_base = self.inspection_params["dump"]

            # Append the latest drag/lift/mean_drag/etc. to episode array.
            self.episode_drags0 = np.append(self.episode_drags0, [self.history_parameters["drag0"].get()[-1]])
            self.episode_drags1 = np.append(self.episode_drags1, [self.history_parameters["drag1"].get()[-1]])
            self.episode_drags2 = np.append(self.episode_drags2, [self.history_parameters["drag2"].get()[-1]])

            self.episode_areas = np.append(self.episode_areas, [self.history_parameters["recirc_area"].get()[-1]])

            self.episode_lifts0 = np.append(self.episode_lifts0, [self.history_parameters["lift0"].get()[-1]])
            self.episode_lifts1 = np.append(self.episode_lifts1, [self.history_parameters["lift1"].get()[-1]])
            self.episode_lifts2 = np.append(self.episode_lifts2, [self.history_parameters["lift2"].get()[-1]])

            self.episode_mean_drag = np.append(self.episode_mean_drag, [self.history_parameters["mean_drag"].get()[-1]])
            self.episode_mean_lift = np.append(self.episode_mean_lift, [self.history_parameters["mean_lift"].get()[-1]])

            self.episode_reward = np.append(self.episode_reward, [self.history_parameters["reward"].get()[-1]])

            if (self.last_episode_number != self.episode_number and "single_run" in self.inspection_params and self.inspection_params["single_run"] == False):
                self.last_episode_number = self.episode_number

                # Calculate average values over an episode
                avg_drag0 = np.average(self.episode_drags0[len(self.episode_drags0)//2:])
                avg_drag1 = np.average(self.episode_drags1[len(self.episode_drags1)//2:])
                avg_drag2 = np.average(self.episode_drags2[len(self.episode_drags2)//2:])

                avg_area = np.average(self.episode_areas[len(self.episode_areas)//2:])

                avg_lift0 = np.average(self.episode_lifts0[len(self.episode_lifts0)//2:])
                avg_lift1 = np.average(self.episode_lifts1[len(self.episode_lifts1)//2:])
                avg_lift2 = np.average(self.episode_lifts2[len(self.episode_lifts2)//2:])

                avg_mean_drag = np.average(self.episode_mean_drag[len(self.episode_mean_drag)//2])
                avg_mean_lift = np.average(self.episode_mean_lift[len(self.episode_mean_lift)//2])

                avg_reward = np.average(self.episode_reward[len(self.episode_reward)//2])

                # Dump the episode average results to .csv file
                name = "output.csv"
                if (not os.path.exists("saved_models")):
                    os.mkdir("saved_models")
                if (not os.path.exists("saved_models/"+name)):
                    with open("saved_models/"+name, "w") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")

                        spam_writer.writerow(["Episode", "AvgDrag0", "AvgDrag1", "AvgDrag2", "AvgRecircArea",
                                            "AvgLift0", "AvgLift1", "AvgLift2", "Qs0", "Qs1", "Qs2", "Reward", "mean_drag", "mean_lift"])

                        spam_writer.writerow([self.last_episode_number, avg_drag0, avg_drag1, avg_drag2, avg_area,
                                            avg_lift0, avg_lift1, avg_lift2, self.Qs[0], self.Qs[1], self.Qs[2], avg_reward, avg_mean_drag, avg_mean_lift])

                else:
                    with open("saved_models/"+name, "a") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        # 1D arrays
                        spam_writer.writerow([self.last_episode_number, avg_drag0, avg_drag1, avg_drag2, avg_area,
                                            avg_lift0, avg_lift1, avg_lift2, self.Qs[0], self.Qs[1], self.Qs[2], avg_reward, avg_mean_drag, avg_mean_lift])

                # Reset arrays for next episode
                self.episode_drags0 = np.array([])
                self.episode_drags1 = np.array([])
                self.episode_drags2 = np.array([])

                self.episode_areas = np.array([])

                self.episode_lifts0 = np.array([])
                self.episode_lifts1 = np.array([])
                self.episode_lifts2 = np.array([])

                self.episode_mean_drag = np.array([])
                self.episode_mean_lift = np.array([])

                self.episode_reward = np.array([])

                # Save the result of best episode during training (1 per parallel environment)
                if (os.path.exists("saved_models/output.csv")):
                    if (not os.path.exists("best_model")):
                        shutil.copytree("saved_models", "best_model")

                    else:
                        with open("saved_models/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            last_iter = lastrow[1]

                        with open("best_model/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            best_iter = lastrow[1]

                        if float(best_iter) < float(last_iter):
                            print("best_model updated")
                            if (os.path.exists("best_model")):
                                shutil.rmtree("best_model")
                            shutil.copytree("saved_models", "best_model")

            # Dump flow to .vtk/.pvd file for later visualization in ParaView
            if self.n_iter_make_ready is None:
                # Plot during single run simulation
                if ("single_run" in self.inspection_params and self.inspection_params["single_run"] == True):
                    if self.solver_step % (modulo_base) == 0:
                        if not self.initialized_output:
                            self.u_out = File('results/u_out.pvd')
                            self.p_out = File('results/p_out.pvd')
                            self.initialized_output = True

                        if(not self.area_probe is None):
                            self.area_probe.dump(self.area_probe)

                        self.u_out << self.flow.u_
                        self.p_out << self.flow.p_

                else:
                    # WARNING: This will take several GB per Episode!!
                    # If (self.episode_number % 1) every episode will dump fields.
                    # Each environment is separate, i.e. 800 episodes / 10 environment = 80 episodes per environment.
                    # self.episode_number % 80 will dump the last episode only. (episode_number % 40) will dump the middle and the last episode
                    if self.episode_number % 5000 == 0:
                        if self.solver_step % (modulo_base) == 0:
                            if not self.initialized_output:
                                # Save the .pvd files of each episode to separate result folders to avoid overwriting previous episodes.
                                self.u_out = File('results{}/u_out.pvd'.format(self.episode_number))
                                self.p_out = File('results{}/p_out.pvd'.format(self.episode_number))
                                self.initialized_output = True

                            if(not self.area_probe is None):
                                self.area_probe.dump(self.area_probe)

                            self.u_out << self.flow.u_
                            self.p_out << self.flow.p_

    def __str__(self):
        print('')

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.ready_to_use = False


    def reset(self):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """
        if self.solver_step > 0 and not self.flag_need_reset:
            mean_accumulated_drag = self.accumulated_drag / self.solver_step
            mean_accumulated_lift = self.accumulated_lift / self.solver_step

            if self.verbose > -1:
                print("mean accumulated drag on the whole episode: {}".format(mean_accumulated_drag))

        chance = random.random()

        probability_hard_reset = 0.2

        # 20% chance for a complete reset
        if chance < probability_hard_reset or self.flag_need_reset:
            self.start_class(complete_reset=True)
            self.flag_need_reset = False
        else:
            self.start_class(complete_reset=False)

        next_state = np.transpose(np.array(self.probes_values))
        if self.verbose > 0:
            print(next_state)

        self.episode_number += 1

        return(next_state)


    def execute(self, actions=None):
        """
        Executes action, observes next state(s) and reward.
        Args:
            actions: Actions to execute.
        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        try:
            action = actions

            if self.verbose > 1:
                print("--- call execute ---")

            if action is None:
                if self.verbose > -1:
                    print("Careful, no action given; by default no rotation!")

                num_cylinders = len(self.geometry_params["cylinder_center"])
                action = np.zeros((num_cylinders, ))

            if self.verbose > 2:
                print(action)

            self.previous_action = self.action
            self.action = action

            # To execute several numerical integration steps
            self.last_actuation = 0
            self.time = 0

            # Reset numerical timestep counter between every new action given to execute()
            self.current_numerical_step = 0

            while (self.time - self.last_actuation) < self.duration_execute:
                self.current_dt = self.solver_params['dt'] # Possible to add scaling factor to make variable dt. (not tested)
                self.time += self.current_dt

                if "smooth_control" in self.optimization_params:
                    # To avoid very sudden changes in rotation that can break simulation we apply smoothing.
                    self.Qs += self.optimization_params["smooth_control"] * (np.array(action) - self.Qs)
                else:
                    self.Qs = np.transpose(np.array(action))

                self.u_, self.p_, self.status = self.flow.evolve(self.Qs)

                # Display information?
                self.dump_values()
                self.output_data()

                # Locally reset numerical solver step counter
                # current_numerical_step is reset every time a new action is given by the agent.
                self.current_numerical_step += 1

                # Done with one numerical solver step. Resets for every episode, not every action.
                self.solver_step += 1

                # sample probes, drag, and lift
                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                self.drag = [dp.sample(self.u_, self.p_) for dp in self.drag_probes]
                self.lift = [lp.sample(self.u_, self.p_) for lp in self.lift_probes]
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                # write sampled data to the history buffers
                self.write_history_parameters()

                self.accumulated_drag += np.mean(self.drag)
                self.accumulated_lift += np.mean(self.lift)

            self.last_actuation = self.time

            next_state = np.transpose(np.array(self.probes_values))

            if self.verbose > 2:
                print(next_state)

            terminal = False

            if self.verbose > 2:
                print(terminal)

            reward = self.compute_reward()
            self.reward = reward

            if self.verbose > 2:
                print(reward)

            if self.verbose > 1:
                print("--- done execute ---")

        except:
            # If exception, something has gone wrong.
            print("------- hit exception in execute -------")
            self.flag_need_reset = True
            terminal = True
            reward = -100
            next_state = np.transpose(np.zeros(np.array(self.probes_values).shape))  # necessary to avoid contaminating buffers of tensorflow with NaNs or Infs, which will crash training later on

        if self.found_invalid_values(next_state) or self.found_invalid_values(reward):
            # Looks for NaN or Inf values in the state/reward
            self.flag_need_reset = True
            print("------- hit NaN in state or reward in execute -------")
            terminal = True
            reward = -100
            next_state = np.transpose(np.zeros(np.array(self.probes_values).shape))

        return (next_state, terminal, reward)


    def found_invalid_values(self, to_check, abs_threshold=20):
        """Check arrays for NaN/Inf values."""
        if type(to_check) is np.ndarray:
            bool_ret = np.isnan(to_check).any() or np.isinf(to_check).any() or (np.abs(to_check) > abs_threshold).any()
        else:
            bool_ret = self.found_invalid_values(np.array(to_check))

        return(bool_ret)


    def compute_reward(self):
        """Compute reward for given action."""
        # Used for drag reduction
        if self.reward_function == 'plain_drag_lift':
            values_drag_in_last_execute = np.mean(self.history_parameters["mean_drag"].get()[-self.current_numerical_step:])
            values_lift_in_last_execute = np.mean(self.history_parameters["mean_lift"].get()[-self.current_numerical_step:])

            return (values_drag_in_last_execute + np.abs(self.inspection_params["mean_drag"]) - 0.2 * np.abs(values_lift_in_last_execute))

        # Used for drag increase
        if self.reward_function == 'more_drag_simple_actuation':
            values_drag_in_last_execute = np.mean( self.history_parameters["mean_drag"].get()[-self.current_numerical_step:])
            values_lift_in_last_execute = np.mean(self.history_parameters["mean_lift"].get()[-self.current_numerical_step:])
            penalty = - 0.1 * math.sqrt(self.Qs[0]**2 + self.Qs[1]**2 + self.Qs[2]**2)

            return (- values_drag_in_last_execute - np.abs(self.inspection_params["mean_drag"]) - 0.2 * np.abs(values_lift_in_last_execute) + penalty)

        else:
            raise RuntimeError("reward function {} not yet implemented".format(self.reward_function))

    #@property
    def states(self):
        """
        Returns:
            States specification, with the following attributes
                (required):
                - type: 'float'
                - shape: integer, or list/tuple of integers (required).
        """
        if self.output_params["probe_type"] == 'pressure':
            return dict(type='float',
                        shape=(len(self.output_params["locations"]) * \
                            self.optimization_params["num_steps_in_pressure_history"],)
                        )
        elif self.output_params["probe_type"] == 'velocity':
            return dict(type='float',
                        shape=(2 * len(self.output_params["locations"]) * \
                            self.optimization_params["num_steps_in_pressure_history"],)
                        )

    def actions(self):
        """
        Returns:
            actions (spec, or dict of specs): Actions specification, with the following attributes
                (required):
                - type: 'float' (required).
                - shape: list/tuple of integers (default: []).
                - min_value and max_value: float
        """
        return dict(type='float',
                    shape=(len(self.geometry_params["cylinder_center"]), ),
                    min_value=self.optimization_params["min_rotation_cyl"],
                    max_value=self.optimization_params["max_rotation_cyl"]
                    )
