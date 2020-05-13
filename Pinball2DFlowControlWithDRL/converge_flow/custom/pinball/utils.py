import dolfin as df
import numpy as np


def as_mpi4py_comm(comm):
    '''Drop comm to mpi4py for better API'''
    return comm.tompi4py() if df.__version__ == '2017.2.0' else comm


def mpi_comm_world():
    '''Compat for dolfin world comm'''
    return df.mpi_comm_world() if df.__version__ == '2017.2.0' else df.MPI.comm_world


def mpi_comm_self():
    '''Drop comm to mpi4py for better API'''
    return df.mpi_comm_self() if df.__version__ == '2017.2.0' else df.MPI.comm_self


def unique_points(points):
    '''Nice: https://stackoverflow.com/a/20271006'''
    x, y = points.T
    b = x + y*1.0j 
    idx = np.unique(b, return_index=True)[1]
    return points[idx]
