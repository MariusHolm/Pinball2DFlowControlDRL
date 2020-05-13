FROM quay.io/fenicsproject/stable:2018.1.0
USER root
RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get -y install  wget bzip2 gcc ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        git mercurial subversion vim tmux build-essential curl grep sed dpkg gmsh htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER root

# Choose correct MPI version after gmsh installation.
RUN update-alternatives --set mpirun /usr/bin/mpirun.mpich

# Make a local directory and clone the public repository
RUN mkdir local \
    && cd local \
    && git clone https://github.com/tensorforce/tensorforce.git \
    && cd tensorforce \
    && git checkout 0.5.0 -b tensorforce_0_5_0 \
    && pip3 install -e .[tf] \
    && git clone git@github.com:MariusHolm/Pinball2DFlowControlDRL.git

USER root
