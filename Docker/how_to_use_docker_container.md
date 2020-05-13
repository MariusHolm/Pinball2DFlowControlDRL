# Docker setup
On a virtual machine running Ubuntu we use the following commands to setup a Docker container with the necessary code. (The same commands are applicable for laptops/desktops with Ubuntu OS). 

Commands done in the VM or on a personal computer (in Ubuntu OS) are preceded by:

​	user@computer:~$ 

While commands done in a Docker container are preceded by:

​	fenics@2a6ff79d61de:~$

## Build the Docker image yourself from Dockerfile

A complete Docker image can be built by with the provided Dockerfile which creates a Docker image which can run all of the necessary simulations. (Also possible to download a Docker image with a few changes needed to run everything.)

```shell
1.
# Check that docker is available. If not, install following instructions at https://docs.docker.com/install/linux/docker-ce/ubuntu/
	user@computer:~$ sudo docker -v
	Docker version 18.09.9, build 1752eb3

2. 
# Clone the repository or copy the Dockerfile to the desired location.
	user@computer:~$ git clone git@github.com:MariusHolm/Pinball_public_temporary.git

3.
# Go to the folder where the Dockerfile is saved (/Docker/ in the repository) and build the image by doing (This can take some time):
	user@computer:~$ sudo docker build . -t "given-image-tag"

4.
# Image is now built. Can check by doing:
	user@computer:~$ sudo docker images
	REPOSITORY			Tag			IMAGE ID		CREATED	 		SIZE
	"given-image-tag"	latest		e97fd73752be	1 minute ago	3.1GB

5.
# Spin a container out of the image. If you want the Docker container to be self-contained skip to 5b). If you want to share a folder with the Docker container use the command in 5a). This will share the current folder of the user@computer [$(pwd)] with the Docker container.
	a) 
	user@computer:~$ sudo docker run -ti -v "$(pwd):/home/fenics/shared" --name "name-of-container" "name_of_docker_image" 
	
	b)
	user@computer:~$ sudo docker run -ti --name "name-of-container" "name_of_docker_image"

	# Both commands should output:
        # FEniCS stable version image
    
        Welcome to FEniCS/stable!
    
        This image provides a full-featured and optimized build of the stable
        release of FEniCS.
    
        To help you get started this image contains a number of demo
        programs. Explore the demos by entering the 'demo' directory, for
        example:
    
            cd ~/demo/python/documented/poisson
            python3 demo_poisson.py
        fenics@2a6ff79d61de:~$ 

# Exit the container
6. 
	fenics@565f6528e97b:~$ exit
	
# Start the container in the background
7.
	user@computer:~$ sudo docker start "name-of-container"
	"name-of-container"

# Open a terminal in the container
8. 
	user@computer:~$ sudo docker exec -ti -u fenics "name-of-container" /bin/bash -l

        # FEniCS stable version image

        Welcome to FEniCS/stable!

        This image provides a full-featured and optimized build of the stable
        release of FEniCS.

        To help you get started this image contains a number of demo
        programs. Explore the demos by entering the 'demo' directory, for
        example:

            cd ~/demo/python/documented/poisson
            python3 demo_poisson.py
        fenics@2a6ff79d61de:~$ 

# The container is now ready to start simulations.

# To exit the container. 
9.
	fenics@2a6ff79d61de:~$ exit
```



## Download and edit a full Docker image

Instead of building the image from a Dockerfile it is also possible to create a working Docker container by downloading the Docker container presented in https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel (link to Docker image: https://folk.uio.no/jeanra/Informatics/cylinder2dflowcontrol_Parallel_v1.tar)

Using this Docker image will also require 2 minor changes in the python script `launch_parallel_training.py`, see comments in the code for details.

```shell
1.
# Check that docker is available. If not, install following instructions at https://docs.docker.com/install/linux/docker-ce/ubuntu/
	user@computer:~$ sudo docker -v
	Docker version 18.09.9, build 1752eb3

2. 
# Download the Docker image
	user@computer:~$ wget -4 "https://folk.uio.no/jeanra/Informatics/cylinder2dflowcontrol_Parallel_v1.tar"

3.
# Load the image
	user@computer:~$ sudo docker load -i "https://folk.uio.no/jeanra/Informatics/cylinder2dflowcontrol_Parallel_v1.tar"
	
3/4.
# List Docker images to see that the image is loaded correctly.
# The loaded image might not have a tag="name_of_docker_image". Then use command "sudo docker tag IMAGE_ID name_of_docker_image" to name the image.
	user@computer:~$ sudo docker images


4.
# Spin a container out of the image. If you want the Docker container to be self-contained skip to 5b). If you want to share a folder with the Docker container use the command in 5a). This will share the current folder of the user@computer [$(pwd)] with the Docker container.
	a) 
	user@computer:~$ sudo docker run -ti -v "$(pwd):/home/fenics/shared" --name "name-of-container" "name_of_docker_image" 
	
	b)
	user@computer:~$ sudo docker run -ti --name "name-of-container" "name_of_docker_image"

	# Both commands should output:
        # FEniCS stable version image
    
        Welcome to FEniCS/stable!
    
        This image provides a full-featured and optimized build of the stable
        release of FEniCS.
    
        To help you get started this image contains a number of demo
        programs. Explore the demos by entering the 'demo' directory, for
        example:
    
            cd ~/demo/python/documented/poisson
            python3 demo_poisson.py
        fenics@2a6ff79d61de:~$ 
   
5. 
# Exit the container
 		fenics@2a6ff79d61de:~$ exit

6.
# Start the container in the background
	user@computer:~$ sudo docker start "name-of-container"
	"name-of-container"

7. 
# Open a terminal in the container
	user@computer:~$ sudo docker exec -ti -u fenics "name-of-container" /bin/bash -l

        # FEniCS stable version image
    
        Welcome to FEniCS/stable!
    
        This image provides a full-featured and optimized build of the stable
        release of FEniCS.
    
        To help you get started this image contains a number of demo
        programs. Explore the demos by entering the 'demo' directory, for
        example:
    
            cd ~/demo/python/documented/poisson
            python3 demo_poisson.py
        fenics@2a6ff79d61de:~$ 


8. 
# Install gmsh software to create mesh for flow initialization.
	user@computer:~$ sudo apt-get update
	
	user@computer:~$ sudo apt-get install gmsh

# check that gmsh is version 3.0.6 by doing:
	user@computer:~$ gmsh --version
	3.0.6

9.
# Installing gmsh installs another MPI version, so we need to fix mpi4py so we can run flow initialization with MPI. Enter the number found in the "Selection" column, corresponding to /usr/bin/mpirun.mpich.
	fenics@2a6ff79d61de:~$ sudo update-alternatives --config mpirun
	
        There are 2 choices for the alternative mpirun (providing /usr/bin/mpirun).
    
          Selection    Path                     Priority   Status
        ------------------------------------------------------------
        * 0            /usr/bin/mpirun.openmpi   50        auto mode
          1            /usr/bin/mpirun.mpich     40        manual mode
          2            /usr/bin/mpirun.openmpi   50        manual mode
    
        Press <enter> to keep the current choice[*], or type selection number:
   
# Without step 9. the flow initialization in folder "converge_flow/" will not work with MPI, and thus be a lot slower.


# Container is now ready to use! 
#Simply enter the following in the container to spin up the container (Container must have been started ("docker start") previous to ("docker exec").)
	user@computer:~$ sudo docker exec -ti -u fenics "name-of-container" /bin/bash -l

```