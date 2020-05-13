# Pinball2DFlowControlDRL 

This repository contains the code used in the Master thesis **Using Deep Reinforcement Learning for Active Flow Control** by Marius Holm (2020) at the University of Oslo. The work is a continuation of the previous work done by Rabault et al., Journal of Fluid Mechanics (2019), preprint available at https://arxiv.org/pdf/1808.07664.pdf, code available at https://github.com/jerabaul29/Cylinder2DFlowControlDRL. 

Due to the increased complexity of the "fluidic pinball" system, we use code that allows us to perform DRL training in parallel. This corresponds to the code and method presented in "Accelerating Deep Reinforcement Learning strategies of Flow Control through a multi-environment approach", Rabault and Kuhnle, Physics of Fluids (2019), preprint accessible at https://arxiv.org/abs/1906.10382, code available at https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel.

In this repository we present code that allows parallel training of active flow control of the "fluidic pinball" system introduced by  Noack and Morzynski (2017) http://berndnoack.com/FlowControl.php.

If you find this work useful in your own research please cite the work as:

```markdown
Holm, M., Rabault, J. (2020). 
Using Deep Reinforcement Learning for Active Flow Control.
```

## Getting started

The main code of the project is located in **Pinball2DFlowControlWithDRL**. A detailed README is provided inside the folder. 

In the **Docker** folder we provide a *Dockerfile* to build a Docker container with all necessary packages. A detailed README explains the procedure of building the container, with the option to download and edit a pre-made container used in the project found at https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel .