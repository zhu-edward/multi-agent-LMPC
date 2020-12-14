# Multi Agent LMPC Examples

This repository collects the multi-agent LMPC examples shown in the folloing publications

- Zhu, E., St&uuml;rz, Y., Rosolia, R., Borrelli, F., "Trajectory Optimization for Nonlinear Multi-Agent Systems using Decentralized Learning Model Predictive Control", Presented at Conference on Decision and Control 2020. [[arXiv](https://arxiv.org/abs/2004.01298)]
<!-- - St&uuml;rz, Y., Zhu, E., Rosolia, R., Borrelli, F., "Distributed Learning Model Predictive Control for Linear Systems", Presented at Conference on Decision and Control 2020, Dec 14-18, 2020. [[arXiv](https://arxiv.org/abs/2006.13406)] -->

The authors are with the Model Predictive Control Lab at UC Berkeley
- Edward Zhu (edward.zhu@berkeley.edu)
- Yvonne St&uuml;rz (y.stuerz@berkeley.edu)

## Directory Structre

`decentralized_LMPC` collects examples relating to the first paper above where a decentralized LMPC policy is implemented in multiple multi-agent navigation tasks. In this approach, synthesis of the decoupled MPC policies is done in a centralized manner between iterations of task execution. Online execution of the decoupled policies can be done completely in parallel with no communication between agents
- `3_agent_nl_demo`: A simple navigation task for 3 car-like agents navigating an intersection
- `3_agent_nl_centralized_demo`: The above task with centralized policy synthesis and control (for comparison with the decentralized approach)
- `multi_agent_nl_demo`: A more complex task with 10 agents performing collision free manuevers in a shared space. Agents are initialized in a circle
- `multi_agent_rand_nl_demo`: A more complex task with 10 agents performing collision free manuevers in a shared space. Agents are initialized in arbitrary positions

<!-- `distributed_LMPC` collects examples relating to the second paper where a distributed LMPC policy is implemented using an ADMM based approach -->

## Native/VirtualEnv Installation

The code is written in Python 3 and the following packages are required

```
sudo apt install coinor-libipopt-dev libblas-dev ffmpeg
sudo python3 -m pip install cvxpy ipopt casadi scikit-learn
```

## Docker Installation

We also provide Dockerfiles and scripts to build a Docker image and instantiate containers to run the code in this repository. To do so, first we need to [install Docker CE](https://docs.docker.com/install/). Instructions for installing via the `apt` repository for [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/) are included below. Instructions for MacOS can be found [here](https://docs.docker.com/docker-for-mac/install/).

___Ubuntu___

```
# Update apt package index
sudo apt update

# Install packages to allow apt to use a repository over HTTPS
sudo apt install -y apt-transport-https \
  ca-certificates \
  curl \
  software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"

# Update apt package index
sudo apt update

# Install Docker CE
sudo apt install docker-ce

# Add user to docker group
export user=$(whoami)
sudo usermod -aG docker $user
```

Docker related files are included in the `docker` directory. The bash script `start_LMPC_docker.sh` builds the image, which extends the `continuumio/anaconda` image by adding the cvxpy and ipopt Python libraries.

### Running with Docker

In order to (re)build the image and start the container, run the following command from the base directory of the repository:

```
bash docker/start_LMPC_docker.sh -r
```

The `-r` flag triggers a rebuild of the image. There is no harm in always including this flag to ensure that changes to the Dockerfile are implemented. If the `-r` flag is not specified, a container will be instantiated using the most recent build of the image.

Once the container is running, open up a new terminal and use the following command to start a bash session in the container (you can get the name of the container using `docker container ls`):

```
docker exec -it ${container_name} bash
```

We can then run our code within this environment.

#### Important Notes

- Upon container instantiation the base directory of the repository is mounted to `/home/LMPC` within the container
- `python-tk` is installed and `TkAgg` needs to be selected as the `matplotlib` backend as opposed to the default `Qt5Agg` (which doesn't seem to work). This can be done by including the following code once __before__ the first time `matplotlib.pyplot` is imported:
```
import matplotlib
matplotlib.use('TkAgg')
```
- A Jupyter notebook server can be started within the container using the command:
```
/opt/conda/bin/jupyter notebook --notebook-dir=/home/LMPC --ip=0.0.0.0 --port=8888 --allow-root
```
Once it is up and running, enter the following URL into a web browser:
```
localhost:8888/?token=[server_token]
```
where `server_token` can be found in the terminal print-out after starting the notebook server
