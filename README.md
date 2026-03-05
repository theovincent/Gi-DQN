# Implementation of Gradient iterated Deep Q-Network (Gi-DQN)

## User installation
We recommend using Python 3.11.5. In the folder where the code is, create a Python virtual environment, activate it, update pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as:```pytest```

## Running experiments
The script `launch_job/atari/launch.sh` trains an Gi-DQN ($K=5$, $\beta=1$) agent with the linear shared CNN architecture on a local machine, on the game _Breakout_.
