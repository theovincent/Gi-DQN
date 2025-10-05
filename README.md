# Gradient iterated Deep Q-Network (Gi-DQN)

## User installation
CPU installation:
```bash
python3 -m venv env_cpu
source env_cpu/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```
GPU installation:
```bash
python3 -m venv env_gpu
source env_gpu/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as:```pytest```