srun --job-name "requirements" --cpus-per-task 4 --mem-per-cpu 8G --time 4:00:00 --immediate=3600 \
    --container-mounts=/netscratch/$USER/Gi-DQN:/netscratch/$USER/Gi-DQN,/home/$USER/.netrc:/root/.netrc --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
    --container-workdir=/netscratch/$USER/Gi-DQN --gres=gpu:1 --pty bash

apt-get install ffmpeg libsm6 libxext6 bc -y