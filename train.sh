
#!/bin/bash


## 训练1024维向量

# train_rqkmeans
# project_dir="output/rqkmeans/$(date +%Y-%m-%d_%H-%M-%S)"
# mkdir -p ${project_dir}

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#     --num_processes 2 \
#     train_rqkmeans.py \
#     --config config/1024/rqkmeans.yaml \
#     --project_dir ${project_dir} \
#     > "${project_dir}/accelerate.log" 2>&1 &



# train_rqvae
project_dir="output/rqvae/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p ${project_dir}

CUDA_VISIBLE_DEVICES=5,7 accelerate launch \
    --num_processes 2 \
    train_rqvae.py \
    --config config/1024/rqvae.yaml \
    --project_dir ${project_dir} \
    > "${project_dir}/accelerate.log" 2>&1 &

tensorboard --logdir="output/rqvae" --host $POD_IP --port 8080