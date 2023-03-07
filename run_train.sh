/home/jiaanwang/anaconda3/envs/glm/bin/python3.8 train.py \
                                                --model-type xglm \
                                                --load_model_dir /home/fengwang/ASR/model/xglm1.7b \
                                                --save_train_model_file /home/fengwang/ASR/XGLM/checkpoint 
                                                

# /home/jiaanwang/anaconda3/envs/glm/bin/python3.8 -m torch.distributed.launch --nproc_per_node 2 train.py \
#                                                                                 --model-type glm \
#                                                                                 --load_model_dir /home/fengwang/ASR/model/bloom_3b \
#                                                                                 --save_train_model_file /home/fengwang/ASR/XGLM/checkpoint