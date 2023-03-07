NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --include localhost:0 --master_port $(shuf -n 1 -i 10000-65535) train_deepseed.py \
                                                                                            --deepspeed \
                                                                                            --deepspeed_config /home/fengwang/ASR/XGLM/config_xglm.json \
                                                                                            --model-type bloom \
                                                                                            --load_model_dir /home/fengwang/ASR/model/bloom_3b \
                                                                                            --save_train_model_file /home/fengwang/ASR/bloom/checkpoint