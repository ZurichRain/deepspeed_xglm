/home/jiaanwang/anaconda3/envs/glm/bin/python3.8 inference.py \
                                                --model-type xglm \
                                                --load_model_dir /home/fengwang/ASR/model/xglm1.7b \
                                                --load_checkpoint_dir /home/fengwang/ASR/XGLM/checkpoint/model_3b_fp16_0.ckpt \
                                                --num_beams 3 \
                                                --top_k 40 \
                                                --top_p 1.0