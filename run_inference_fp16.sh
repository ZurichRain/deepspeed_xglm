/home/jiaanwang/anaconda3/envs/glm/bin/python3.8 inference_fp16.py \
                                                --model-type bloom \
                                                --load_model_dir /home/fengwang/ASR/model/bloom_3b \
                                                --load_checkpoint_dir /home/fengwang/ASR/bloom/checkpoint/model_3b_fp16_1.ckpt \
                                                --num_beams 3 \
                                                --top_k 40 \
                                                --top_p 1.0