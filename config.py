import torch

class Config:
    def __init__(self):
        self.fp16 = True
        self.useddp = False
        
        self.batch_size = 32
        self.epoch_num = 5
        self.save_train_model_file = '/home/fengwang/ASR/XGLM/checkpoint'
        self.model_dir = '/home/fengwang/ASR/model/xglm1.7b'

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.use_tqdm = True
        self.learning_rate = 1e-5
        self.loss_scale = 0
        self.clip_grad = -1
        # self.model_dir = '/home/fengwang/ASR/model/xglm1.7b'
        