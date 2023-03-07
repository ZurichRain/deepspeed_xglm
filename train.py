import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
import sys
sys.path.append('XGLM')
# from transformers import AutoTokenizer, BloomForCausalLM
import torch
from tqdm import tqdm
import logging
import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup 
from torch.utils.data import DataLoader
import json
from torch.cuda.amp import autocast, GradScaler
from fp16.fp16 import FP16_Optimizer, DynamicLossScaler
from util import seed_everything
from config import Config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse



seed_everything()
logging.getLogger().setLevel(logging.INFO)
config = Config()

def train_epoch(train_loader, model, optimizer, epoch ,scheduler=None, scaler=None):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    prey_lis=[]
    truy_lis=[]
    
    if config.use_tqdm:
        train_bar = enumerate(tqdm(train_loader))
    else:
        train_bar = enumerate(train_loader)
    for idx, batch_samples in train_bar:
        optimizer.zero_grad()
        cbatch=batch_samples
        # print(cbatch)
        
        while True:
            # loss = model(**cbatch)
            loss = model(cbatch['bloom_data'], cbatch['bloom_mask_data'], label = cbatch['label'])
            reduced_loss = loss.detach().clone().view(1)
            if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
                train_losses += reduced_loss.item()
                if config.fp16:
                    # scaler.scale(loss).backward()
                    optimizer.backward(loss, update_master_grads=False)
                    optimizer.update_master_grads()
                    if config.clip_grad > 0.0:
                        optimizer.clip_master_grads(config.clip_grad)
                else:
                    loss.backward()
                optimizer.step()
                if not config.fp16 and scheduler is not None:
                    scheduler.step()
                break
            else:
                print("Found NaN loss, skip backward")
                del loss

        torch.cuda.empty_cache()
    
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train( model:torch.nn.Module, optimizer,train_loader, scheduler=None, scaler=None):
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    
    for epoch in range(1, config.epoch_num + 1):
        if config.useddp:
            train_loader.sampler.set_epoch(epoch)
        train_epoch(train_loader, model, optimizer, epoch , scheduler, scaler)

        # torch.save(model,config.save_train_model_file+'/model_'+str(epoch))
        if config.useddp:
            if dist.get_rank() == 0:
                torch.save(model.state_dict(),config.save_train_model_file+'/model_3b_fp16_'+str(epoch)+'.ckpt')
        else:
            torch.save(model.state_dict(),config.save_train_model_file+'/model_3b_fp16_'+str(epoch%3)+'.ckpt')
        # torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

    logging.info("Training Finished!")



def get_data(path):
    data=[]
    with open(path,'r') as f:
        for line in f.readlines():
            data.append({
                'text':line.strip(),
                'label':line.strip()
            })
    return data

# .half()
# scaler = GradScaler()
if config.useddp:
    dist.init_process_group(backend='nccl')
from model import CodeMixXglm, FP16_Module, CodeMixBloom
from dataset import ChmixEnGen

parser = argparse.ArgumentParser(description='PyTorch CodeMix Model')
parser.add_argument('--model-type', type=str, default='xglm', help='used model for training')
parser.add_argument('--load_model_dir', type=str, help='used local model dir for training')
parser.add_argument('--save_train_model_file', type=str, help='save checkpoint dir')
args = parser.parse_args()
config.model_dir = args.load_model_dir
config.save_train_model_file = args.save_train_model_file

print('using model {}'.format(args.model_type))
if args.model_type == 'xglm':    
    model = CodeMixXglm(config)
else:
    model = CodeMixBloom(config)
# model = CodeMix()
# model = CodeMixBloom()
model.to(config.device)
if config.useddp:
    model = DDP(model, device_ids=[0], output_device=0)

if config.fp16:
    # model.half()
    model = FP16_Module(model)

train_data_path = '/home/fengwang/ASR/data/new_zh_en_data/all_data/all_train.txt'
train_data_lis = get_data(train_data_path)
train_dataset = ChmixEnGen(config,train_data_lis)
train_size = len(train_dataset)
if config.useddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn,
                              sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)

optimizer = optim.AdamW(model.parameters(), lr = config.learning_rate)

if config.fp16:
    optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale=config.loss_scale,
                                dynamic_loss_scale=True,
                                dynamic_loss_args={
                                    'scale_window': 1000,
                                    'min_scale': 1, 
                                    'delayed_shift': 2})
    optimizer._model_params_to_master_params()

scheduler= None
if not config.fp16:
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
train(model, optimizer, train_loader, scheduler=scheduler)




