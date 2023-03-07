import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('XGLM')
from transformers import AutoTokenizer
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
import deepspeed
import mpu



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
                
                # if config.fp16:
                #     # scaler.scale(loss).backward()
                #     optimizer.backward(loss, update_master_grads=False)
                #     optimizer.update_master_grads()
                #     if config.clip_grad > 0.0:
                #         optimizer.clip_master_grads(config.clip_grad)
                # else:
                #     loss.backward()
                model.backward(loss)
                model.step()
                # optimizer.step()
                # if not config.fp16 and scheduler is not None:
                #     scheduler.step()
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
            torch.save(model.module.state_dict(),config.save_train_model_file+'/model_3b_fp16_'+str(epoch%3)+'.ckpt')
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

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank

    torch.cuda.set_device(device) # 设置model使用的设备
    # Call the init process
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '6000')
    args.master_port = '4568'
    init_method += args.master_ip + ':' + args.master_port

    torch.distributed.init_process_group(
        backend = 'nccl',
        world_size=args.world_size, rank=args.rank, # world_size 设置了当前组内的进程数
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(1)

# .half()
# scaler = GradScaler()
# if config.useddp:
#     dist.init_process_group(backend='nccl')
from model import CodeMixXglm, FP16_Module, CodeMixBloom
from dataset import ChmixEnGen

parser = argparse.ArgumentParser(description='PyTorch CodeMix Model')
parser.add_argument('--model-type', type=str, default='xglm', help='used model for training')
parser.add_argument('--load_model_dir', type=str, help='used local model dir for training')
parser.add_argument('--save_train_model_file', type=str, help='save checkpoint dir')
parser.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

parser = deepspeed.add_config_arguments(parser) #添加deepseed的配置
args = parser.parse_args()

args.rank = int(os.getenv('RANK', '0'))
args.world_size = int(os.getenv("WORLD_SIZE", '1'))
if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_config is not None:
    with open(args.deepspeed_config) as file:
        deepspeed_config = json.load(file)
    if "train_micro_batch_size_per_gpu" in deepspeed_config:
        args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
    if "gradient_accumulation_steps" in deepspeed_config:
        args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
    else:
        args.gradient_accumulation_steps = 1
    # if "optimizer" in deepspeed_config:
    #     optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
    #     args.lr = optimizer_params_config.get("lr", args.lr)
    #     args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)

initialize_distributed(args)
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
    model.half()
    model = FP16_Module(model)

train_data_path = '/home/fengwang/ASR/data/new_zh_en_data/all_data/all_train.txt'
train_data_lis = get_data(train_data_path)
tokenizer = AutoTokenizer.from_pretrained(config.model_dir)

# if mpu.get_model_parallel_rank() == 0:
#     num_tokens = tokenizer.num_tokens
#     before = num_tokens
#     after = before
#     multiple = args.make_vocab_size_divisible_by
#     '''
#         这里保证token数量补充到可以划分
#     '''
#     while (after % multiple) != 0:
#         after += 1
#     print('> padded vocab (size: {}) with {} dummy '
#                     'tokens (new size: {})'.format(before, after - before, after))
#     print('> found end-of-document token: {}'.format(eod_token))
#     # print(torch.cuda.device_count())
#     # exit()
#     token_counts = torch.cuda.LongTensor([after, eod_token])
# else:
#     token_counts = torch.cuda.LongTensor([0, 0])
# # Broadcast num tokens.
# torch.distributed.broadcast(token_counts,
#                             mpu.get_model_parallel_src_rank(),
#                             group=mpu.get_model_parallel_group())


train_dataset = ChmixEnGen(config,train_data_lis)
train_size = len(train_dataset)
if config.useddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn,
                              sampler=train_sampler)
else:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=train_dataset.collate_fn,
                              sampler=train_sampler)

optimizer = optim.AdamW(model.parameters(), lr = config.learning_rate)
model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer = optimizer,
                model_parameters = model.parameters(),
                args=args,
                mpu=mpu,
                dist_init_required=False
            )
optimizer.refresh_fp32_params()
# if config.fp16:
#     optimizer = FP16_Optimizer(optimizer,
#                                 static_loss_scale=config.loss_scale,
#                                 dynamic_loss_scale=True,
#                                 dynamic_loss_args={
#                                     'scale_window': 1000,
#                                     'min_scale': 1, 
#                                     'delayed_shift': 2})
#     optimizer._model_params_to_master_params()

scheduler= None
if not config.fp16:
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
train(model, optimizer, train_loader, scheduler=scheduler)




