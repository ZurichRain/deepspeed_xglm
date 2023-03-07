import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
import sys
sys.path.append('XGLM')
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import logging
import json
from config import Config
import argparse
config = Config()

logging.getLogger().setLevel(logging.INFO)
from model import CodeMixXglm, FP16_Module, CodeMixBloom

parser = argparse.ArgumentParser(description='PyTorch CodeMix Model')
parser.add_argument('--model-type', type=str, default='xglm', help='used model for eval')
parser.add_argument('--load_model_dir', type=str, help='used model struct dir for eval')
parser.add_argument('--load_checkpoint_dir', type=str, help='used checkpoint model dir for eval')
parser.add_argument('--num_beams', type=int, default=3)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--top_p', type=float, default=1.0)
args = parser.parse_args()
config.model_dir = args.load_model_dir
if args.model_type == 'xglm':
    model = CodeMixXglm(config)
else:
    model = CodeMixBloom(config)

# model.load_state_dict(torch.load('/home/fengwang/ASR/XGLM/checkpoint/model_3b_2.ckpt'))
# print(model)
model.load_state_dict(torch.load(args.load_checkpoint_dir))
model.to(config.device)
tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s>'))], dtype=torch.long, device=config.device)
attention_mask = torch.tensor([[1]],dtype=torch.long,device=config.device)
outputs=model.generate(input_ids=input_ids,attention_mask=attention_mask,do_sample=True, num_beams = args.num_beams, top_k=args.top_k, top_p=args.top_p)
outputs=tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(outputs[0])
