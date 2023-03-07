import torch
from config import Config
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class ChmixEnGen(Dataset):
    def __init__(self, config, seq_data_lis):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
        self.seq_data_lis = seq_data_lis
        self.device = config.device
        self.dataset = self.preprocess()

    def preprocess(self):
        self.get_seq_bloom_tok()

        input_data=[]
        for data in self.seq_data_lis:
            cur_dict=dict()
            cur_dict['seq_bloom_tok'] = data['seq_bloom_tok']
            cur_dict['label'] = data['seq_bloom_label']
            input_data.append(cur_dict)
        return input_data
    

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)
    
    def get_seq_bloom_tok(self):
        for seq_data in self.seq_data_lis:
            seq_data['seq_bloom_tok'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<s>'+seq_data['text']))
            seq_data['seq_bloom_label'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq_data['label']))
        
    def collate_fn(self, batch):
        seq_bloom_tok = [data['seq_bloom_tok'] for data in batch]
        label = [data['label'] for data in batch]
        batch_size = len(seq_bloom_tok)
        max_bloomseq_len = max([len(s) for s in seq_bloom_tok])
        batch_bloom_data = [[0 for i in range(max_bloomseq_len)]for j in range(batch_size)]
        batch_bloom_mask_data = [[0 for i in range(max_bloomseq_len)]for j in range(batch_size)]
        for j in range(batch_size):
            cur_len = len(seq_bloom_tok[j])
            batch_bloom_data[j][:cur_len] = seq_bloom_tok[j]
            batch_bloom_mask_data[j][:cur_len] = [1 for _ in range(cur_len)]

        batch_label = [[-100 for i in range(max_bloomseq_len)]for j in range(batch_size)]
        for j in range(batch_size):
            cur_len = len(label[j])
            batch_label[j][:cur_len] = label[j]

        batch_bloom_data = torch.tensor(batch_bloom_data, dtype=torch.long).to(self.device)
        batch_bloom_mask_data = torch.tensor(batch_bloom_mask_data, dtype=torch.long).to(self.device)
        batch_label = torch.tensor(batch_label, dtype=torch.long).to(self.device)
        return {
            'bloom_data':batch_bloom_data,
            'bloom_mask_data':batch_bloom_mask_data,
            'label': batch_label
        }