import argparse
import sys
sys.path.append("/sise/home/noagoren/RecSysFinal")
#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import inspect

# Import for RAG
import faiss
                    
from src.param import parse_args
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from transformers.file_utils import is_apex_available

from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining

# Add imports for RAG dataset
from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader

_use_native_amp = False
_use_apex = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
# try:
#     from packaging import version
#     if version.parse(torch.__version__) < version.parse("1.6"):
#         if is_apex_available():
#             from apex import amp
#         _use_apex = True
#     else:
#         _use_native_amp = True
#         from torch.cuda.amp import autocast
# except ImportError:
#     if is_apex_available():
#         from apex import amp
#     _use_apex = True

from src.trainer_base import TrainerBase

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run P5 model evaluation with specified arguments.")
parser.add_argument("--backbone", type=str, default="t5-small", help="Model backbone (e.g., 't5-small', 't5-base').")
parser.add_argument("--output", type=str, default="snap/results", help="Output directory for results.")
parser.add_argument("--load", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--data", type=str, default="beauty", help="Dataset to use (e.g., beauty, sports, toys).")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
# parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
# parser.add_argument("--rag", action="store_true", help="Enable RAG for all tasks.")
args_parsed = parser.parse_args()

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
args = DotDict()

args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = args_parsed.data
args.valid = args_parsed.data
args.test = args_parsed.data
args.batch_size = args_parsed.batch_size
args.optim = 'adamw' 
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.losses = 'rating,sequential,explanation,review,traditional'
args.backbone = args_parsed.backbone
args.output = args_parsed.output
args.epoch = 10
args.local_rank = 0
args.comment = 'rag'
args.train_topk = -1
args.valid_topk = -1
args.dropout = 0.1
args.tokenizer = 'p5'
args.max_text_length = 512
args.do_lower_case = False
args.word_mask_rate = 0.15
args.gen_max_length = 64
args.weight_decay = 0.01
args.adam_eps = 1e-6
args.gradient_accumulation_steps = 1
# args.seed = 2022
# args.load = args_parsed.load

# # Set seeds
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# np.random.seed(args.seed)

'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


# Whole word embedding
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss') # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 0
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

comments = []
dsets = []
if 'toys' in args.train:
    dsets.append('toys')
if 'beauty' in args.train:
    dsets.append('beauty')
if 'sports' in args.train:
    dsets.append('sports')
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)

if args.local_rank in [0, -1]:
    print(args)

def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone
    
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)
    
    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model

# Create config, tokenizer, and model
config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone
    
tokenizer = create_tokenizer(args)

model_class = P5Pretraining
model = create_model(model_class, config)

model = model.cuda()

if 'p5' in args.tokenizer:
    model.resize_token_embeddings(tokenizer.vocab_size)
    
model.tokenizer = tokenizer

args.load = args_parsed.load


def initialize_whole_word_embeddings(model):
    """
    Initialize whole word embeddings from token embeddings without fine-tuning
    """
    print("\nInitializing whole word embeddings from regular embeddings")
    
    if hasattr(model.encoder, 'whole_word_embeddings'):
        if model.encoder.whole_word_embeddings.weight.shape != model.encoder.embed_tokens.weight.shape:
            vocab_size, hidden_size = model.encoder.embed_tokens.weight.shape
            model.encoder.whole_word_embeddings = nn.Embedding(vocab_size, hidden_size).to('cuda')
            print(f"Recreated whole_word_embeddings with correct shape: {model.encoder.whole_word_embeddings.weight.shape}")
        
        with torch.no_grad():
            model.encoder.whole_word_embeddings.weight.data.copy_(model.encoder.embed_tokens.weight.data)




def debug_whole_word_embeddings():
    print("\nDebugging whole word embeddings:")
    print("Model has whole_word_embeddings attribute:", hasattr(model.encoder, 'whole_word_embeddings'))
    
    if hasattr(model.encoder, 'whole_word_embeddings'):
        print("Shape of whole_word_embeddings:", model.encoder.whole_word_embeddings.weight.shape)
        print("Shape of embed_tokens:", model.encoder.embed_tokens.weight.shape)
        
        # Check if they have the same dimensions
        if model.encoder.whole_word_embeddings.weight.shape != model.encoder.embed_tokens.weight.shape:
            print("WARNING: Shapes do not match! Fixing the whole_word_embeddings dimension...")
            # Create a new embedding layer with the correct dimensions
            vocab_size, hidden_size = model.encoder.embed_tokens.weight.shape
            model.encoder.whole_word_embeddings = nn.Embedding(vocab_size, hidden_size).to('cuda')
            print("New shape of whole_word_embeddings:", model.encoder.whole_word_embeddings.weight.shape)
            print("Now shapes match, weights can be copied correctly")
        else:
            print("Shapes match, weights can be copied")
            
    # Check parameter initialization status
    has_uninit_values = torch.isnan(model.encoder.whole_word_embeddings.weight).any()
    print("Whole word embeddings contain NaN values:", has_uninit_values)
    
    # Check if the embedding is used in forward pass
    model_type = 't5' if 't5' in args.backbone else 'unknown'
    if model_type == 't5':
        print("This is a T5 model - checking relevant code paths...")
        print("Model inputs in forward pass include whole_word_ids:", 'whole_word_ids' in inspect.signature(model.encoder.forward).parameters)

# Load Model Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, device)
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)

ckpt_path = args.load
load_checkpoint(ckpt_path)


class RAGEnhancedP5Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, embedder, item_index, item_ids, 
                 review_index, review_meta, meta_data, meta_dict, id2item, user_items,
                 mode='test', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        # RAG components
        self.embedder = embedder
        self.item_index = item_index
        self.item_ids = item_ids
        self.review_index = review_index
        self.review_meta = review_meta
        self.meta_data = meta_data
        self.meta_dict = meta_dict
        self.id2item = id2item
        self.user_items = user_items
        

        path = os.getcwd()
        print('Data sources: ', split.split(','))
        self.mode = mode
        if self.mode == 'train':
            self.review_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'review_splits.pkl'))['train']
            self.exp_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'exp_splits.pkl'))['train']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'rating_splits_augmented.pkl'))['train']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'val':
            self.review_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'review_splits.pkl'))['val']
            self.exp_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'exp_splits.pkl'))['val']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'rating_splits_augmented.pkl'))['val']
            else:
                self.rating_data = self.review_data
        elif self.mode == 'test':
            self.review_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
            if os.path.exists(os.path.join(path, 'notebooks/data', split, 'zeroshot_exp_splits.pkl')):
                self.zeroshot_exp_data = load_pickle(os.path.join(path, 'notebooks/data', split, 'zeroshot_exp_splits.pkl'))
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join(path, 'notebooks/data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items_data = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items_data[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join(path, 'notebooks/data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join(path, 'notebooks/data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        
        self.user_id2name = load_pickle(os.path.join(path, 'notebooks/data', split, 'user_id2name.pkl'))
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    # compute_datum_info function intends to plan which data sample to be used for which task group according to the sample numbers
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                # The first group of sequential prompts (directly predict next item): 2-1 to 2-6 and 2-13
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 13 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of sequential prompts (predict next item from a candidate list): 2-7 to 2-10
                if sum([6 < int(ind.split('-')[1]) <= 10 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
                # The third group of sequential prompts (predict yes or no for each user-item pair): 2-11 to 2-12
                if sum([10 < int(ind.split('-')[1]) <= 12 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][2]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][2]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'review':
                self.total_length += len(self.review_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'traditional':
                # The first group of direct recommendation prompts (predict yes or no for each user-item pair): 5-1 to 5-4
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # The second group of direct recommendation prompts (choose one item from 100 candidates): 5-5 to 5-8
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.user2id) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'zeroshot':
                if sum([0 < int(ind.split('-')[1]) <= 7 for ind in self.task_list[key]]):
                    self.total_length += len(self.zeroshot_exp_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
            else:
                raise NotImplementedError
    
    # use Gaussian sampling to augment rating scores
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'rating':            
            rating_datum = self.rating_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['rating'][task_candidates[task_idx]]
            assert task_template['task'] == 'rating'
            
            # Process based on task template ID
            if task_template['id'] == '1-6':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                user_id = self.user2id[rating_datum['reviewerID']]
                target_item = self.item2id[rating_datum['asin']]
                source_text = task_template['source'].format(user_desc, target_item)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
                
                # RAG enhancement for task 1-6
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Rating prediction for user preferences similar to {user_desc}"
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '1-10':
                if 'reviewerName' in rating_datum:
                    user_desc = rating_datum['reviewerName']
                else:
                    user_desc = rating_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title)
                target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
                
                # RAG enhancement for task 1-10
                user_id = self.user2id[rating_datum['reviewerID']]
                target_item = self.item2id[rating_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Rating prediction for items similar to {title} for users with preferences like {user_desc}"
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"

        elif task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] == '2-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_id, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
                
                # RAG enhancement for task 2-3
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Recommend next items similar to user {user_id} purchase history"
                    for item in purchase_history[-3:]:  # Add recent items to query
                        item_id = str(item)
                        if item_id in self.id2item:
                            item_asin = self.id2item[item_id]
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    query += f", {item_meta['title']}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '2-13':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' -> '.join(purchase_history))
                target_text = task_template['target'].format(target_item)
                
                # RAG enhancement for task 2-13
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Recommended items for a user who likes"
                    for item in purchase_history[-3:]:  # Add recent items to query
                        item_id = str(item)
                        if item_id in self.id2item:
                            item_asin = self.id2item[item_id]
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    query += f", {item_meta['title']}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"

        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            
            if task_template['id'] == '3-3':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title)
                target_text = task_template['target'].format(exp_datum['explanation'])
                
                # RAG enhancement for task 3-3
                user_id = self.user2id[exp_datum['reviewerID']]
                target_item = self.item2id[exp_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Explanation for {int(exp_datum['overall'])} star rating of {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Reviews about {title} with {int(exp_datum['overall'])} stars"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    if retrieved_reviews:
                        enhanced_context += "\nRelevant review snippets: "
                        for review in retrieved_reviews:
                            if 'reviewText' in review:
                                text = review['reviewText'][:100] + "..." if len(review['reviewText']) > 100 else review['reviewText']
                                enhanced_context += f'"{text}" '
                        enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '3-9':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title)
                target_text = task_template['target'].format(exp_datum['explanation'])
                
                # RAG enhancement for task 3-9
                user_id = self.user2id[exp_datum['reviewerID']]
                target_item = self.item2id[exp_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Explanation about {title} focusing on {exp_datum['feature']}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Reviews mentioning {exp_datum['feature']} about {title}"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    if retrieved_reviews:
                        enhanced_context += "\nRelevant review snippets: "
                        for review in retrieved_reviews:
                            if 'reviewText' in review:
                                text = review['reviewText'][:100] + "..." if len(review['reviewText']) > 100 else review['reviewText']
                                enhanced_context += f'"{text}" '
                        enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '3-12':
                if 'reviewerName' in exp_datum:
                    user_desc = exp_datum['reviewerName']
                else:
                    user_desc = exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']])
                target_text = task_template['target'].format(exp_datum['explanation'])
                
                # RAG enhancement for task 3-12
                user_id = self.user2id[exp_datum['reviewerID']]
                target_item = self.item2id[exp_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Explanation focused on {exp_datum['feature']} for {int(exp_datum['overall'])} star rating by {user_desc}"
                    
                    if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                        title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                        query += f" about {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Reviews mentioning {exp_datum['feature']} with {int(exp_datum['overall'])} stars"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    if retrieved_reviews:
                        enhanced_context += "\nRelevant review snippets: "
                        for review in retrieved_reviews:
                            if 'reviewText' in review:
                                text = review['reviewText'][:100] + "..." if len(review['reviewText']) > 100 else review['reviewText']
                                enhanced_context += f'"{text}" '
                        enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"

        elif task_name == 'review':
            review_datum = self.review_data[datum_idx]
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['review'][task_candidates[task_idx]]
            assert task_template['task'] == 'review'
            
            if task_template['id'] == '4-1':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(review_datum['summary'])
                
                # RAG enhancement for task 4-1
                user_id = self.user2id[review_datum['reviewerID']]
                target_item = self.item2id[review_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Summarize review about"
                    
                    if 'title' in self.meta_data[self.meta_dict[review_datum['asin']]]:
                        title = self.meta_data[self.meta_dict[review_datum['asin']]]['title']
                        query += f" {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Review summaries similar to {review_datum['summary'] if 'summary' in review_datum else ''}"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '4-2':
                source_text = task_template['source'].format(self.user2id[review_datum['reviewerID']], review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
                
                # RAG enhancement for task 4-2
                user_id = self.user2id[review_datum['reviewerID']]
                target_item = self.item2id[review_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Predict rating from review text"
                    
                    if 'title' in self.meta_data[self.meta_dict[review_datum['asin']]]:
                        title = self.meta_data[self.meta_dict[review_datum['asin']]]['title']
                        query += f" about {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Reviews with {int(review_datum['overall'])} star ratings"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    if retrieved_reviews:
                        enhanced_context += "\nRelevant review snippets with similar ratings: "
                        for review in retrieved_reviews:
                            if 'reviewText' in review:
                                text = review['reviewText'][:100] + "..." if len(review['reviewText']) > 100 else review['reviewText']
                                enhanced_context += f'"{text}" '
                        enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
                    
            elif task_template['id'] == '4-4':
                if 'reviewerName' in review_datum:
                    user_desc = review_datum['reviewerName']
                else:
                    user_desc = review_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, review_datum['reviewText'])
                target_text = task_template['target'].format(int(review_datum['overall']))
                
                # RAG enhancement for task 4-4
                user_id = self.user2id[review_datum['reviewerID']]
                target_item = self.item2id[review_datum['asin']]
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Predict rating from review by {user_desc}"
                    
                    if 'title' in self.meta_data[self.meta_dict[review_datum['asin']]]:
                        title = self.meta_data[self.meta_dict[review_datum['asin']]]['title']
                        query += f" about {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    # Retrieve relevant reviews
                    retrieved_reviews = []
                    review_query = f"Reviews with {int(review_datum['overall'])} star ratings by {user_desc}"
                    review_query_embedding = self.embedder.encode([review_query])
                    faiss.normalize_L2(review_query_embedding)
                    scores, indices = self.review_index.search(review_query_embedding, 3)
                    for idx in indices[0]:
                        if idx < len(self.review_meta):
                            retrieved_reviews.append(self.review_meta[idx])
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    if retrieved_reviews:
                        enhanced_context += "\nRelevant review snippets with similar ratings: "
                        for review in retrieved_reviews:
                            if 'reviewText' in review:
                                text = review['reviewText'][:100] + "..." if len(review['reviewText']) > 100 else review['reviewText']
                                enhanced_context += f'"{text}" '
                        enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"

        elif task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                target_candidates = sequence[1:-2]
                target_idx = random.randint(0, len(target_candidates)-1) # random choose the target index for target_candidates
                target_item = target_candidates[target_idx]
            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1)
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)

                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, candidate_samples[0])
                    target_text = task_template['target'].format('no')

                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Recommend items for user {user_id} preferences"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item if rand_prob > 0.5 else None)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"

            elif task_template['id'] == '5-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        if self.sample_type == 'random':
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                        else:
                            sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                        sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                        candidate_samples.extend(sample_ids)
                    candidate_samples = candidate_samples[:candidate_num]
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[0]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
                
                # RAG enhancement for task 5-4
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    query = f"Items to recommend for user {user_id}"
                    if rand_prob > 0.5:
                        query += f" including {title}"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=3
                    )
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item if rand_prob > 0.5 else None)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
            elif task_template['id'] == '5-5' or task_template['id'] == '5-8':
                user_seq = self.user_items[user_id]
                candidate_samples = []
                candidate_num = 99
                while len(candidate_samples) < candidate_num:
                    if self.sample_type == 'random':
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False)
                    else:
                        sample_ids = np.random.choice(self.all_item, candidate_num, replace=False, p=self.probability)
                    sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in candidate_samples]
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.extend([target_item])
                random.shuffle(candidate_samples)
                
                # For 5-5 format with user_desc
                if task_template['id'] == '5-5':
                    source_text = task_template['source'].format(user_desc, ' , '.join(candidate_samples))
                # For 5-8 format with user_id
                else:
                    source_text = task_template['source'].format(user_id, ' , '.join(candidate_samples))
                    
                target_text = task_template['target'].format(target_item)
                
                # RAG enhancement for tasks 5-5 and 5-8
                if user_id in self.user_items:
                    user_history = self.user_items.get(user_id, [])
                    
                    # Create query to find similar items to user's preferred items
                    query = f"Recommend items for user with the following preferences:"
                    recent_items = user_history[-5:] if len(user_history) >= 5 else user_history
                    for item_id in recent_items:
                        if item_id in self.id2item:
                            item_asin = self.id2item[item_id]
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    query += f" {item_meta['title']},"
                    
                    retrieved_items, scores = retrieve_relevant_items(
                        query, user_history, self.embedder, self.item_index, self.item_ids, 
                        self.meta_data, self.meta_dict, self.id2item, n_results=5
                    )
                    
                    enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                    
                    if retrieved_items:
                        enhanced_context += "\nSimilar items that might be relevant: "
                        retrieved_info = []
                        for item_asin in retrieved_items:
                            if item_asin in self.meta_dict:
                                item_meta = self.meta_data[self.meta_dict[item_asin]]
                                if 'title' in item_meta:
                                    retrieved_info.append(f"{item_meta['title']}")
                        enhanced_context += ", ".join(retrieved_info)
                        enhanced_context += ".\n"
                    
                    # For these tasks, also add specific information about the target item
                    target_asin = self.id2item[str(target_item)]
                    if target_asin in self.meta_dict:
                        target_meta = self.meta_data[self.meta_dict[target_asin]]
                        if 'title' in target_meta:
                            enhanced_context += f"\nTarget item information: {target_meta['title']}"
                            if 'description' in target_meta:
                                if isinstance(target_meta['description'], list) and len(target_meta['description']) > 0:
                                    enhanced_context += f" - {target_meta['description'][0][:100]}..."
                                elif isinstance(target_meta['description'], str):
                                    enhanced_context += f" - {target_meta['description'][:100]}..."
                            enhanced_context += "\n"
                    
                    source_text = f"{enhanced_context}\n{source_text}"
        
        input_ids = self.tokenizer.encode(
        source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
    
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        # Prepare output dictionary
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']
        out_dict['loss_weight'] = loss_weight

        return out_dict

    # def __getitem__(self, idx):
    #     out_dict = {}
    #     out_dict['args'] = self.args
        
    #     loss_weight = 1.0
        
    #     datum_info_idx = self.datum_info[idx]
    #     assert datum_info_idx[0] == idx
    #     if len(datum_info_idx) == 3:
    #         task_name = datum_info_idx[1]
    #         datum_idx = datum_info_idx[2]
    #     elif len(datum_info_idx) == 4:
    #         task_name = datum_info_idx[1]
    #         datum_idx = datum_info_idx[2]
    #         task_idx = datum_info_idx[3]
    #     else:
    #         raise NotImplementedError
        
    #     # Extract data based on task type
    #     if task_name == 'rating':            
    #         rating_datum = self.rating_data[datum_idx]
    #         task_candidates = self.task_list[task_name]
    #         task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
    #         task_template = self.all_tasks['rating'][task_candidates[task_idx]]
    #         assert task_template['task'] == 'rating'
            
    #         # Original data processing based on task template ID
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         if task_template['id'] == '1-1':
    #             user_id = self.user2id[rating_datum['reviewerID']]
    #             target_item = self.item2id[rating_datum['asin']]
    #             source_text = task_template['source'].format(user_id, target_item)
    #             target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
    #         elif task_template['id'] == '1-2':
    #             user_id = self.user2id[rating_datum['reviewerID']]
    #             if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
    #                 title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
    #         elif task_template['id'] == '1-6':
    #             if 'name' in self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]:
    #                 user_desc = self.user_data[self.user_meta_dict[rating_datum['reviewerID']]]['name']
    #             else:
    #                 user_desc = rating_datum['reviewerID']
    #             source_text = task_template['source'].format(user_desc, self.item2id[rating_datum['asin']])
    #             target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
    #             # else:
    #             #     title = 'unknown title'
    #             # source_text = task_template['source'].format(user_id, title) 
    #             # target_text = task_template['target'].format(self.gaussian_sampling(rating_datum))
    #         # ... rest of your template handling from pretrain_data.py ...
    #         # Just showing a couple examples to save space
            
    #         # Now enhance the source text with RAG
    #         if user_id in self.user_items:
    #             user_history = self.user_items.get(user_id, [])
                
    #             # Create query for RAG retrieval
    #             query = f"Rating prediction for user {user_id} and item {target_item}"
    #             if 'title' in self.meta_data[self.meta_dict[rating_datum['asin']]]:
    #                 title = self.meta_data[self.meta_dict[rating_datum['asin']]]['title']
    #                 query += f" titled: {title}"
                
    #             # Retrieve relevant items
    #             retrieved_items, scores = retrieve_relevant_items(
    #                 query, user_history, self.embedder, self.item_index, self.item_ids, 
    #                 self.meta_data, self.meta_dict, self.id2item, n_results=3
    #             )
                
    #             # Generate enhanced context
    #             enhanced_context = generate_enhanced_context(user_id, user_history, self.meta_data, self.meta_dict, self.id2item, target_item)
                
    #             # Add retrieved items to context
    #             if retrieved_items:
    #                 enhanced_context += "\nSimilar items that might be relevant: "
    #                 retrieved_info = []
    #                 for item_asin in retrieved_items:
    #                     if item_asin in self.meta_dict:
    #                         item_meta = self.meta_data[self.meta_dict[item_asin]]
    #                         if 'title' in item_meta:
    #                             retrieved_info.append(f"{item_meta['title']}")
    #                 enhanced_context += ", ".join(retrieved_info)
    #                 enhanced_context += ".\n"
                
    #             source_text = f"{enhanced_context}\n{source_text}"
        
    #     elif task_name == 'sequential':
    #         # Process sequential data and add RAG enhancement (similar pattern)
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         # Then add RAG enhancement
        
    #     elif task_name == 'explanation':
    #         # Process explanation data and add RAG enhancement (similar pattern)
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         # Then add RAG enhancement
        
    #     elif task_name == 'review':
    #         # Process review data and add RAG enhancement (similar pattern)
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         # Then add RAG enhancement
        
    #     elif task_name == 'traditional':
    #         # Process traditional data and add RAG enhancement (similar pattern)
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         # Then add RAG enhancement
        
    #     elif task_name == 'zeroshot':
    #         # Process zeroshot data and add RAG enhancement (similar pattern)
    #         # ... Your existing code for different templates (from P5_Amazon_Dataset) ...
    #         # Then add RAG enhancement
        
    #     else:
    #         raise NotImplementedError
        
    #     # Encode the input and target
    #     input_ids = self.tokenizer.encode(
    #             source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
    #     tokenized_text = self.tokenizer.tokenize(source_text)
    #     whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
    #     assert len(whole_word_ids) == len(input_ids)
        
    #     target_ids = self.tokenizer.encode(
    #             target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

    #     # Prepare output dictionary
    #     out_dict['input_ids'] = torch.LongTensor(input_ids)
    #     out_dict['input_length'] = len(input_ids)
    #     out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
    #     out_dict['target_ids'] = torch.LongTensor(target_ids)
    #     out_dict['target_length'] = len(target_ids)

    #     out_dict['source_text'] = source_text
    #     out_dict['tokenized_text'] = tokenized_text
    #     out_dict['target_text'] = target_text

    #     out_dict['task'] = task_template['task']

    #     out_dict['loss_weight'] = loss_weight

    #     return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry

def get_rag_enhanced_loader(args, task_list, sample_numbers, embedder, item_index, item_ids, 
                           review_index, review_meta, meta_data, meta_dict, id2item, user_items,
                           split='toys', mode='test', batch_size=16, workers=4, distributed=False):
    """
    Creates a dataloader with RAG enhancement fully integrated.
    """
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)

    if split == 'yelp':
        from all_yelp_templates import all_tasks as task_templates
    else:
        from all_amazon_templates import all_tasks as task_templates

    # Create the RAG-enhanced dataset
    dataset = RAGEnhancedP5Dataset(
        task_templates,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        mode=mode,
        split=split,
        rating_augment=False
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader


class ImprovedEmbedder:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def encode(self, texts, batch_size=8):
        """Generate embeddings using attention pooling over T5 encoder outputs"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # Filter out empty texts
            filtered_texts = [text if text.strip() != "" else "unknown item" for text in batch_texts]
            
            # Tokenize
            inputs = self.tokenizer(filtered_texts, 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=256,  # Shorter context for efficiency
                                    return_tensors="pt").to(self.device)
                                    # return_tensors="pt")
            
            with torch.no_grad():
                try:
                    # Get encoder outputs directly (without using whole_word_ids)
                    encoder_outputs = self.model.encoder(
                        input_ids=inputs['input_ids'], 
                        attention_mask=inputs.get('attention_mask', None)
                    )
                    
                    # Get hidden states based on return type
                    if isinstance(encoder_outputs, tuple):
                        hidden_states = encoder_outputs[0]
                    else:
                        hidden_states = encoder_outputs.last_hidden_state
                    
                    # Use attention pooling for better representation
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        # Create attention weights using mask
                        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        embeddings = sum_embeddings / sum_mask
                    else:
                        # Simple mean if no mask
                        embeddings = torch.mean(hidden_states, dim=1)
                    
                    # Normalize embeddings for cosine similarity
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu().numpy())
                
                except Exception as e:
                    print(f"Error in encoding batch: {e}")
                    # Create fallback embeddings
                    fallback_size = hidden_states.shape[-1] if 'hidden_states' in locals() else 512
                    fallback_embeddings = np.zeros((len(filtered_texts), fallback_size))
                    all_embeddings.append(fallback_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

def build_review_index(review_data, embedder, device='cuda'):
    """Build a FAISS index for review content to enable rich retrieval"""
    print("Building review content index for RAG...")
    review_texts = []
    review_meta = []  # Store metadata about each review
    
    # Extract review texts
    for review in tqdm(review_data):
        if 'reviewText' in review and review['reviewText'].strip():
            review_texts.append(review['reviewText'])
            review_meta.append({
                'asin': review['asin'],
                'user_id': review['reviewerID'],
                'rating': review['overall']
            })
    
    # Generate embeddings in batches
    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(review_texts), batch_size)):
        batch_texts = review_texts[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_texts)
        embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    return index, review_meta, all_embeddings


def build_item_index(meta_data, embedder, device='cuda'):
    """
    Build a FAISS index for item embeddings to enable fast retrieval
    """
    print("Building item index for RAG...")
    item_texts = []
    item_ids = []
    
    # Extract item descriptions and titles
    for i, item in enumerate(tqdm(meta_data)):
        # Combine title and description if available
        item_text = item.get('title', '')
        if 'description' in item:
            if isinstance(item['description'], list) and len(item['description']) > 0:
                item_text += " " + " ".join(item['description'])
            elif isinstance(item['description'], str):
                item_text += " " + item['description']
                
        # Add features/attributes if available
        if 'feature' in item and isinstance(item['feature'], list):
            item_text += " " + " ".join(item['feature'])
            
        item_texts.append(item_text)
        item_ids.append(item['asin'])
    
    # Generate embeddings
    embeddings = []
    batch_size = 8  # Smaller batch size for memory efficiency
    for i in tqdm(range(0, len(item_texts), batch_size)):
        batch_texts = item_texts[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_texts)
        embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity (cosine)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    return index, item_ids, all_embeddings
def retrieve_relevant_items(query, user_history, embedder, index, item_ids, meta_data, meta_dict, id2item, n_results=3):
    """Enhanced retrieval with better error handling"""
    # Encode the query
    try:
        query_embedding = embedder.encode([query])

        
        
        # Check if we got a valid embedding
        if query_embedding.size == 0 or np.isnan(query_embedding).any():
            print("Warning: Invalid query embedding generated")
            return [], []
        
        # if torch.cuda.is_available():
        #     query_embedding = torch.tensor(query_embedding, device='cuda')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        # Search for similar items
        scores, indices = index.search(query_embedding, min(n_results * 2, index.ntotal))  # Get more candidates
        

        # Get the corresponding item IDs
        candidate_items = [item_ids[idx] for idx in indices[0] if idx < len(item_ids)]
        candidate_scores = scores[0][:len(candidate_items)]
        
        # Apply a simple boosting based on user history
        reranked_items = []
        for i, item_asin in enumerate(candidate_items):
            if i >= len(candidate_scores):
                break
                
            score = candidate_scores[i]
            reranked_items.append((item_asin, score))

        # Sort by score and return
        reranked_items.sort(key=lambda x: x[1], reverse=True)
        
        # Take top n results
        result_items = [item for item, _ in reranked_items[:n_results]]
        result_scores = [score for _, score in reranked_items[:n_results]]
        
        return result_items, result_scores
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return [], []  # Return empty results on error

def setup_rag_p5(model, tokenizer, meta_data):
    """Set up RAG components using the existing P5 model with improvements"""
    print("Setting up improved RAG components...")
    
    # Initialize whole word embeddings if not done already
    initialize_whole_word_embeddings(model)
    
    # Create embedder using the model's encoder
    embedder = ImprovedEmbedder(model, tokenizer)
    
    # Build item index
    item_index, item_ids, item_embeddings = build_item_index(meta_data, embedder)
    
    # Verify the index is working
    print(f"FAISS index contains {item_index.ntotal} items")
    
    # Test retrieval with a random query
    if meta_data and 'title' in meta_data[0]:
        test_query = meta_data[0]['title']
        print(f"Testing retrieval with query: '{test_query}'")
        retrieved_items, scores = retrieve_relevant_items(
            test_query, [], embedder, item_index, item_ids, meta_data, 
            {m['asin']: i for i, m in enumerate(meta_data)}, {}, 3
        )
        print(f"Top retrieved items: {retrieved_items}")
        print(f"Retrieval scores: {scores}")
    
    return embedder, item_index, item_ids

def generate_enhanced_context(user_id, user_history, meta_data, meta_dict, id2item, target_item=None):
    """Generate rich context for RAG enhancement"""
    
    # Get user's recent history
    recent_items = user_history[-5:] if len(user_history) >= 5 else user_history
    
    # Calculate item frequency in user history
    item_frequency = {}
    for item_id in user_history:
        if item_id not in item_frequency:
            item_frequency[item_id] = 0
        item_frequency[item_id] += 1
    
    # Get most frequent items (user preferences)
    frequent_items = sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Extract category preferences
    category_counts = {}
    for item_id in user_history:
        if item_id in id2item:
            item_asin = id2item[item_id]
            if item_asin in meta_dict:
                item_meta = meta_data[meta_dict[item_asin]]
                if 'category' in item_meta:
                    categories = item_meta['category'] if isinstance(item_meta['category'], list) else [item_meta['category']]
                    for category in categories:
                        if category not in category_counts:
                            category_counts[category] = 0
                        category_counts[category] += 1
    
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Build context string
    context = f"User {user_id} profile:\n"
    
    # Add top categories
    if top_categories:
        context += "Favorite categories: "
        context += ", ".join([f"{cat}" for cat, _ in top_categories])
        context += ".\n"
    
    # Add information about frequent items
    if frequent_items:
        context += "Frequently interacted with: "
        freq_items_info = []
        for item_id, freq in frequent_items:
            if item_id in id2item:
                item_asin = id2item[item_id]
                if item_asin in meta_dict:
                    item_meta = meta_data[meta_dict[item_asin]]
                    if 'title' in item_meta:
                        freq_items_info.append(f"{item_meta['title']}")
        context += ", ".join(freq_items_info)
        context += ".\n"
    
    # Add information about recent items
    context += "Recent activity: "
    recent_items_info = []
    for item_id in recent_items:
        if item_id in id2item:
            item_asin = id2item[item_id]
            if item_asin in meta_dict:
                item_meta = meta_data[meta_dict[item_asin]]
                if 'title' in item_meta:
                    recent_items_info.append(f"{item_meta['title']}")
    context += ", ".join(recent_items_info[-3:])  # Last 3 for brevity
    context += ".\n"
    
    # Add target item information if provided
    if target_item and target_item in id2item:
        target_asin = id2item[target_item]
        if target_asin in meta_dict:
            target_meta = meta_data[meta_dict[target_asin]]
            context += "Candidate item: "
            if 'title' in target_meta:
                context += f"{target_meta['title']}. "
            if 'description' in target_meta:
                desc = target_meta['description']
                if isinstance(desc, list) and desc:
                    context += f"Description: {desc[0][:100]}... "
                elif isinstance(desc, str):
                    context += f"Description: {desc[:100]}... "
            if 'feature' in target_meta and isinstance(target_meta['feature'], list) and target_meta['feature']:
                context += f"Features: {', '.join(target_meta['feature'][:3])}."
    
    return context

# Load meta data for RAG
print("Loading meta data...")
meta_data = []
path = os.getcwd()
for meta in parse(os.path.join(path, 'notebooks/data', args.test, 'meta.json.gz')):
# for meta in parse(os.path.join('../data', args.test, 'meta.json.gz')):
    meta_data.append(meta)

meta_dict = {m['asin']: i for i, m in enumerate(meta_data)}

print("Loading review data...")
review_data = []
for review in load_pickle(os.path.join(path, 'notebooks/data', args.test, 'review_splits.pkl'))['test']:
    review_data.append(review)




print("Setting up item-based RAG components...")
embedder, item_index, item_ids = setup_rag_p5(model, tokenizer, meta_data)

print("Setting up review-based RAG components...")
review_index, review_meta, review_embeddings = build_review_index(review_data, embedder)

from src.all_amazon_templates import all_tasks as task_templates

from src.all_amazon_templates import all_tasks as task_templates

# Load data splits
path = os.getcwd()
data_splits = load_pickle(f'{path}/notebooks/data/{args.test}/rating_splits_augmented.pkl')
test_review_data = data_splits['test']
print(f"Number of test reviews: {len(test_review_data)}")
# print(f"Sample test review: {test_review_data[0]}")

data_maps = load_json(os.path.join(path, 'notebooks/data', args.test, 'datamaps.json'))
print(f"Number of users: {len(data_maps['user2id'])}")
print(f"Number of items: {len(data_maps['item2id'])}")

# Load user item interactions for RAG enhancement
user_items = {}
sequential_data = ReadLineFromFile(os.path.join(path, 'notebooks/data', args.test, 'sequential_data.txt'))
for line in sequential_data:
    user, items = line.strip().split(' ', 1)
    items = items.split(' ')
    items = [int(item) for item in items]
    user_items[user] = items

id2item = data_maps['id2item']

print(f"Loaded interaction history for {len(user_items)} users")


# Import necessary modules for evaluation
from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)


# print("################ Evaluation - Rating ################")
# print("Testing Rating with scale 1-10")
# test_task_list = {'rating': ['1-10']}
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

# zeroshot_test_loader = get_rag_enhanced_loader(
#         args,
#         test_task_list,
#         test_sample_numbers,
#         embedder,
#         item_index,
#         item_ids,
#         review_index,
#         review_meta,
#         meta_data,
#         meta_dict,
#         id2item,
#         user_items,
#         split=args.test, 
#         mode='test', 
#         batch_size=args.batch_size,
#         workers=0,
#         distributed=args.distributed
# )
# print(f"Number of batches: {len(zeroshot_test_loader)}")

# gt_ratings = []
# pred_ratings = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         gt_ratings.extend(batch['target_text'])
#         pred_ratings.extend(results)
        
# predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if p in [str(i/10.0) for i in list(range(10, 50))]]
# RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
# print('Results for Rating 1-10:')
# print('RMSE {:7.4f}'.format(RMSE))
# MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
# print('MAE {:7.4f}'.format(MAE))

# # Save results to JSON
# rating_results_1_10 = {
#     'task': 'rating_1-10',
#     'metric': {
#         'rmse': RMSE,
#         'mae': MAE
#     }
# }
# with open(os.path.join(args.output, f"{args.test}_{args.backbone}_rating_1-10_rag.json"), 'w') as f:
#     json.dump(rating_results_1_10, f, indent=2)

# print("Testing Rating 1-6")
# test_task_list = {'rating': ['1-6']}
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

# zeroshot_test_loader = get_rag_enhanced_loader(
#         args,
#         test_task_list,
#         test_sample_numbers,
#         embedder,
#         item_index,
#         item_ids,
#         review_index,
#         review_meta,
#         meta_data,
#         meta_dict,
#         id2item,
#         user_items,
#         split=args.test, 
#         mode='test', 
#         batch_size=args.batch_size,
#         workers=0,
#         distributed=args.distributed
# )
# print(f"Number of batches: {len(zeroshot_test_loader)}")

# gt_ratings = []
# pred_ratings = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         gt_ratings.extend(batch['target_text'])
#         pred_ratings.extend(results)
        
# predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if p in [str(i/10.0) for i in list(range(10, 50))]]
# RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
# print('Results for Rating with scale 1-6:')
# print('RMSE {:7.4f}'.format(RMSE))
# MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
# print('MAE {:7.4f}'.format(MAE))

# # Save results to JSON
# rating_results_1_6 = {
#     'task': 'rating_1-6',
#     'metric': {
#         'rmse': RMSE,
#         'mae': MAE
#     }
# }
# with open(os.path.join(args.output, f"{args.test}_{args.backbone}_rating_1-6_rag.json"), 'w') as f:
#     json.dump(rating_results_1_6, f, indent=2)

# print("################ Evaluation - Sequential ################")
# print("Testing Sequential with prompt 2-13")
# test_task_list = {'sequential': ['2-13']}
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

# zeroshot_test_loader = get_rag_enhanced_loader(
#         args,
#         test_task_list,
#         test_sample_numbers,
#         embedder,
#         item_index,
#         item_ids,
#         review_index,
#         review_meta,
#         meta_data,
#         meta_dict,
#         id2item,
#         user_items,
#         split=args.test, 
#         mode='test', 
#         batch_size=args.batch_size,
#         workers=0,
#         distributed=args.distributed
# )
# print(f"Number of batches: {len(zeroshot_test_loader)}")

# all_info = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         beam_outputs = model.generate(
#                 batch['input_ids'].to('cuda'), 
#                 max_length=50, 
#                 num_beams=20,
#                 no_repeat_ngram_size=0, 
#                 num_return_sequences=20,
#                 early_stopping=True
#         )
#         generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
#         for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
#             new_info = {}
#             new_info['target_item'] = item[1]
#             new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
#             all_info.append(new_info)

# gt = {}
# ui_scores = {}
# for i, info in enumerate(all_info):
#     gt[i] = [int(info['target_item'])]
#     pred_dict = {}
#     for j in range(len(info['gen_item_list'])):
#         try:
#             pred_dict[int(info['gen_item_list'][j])] = -(j+1)
#         except:
#             pass
#     ui_scores[i] = pred_dict



# print('Results for sequential 2-13:')
# metrics_5_2_13 = evaluate_all(ui_scores, gt, 5)[1]
# metrics_10_2_13 = evaluate_all(ui_scores, gt, 10)[1]

# # Save results to JSON
# sequential_results_2_13 = {
#     'task': 'sequential_2-13',
#     'metric_5': metrics_5_2_13,
#     'metric_10': metrics_10_2_13
# }
# with open(os.path.join(args.output, f"{args.test}_{args.backbone}_sequential_2-13_rag.json"), 'w') as f:
#     json.dump(sequential_results_2_13, f, indent=2)


# print("Testing Sequential with prompt 2-3")
# test_task_list = {'sequential': ['2-3']}
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

# zeroshot_test_loader = get_rag_enhanced_loader(
#         args,
#         test_task_list,
#         test_sample_numbers,
#         embedder,
#         item_index,
#         item_ids,
#         review_index,
#         review_meta,
#         meta_data,
#         meta_dict,
#         id2item,
#         user_items,
#         split=args.test, 
#         mode='test', 
#         batch_size=args.batch_size,
#         workers=0,
#         distributed=args.distributed
# )
# print(f"Number of batches: {len(zeroshot_test_loader)}")

# all_info = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         beam_outputs = model.generate(
#                 batch['input_ids'].to('cuda'), 
#                 max_length=50, 
#                 num_beams=20,
#                 no_repeat_ngram_size=0, 
#                 num_return_sequences=20,
#                 early_stopping=True
#         )
#         generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
#         for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
#             new_info = {}
#             new_info['target_item'] = item[1]
#             new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
#             all_info.append(new_info)
            
# gt = {}
# ui_scores = {}
# for i, info in enumerate(all_info):
#     gt[i] = [int(info['target_item'])]
#     pred_dict = {}
#     for j in range(len(info['gen_item_list'])):
#         try:
#             pred_dict[int(info['gen_item_list'][j])] = -(j+1)
#         except:
#             pass
#     ui_scores[i] = pred_dict
    
# print('Results for sequential 2-3:')
# metrics_5_2_3 = evaluate_all(ui_scores, gt, 5)[1]
# metrics_10_2_3 = evaluate_all(ui_scores, gt, 10)[1]

# # Save results to JSON
# sequential_results_2_3 = {
#     'task': 'sequential_2-3',
#     'metric_5': metrics_5_2_3,
#     'metric_10': metrics_10_2_3
# }
# with open(os.path.join(args.output, f"{args.test}_{args.backbone}_sequential_2-3_rag.json"), 'w') as f:
#     json.dump(sequential_results_2_3, f, indent=2)

print("################ Evaluation - Explanation ################")
print("Testing Explanation with prompt 3-12")
test_task_list = {'explanation': ['3-12']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=9,
                num_beams=12,
                num_return_sequences=1,
                num_beam_groups=3,
                repetition_penalty=0.7
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('Results for Explanation 3-12:')
print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

# Save results to JSON
explanation_results_3_12 = {
    'task': 'explanation_3-12',
    'metric': {
        'bleu1': BLEU1,
        'bleu4': BLEU4,
        'rouge': ROUGE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_explanation_3-12_rag.json"), 'w') as f:
    json.dump(explanation_results_3_12, f, indent=2)

print("Testing Explanation with prompt 3-9")
test_task_list = {'explanation': ['3-9']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=10,
                num_beams=12,
                num_return_sequences=1,
                num_beam_groups=3
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('Results for Explanation 3-9:')
print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

# Save results to JSON
explanation_results_3_9 = {
    'task': 'explanation_3-9',
    'metric': {
        'bleu1': BLEU1,
        'bleu4': BLEU4,
        'rouge': ROUGE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_explanation_3-9_rag.json"), 'w') as f:
    json.dump(explanation_results_3_9, f, indent=2)

print("Testing Explanation with prompt 3-3")
test_task_list = {'explanation': ['3-3']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=10
        )
        results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tokens_predict.extend(results) 
        tokens_test.extend(batch['target_text'])
        
new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('Results for Explanation 3-3:')
print('BLEU-1 {:7.4f}'.format(BLEU1))
print('BLEU-4 {:7.4f}'.format(BLEU4))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

# Save results to JSON
explanation_results_3_3 = {
    'task': 'explanation_3-3',
    'metric': {
        'bleu1': BLEU1,
        'bleu4': BLEU4,
        'rouge': ROUGE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_explanation_3-3_rag.json"), 'w') as f:
    json.dump(explanation_results_3_3, f, indent=2)

print("################ Evaluation - Review ################")
print("Testing Review with prompt 4-4")
test_task_list = {'review': ['4-4']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

gt_ratings = []
pred_ratings = []
max_eval_samples = 50  # Limit evaluation to 50 batches for speed
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i >= max_eval_samples:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
print('Results for Review 4-4:')
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))

# Save results to JSON
review_results_4_4 = {
    'task': 'review_4-4',
    'metric': {
        'rmse': RMSE,
        'mae': MAE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_review_4-4_rag.json"), 'w') as f:
    json.dump(review_results_4_4, f, indent=2)

print("Testing Review with prompt 4-2")
test_task_list = {'review': ['4-2']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i >= max_eval_samples:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)
        
predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
print('Results for Review 4-2:')
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))

# Save results to JSON
review_results_4_2 = {
    'task': 'review_4-2',
    'metric': {
        'rmse': RMSE,
        'mae': MAE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_review_4-2_rag.json"), 'w') as f:
    json.dump(review_results_4_2, f, indent=2)

print("Testing Review with prompt 4-1")
test_task_list = {'review': ['4-1']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i >= max_eval_samples:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        tokens_predict.extend(results)
        tokens_test.extend(batch['target_text'])

new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU2 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=2, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)
print('Results for Review 4-1:')
print('BLEU-2 {:7.4f}'.format(BLEU2))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

# Save results to JSON
review_results_4_1 = {
    'task': 'review_4-1',
    'metric': {
        'bleu2': BLEU2,
        'rouge': ROUGE
    }
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_review_4-1_rag.json"), 'w') as f:
    json.dump(review_results_4_1, f, indent=2)


print("################ Evaluation - Traditional ################")
print("Testing Traditional with prompt 5-8")
test_task_list = {'traditional': ['5-8']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader), miniters=1000):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
            batch['input_ids'].to('cuda'),
            max_length=50,
            num_beams=20,
            no_repeat_ngram_size=0,
            num_return_sequences=20,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs.sequences, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)


gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict

print('Results for traditional 5-8:')    
res_topk_1 = evaluate_all(ui_scores, gt, 1)[1]
res_topk_5 = evaluate_all(ui_scores, gt, 5)[1]
res_topk_10 = evaluate_all(ui_scores, gt, 10)[1]
# Save results to JSON
traditional_results_5_8 = {
    'task': 'traditional_5-8',
    'metric_1': res_topk_1,
    'metric_5': res_topk_5,
    'metric_10': res_topk_10
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_traditional_5-8_rag.json"), 'w') as f:
    json.dump(traditional_results_5_8, f, indent=2)


print("Testing Traditional with prompt 5-5")
test_task_list = {'traditional': ['5-5']}
test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader), miniters=1000):
    with torch.no_grad():
        results = model.generate_step(batch)
        beam_outputs = model.generate(
            batch['input_ids'].to('cuda'),
            max_length=50,
            num_beams=20,
            no_repeat_ngram_size=0,
            num_return_sequences=20,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        generated_sents = model.tokenizer.batch_decode(beam_outputs.sequences, skip_special_tokens=True)
        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            new_info = {}
            new_info['target_item'] = item[1]
            new_info['gen_item_list'] = new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
            all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict

print('Results for traditional 5-5:')
res_topk_1 = evaluate_all(ui_scores, gt, 1)[1]
res_topk_5 = evaluate_all(ui_scores, gt, 5)[1]
res_topk_10 = evaluate_all(ui_scores, gt, 10)[1]
# Save results to JSON
traditional_results_5_5 = {
    'task': 'traditional_5-5',
    'metric_1': res_topk_1,
    'metric_5': res_topk_5,
    'metric_10': res_topk_10
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_traditional_5-5_rag.json"), 'w') as f:
    json.dump(traditional_results_5_5, f, indent=2)

print("Testing Traditional with prompt 5-1")
test_task_list = {'traditional': ['5-1']}
test_sample_numbers = {'traditional': 100}

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

m = nn.Softmax(dim=1)

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader), miniters=1000):
    with torch.no_grad():
        beam_outputs = model.generate(
            batch['input_ids'].to('cuda'), 
            max_length=50, 
            num_beams=1, 
            no_repeat_ngram_size=2, 
            num_return_sequences=1, 
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        gen_yes_probs = m(beam_outputs.scores[1][:, [4273, 150]])[:, 0].to(device) # 4273 -> 'yes', 150 -> 'no'
        sorted, indices = torch.sort(gen_yes_probs, descending=True)
        new_info = {}
        all_item_ids = [_.split(' ?')[0].split('item_')[-1] for _ in batch['source_text']]
        new_info['target_item'] = all_item_ids[0]
        new_info['gen_item_list'] = [all_item_ids[_] for _ in indices[:20]]
        all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict

print('Results for traditional 5-1:')
res_topk_1 = evaluate_all(ui_scores, gt, 1)[1]
res_topk_5 = evaluate_all(ui_scores, gt, 5)[1]
res_topk_10 = evaluate_all(ui_scores, gt, 10)[1]
# Save results to JSON
traditional_results_5_1 = {
    'task': 'traditional_5-1',
    'metric_1': res_topk_1,
    'metric_5': res_topk_5,
    'metric_10': res_topk_10
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_traditional_5-1_rag.json"), 'w') as f:
    json.dump(traditional_results_5_1, f, indent=2)


print("Testing Traditional with prompt 5-4")
test_task_list = {'traditional': ['5-4']}
test_sample_numbers = {'traditional': 100}

all_item_ids = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    all_item_ids.append([_.split(' ?')[0].split('item_')[-1] for _ in batch['source_text']])

zeroshot_test_loader = get_rag_enhanced_loader(
        args,
        test_task_list,
        test_sample_numbers,
        embedder,
        item_index,
        item_ids,
        review_index,
        review_meta,
        meta_data,
        meta_dict,
        id2item,
        user_items,
        split=args.test, 
        mode='test', 
        batch_size=args.batch_size,
        workers=0,
        distributed=args.distributed
)
print(f"Number of batches: {len(zeroshot_test_loader)}")

m = nn.Softmax(dim=1)

all_info = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    with torch.no_grad():
        beam_outputs = model.generate(
            batch['input_ids'].to('cuda'), 
            max_length=50, 
            num_beams=1, 
            no_repeat_ngram_size=2, 
            num_return_sequences=1, 
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        gen_yes_probs = m(beam_outputs.scores[1][:, [4273, 150]])[:, 0].to(device) # 4273 -> 'yes', 150 -> 'no'
        sorted, indices = torch.sort(gen_yes_probs, descending=True)
        new_info = {}
        new_info['target_item'] = all_item_ids[i][0]
        new_info['gen_item_list'] = [all_item_ids[i][_] for _ in indices[:20]]
        all_info.append(new_info)
            
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict

print('Results for traditional 5-4:')
res_topk_1 = evaluate_all(ui_scores, gt, 1)[1]
res_topk_5 = evaluate_all(ui_scores, gt, 5)[1]
res_topk_10 = evaluate_all(ui_scores, gt, 10)[1]
# Save results to JSON
traditional_results_5_4 = {
    'task': 'traditional_5-4',
    'metric_1': res_topk_1,
    'metric_5': res_topk_5,
    'metric_10': res_topk_10
}
with open(os.path.join(args.output, f"{args.test}_{args.backbone}_traditional_5-4_rag.json"), 'w') as f:
    json.dump(traditional_results_5_4, f, indent=2)