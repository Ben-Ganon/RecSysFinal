import argparse
import sys
sys.path.append("/sise/home/ganonb/RecSysProject")
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
# from packaging import version
from collections import defaultdict
import transformers
from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from src.param import parse_args
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
# if version.parse(torch.__version__) < version.parse("1.6"):
from transformers.file_utils import is_apex_available
if is_apex_available():
    from apex import amp
_use_apex = True
# else:
#     _use_native_amp = True

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


# In[ ]:

parser = argparse.ArgumentParser(description="Run model evaluation with specified arguments.")
parser.add_argument("--backbone", type=str, required=True, help="Model backbone (e.g., 't5-small').")
parser.add_argument("--output", type=str, required=True, help="Output directory for results.")
parser.add_argument("--load", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--data", type=str, required=True)
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
args.test =  args_parsed.data
args.batch_size = 100
args.optim = 'adamw'
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.losses = 'rating,sequential,explanation,review,traditional'
# args.backbone = 't5-small' # small or base
args.backbone = args_parsed.backbone
# args.output = 'snap/sports-small'
args.output = args_parsed.output
args.epoch = 10
args.local_rank = 0

args.comment = ''
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




'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

'''
Whole word embedding
'''
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss') # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 0 # Change GPU ID
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

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M')

if args.local_rank in [0, -1]:
    run_name = f'{current_time}_GPU{args.world_size}'
    if len(comments) > 0:
        run_name += f'_{comment}'
    args.run_name = run_name
    print(args)


# In[ ]:


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


# In[ ]:


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


# #### Load Model

# In[ ]:


# args.load = "../snap/sports-small-group5-part1.pth"
args.load = args_parsed.load

# Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cuda')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)

ckpt_path = args.load
load_checkpoint(ckpt_path)

from src.all_amazon_templates import all_tasks as task_templates


# #### Check Test Split

# In[ ]:


path = os.getcwd()
data_splits = load_pickle(f'{path}/data/sports/rating_splits_augmented.pkl')
# data_splits = load_pickle('../data/sports/rating_splits_augmented.pkl')
test_review_data = data_splits['test']


# In[ ]:


len(test_review_data)


# In[ ]:


test_review_data[0]


# In[ ]:


path = os.getcwd()
data_maps = load_json(os.path.join(f'{path}/data', 'sports', 'datamaps.json'))
# data_maps = load_json(os.path.join('../data', 'sports', 'datamaps.json'))
print(len(data_maps['user2id'])) # number of users
print(len(data_maps['item2id'])) # number of items


# ### Test P5

# In[ ]:


from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader, P5_Yelp_Dataset
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all


# In[ ]:


from multiprocessing import Pool
import math
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

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


# In[ ]:


class P5_Amazon_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type

        print('Data sources: ', split.split(','))
        self.mode = mode
        assert self.mode == 'test'
        if self.mode == 'test':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError

        self.sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1

        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items

        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('data', split, 'negative_samples.txt'))

        datamaps = load_json(os.path.join('data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']

        self.user_id2name = load_pickle(os.path.join('data', split, 'user_id2name.pkl'))

        self.meta_data = []
        for meta in parse(os.path.join('data', split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i

        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

    def compute_datum_info(self):
        curr = 0
        assert 'traditional' in self.task_list.keys()
        key = 'traditional'
        assert 0 < int(self.task_list[key].split('-')[1]) <= 4
        self.total_length += len(self.user2id) * 100
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, key, i // 100, i % 100))
        curr = self.total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        loss_weight = 1.0

        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            candidate_item_idx = datum_info_idx[3]
        else:
            raise NotImplementedError

        if task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            assert self.mode == 'test'
            if candidate_item_idx == 0:
                target_item = sequence[-1]

            task_candidates = self.task_list[task_name]
            task_template = self.all_tasks['traditional'][task_candidates]
            assert task_template['task'] == 'traditional'

            if task_template['id'] == '5-1':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(user_id, candidate_samples[candidate_item_idx-1])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(candidate_samples[candidate_item_idx-1], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                if candidate_item_idx == 0:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                if candidate_item_idx == 0:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            else:
                raise NotImplementedError

        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)

        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

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
                ## length-aware loss normalization
                if entry['target_length'] > 0:
                    loss_weights[i] = entry['loss_weight'] / entry['target_length']
                else:
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


def get_loader(args, task_list, sample_numbers, split='toys', mode='train',
               batch_size=16, workers=4, distributed=False):

    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case)

    if split == 'yelp':
        from all_yelp_templates import all_tasks as task_templates

        dataset = P5_Yelp_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
    else:
        from all_amazon_templates import all_tasks as task_templates

        dataset = P5_Amazon_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
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


# #### Evaluation - Traditional (Prompt 5-1)

# In[ ]:


test_task_list = {'traditional': '5-1'
}
test_sample_numbers = {'traditional': 100}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test,
        mode='test',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))


# In[ ]:


m = nn.Softmax(dim=1)


# In[ ]:


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
        gen_yes_probs = m(beam_outputs.scores[1][:, [4273, 150]])[:, 0].cpu() # 4273 -> 'yes', 150 -> 'no'
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

evaluate_all(ui_scores, gt, 1)
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)


# In[ ]:





# In[ ]:


all_item_ids = []
for i, batch in tqdm(enumerate(zeroshot_test_loader), miniters=1000):
    all_item_ids.append([_.split(' ?')[0].split('item_')[-1] for _ in batch['source_text']])


# #### Evaluation - Traditional (Prompt 5-4)

# In[ ]:


test_task_list = {'traditional': '5-4'
}
test_sample_numbers = {'traditional': 100}

zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test,
        mode='test',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
)
print(len(zeroshot_test_loader))

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
        gen_yes_probs = m(beam_outputs.scores[1][:, [4273, 150]])[:, 0].cpu() # 4273 -> 'yes', 150 -> 'no'
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

res_topk_1 = evaluate_all(ui_scores, gt, 1)[1]
res_topk_5 = evaluate_all(ui_scores, gt, 5)[1]
res_topk_10 = evaluate_all(ui_scores, gt, 10)[1]

res_topk_1_filepath = f"{args.output}/{args_parsed.load}_topk_1.json"
res_topk_5_filepath = f"{args.output}/{args_parsed.load}_topk_5.json"
res_topk_10_filepath = f"{args.output}/{args_parsed.load}_topk_10.json"

json.dump(res_topk_1, open(res_topk_1_filepath, "w"))
json.dump(res_topk_5, open(res_topk_5_filepath, "w"))
json.dump(res_topk_10, open(res_topk_10_filepath, "w"))

