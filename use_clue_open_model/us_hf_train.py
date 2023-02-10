#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    :   us_hf_train.py
# @Time    :   2023/02/06 14:23:52
# @Author  :   Adolf 
# @Desc    :   None
import torch
from datasets import load_dataset
from functools import partial

# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration

def train():
    tokenizer = AutoTokenizer.from_pretrained("model/language_model/t5-base-chinese-cluecorpussmall/")
    model = T5ForConditionalGeneration.from_pretrained("model/language_model/t5-base-chinese-cluecorpussmall/")

    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.bos_token = tokenizer.cls_token

    dataset = load_dataset('text', data_files={'train': "data/DuReaderQG/train.json",
                                                'dev': "data/DuReaderQG/dev.json"})    
    print(dataset)

    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=256,
        max_target_seq_len=32,
    )
    dataset = dataset.map(convert_func, batched=True)

train()