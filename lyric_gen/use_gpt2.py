#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:14
# @Author  : Adolf
# @Site    : 
# @File    : use_gpt2.py
# @Software: PyCharm
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("model/language_model/gpt2-chinese-lyric")
model = GPT2LMHeadModel.from_pretrained("model/language_model/gpt2-chinese-lyric")

text_generator = TextGenerationPipeline(model, tokenizer)
res = text_generator("以“谎言，伤痛，心爱，掩饰，退一步，寻找”为主题写一首歌：", max_length=300, do_sample=True)
print(res)
