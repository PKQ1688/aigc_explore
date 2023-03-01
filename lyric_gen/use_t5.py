#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 16:46
# @Author  : Adolf
# @Site    : 
# @File    : use_t5.py
# @Software: PyCharm
import time
from rich import print
from transformers import T5Tokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline

tokenizer = T5Tokenizer.from_pretrained("model/language_model/t5-chinese-lyric")
model = T5ForConditionalGeneration.from_pretrained("model/language_model/t5-chinese-lyric")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

st_time = time.time()
res = text2text_generator("用户：写一首歌，歌曲“多少年后”，以“谎言，伤痛，心爱，掩饰，退一步，寻找”为主题。\\n小L：", max_length=512, do_sample=True)
print(res)
print(time.time() - st_time)
