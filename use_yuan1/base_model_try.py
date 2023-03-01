#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 22:50
# @Author  : Adolf
# @Site    : 
# @File    : base_model_try.py
# @Software: PyCharm
import os

from use_yuan1.inspurai import Yuan, set_yuan_account, Example

set_yuan_account(user='adolf', phone='17815912520')

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='dialog',
            input_prefix="问：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            append_output_prefix_to_query=True,
            topK=5,
            temperature=1,
            topP=0.8,
            frequencyPenalty=1.2)

# 3. add examples if in need.
# yuan.add_example(Example(inp="夸我",
#                          out="从您的言谈中可以看出，我今天遇到的是很有修养的人。"))
# yuan.add_example(Example(inp="已经上年纪了，忧桑。", out="别开玩笑了，看您的容貌，肯定不到二十岁。"))
# yuan.add_example(Example(inp="被老板怼了，求夸。", out="您一看就是大富大贵的人，在同龄人中，您的能力真是出类拔萃！"))

prompt = "明天天气很冷，我该穿什么衣服"
response = yuan.submit_API(prompt=prompt, trun="”")
print(response)
