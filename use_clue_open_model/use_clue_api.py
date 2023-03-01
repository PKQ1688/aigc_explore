#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 00:57
# @Author  : Adolf
# @Site    : 
# @File    : use_clue_api.py
# @Software: PyCharm
import clueai

cl = clueai.Client('ZQ1gydAgQqiA4le_dCeXx10000011011')

prompt = '''
帮我生成一些歌词：
歌名：多少年后
主题：谎言，伤痛，心爱，掩饰，退一步，寻找
歌词：
'''

prediction = cl.generate(
    model_name='clueai-large',
    prompt=prompt)
# 需要返回得分的话，指定return_likelihoods="GENERATION"

# print the predicted text
print('prediction: {}'.format(prediction.generations[0].text))
