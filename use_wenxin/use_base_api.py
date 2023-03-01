#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 00:39
# @Author  : Adolf
# @Site    : 
# @File    : use_base_api.py
# @Software: PyCharm
import wenxin_api
from wenxin_api.tasks.free_qa import FreeQA

wenxin_api.ak = "3wvxG7CAQsKd4k2wOPvXdmX6YCBr7rP8"
wenxin_api.sk = "389RZROWoNKoy997IznGULYarMhbPGa4"
input_dict = {
    "text": "问题：我被邻居家的狗咬了怎么办？回答：",
    "seq_len": 512,
    "topp": 0.5,
    "penalty_score": 1.2,
    "min_dec_len": 2,
    "min_dec_penalty_text": "。?：！[<S>]",
    "is_unidirectional": 0,
    "task_prompt": "qa",
    "mask_type": "paragraph"
}
rst = FreeQA.create(**input_dict)
print(rst)
