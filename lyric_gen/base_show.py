#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 18:15
# @Author  : Adolf
# @Site    : 
# @File    : base_show.py
# @Software: PyCharm
import streamlit as st
from transformers import BertTokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline

# 加载缓存文件
if "t5_token" not in st.session_state:
    st.session_state['t5_token'] = T5Tokenizer.from_pretrained("model/language_model/t5-chinese-lyric")

if "t5_model" not in st.session_state:
    st.session_state["t5_model"] = T5ForConditionalGeneration.from_pretrained("model/language_model/t5-chinese-lyric")

if "t5_pipeline" not in st.session_state:
    st.session_state["t5_pipeline"] = Text2TextGenerationPipeline(st.session_state["t5_model"],
                                                                  st.session_state['t5_token'])

# if "gpt_token" not in st.session_state:
#     st.session_state['gpt_token'] = BertTokenizer.from_pretrained("model/language_model/gpt2-chinese-lyric")
#
# if "gpt_model" not in st.session_state:
#     st.session_state["gpt_model"] = GPT2LMHeadModel.from_pretrained("model/language_model/gpt2-chinese-lyric")
#
# if "gpt_pipeline" not in st.session_state:
#     st.session_state["gpt_pipeline"] = Text2TextGenerationPipeline(st.session_state["gpt_model"],
#                                                                    st.session_state['gpt_token'])

tab1, tab2 = st.tabs(["歌词生成器", "歌词续写"])

with tab1:
    lyric_name = st.text_input(value="多年之后", label="歌曲名")
    lyric_theme = st.text_input(value="谎言，伤痛，心爱，掩饰，退一步，寻找", label="主题")

    run = st.button("开始生成")
    if run:
        res = st.session_state["t5_pipeline"](
            f"用户：写一首歌，歌手“AI助手”，歌曲“{lyric_name}”，以“{lyric_theme}”为主题。\\n小L：",
            max_length=512, do_sample=True)
        res = res[0]["generated_text"]
        res_list = res.split("\\n")
        # st.write(res)
        # st.write(res_list)
        for con in res_list:
            st.write(con)

with tab2:
    st.text_input(value="最美的不是下雨天，是曾与你躲过雨的屋檐", label="要续写的歌词")
    run = st.button("开始续写")
    # if run:
    #     res = st.session_state["gpt_pipeline"]("最美的不是下雨天，是曾与你躲过雨的屋檐", max_length=100, do_sample=True)
    #     res = res[0]["generated_text"]
    #     st.write(res.replace(" ", ""))
