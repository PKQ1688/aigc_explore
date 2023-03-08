#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    :   label_tool.py
# @Time    :   2023/02/07 10:54:08
# @Author  :   Adolf 
# @Desc    :   None
import torch
import random
import pandas as pd
import streamlit as st
from itertools import combinations
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Rank Labeler Tool",
    page_icon='🏖️',
    layout="wide"
)

MODEL_CONFIG = {
    'model_name': 'uer/gpt2-chinese-cluecorpussmall',  # backbone
    'device': 'cuda:0',  # 使用设备
    'dataset_file': 'data/human_labeled/total_dataset.tsv',  # 标注数据集的存放文件
    'rank_list_len': 4,  # 排序列表的长度
    'max_gen_seq_len': 40,  # 生成答案最大长度
    'random_prompts': [  # 随机prompt池
        '今天我去了',
        '这部电影很',
        '刚收到货，感觉',
        '这部电影很',
        '说实话，真的很',
        '这次购物总的来说体验很'
    ]
}

######################## 会话缓存初始化 ###########################
# if 'model_config' not in st.session_state:
# st.session_state['model_config'] = MODEL_CONFIG

if 'model' not in st.session_state:
    # model_name = st.session_state['model_config']['model_name']
    # st.session_state['model'] = T5Tokenizer.from_pretrained(model_name)
    st.session_state['model'] = T5ForConditionalGeneration.from_pretrained("model/language_model/ChatYuan-large-v1")
    st.session_state['model'].to(device)

if 'tokenizer' not in st.session_state:
    # model_name = st.session_state['model_config']['model_name']
    st.session_state['tokenizer'] = T5Tokenizer.from_pretrained("model/language_model/ChatYuan-large-v1")

# if 'generator' not in st.session_state:
#     st.session_state['generator'] = T5ForConditionalGeneration(
#         st.session_state['model'],
#         st.session_state['tokenizer'],
#         device=MODEL_CONFIG['device']
#     )

if 'current_results' not in st.session_state:
    st.session_state['current_results'] = [''] * MODEL_CONFIG['rank_list_len']

if "results_combinations" not in st.session_state:
    st.session_state['results_combinations'] = ["初始化结果"]

if 'current_prompt' not in st.session_state:
    st.session_state['current_prompt'] = '我被邻居家的狗咬了怎么办？'


######################### 函数定义区 ##############################
# 基础的预处理和后处理
def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")


def answer(text, sample=True, top_p=1, temperature=0.7):
    """sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样"""
    text = preprocess(text)
    encoding = st.session_state['tokenizer'](text=[text], truncation=True, padding=True, max_length=768,
                                             return_tensors="pt").to(device)
    if not sample:
        out = st.session_state['model'].generate(**encoding, return_dict_in_generate=True, output_scores=False,
                                                 max_new_tokens=512,
                                                 num_beams=1, length_penalty=0.6)
    else:
        out = st.session_state['model'].generate(**encoding, return_dict_in_generate=True, output_scores=False,
                                                 max_new_tokens=512,
                                                 do_sample=True, top_p=top_p, temperature=temperature,
                                                 no_repeat_ngram_size=3)
    out_text = st.session_state['tokenizer'].batch_decode(out["sequences"], skip_special_tokens=True)
    # st.session_state['current_results'] = postprocess(out_text[0])
    return postprocess(out_text[0])


def generate_text():
    """
    模型生成文字。
    """
    current_results = []
    for _ in range(MODEL_CONFIG['rank_list_len']):
        res = answer(text=st.session_state['current_prompt'])
        current_results.append(res)
    st.session_state['current_results'] = current_results


def data_show(idx):
    one_res = st.session_state['results_combinations'][idx]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<font color=#008000> {one_res[0]} </font>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<font color=blue> {one_res[1]} </font>", unsafe_allow_html=True)

    with col1:
        # st.radio(label="上一句")
        selected_answers = st.radio(label="", options=["左", "右"])
    with col2:
        st.write("")
        st.write("")
        st.write("")
        next_ = st.button("next")

    return selected_answers, next_


######################### 页面定义区（侧边栏） ########################

st.sidebar.title('Rank Labeler Tool')

label_tab, dataset_tab = st.tabs(['Label', 'Dataset'])

with label_tab:
    random_button = st.button('随机 prompt',
                              help='从prompt池中随机选择一个prompt，可通过修改源码中 MODEL_CONFIG["random_prompts"] 参数来自定义prompt池。')
    if random_button:
        prompt_text = random.choice(MODEL_CONFIG['random_prompts'])
    else:
        prompt_text = st.session_state['current_prompt']

    query_txt = st.text_input('prompt: ', prompt_text)
    if query_txt != st.session_state['current_prompt']:
        st.session_state['current_prompt'] = query_txt
        generate_text()
        st.session_state['results_combinations'] = list(combinations(st.session_state["current_results"], 2))
        # st.write(st.session_state['current_results'])

    with st.expander('💡 判断左边的结果是否比右边的结果好', expanded=True):
        if st.session_state['current_results'][0] == '':
            generate_text()
            st.session_state['results_combinations'] = list(combinations(st.session_state["current_results"], 2))

        i = 0
        selected_answers, next_ = data_show(i)
        i = i + 1
        while next_ and i < len(st.session_state['results_combinations']) - 1:
            data_show(i)
        # columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
        # columns = 3
        # st.write("判断左边的结果是否比右边的结果好")

        # rank_results = [-1] * MODEL_CONFIG['rank_list_len']
        # rank_choices = [-1] + [i + 1 for i in range(MODEL_CONFIG['rank_list_len'])]
        # for i, c in enumerate(columns):
        #     with c:
        #         choice = st.selectbox(f'句子{i+1}排名', rank_choices, help='为当前句子选择排名，排名越小，得分越高。（-1代表当前句子暂未设置排名）')
        #         if choice != -1 and choice in rank_results:
        #             st.info(f'当前排名[{choice}]已经被句子[{rank_results.index(choice)+1}]占用，请先将占用排名的句子置为-1再为当前句子分配该排名。')
        #         else:
        #             rank_results[i] = choice
        #         color = RANK_COLOR[i] if i < len(RANK_COLOR) else 'white'
        #         # st.write(color)
        #         st.markdown(f":{color}[{st.session_state['current_results'][i]}]")

with dataset_tab:
    st.write('Dataset')
