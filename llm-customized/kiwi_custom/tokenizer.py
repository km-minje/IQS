import os
import pandas as pd
import json

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords


def custom_kiwi():
    auto_dir = "autopedia.xlsx"
    dict_dir = "cft_dictionary.xlsx"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    auto_dir = os.path.join(current_dir, auto_dir)
    dict_dir = os.path.join(current_dir, dict_dir)
    
    kiwi = Kiwi()
    # autopedia 데이터 추가
    auto_df = pd.read_excel(auto_dir, engine='openpyxl', sheet_name=1)
    auto_keywords = list(auto_df["용어명(*)[한국어]"]) + list(auto_df["용어명(*)[영어]"])
    for keyword in auto_keywords:
        keyword = str(keyword)
        if not keyword.isdigit():
            kiwi.add_user_word(keyword, 'NNP', 0)
    # CFT 사전 데이터 추가
    cft_df = pd.read_excel(dict_dir, engine='openpyxl', sheet_name=1)
    cft_keywords = list(cft_df["키워드"])
    for keyword in cft_keywords:
        keyword = str(keyword)
        if not keyword.isdigit():
            kiwi.add_user_word(keyword, 'NNP', 0)
    # entity 데이터 parse
    entity_df = pd.read_excel(dict_dir, engine='openpyxl', sheet_name=2)
    entity_dict = {entity_df['키워드'][idx]:entity_df['이름'][idx] for idx in range(len(entity_df["키워드"]))} 
        
    return kiwi, entity_dict


def entity_match(data_dir):
    cft_df = pd.read_excel(data_dir, engine='openpyxl')
    id_list = list(cft_df["관리번호"])
    prob_list = list(cft_df["Unnamed: 28"])
    solv_list = list(cft_df["대책"])
    kiwi, entity_dict = custom_kiwi()
    stopwords = Stopwords() # custom 해야 사용할 수 있을듯(default는 성능 안좋음)
    out_dict = {}
    for idx in range(1, 30): # 30개만 test
        a_prob = str(prob_list[idx])
        a_solv = str(solv_list[idx])
        p_dict = {}
        s_dict = {}
        p_tokens = kiwi.tokenize(a_prob, normalize_coda=False)
        s_tokens = kiwi.tokenize(a_solv, normalize_coda=False)
        
        p_noun_list = []
        p_entity_list = []
        s_noun_list = []
        s_entity_list = []
        for idx2 in range(len(p_tokens)-1):
            if (p_tokens[idx2].tag in ["XR", "NNG"]) & (p_tokens[idx2+1].tag == "XSN"): # '어근'/'일반명사' 다음 '명사파생 접미사'가 나오면 합침
                p_noun_list.append(p_tokens[idx2].form + p_tokens[idx2+1].form)
            elif p_tokens[idx2].tag in ["NNG", "NNP", "NNB", "NR", "NP"]: # '명사'만 추출
                p_noun_list.append(p_tokens[idx2].form)
        for a_noun in p_noun_list:
            if a_noun in entity_dict.keys(): # rule-based entity 추출
                p_entity_list.append((a_noun, entity_dict[a_noun]))
        for idx3 in range(len(s_tokens)-1):
            if (s_tokens[idx3].tag in ["XR", "NNG"]) & (s_tokens[idx3+1].tag == "XSN"):
                s_noun_list.append(s_tokens[idx3].form + s_tokens[idx3+1].form + "test")
            elif s_tokens[idx3].tag in ["NNG", "NNP", "NNB", "NR", "NP"]:
                s_noun_list.append(s_tokens[idx3].form)  
        for a_noun in s_noun_list:
            if a_noun in entity_dict.keys():
                s_entity_list.append((a_noun, entity_dict[a_noun]))

        p_dict["sentence"] = a_prob
        p_dict["tokens"] = p_noun_list
        p_dict["entities"] = p_entity_list
        s_dict["sentence"] = a_solv
        s_dict["tokens"] = s_noun_list
        s_dict["entities"] = s_entity_list
        out_dict[id_list[idx]] = [p_dict, s_dict]

    out_path = 'cft_tokenize_entity_load_test.json'
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        json.dump(out_dict, json_file, ensure_ascii=False, indent='\t')
    
if __name__ == "__main__":
    cft_data_dir = "cft_data.xlsx"
    entity_match(cft_data_dir)