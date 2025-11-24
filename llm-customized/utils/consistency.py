import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict

import warnings

import pandas as pd

from utils.metrics import compute_metrics
warnings.filterwarnings("ignore", category=UserWarning)

import re

def contains_chinese(text: str) -> bool: return bool(re.search(r'[\u4e00-\u9fff]', text))

def paraphrase_problem(problem, num_paths=5, llm=None):
    llm._max_tokens=512
    output_parser = JsonOutputParser()
    
    prompt_template = PromptTemplate(
        input_variables=["num_path", "problem"],
        template="""Paraphrase the following problem using a unique approach. This is unique path number {num_path}.
        Paraphrasing 'problem' into another form. with Korean not Chinese.
        If don't understand the meaning of words in the contexts cleary, use word itself directly.

        Problem: {problem}
        Reasoning path: {num_path}
        {{
            "ParaphrasedProblem": ""
        }}
        """,
    )
    
    prompt_template = PromptTemplate(input_variables=["num_path", "problem"],template="""Paraphrase the following problem simply with Korean not Chinese by using a unique reasoning path.\nIf don't understand the meaning of words in the contexts cleary, use word itself directly.\nProblem: {problem}\nReasoning path: {num_path}\n{{\n    "ParaphrasedProblem": ""\n}}\n""",)
    prompt_template = PromptTemplate(input_variables=["num_path", "problem"],template="""Paraphrase the following problem and parse to json format simply with Korean not Chinese by using a unique reasoning path.\nMost of words are related with engineering or vehicle.\nProblem: {problem}\nReasoning path: {num_path}\n{{\n    "ParaphrasedProblem": ""\n}}\n""",)

    paths = []
    import pdb; pdb.set_trace() 
    for i in range(num_paths):   
        chain = prompt_template | llm | output_parser
        while True:
            try:
                response =  chain.invoke({"problem": problem, "num_path": i+1})
                if "ParaphrasedProblem" in response:
                    if not contains_chinese(response["ParaphrasedProblem"]):
                        paths.append(response["ParaphrasedProblem"])
                        break
                    else:
                        print("중국어 포함, 재생성")
                else:
                    print("의역 실패, 재생성")
            except:
                print("re invoke")

    return paths

def generate_path(problem, num_path, documents, llm=None):
    response_schemas = [
        ResponseSchema(name="Answer", description="Answer only by korean language"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["problem", "num_path", "documents"],
        template="""Solve the following problem using a unique approach. This is reasoning path {num_path}.
        Use the following documents to answer the given problem with Korean not Chinese.
        Documents: {documents}
        Problem: {problem}
        Reasoning path: {num_path}
        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions},
    )

    
    chain = prompt_template | llm | output_parser
    import pdb; pdb.set_trace() 
    
    while True:
        try:
            response =  chain.invoke({"problem": problem, "num_path": num_path+1, "documents": documents})
            if "Answer" in response:
                if not contains_chinese(response["Answer"]):
                    return response['Answer']
                else:
                    print("중국어 포함, 재생성")
            else:
                print("답변 생성 실패, 재생성")
        except:
            print("re invoke")

            

def aggregate_results(paths, llm):
    llm._max_tokens = 1024
    response_schemas = [
        ResponseSchema(name="Analysis", description="Analyze each path's consistency individually by korean language"),
        ResponseSchema(name="ConsistentAnswers", description="list of reasoning path number, e.g. 1,2,4"),
        ResponseSchema(name="AggregatedAnswer", description="Generated trustworthy answer by korean language"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    prompt_template = PromptTemplate(
        input_variables=["paths"],
        template="""Analyze the following reasoning paths and determine the most consistent answers. If there are discrepancies, explain why and provide the most likely correct answer.
        Generate trustworthy answer with Korean not Chinese by using consistent answers.
        Reasoning paths:
        {paths}
        
        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt_template | llm | output_parser
    import pdb; pdb.set_trace()
    while True:
        try:
            response = chain.invoke({"\n".join(paths)})
            if ("Analysis" in response) and ("ConsistentAnswers" in response) and ("AggregatedAnswer" in response):
                if ((not contains_chinese(response["Analysis"])) 
                    and (not contains_chinese(response["ConsistentAnswers"])) 
                    and (not contains_chinese(response["AggregatedAnswer"]))):
                    return response
                else:
                    print("중국어 포함, 재생성")
            else:
                print("답변 생성 실패, 재생성")
        except:
            print("re invoke")


def self_consistency_check(problem, paths, aggregated_result, llm):
    response_schemas = [
        ResponseSchema(name="Evaluation", description="Evaluation consider factors like logical consistency, adherence to known facts, and potential biases"),
        ResponseSchema(name="LogicalConsistency", description="score 1 to 10"),
        ResponseSchema(name="AdherenceToKnownFacts", description="score 1 to 10"),
        ResponseSchema(name="PotentialBiases", description="score 1 to 10"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["problem", "paths", "result"],
        template="""Evaluate the consistency and reliability of the following Result aggregated by given Reasoning Paths for the given Problem.
        Generate evaluation sentences with Korean not Chinese.
        Generated texts must include score for each factors (logical consistency, adherence to known facts, potential biases) on a scale of 1 to 10.
        Problem: {problem}
        Reasoning Paths: {paths}
        Result: {result}
        
        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt_template | llm | output_parser
    import pdb; pdb.set_trace()
    while True:
        try:
            chain.invoke({"problem": problem, "paths": "\n".join(paths),"result": aggregated_result})         
            response = chain.invoke({
                "problem": problem, 
                "paths": "\n".join(paths),
                "result": aggregated_result
            })
            if (("Evaluation" in response) and 
                ("LogicalConsistency" in response) and 
                ("AdherenceToKnownFacts" in response) and
                ("PotentialBiases" in response)):
                if ((not contains_chinese(response["Evaluation"])) 
                    and (not contains_chinese(response["LogicalConsistency"])) 
                    and (not contains_chinese(response["AdherenceToKnownFacts"])) 
                    and (not contains_chinese(response["PotentialBiases"]))):
                    return response
                else:
                    print("중국어 포함, 재생성")
            else:
                print("답변 생성 실패, 재생성")
        except:
            print("re invoke")


import time
def check_self_consistency_w_retrievers(llm, retriever, relevance_dict, query_dict, n_path=5, k=10, max_query=100, file_path='test.csv'):
    all_scores = {name: defaultdict(list) for name in range(n_path)}
    all_consistency = {name: defaultdict(list) for name in range(n_path)}
    file_path = file_path

    if max_query > len(relevance_dict):
        n_query = len(relevance_dict)
    else:
        n_query = max_query
    
    random_key = random.sample(list(relevance_dict.keys()), n_query)
    sampled_dict = {key: relevance_dict[key] for key in list(relevance_dict.keys())[:n_query]}
    
    for query_id, true_doc_ids in tqdm(sampled_dict.items(), desc="Evaluating"):
        query_text = query_dict.get(query_id)
        query_texts = paraphrase_problem(query_text, n_path, llm)
        if not query_text:
            continue
        
        # 중복제거
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if query_id in existing_df['qid'].values:
                continue
        
        st = time.time()
        paths = []
        for i in range(n_path):
            try:    
                if (hasattr(retriever, "_instruction") and not retriever._update_queries):
                    queries = np.array([query for key, query in query_dict.items()][:n_query])
                    retriever.update_querie_embeddings(queries)

                docs_and_scores = retriever.similarity_search_with_score(query=query_text, k=k)
                pred_ids = []
                scores = []
                docs = []
                for doc, score in docs_and_scores:      
                    pred_ids.append(doc.metadata["_id"])
                    scores.append(score)
                    docs.append(doc)
                    
                query = query_texts[i]

                documents = ''.join([doc.page_content for doc in docs[:k]])
                path = generate_path(problem=query, num_path=i, documents=documents, llm=llm)
                paths.append(f"Reasoning Path: {i+1}\n" + path)

                metrics = compute_metrics(pred_ids, true_doc_ids, scores, k=k)
                
                for metric, val in metrics.items():
                    all_scores[i][metric].append(val)

            except Exception as e:
                print(f"❗ Error in retriever with query '{query_id}': {e}")
                print(query_text, true_doc_ids)
                continue
        et = time.time()
        path_generation_time = et-st
        path_generation_time = round(path_generation_time, 2)
        
        st = time.time()
        aggregated_result = aggregate_results(paths=paths, llm=llm)
        et = time.time()
        aggregation_time = et-st
        aggregation_time = round(aggregation_time, 2)
        
        st = time.time()
        consistency_evaluation = self_consistency_check(
            query_text, 
            paths, 
            aggregated_result["AggregatedAnswer"], 
            llm
        )
        et = time.time()
        consistency_time = et-st
        consistency_time = round(consistency_time, 2)
        
        
        try:
            lc = int(consistency_evaluation['LogicalConsistency']) if 'LogicalConsistency' in consistency_evaluation else -1
            atkf = int(consistency_evaluation['AdherenceToKnownFacts']) if 'AdherenceToKnownFacts' in consistency_evaluation else -1
            pb = int(consistency_evaluation['PotentialBiases']) if 'PotentialBiases' in consistency_evaluation else -1
            aggregation = aggregated_result
            evaluation = consistency_evaluation['Evaluation']
            all_consistency['Logical consistency'] = lc 
            all_consistency['Adherence to known facts'] = atkf
            all_consistency['Potential biases'] = pb
            
            new_row = {
                'qid': query_id,
                'Original query': query_text,
                'Paraphrased query': query_texts,
                'Analysis each path': aggregation["Analysis"],
                'Consistent Answers': aggregation["ConsistentAnswers"],
                'Aggregated Answer': aggregation["AggregatedAnswer"],
                'Evaluation': evaluation,
                'Logical consistency': lc,
                'Adherence to known facts': atkf,
                'Potential biases': pb,
                'time-paraphrasing': path_generation_time,
                'time-aggregation': aggregation_time,
                'time-consistency': consistency_time,
            }

            # To Dataframe
            new_df = pd.DataFrame([new_row])
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                if query_id in existing_df['qid'].values:
                    continue
                else:
                    new_df = pd.DataFrame([new_row])
                    new_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                new_df = pd.DataFrame([new_row])
                new_df.to_csv(file_path, mode='w', header=True, index=False)
        except Exception as e:
            print('not existing keys')
            
    return all_scores, all_consistency


def get_result(path:str):
    df = pd.read_csv(path)

    lc = df['LogicalConsistency']
    adkf = df['AdherenceToKnownFacts']
    pb = df['PotentialBiases']

    avg_lc = sum(lc)/len(lc)
    avg_adkf = sum(adkf)/len(adkf)
    avg_pb = sum(pb)/len(pb)

    print(f"n = {len(lc)}")
    print(f"logical consistency: {avg_lc:.2f}")
    print(f"Adherence to known facts: {avg_adkf:.2f}")
    print(f"Potential biases: {avg_pb:.2f}")
    return avg_lc, avg_adkf, avg_pb
    