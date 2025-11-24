import pandas as pd
from kiwipiepy import Kiwi
from konlpy.tag import Okt, Kkma
from mecab import MeCab
from kiwi_custom import tokenizer as custom_tokenizer
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# Mapping of tokenizer names to functions
tokenizer_map = {
    "kiwi": Kiwi(),
    "custom_kiwi": custom_tokenizer.custom_kiwi()[0],
    "okt": Okt(),
    "mecab": MeCab(),
    "kkma": Kkma(),
}

import re


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def get_tokenizer(name: str):
    try:
        return tokenizer_map[name]
    except KeyError:
        raise ValueError(
            f"Tokenizer '{name}' is not supported. Choose from: {list(tokenizer_map.keys())}"
        )


def enrich_df(df, tokenizer_name):
    tokenizer = get_tokenizer(tokenizer_name)
    # Show column mapping
    columns = df.columns.tolist()
    column_mapping = {i + 1: col for i, col in enumerate(columns)}
    print("\nColumn Mapping:")
    for num, col in column_mapping.items():
        print(f"{num}: {col}")

    # User input
    enrich_columns = {}
    while True:
        selected_input = input(
            "\nSelect columns (type 'a' for add enrich columns, 'q' for quit: "
        )
        if selected_input.strip().lower() == "q":
            break
        elif selected_input.strip().lower() == "a":
            enrich_column_name = input("\n enrich column name: ")
            selected_enrich_columns = input(
                "\nSelect enrich columns (type 'd' for default column or comma-separated numbers like 1,2,4): "
            )
            if selected_enrich_columns.strip().lower() == "d":
                selected_enrich_columns = columns
            else:
                try:
                    selected_enrich_indices = [
                        int(i.strip()) for i in selected_enrich_columns.split(",")
                    ]
                    selected_enrich_columns = [
                        column_mapping[i] for i in selected_enrich_indices
                    ]
                except:
                    raise Exception(ValueError, KeyError)
            enrich_columns[enrich_column_name] = selected_enrich_columns

        else:
            print("select given key")

    new_df = pd.DataFrame()
    save_df = pd.DataFrame()
    save_list = []
    for enrich_column_name, selected_enrich_columns in enrich_columns.items():

        def row_to_column(row):
            # return "\n".join([f"▶{col}: {row[col]}" for col in selected_enrich_columns])
            return "\n".join([f"{row[col]}" for col in selected_enrich_columns])

        new_df = df.apply(row_to_column, axis=1)

        for idx in range(len(new_df)):

            tokens = tokenizer.tokenize(str(new_df[idx]), normalize_coda=False)
            noun_list = []

            for idx2 in range(len(tokens) - 1):
                if (tokens[idx2].tag in ["XR", "NNG"]) & (
                    tokens[idx2 + 1].tag == "XSN"
                ):
                    noun_list.append(tokens[idx2].form + tokens[idx2 + 1].form)
                elif tokens[idx2].tag in ["NNG", "NNP", "NNB", "NR", "NP"]:
                    noun_list.append(tokens[idx2].form)
            if tokens[-1].tag in ["NNG", "NNP", "NNB", "NR", "NP"]:
                noun_list.append(tokens[-1].form)

            save_list.append(noun_list)

        df[enrich_column_name] = save_list
    return df


def llm_enrich_df(df, llm=None):

    # Show column mapping
    columns = df.columns.tolist()
    column_mapping = {i + 1: col for i, col in enumerate(columns)}
    print("\nColumn Mapping:")
    for num, col in column_mapping.items():
        print(f"{num}: {col}")

    # User input
    enrich_columns = {}
    while True:
        # selected_input = input(
        #     "\nSelect columns (type 'a' for add llm enrich columns, 'q' for quit: "
        # )
        selected_input = "a"
        if selected_input.strip().lower() == "q":
            break
        elif selected_input.strip().lower() == "a":
            # enrich_column_name = input("\n enrich column name: ")
            # selected_enrich_columns = input(
            #     "\nSelect enrich columns (type 'd' for default column or comma-separated numbers like 1,2,4): "
            # )
            enrich_column_name = "요약"
            selected_enrich_columns = "d"
            if selected_enrich_columns.strip().lower() == "d":
                selected_enrich_columns = columns
            else:
                try:
                    selected_enrich_indices = [
                        int(i.strip()) for i in selected_enrich_columns.split(",")
                    ]
                    selected_enrich_columns = [
                        column_mapping[i] for i in selected_enrich_indices
                    ]
                except:
                    raise Exception(ValueError, KeyError)
            enrich_columns[enrich_column_name] = selected_enrich_columns
            break  # temp
        else:
            print("select given key")

    response_schemas = [
        ResponseSchema(name="문제점", description="Answer only by korean language"),
        ResponseSchema(name="원인", description="Answer only by korean language"),
        ResponseSchema(name="해결책", description="Answer only by korean language"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    template = """주어진 데이터는 차량 도메인에서 쓰이는 용어들이 포함되어 있어. 
    해당 데이터를 토대로 내용을 풍부하게 설명해줘. 
    단 영어 단어를 한글로 풀어서 설명해줘.
    
    차량데이터:{text}
    
    **추가 설명 없이** 다음과 같이 정래해줘
    
    문제점:
    원인:
    해결책:
    """

    template = """주어진 데이터는 차량 도메인에서 쓰이는 용어들이 포함되어 있어. \n해당 데이터를 토대로 내용을 풍부하게 설명해줘. \n단 영어 단어를 한글로 풀어서 설명해줘.\n\n차량데이터:{text}\n\n**추가 설명 없이** 다음과 같이 정래해줘\n\n문제점:\n원인:\n해결책:"""

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    # format_instructions = output_parser.get_format_instructions()
    # prompt_template = PromptTemplate(
    #     input_variables=["text"],
    #     template=template,
    #     partial_variables={"format_instructions": format_instructions},
    # )
    # chain = prompt_template | llm | output_parser
    chain = prompt_template | llm

    new_df = pd.DataFrame()
    save_df = pd.DataFrame()
    save_list = []
    for enrich_column_name, selected_enrich_columns in enrich_columns.items():

        def row_to_column(row):
            # return "\n".join([f"▶{col}: {row[col]}" for col in selected_enrich_columns])
            return "\n\n".join(
                [f"▶{col}:\n{row[col]}" for col in selected_enrich_columns]
            )

        new_df = df.apply(row_to_column, axis=1)
        # import pdb

        # pdb.set_trace()
        idx = df.index.start
        responses = []
        while True:
            if idx == df.index.stop:
                break
            try:
                # import pdb

                # pdb.set_trace()
                response = chain.invoke({"text": new_df[idx]})
                responses.append(response.content)
                idx += 1
                import os

                file_path = "gemma_hyundai_corpus.csv"
                save_df = df.loc[[idx - 1]].copy(deep=False)
                save_df[enrich_column_name] = response.content
                # save_df["요약"] = text_response

                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path)

                    if save_df["순번"].values in existing_df["순번"].values:
                        continue
                    else:
                        save_df.to_csv(file_path, mode="a", header=False, index=False)
                else:
                    save_df.to_csv(file_path, mode="w", header=True, index=False)
                    # save_df.to_csv(file_path, mode="w", header=True)
            except:
                print("re invoke")
        df[enrich_column_name] = responses
    return df
