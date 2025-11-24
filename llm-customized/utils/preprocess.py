import psutil
import os
import signal
import json
import torch
import gc
import subprocess


# Loading data functions
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def kill_ollama_models():
    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline and "ollama" in cmdline[0] and "run" in cmdline:
                os.kill(proc.pid, signal.SIGTERM)
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def kill_all_ollama_processes():
    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "ollama" in proc.info["name"].lower():
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def is_ollama_running():
    for proc in psutil.process_iter(["name", "cmdline"]):
        if "ollama" in proc.info["name"] and "serve" in " ".join(proc.info["cmdline"]):
            return True
    return False


def clean_up():
    torch.cuda.empty_cache()
    gc.collect()


import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file_to_df(file_path):
    file_type = file_path.split(".")[-1].lower()
    if file_type == "csv":
        # Load CSV
        df = pd.read_csv(file_path, header=0)
        columns = df.columns.tolist()

        # # Show column mapping
        # column_mapping = {i + 1: col for i, col in enumerate(columns)}
        # print("\nColumn Mapping:")
        # for num, col in column_mapping.items():
        #     print(f"{num}: {col}")

        # # User input
        # selected_input = input(
        #     "\nSelect columns (type 'd' for all columns, or comma-separated numbers like 1,2,4): "
        # )

        # if selected_input.strip().lower() == "d":
        #     selected_columns = columns
        # else:
        #     try:
        #         selected_indices = [int(i.strip()) for i in selected_input.split(",")]
        #         selected_columns = [column_mapping[i] for i in selected_indices]
        #     except (ValueError, KeyError):
        #         print("Invalid input. Please enter valid column numbers.")
        #         return df

        # # Create new 'text' column based on selected columns
        # def row_to_text(row):
        #     return "\n\n".join([f"▶{col}: {row[col]}" for col in selected_columns])

        # df["text"] = df.apply(row_to_text, axis=1)
        return df

    elif file_type in ["json", "jsonl"]:
        # Load JSON or JSONL file
        with open(file_path, "r", encoding="utf-8") as f:
            if file_type == "json":
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f]

        df = pd.DataFrame(data)
        columns = df.columns.tolist()

        # # Show column mapping
        # column_mapping = {i + 1: col for i, col in enumerate(columns)}
        # print("\nColumn Mapping:")
        # for num, col in column_mapping.items():
        #     print(f"{num}: {col}")

        # # User input
        # selected_input = input(
        #     "\nSelect columns (type 'd' for default : all columns, or comma-separated numbers like 1,2,4): "
        # )

        # if selected_input.strip().lower() == "d":
        #     selected_columns = columns
        # else:
        #     try:
        #         selected_indices = [int(i.strip()) for i in selected_input.split(",")]
        #         selected_columns = [column_mapping[i] for i in selected_indices]
        #     except (ValueError, KeyError):
        #         print("Invalid input. Please enter valid column numbers.")
        #         return df

        # # Create 'text' column
        # def row_to_text(row):
        #     return "\n\n".join([f"▶{col}: {row[col]}" for col in selected_columns])

        # df["text"] = df.apply(row_to_text, axis=1)
        return df

    elif file_type == "txt":
        # Load TXT: one line = one row
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        df = pd.DataFrame(lines, columns=["text"])
        return df

    else:
        raise ValueError("Unsupported file type")


def chunk_df(df):
    # chunk file
    chunk_size = input("\nSelect chunk_size (type 'd' for default 5000, or int): ")
    chunk_overlap = input(
        "\nSelect chunk_overlap_size (type 'd' for default 200, or int): "
    )
    if chunk_size.strip().lower() == "d":
        chunk_size = 5000
    else:
        chunk_size = int(chunk_size)

    if chunk_overlap.strip().lower() == "d":
        chunk_overlap = 200
    else:
        chunk_overlap = int(chunk_overlap)
        # User input

    columns = df.columns.tolist()
    column_mapping = {i + 1: col for i, col in enumerate(columns)}
    print("\nColumn Mapping:")
    for num, col in column_mapping.items():
        print(f"{num}: {col}")

    # User input
    enrich_columns = {}
    selected_input = input(
        "\nSelect columns for agrregating&chunking (type 'd' for default : all columns, or comma-separated numbers like 1,2,4): "
    )

    if selected_input.strip().lower() == "d":
        selected_columns = columns
    else:
        try:
            selected_indices = [int(i.strip()) for i in selected_input.split(",")]
            selected_columns = [column_mapping[i] for i in selected_indices]
        except:
            raise Exception(ValueError, KeyError)

    def row_to_text(row):
        return "\n\n".join([f"▶{col}:\n{row[col]}" for col in selected_columns])

    df["text"] = df.apply(row_to_text, axis=1)

    # 텍스트 분할기 설정
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", ".", " ", ""],
    )

    # 분할된 결과를 저장할 리스트
    chunk_data = []

    for idx, row in df.iterrows():
        text = row["text"]
        metadata = row.drop("text").to_dict()  # 'text' 외의 컬럼들 메타데이터로 저장

        # 텍스트를 분할
        chunks = splitter.split_text(text)

        # 분할된 각 chunk에 메타데이터와 원본 인덱스 부여
        for i, chunk in enumerate(chunks):
            chunk_data.append(
                {
                    "chunk": chunk,
                    "index": len(chunk_data) + 1,
                    "source_index": idx,
                    **metadata,
                }
            )

    # 결과를 DataFrame으로 반환
    return pd.DataFrame(chunk_data)


# if __name__ == "__main__":
#     load_jsonl()
