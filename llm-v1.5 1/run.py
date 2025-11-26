import os
from datetime import datetime
import warnings
import numpy as np
import click

warnings.filterwarnings("ignore", category=UserWarning)

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from embeddings import (
    OpenAIScoreRerankers,
    OpenAIEmbeddings,
    FlagEmbedding,
    FlagEmbeddingReranker,
)

from llms import OllamaChatModel

from utils import (
    load_jsonl,
    clean_up,
    load_file_to_df,
    chunk_df,
    enrich_df,
    llm_enrich_df,
)

from utils import (
    load_or_create_bm25_retriever,
    load_or_create_ensemble_retriever,
    load_or_create_faiss_retriever,
)

from transformers import set_seed
from elasticsearch import Elasticsearch


# Set CUDA devices
def set_cuda_devices(devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices


@click.command()
@click.option(
    "--config",
    default="configs/retriever_short_config.yaml",
    help="Path to the config file",
)
def main(config):
    try:
        # index_name = "test_llm"
        # index_name = "hyundai_llm"
        # index_name = "hyundai_llm_32"
        index_name = "hyundai_llm_short_32"
        query_path = "datasets\\hyundai_llm_1n_short\\hyundai_queries.jsonl"
        queries = load_jsonl(query_path)
        query_dict = {q["_id"]: q["text"] for q in queries}

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        host = "http://localhost:9200"
        user = "elastic"
        password = "elastic"
        pipeline_name = None
        all_config = {
            "name": index_name,
            "es": {
                "host": "http://localhost:9200",
                "id": "elastic",
                "pw": "elastic",
            },
        }

        serving_model_name = "bge-m3-automotive-ft-v1"

        if index_name.startswith("hyundai_llm_short"):
            file_path = "datasets\\hyundai_llm_1n_short\\hyundai_corpus.jsonl"

        elif index_name.startswith("hyundai_llm"):
            file_path = "datasets\\hyundai_llm_1n\\hyundai_corpus.jsonl"

        select_local = input("local 모델 : 1, vm serving 모델 : 2, es 연동 : 3\n")
        # select_local = '2'
        if select_local == "1":
            config = {"path": "D:\\project\\models\\bge-m3-finetune3"}
            embedding_model = FlagEmbedding(config)

            config = {"path": "D:\\project\\models\\reranker"}
            reranker = FlagEmbeddingReranker(config)
            config = {"path": "gemma3:27b", "max_tokens": 512}
            llm_model = OllamaChatModel(config)
        elif select_local == "2":
            config = {
                "model": "bge-m3-automotive-ft-v2",
                "url": "http://10.129.38.197/embedding-custom-bge-m3/v1",
                "api": "EMPTY",
            }
            embedding_model = OpenAIEmbeddings(config)

            config = {
                "model": "bge-reranker-v2-m3-automotive-ft-v1",
                "url": "http://10.129.38.197/reranker/v1/score",
                "api": "EMPTY",
            }
            reranker = OpenAIScoreRerankers(config)

            llm_model = ChatOpenAI(
                model="Qwen/Qwen2.5-72B-Instruct",
                base_url="http://10.129.38.194/v1",
                api_key="EMPTY",
                temperature=0.5,
                max_tokens=512,
                timeout=None,
                max_retries=2,
            )
        elif select_local == "3":  # elastic-search

            es = Elasticsearch(
                hosts=[host],
                basic_auth=(user, password),
            )
            # create inference
            from utils.elasticsearch import register_inference_model_http

            serving_model_name = "bge-m3-automotive-ft-v1"

            register_inference_model_http(
                host,
                user,
                password,
                serving_model_name,
                "text_embedding",
                "openai",
                "http://host.docker.internal:8001/v1/embeddings",
            )
            # register_inference_model_http(host, user, password, "bge-m3-automotive-ft-v2", "text_embedding", "openai","http://10.129.38.197/embedding-custom-bge-m3/v1/embeddings")

            # ingest models (only for embedding)
            from utils.elasticsearch import create_ingest_pipeline

            pipeline_name = "bge-m3-automotive-ft-v1-pipeline"
            create_ingest_pipeline(
                host, user, password, serving_model_name, pipeline_name
            )

        else:
            raise Exception

        df = load_file_to_df(file_path)

        turn_on_enrich = input(
            "Select enrich: \n 1: tokenize enrich\n 2: llm enrich\n 3: no enrich\n"
        )
        if turn_on_enrich == "1":
            df_enrich = enrich_df(df, "custom_kiwi")
        elif turn_on_enrich == "2":
            df_enrich = llm_enrich_df(df, llm_model)
        elif turn_on_enrich == "3":
            print("")

        df_chunk = chunk_df(df_enrich if turn_on_enrich in ["1", "2"] else df)
        7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36
        docs = []
        ids = []
        for i, row in df_chunk.iterrows():
            docs.append(
                Document(
                    page_content=f"{row['chunk']}", metadata={"_id": int(row["순번"])}
                )
                # Document(page_content=f"{row['chunk']}", metadata={"_id": row["index"]})
            )
            ids.append(row["순번"])

        if select_local in ["3", 3]:

            from utils.elasticsearch import create_index

            create_index(host, user, password, index_name, pipeline_name)

            from utils.elasticsearch import bulk_index_with_pipeline, add_index_document
            from retrievers import get_tokenizer

            tokenizer = get_tokenizer("mecab")
            bulk_size = 300
            embedding_model = OpenAIEmbeddings(
                model="bge-m3-automotive-ft-v2",
                base_url="http://10.129.38.197/embedding-custom-bge-m3/v1",
                api_key="EMPTY",
            )
            for i in range(0, len(docs), bulk_size):
                # bulk_index_with_pipeline(host, user, password, index_name, docs[i:i+bulk_size], pipeline_id=pipeline_name, tokenizer=tokenizer, embedding_model=embedding_model)
                bulk_index_with_pipeline(
                    host,
                    user,
                    password,
                    index_name,
                    docs[i : i + bulk_size],
                    pipeline_id=None,
                    tokenizer=tokenizer,
                    embedding_model=embedding_model,
                )

            # bm25 retriever with similarity search
            from utils.load_retriever import create_es_bm25_retriever

            config = {"tokenizer": "mecab", "index": index_name, "field": "text"}
            bm25_retriever = create_es_bm25_retriever("es_bm25", config, all_config)
            bm25_retriever.similarity_search_with_score(query, 5)

            # vector retriever with similarity search
            from utils.load_retriever import create_es_knn_retriever

            embedding_models = {"bge-m3": embedding_model}
            config = {
                "embedding_model": "bge-m3",
                "serving_model": serving_model_name,
                "index": index_name,
                "field": "text_embedding",
            }
            embedding_models = {"bge-m3": embedding_model}
            config = {
                "embedding_model": "bge-m3",
                "index": index_name,
                "field": "text_embedding",
            }
            vectordb_retriever = create_es_knn_retriever(
                "es_knn", config, embedding_models, all_config
            )

            # hybrid retriever with similarity search
            from utils.load_retriever import create_es_ensemble_retriever

            embedding_models = None
            embedding_models = {"bge-m3": embedding_model}

            config = {
                "tokenizer": "mecab",
                "embedding_model": "bge-m3",
                "index": index_name,
                "field": ["text", "text_embedding"],
                "weights": [0.3, 0.7],
                "c": 60.0,
                # "serving_model" : serving_model_name,
            }
            ensemble_retriever = create_es_ensemble_retriever(
                "es_ensemble", config, embedding_models, all_config
            )
            ensemble_retriever.similarity_search_with_score(query, 5)

            # reranker
            from embeddings.ESReranker import ESReranker

            serving_reranker_model_name = "bge-reranker-v2-m3-automotive"
            config = {"serving_model": serving_reranker_model_name}
            reranker = ESReranker(config, all_config)

        else:
            config = {
                "tokenizer": "mecab",
            }
            bm25_retriever = load_or_create_bm25_retriever(
                "bm25", config, docs, all_config
            )

            # query에 명시적으로
            # "Represent this sentence for searching relevant passages:" 추가할 경우
            # instruction True
            embedding_models = {"bge-m3": embedding_model}
            config = {
                "embedding_model": "bge-m3",
                "instruction": False,
            }
            vectordb_retriever = load_or_create_faiss_retriever(
                "faiss",
                config,
                docs,
                embedding_models,
                all_config,
            )

            config = {
                "retrievers": ["bm25_retriever", "vectordb_retriever"],
                "weights": [0.3, 0.7],
                "search_type": "mmr",
                "c": 60.0,
            }
            all_retrievers = {
                "bm25_retriever": bm25_retriever,
                "vectordb_retriever": vectordb_retriever,
            }
            ensemble_retriever = load_or_create_ensemble_retriever(
                "ensemble", config, all_retrievers, all_config
            )

            # 기존에 있는 dataset내 존재하는 쿼리만 쓸경우
            # update_qurie_embeddings 필요
            update_vectordb = input(
                "기존에 있는 dataset내 존재하는 쿼리만 쓸경우 y를 입력\n"
            )
            if update_vectordb.lower() == "y":
                queries = np.array([query for key, query in query_dict.items()][:])
                vectordb_retriever.update_querie_embeddings(queries)

                ensemble_retriever = load_or_create_ensemble_retriever(
                    "ensemble", config, all_retrievers, all_config
                )
        k = 5
        while True:
            query = input("\n쿼리를 입력. 만약 종료하려면 q를 입력\n")
            if query.lower() == "q":
                break

            print(f"query: {query}")
            docs_and_scores = bm25_retriever.similarity_search_with_score(
                query=query, k=k
            )
            vectordb_retriever._instruction = False  # 존재하지 않는 query 사용시 False
            docs_and_scores = vectordb_retriever.similarity_search_with_score(
                query=query, k=k
            )

            docs_and_scores = ensemble_retriever.similarity_search_with_score(
                query=query, k=k
            )

            pred_ids = []
            pred_texts = []
            new_scores = []
            doc_map = {}
            for doc, score in docs_and_scores:
                _id = doc.metadata["_id"]
                text = doc.page_content
                pred_ids.append(_id)
                pred_texts.append(text)

                new_score = reranker.compute_score(query, text)[0]
                new_scores.append(new_score)

                doc_map[_id] = doc

            sorted_pairs = sorted(zip(new_scores, pred_ids), reverse=True)
            new_ids = [doc_id for score, doc_id in sorted_pairs]
            new_docs_sorted = [doc_map[idx] for idx in new_ids]
            new_scores_sorted = [score for score, doc_id in sorted_pairs]
            for doc, score in zip(new_docs_sorted, new_scores_sorted):
                print(doc)  # page_content, meta_data["_id"]로이루어짐
                print(score)

    except KeyboardInterrupt:
        print("keyborad interrupt")
    finally:
        clean_up()


if __name__ == "__main__":
    main()
