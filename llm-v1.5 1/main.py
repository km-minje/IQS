import os
import time
import json
import yaml
import pickle
from datetime import datetime
import warnings
from typing import List, Callable, Tuple
from collections import defaultdict
import pandas as pd
import subprocess
import click

warnings.filterwarnings("ignore", category=UserWarning)

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import (
    Chroma,
    FAISS,
)  # Changed from FAISS to Chroma

from retrievers import get_tokenizer
from retrievers import (
    BM25RetrieverWithScores,
    FaissRetrieverWithScores,
    EnsembleRetrieverWithScores,
)

from llms import create_llm_model

from embeddings import create_embedding_model
from embeddings import (
    CustomHuggingFaceEmbedding,
    FlagEmbedding,
)

from utils import (
    load_jsonl,
    kill_all_ollama_processes,
    kill_ollama_models,
    is_ollama_running,
    clean_up,
    compute_metrics,
    evaluate_retrievers,
    evaluate_rerankers,
    load_or_create_bm25_retriever,
    load_or_create_ensemble_retriever,
    load_or_create_faiss_retriever,
    create_retriever,
    check_self_consistency_w_retrievers,
    get_result,
)

from transformers import set_seed


# Set CUDA devices
def set_cuda_devices(devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"


@click.command()
@click.option(
    "--config", default="configs/retriever_config.yaml", help="Path to the config file"
)
def main(config):
    try:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # if not is_ollama_running():
        #     subprocess.Popen(["ollama", "serve"])
        #     print("Ollama 서버 실행 시작")
        # else:
        #     kill_ollama_models()
        #     kill_all_ollama_processes()
        #     print("Ollama 서버가 이미 실행 중입니다.")

        # Load configuration
        # with open("configs/retriever_config.yaml", "r") as f:
        with open(config, "r") as f:
            config = yaml.safe_load(f)

        # Set CUDA devices
        set_cuda_devices(config["hardware"]["cuda_devices"])

        # Load datasets
        print("📚 Loading datasets...")
        corpus = load_jsonl(config["datasets"]["corpus"])
        queries = load_jsonl(config["datasets"]["queries"])
        test_pairs = load_jsonl(config["datasets"]["dev"])

        corpus_dict = {doc["_id"]: doc for doc in corpus}
        query_dict = {q["_id"]: q["text"] for q in queries}

        # Create Document objects
        end_index = (
            config["datasets"]["corpus_end_index"]
            if "corpus_end_index" in config["datasets"]
            else 1000000000
        )
        docs = []
        ids = []
        for doc in corpus[:end_index]:
            docs.append(
                Document(page_content=f"{doc['text']}", metadata={"_id": doc["_id"]})
            )
            ids.append(doc["_id"])

        relevance_dict = defaultdict(list)
        for pair in test_pairs:
            if any(int(corpus_id) in ids for corpus_id in pair["corpus-id"]):
                if pair["query-id"] not in relevance_dict:
                    relevance_dict[pair["query-id"]] = [pair["corpus-id"]]
                else:
                    relevance_dict[pair["query-id"]].append(pair["corpus-id"])
        print(len(relevance_dict))

        # Initialize embedding models
        print("🧠 Initializing embedding models...")
        embedding_models = {}
        if "embedding_models" in config and config["embedding_models"] is not None:
            for model_id, model_config in config["embedding_models"].items():
                embedding_models[model_id] = create_embedding_model(
                    model_config, config
                )
        
        
        # Initialize llm models
        print("🧠 Initializing llm models...")
        llm_models = {}
        if "llm_models" in config and config["llm_models"] is not None:
            for model_id, model_config in config["llm_models"].items():
                llm_models[model_id] = create_llm_model(model_config, config)
        
        # Create basic retrievers
        all_retrievers = {}
        retriever_creation_times = {}  # Store creation times
        print("🔍 Creating retrievers...")
        if "retrievers" in config and config["embedding_models"] is not None:
            for retriever_id, retriever_config in config["retrievers"].items():
                if retriever_id != "ensembles":
                    print(f"Creating/loading {retriever_config['name']}...")
                    start = time.time()
                    all_retrievers[retriever_id] = create_retriever(
                        retriever_id,
                        retriever_config,
                        docs,
                        embedding_models,
                        config,
                    )
                    end = time.time()
                    elapsed = end - start
                    retriever_creation_times[retriever_id] = round(elapsed, 3)
                    print(f"⏱ Time: {elapsed:.4f}s")

            # Create ensemble retrievers
            print("🔍 Creating ensemble retrievers...")
            if (
                "ensembles" in config["retrievers"]
                and config["retrievers"]["ensembles"] is not None
            ):
                for ensemble_id, ensemble_config in config["retrievers"][
                    "ensembles"
                ].items():
                    print(f"Creating/loading {ensemble_config['name']}...")
                    start = time.time()

                    all_retrievers[ensemble_id] = load_or_create_ensemble_retriever(
                        ensemble_id,
                        ensemble_config,
                        all_retrievers,
                        config,
                    )

                    end = time.time()
                    elapsed = end - start
                    retriever_creation_times[ensemble_id] = round(elapsed, 3)
                    print(f"⏱ Time: {elapsed:.4f}s")

        # Prepare results for output
        formatted_results = {}
        summary_results = {}
        basic_output = {"config": config}
        if "evaluation" in config and config["evaluation"] is not None:
            # Evaluate retrievers
            print("📊 Evaluating retrievers...")
            k = config["evaluation"]["k"]

            results_retrievers = evaluate_retrievers(
                all_retrievers, relevance_dict, query_dict, k=k
            )

            results_rerankers = {}
            if "rerankers" in config and config["rerankers"] is not None:
                # Evaluate reranker
                print("📊 Evaluating rerankers...")
                k = config["evaluation"]["k"]

                rerankers = config["rerankers"]
                results_rerankers = {}
                if config["rerankers"] is not None:
                    results_rerankers = evaluate_rerankers(
                        rerankers,
                        all_retrievers,
                        embedding_models,
                        relevance_dict,
                        query_dict,
                        k=k,
                    )

            results = {
                **results_retrievers,
                **results_rerankers,
            }

            for name, metric_lists in results.items():
                formatted_results[name] = {"metrics": metric_lists, "averages": {}}
                summary_results[name] = {
                    "creation_time": (
                        retriever_creation_times[name]
                        if name in retriever_creation_times
                        else 0.0
                    )
                }

                for metric_name, values in metric_lists.items():
                    if metric_name in config["evaluation"]["metrics"]:
                        avg_val = sum(values) / len(values) if values else 0
                        formatted_results[name]["averages"][metric_name] = avg_val
                        summary_results[name][metric_name] = round(avg_val, 4)

            # find best
            best_id = None
            best_ndcg = 0
            # Print results
            print(f"\n📈 Retriever Performance (Average metrics @{k}):")
            for name, metrics in summary_results.items():
                print(f"\nRetriever: {name}")
                for metric_name, value in metrics.items():
                    print(f"-{metric_name:<10}: {value:.4f}")

                    if metric_name == "ndcg":
                        if value > best_ndcg:
                            best_id = name
                            best_ndcg = value

            # Save detailed results to file
            retriever_output = {"evaluation_k": k, "summary_results": summary_results}

        consistency_output = {}
        if "consistency" in config and config["consistency"] is not None:
            for consistency_id, consistency_config in config["consistency"].items():
                set_seed(consistency_config["seed"])
                print(f"📊 Checking {consistency_id}...")
                start = time.time()
                check_self_consistency_w_retrievers(
                    llm_models[consistency_config["llm_model"]],
                    all_retrievers[consistency_config["retriever"]],
                    relevance_dict,
                    query_dict,
                    consistency_config["path_num"],
                    consistency_config["k"],
                    max_query=100,
                    file_path=f"./{consistency_id}-{consistency_config['seed']}-{now}.csv",
                )

                end = time.time()
                elapsed = end - start

                (
                    logical_consistency,
                    adherence_to_known_facts,
                    potential_bias,
                    time_paraphrasing,
                    time_aggregation,
                    time_consistency,
                ) = get_result(
                    f"./{consistency_id}-{consistency_config['seed']}-{now}.csv"
                )
                consistency_output[consistency_id] = {
                    "logical_consistency": logical_consistency,
                    "adherence_to_known_facts": adherence_to_known_facts,
                    "potential_bias": potential_bias,
                    "time_paraphrasing": time_paraphrasing,
                    "time_aggregation": time_aggregation,
                    "time_consistency": time_consistency,
                }
                print(f"⏱ Time: {elapsed:.4f}s")

        output = {
            **basic_output,
            **retriever_output,
            **consistency_output,
        }
        os.makedirs("results", exist_ok=True)
        with open(f"results/evaluation_results-{now}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to evaluation_results-{now}.json")

    except KeyboardInterrupt:
        print("keyborad interrupt")
    finally:
        clean_up()


if __name__ == "__main__":
    main()
