import os
import pickle
import faiss
import joblib
import dill
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.vectorstores import Chroma, FAISS
from retrievers import get_tokenizer
from retrievers import (
    BM25RetrieverWithScores,
    BM25OkapiRetrieverWithScores,
    FaissRetrieverWithScores,
    FaissIndexRetrieverWithScores,
    CustomFaissIndexRetrieverWithScores,
    CustomColBERTRetrieverWithScores,
    CustomSparseRetrieverWithScores,
    EnsembleRetrieverWithScores,
    ESBM25RetrieverWithScores,
    ESKNNRetrieverWithScores,
    ESEnsembleRetrieverWithScores
)
from embeddings import create_embedding_model
from embeddings import (
    CustomHuggingFaceEmbedding,
    FlagEmbedding,
)

import faiss
from elasticsearch import Elasticsearch

# Check if BM25 retriever exists in cache and load it
def load_or_create_bm25_retriever(retriever_id, config, docs, all_config):
    """
    캐시에서 BM25 검색기를 로드하거나 없을 시 새로 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param docs: 사용할 문서 목록
    :param all_config: 전체 설정 딕셔너리
    :return: BM25OkapiRetrieverWithScores 인스턴스
    """
    cache_dir = f"{all_config['name']}/retrievers"  # 캐시 디렉토리 경로 지정
    tokenizer_name = config["tokenizer"]  # 토크나이저 이름 가져오기

    cache_file = os.path.join(cache_dir, f"{retriever_id}")  # 캐시 파일 경로
    tokenizer_func = get_tokenizer(tokenizer_name)  # 토크나이저 함수 가져오기
    # 캐시에서 검색기 존재 여부 확인
    if os.path.exists(cache_file):
        print(f"🔄 Loading cached BM25 retriever '{retriever_id}' from {cache_file}")
        return BM25OkapiRetrieverWithScores.load(
            path=cache_file, preprocess_func=tokenizer_func
        )  # 캐시된 검색기 로드

    # 캐시에 없으면 새로운 검색기 생성
    print(f"🔄 Creating new BM25 retriever '{retriever_id}'")
    retriever = BM25OkapiRetrieverWithScores.from_documents(
        docs, preprocess_func=tokenizer_func  # 문서와 토크나이저로 검색기 생성
    )

    #retriever.save(cache_file)  # 캐시에 저장
    return retriever


# Check if faiss DB exists and load it, otherwise create it
def load_or_create_faiss_retriever(
    retriever_id, config, docs, embedding_models, all_config
):
    """
    캐시에서 Faiss 검색기를 로드하거나 없을 시 새로 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param docs: 사용할 문서 목록
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: CustomFaissIndexRetrieverWithScores 인스턴스
    """
    cache_dir = f"{all_config['name']}/retrievers"  # 캐시 디렉토리 경로 지정
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)  # 캐시 디렉토리 생성
        
    embedding_model = embedding_models[config["embedding_model"]]  # 임베딩 모델 가져오기
    embedding_model_name = config["embedding_model"]  # 임베딩 모델 이름

    cache_file = os.path.join(cache_dir, f"{retriever_id}")
    persist_directory = os.path.join(cache_dir, f"{retriever_id}")

    # 이미 존재하는 DB가 있는지 확인
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(
            f"🔄 Loading existing faiss DB for '{retriever_id}' from {persist_directory}"
        )
        try:
            # 기존 DB 로드 시도
            retriever = CustomFaissIndexRetrieverWithScores.load(
                cache_file, embedding_model
            )
            return retriever  # 로드된 검색기 반환
        except Exception as e:
            print(f"❗ Error loading existing faiss DB: {e}")
            print(f"🔄 Will create a new faiss DB instead")

    # 새로운 Faiss DB 생성 및 지속화
    print(
        f"🔄 Creating new faiss DB for '{retriever_id}' with {embedding_model_name}..."
    )
    os.makedirs(persist_directory, exist_ok=True)

    retriever = CustomFaissIndexRetrieverWithScores.from_documents(
        docs=docs,
        embed_model=embedding_model,
        instruction=config.get("instruction"),
    )
    #retriever.save(cache_file)  # 캐시에 저장

    return retriever

def load_or_create_colbert_retriever(
    retriever_id, config, docs, embedding_models, all_config
):
    """
    캐시에서 Faiss 검색기를 로드하거나 없을 시 새로 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param docs: 사용할 문서 목록
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: CustomFaissIndexRetrieverWithScores 인스턴스
    """
    cache_dir = f"{all_config['name']}/retrievers"  # 캐시 디렉토리 경로 지정
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)  # 캐시 디렉토리 생성
        
    embedding_model = embedding_models[config["embedding_model"]]  # 임베딩 모델 가져오기
    embedding_model_name = config["embedding_model"]  # 임베딩 모델 이름

    cache_file = os.path.join(cache_dir, f"{retriever_id}")
    persist_directory = os.path.join(cache_dir, f"{retriever_id}")

    # 이미 존재하는 DB가 있는지 확인
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(
            f"🔄 Loading existing faiss DB for '{retriever_id}' from {persist_directory}"
        )
        try:
            # 기존 DB 로드 시도
            retriever = CustomColBERTRetrieverWithScores.load(
                cache_file, embedding_model
            )
            return retriever  # 로드된 검색기 반환
        except Exception as e:
            print(f"❗ Error loading existing faiss DB: {e}")
            print(f"🔄 Will create a new faiss DB instead")

    # 새로운 Faiss DB 생성 및 지속화
    print(
        f"🔄 Creating new faiss DB for '{retriever_id}' with {embedding_model_name}..."
    )
    os.makedirs(persist_directory, exist_ok=True)

    retriever = CustomColBERTRetrieverWithScores.from_documents(
        docs=docs,
        embed_model=embedding_model,
        instruction=config.get("instruction"),
    )
    #retriever.save(cache_file)  # 캐시에 저장

    return retriever


def load_or_create_sparse_retriever(
    retriever_id, config, docs, embedding_models, all_config
):
    """
    캐시에서 Faiss 검색기를 로드하거나 없을 시 새로 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param docs: 사용할 문서 목록
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: CustomFaissIndexRetrieverWithScores 인스턴스
    """
    cache_dir = f"{all_config['name']}/retrievers"  # 캐시 디렉토리 경로 지정
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)  # 캐시 디렉토리 생성
        
    embedding_model = embedding_models[config["embedding_model"]]  # 임베딩 모델 가져오기
    embedding_model_name = config["embedding_model"]  # 임베딩 모델 이름

    cache_file = os.path.join(cache_dir, f"{retriever_id}")
    persist_directory = os.path.join(cache_dir, f"{retriever_id}")

    # 이미 존재하는 DB가 있는지 확인
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(
            f"🔄 Loading existing faiss DB for '{retriever_id}' from {persist_directory}"
        )
        try:
            # 기존 DB 로드 시도
            retriever = CustomSparseRetrieverWithScores.load(
                cache_file, embedding_model
            )
            return retriever  # 로드된 검색기 반환
        except Exception as e:
            print(f"❗ Error loading existing faiss DB: {e}")
            print(f"🔄 Will create a new faiss DB instead")

    # 새로운 Faiss DB 생성 및 지속화
    print(
        f"🔄 Creating new faiss DB for '{retriever_id}' with {embedding_model_name}..."
    )
    os.makedirs(persist_directory, exist_ok=True)

    retriever = CustomSparseRetrieverWithScores.from_documents(
        docs=docs,
        embed_model=embedding_model,
        instruction=config.get("instruction"),
    )
    #retriever.save(cache_file)  # 캐시에 저장

    return retriever


def load_or_create_ensemble_retriever(
    ensemble_id, ensemble_config, all_retrievers, all_config
):
    """
    앙상블 검색기를 로드하거나 새로 생성합니다.
    
    :param ensemble_id: 앙상블 검색기 식별자
    :param ensemble_config: 앙상블 검색기 설정
    :param all_retrievers: 이미 로드된 검색기 목록
    :param all_config: 전체 설정 딕셔너리
    :return: EnsembleRetrieverWithScores 인스턴스
    """
    cache_dir = f"{all_config['name']}/retrievers"

    # 모든 구성 검색기가 캐시에서 존재하는지 확인
    all_retrievers_exist = all(
        r_id in all_retrievers for r_id in ensemble_config["retrievers"]
    )

    if all_retrievers_exist:
        # 모든 검색기가 존재하면 앙상블 검색기 생성
        print(f"🔄 Creating new ensemble retriever '{ensemble_id}'")
        component_retrievers = [
            all_retrievers[r_id] for r_id in ensemble_config["retrievers"]
        ]

        ensemble = EnsembleRetrieverWithScores(
            retrievers=component_retrievers,
            weights=ensemble_config["weights"],
            search_type=ensemble_config["search_type"],
            c=ensemble_config["c"],
        )

    return ensemble


def create_es_bm25_retriever(retriever_id, config, all_config):
    """
    Elasticsearch 기반 BM25 검색기를 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param all_config: 전체 설정 딕셔너리
    :return: ESBM25RetrieverWithScores 인스턴스
    """
    tokenizer_func = None  # 토크나이저 함수 초기화
    if config["tokenizer"]:
        tokenizer_name = config["tokenizer"]  # 토크나이저 이름 가져오기
        tokenizer_func = get_tokenizer(tokenizer_name)  # 토크나이저 함수 가져오기

    # Elasticsearch 연결 설정
    host = all_config["es"]["host"]
    id = all_config["es"]["id"]
    pw = all_config["es"]["pw"]
    
    es = Elasticsearch(
        [host],
        basic_auth=(id, pw)
    )
    index = config["index"]  # 인덱스 이름
    target_field = config["field"]  # 검색할 필드
    
    retriever = ESBM25RetrieverWithScores(es, index, target_field, tokenizer_func)  # BM25 검색기 생성
    print(f"🔄 Creating new ES BM25 retriever '{retriever_id}'")
    
    return retriever


def create_es_knn_retriever(retriever_id, config, embedding_models, all_config):
    """
    Elasticsearch 기반 KNN 검색기를 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: ESKNNRetrieverWithScores 인스턴스
    """
    embedding_model = None  # 임베딩 모델 초기화
    if config.get("embedding_model"):
        embedding_model = embedding_models[config.get("embedding_model")]  # 임베딩 모델 가져오기
        
    serving_model = None  # 서빙 모델 초기화
    if config.get("serving_model"):
        serving_model = config.get("serving_model")  # 서빙 모델 가져오기
    
    # Elasticsearch 연결 설정
    host = all_config["es"]["host"]
    id = all_config["es"]["id"]
    pw = all_config["es"]["pw"]
    
    es = Elasticsearch(
        [host],
        basic_auth=(id, pw)
    )
    
    index = config["index"]  # 인덱스 이름
    target_field = config["field"]  # 검색할 필드
    
    retriever = ESKNNRetrieverWithScores(es, index, target_field, embedding_model, serving_model)  # KNN 검색기 생성
    print(f"🔄 Creating new ES knn retriever '{retriever_id}'")
    
    return retriever

def create_es_ensemble_retriever(retriever_id, config, embedding_models, all_config):
    """
    Elasticsearch 기반 앙상블 검색기를 생성합니다.
    
    :param retriever_id: 검색기 식별자
    :param config: 검색기 설정
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: ESEnsembleRetrieverWithScores 인스턴스
    """
    # 토크나이저 함수 가져오기
    tokenizer_func = None
    if config.get("tokenizer"):
        tokenizer_name = config["tokenizer"]
        tokenizer_func = get_tokenizer(tokenizer_name)

    # 임베딩 모델 및 서빙 모델 가져오기
    embedding_model = None
    if config.get("embedding_model"):
        embedding_model = embedding_models[config.get("embedding_model")]
        
    serving_model = None
    if config.get("serving_model"):
        serving_model = config.get("serving_model")
    
    # Elasticsearch 연결 설정
    host = all_config["es"]["host"]
    id = all_config["es"]["id"]
    pw = all_config["es"]["pw"]
    
    es = Elasticsearch(
        [host],
        basic_auth=(id, pw)
    )
    
    index = config["index"]  # 인덱스 이름
    target_field = config["field"]  # 검색할 필드
    weights = config.get("weights")  # 가중치 설정
    c = config.get("c")  # 랭크 조정 상수 설정
    
    retriever = ESEnsembleRetrieverWithScores(es, index, target_field, tokenizer_func, embedding_model, serving_model, weights, c)  # 앙상블 검색기 생성
    print(f"🔄 Creating new ES ensemble retriever '{retriever_id}'")
    
    return retriever

def create_retriever(name, config, docs, embedding_models, all_config):
    """
    주어진 설정에 따라 적절한 검색기를 생성합니다.
    
    :param name: 검색기 이름
    :param config: 검색기 설정
    :param docs: 사용할 문서 목록
    :param embedding_models: 임베딩 모델 딕셔너리
    :param all_config: 전체 설정 딕셔너리
    :return: 생성된 검색기 인스턴스
    """
    if config["type"] == "bm25":
        return load_or_create_bm25_retriever(name, config, docs, all_config)
    elif config["type"] == "chroma":
        pass  # Chroma 검색기 구현 필요
    elif config["type"] == "faiss":
        return load_or_create_faiss_retriever(
            name, config, docs, embedding_models, all_config
        )
    elif config["type"] == "colbert":
        return load_or_create_colbert_retriever(
            name, config, docs, embedding_models, all_config
        )
    elif config["type"] == "sparse":
        return load_or_create_sparse_retriever(
            name, config, docs, embedding_models, all_config
        )
    elif config["type"] == "es_bm25":
        return create_es_bm25_retriever(name, config, all_config)
    elif config["type"] == "es_knn":
        return create_es_knn_retriever(name, config, embedding_models, all_config)
    elif config["type"] == "es_ensemble":
        return create_es_ensemble_retriever(name, config, embedding_models, all_config)      
    else:
        raise ValueError(f"Unknown retriever type: {config['type']}")