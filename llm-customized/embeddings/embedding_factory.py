from embeddings.FlagEmbedding import FlagEmbedding
from embeddings.FlagColBERT import FlagColBERT
from embeddings.FlagSparse import FlagSparse
from embeddings.FlagEmbeddingReranker import FlagEmbeddingReranker
from embeddings.ESReranker import ESReranker
from embeddings.CustomHuggingFaceEmbedding import CustomHuggingFaceEmbedding
from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
from embeddings.OpenAIScoreRerankers import OpenAIScoreRerankers
from langchain_community.embeddings import HuggingFaceEmbeddings


# Get HuggingFace embeddings
def get_hf_emb(config):
    path = config["path"]
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    hf = HuggingFaceEmbeddings(
        model_name=path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        multi_process=False,
    )
    return hf


def create_embedding_model(config, all_config):
    if config["type"] == "huggingface":
        return get_hf_emb(config)
    elif config["type"] == "flag":
        return FlagEmbedding(config)
    elif config["type"] == "colbert":
        return FlagColBERT(config)
    elif config["type"] == "sparse":
        return FlagSparse(config)
    elif config["type"] == "custom":
        return CustomHuggingFaceEmbedding(config)
    elif config["type"] == "reranker":
        return FlagEmbeddingReranker(config)
    elif config["type"] == "api_embedding":
        return OpenAIEmbeddings(config)
    elif config["type"] == "api_reranker":
        return OpenAIScoreRerankers(config)
    elif config["type"] == "es_reranker":
        return ESReranker(config, all_config)
        
    else:
        raise ValueError(f"Unknown embedding model type: {config['type']}")