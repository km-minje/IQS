from elasticsearch import Elasticsearch, helpers
import numpy as np

def bulk_index_with_pipeline(es_host, username, password, index_name, documents, pipeline_id=None, tokenizer=None, embedding_model=None):
    """
    주어진 문서들을 Elasticsearch에 일괄 인덱싱합니다. 파이프라인을 사용할 수 있습니다.
    
    :param es_host: Elasticsearch 호스트
    :param username: Elasticsearch 사용자 이름
    :param password: Elasticsearch 비밀번호
    :param index_name: 문서를 인덱싱할 인덱스 이름
    :param documents: 인덱싱할 문서 목록
    :param pipeline_id: 사용할 (선택적) ingest 파이프라인 ID
    :param tokenizer: (선택적) 문서 텍스트 토크나이저
    :param embedding_model: (선택적) 문서의 임베딩을 생성하기 위한 모델
    """
    es = Elasticsearch(
        [es_host],
        basic_auth=(username, password)  # 기본 인증
    )
    
    actions = []  # 인덱싱 동작을 저장할 리스트
    for doc in documents:
        doc_id = doc.metadata['_id']  # 문서 ID 가져오기
        source = {"text": doc.page_content}  # 문서 내용 저장
        
        # 토크나이저가 제공되면 문서를 토큰화
        if tokenizer:
            source["tokenized_text"] = tokenizer(doc.page_content)
        
        # 파이프라인이 없고 임베딩 모델이 제공되는 경우 임베딩 추가
        if embedding_model and pipeline_id is None:
            source["text_embedding"] = np.array(embedding_model.embed_query(doc.page_content)).astype(np.float32)

        if "_id" in source:  # _id 필드가 있다면 제거
            del source["_id"]

        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_source": source,  # 문서 소스
            "_id": doc_id,  # 문서 ID
        })

    # Elasticsearch에 일괄 인덱싱
    helpers.bulk(es, actions, pipeline=pipeline_id)
    print(f"✅ Indexed {len(actions)} docs with pipeline '{pipeline_id}'")


def add_index_document(es_host, username, password, index_name, documents, pipeline_id=None, tokenizer=None, embedding_model=None):
    """
    주어진 문서들을 Elasticsearch에 인덱싱합니다. 각 문서에 대한 ID를 새롭게 생성합니다.
    
    :param es_host: Elasticsearch 호스트
    :param username: Elasticsearch 사용자 이름
    :param password: Elasticsearch 비밀번호
    :param index_name: 문서를 인덱싱할 인덱스 이름
    :param documents: 인덱싱할 문서 목록
    :param pipeline_id: 사용할 (선택적) ingest 파이프라인 ID
    :param tokenizer: (선택적) 문서 텍스트 토크나이저
    :param embedding_model: (선택적) 문서의 임베딩을 생성하기 위한 모델
    """
    es = Elasticsearch(
        [es_host],
        basic_auth=(username, password)  # 기본 인증
    )
    
    actions = []  # 인덱싱 동작 저장용 리스트
    doc_id = es.count(index=index_name)['count']  # 현재 인덱스에 있는 문서 수로 ID 초기화
    for doc in documents:
        doc_id += 1  # ID 증가
        
        source = {"text": doc.page_content}  # 문서 내용 저장
        if tokenizer:
            source["tokenized_text"] = tokenizer(doc.page_content)  # 문서 토큰화
        
        if embedding_model and pipeline_id is None:
            source["text_embedding"] = embedding_model.embed_query(doc.page_content)  # 임베딩 추가

        if "_id" in source:  # _id 필드 제거
            del source["_id"]

        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_source": source,  # 문서 소스
            "_id": doc_id,  # 문서 ID
        })

    # Elasticsearch에 일괄 인덱싱
    helpers.bulk(es, actions, pipeline=pipeline_id)
    print(f"✅ Indexed {len(actions)} docs with pipeline '{pipeline_id}'")


def create_ingest_pipeline(host, username, password, model_id: str, pipeline_id: str):
    """
    Elasticsearch에서 ingest pipeline을 생성합니다.
    
    :param host: Elasticsearch 호스트
    :param username: Elasticsearch 사용자 이름
    :param password: Elasticsearch 비밀번호
    :param model_id: 등록된 Inference API 모델 ID (예: 'bge-m3-automotive-online')
    :param pipeline_id: 생성할 pipeline 이름 (예: 'hyundai_embedding_pipeline_online')
    """
    es = Elasticsearch(
        [host],
        basic_auth=(username, password)  # 기본 인증
    )
    
    # ingest pipeline 정의
    pipeline_body = {
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "input_output": {
                        "input_field": "text",
                        "output_field": "text_embedding"
                    },
                    "inference_config": {
                        "text_embedding": {}
                    }
                }
            }
        ]
    }

    es.ingest.put_pipeline(id=pipeline_id, body=pipeline_body)  # pipeline 생성
    print(f"✅ Ingest pipeline '{pipeline_id}' created successfully (model_id: {model_id})")


def register_inference_model_http(
    es_host: str,
    username: str,
    password: str,
    model_name: str,
    model_type: str,
    service: str,
    url: str
):
    """
    Elasticsearch Inference 모델 등록 함수 (requests 사용)
    
    :param es_host: Elasticsearch 호스트
    :param username: Elasticsearch 사용자 이름
    :param password: Elasticsearch 비밀번호
    :param model_name: 등록할 모델 이름
    :param model_type: 모델 타입
    :param service: 서비스 명
    :param url: 서비스 URL
    """
    es = Elasticsearch(
        [es_host],
        basic_auth=(username, password)  # 기본 인증
    )
    
    # HTTP 기본 인증을 위한 인코딩
    user_pass = f"{username}:{password}"
    import base64
    b64_auth = base64.b64encode(user_pass.encode()).decode()
    auth_header = f"Basic {b64_auth}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header  # 인증 헤더 설정
    }
    
    # Inference API 모델 등록을 위한 payload
    payload = {
        "service": service,
        "service_settings": {
            "url": url,
            "api_key": "EMPTY",
            "model_id": model_name
        }
    }

    # Elasticsearch에 요청 전송
    meta, response = es.transport.perform_request(
        method="PUT",
        target=f"/_inference/{model_type}/{model_name}",
        headers=headers,
        body=payload
    )

    # 등록 성공 및 실패 확인
    if meta.status in [200, 201]:
        print(f"✅ 모델 등록 성공: {model_name} ({model_type})")
    else:
        print(f"❌ 등록 실패: {meta.status}")

    return response


def create_index(es_host, username, password, index_name, pipeline_id=None):
    """
    Elasticsearch에 새로운 인덱스를 생성합니다.
    
    :param es_host: Elasticsearch 호스트
    :param username: Elasticsearch 사용자 이름
    :param password: Elasticsearch 비밀번호
    :param index_name: 생성할 인덱스 이름
    :param pipeline_id: 사용할 (선택적) ingest 파이프라인 ID
    """
    es = Elasticsearch(
        [es_host],
        basic_auth=(username, password)  # 기본 인증
    )
    
    # 이미 존재하는 인덱스가 있다면 삭제
    if es.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)
    
    print(f"Creating new index: {index_name}")
    
    # 인덱스 생성에 필요한 설정 및 매핑
    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "nori": {
                        "tokenizer": "nori_tokenizer"
                    },
                    "whitespace": {
                        "tokenizer": "whitespace"
                    },
                }
            },
            "index": {
                "similarity": {
                    "bm25": {
                        "type": "BM25",
                        "k1": 1.5,
                        "b": 0.75
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "nori",
                },
                "tokenized_text" : {
                    "type": "text",
                    "analyzer" : "whitespace",
                    "similarity": "bm25",
                },
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "dot_product"
                }
            }
        }
    }
    es.indices.create(index=index_name, body=body)  # 인덱스 생성
    

def is_text_exists(es, index_name, text):
    """
    주어진 텍스트가 인덱스에 존재하는지 확인합니다.
    만약 존재하면 _id값을 반환하고, 없으면 None을 반환합니다.
    
    :param es: Elasticsearch 인스턴스
    :param index_name: 체크할 인덱스 이름
    :param text: 체크할 텍스트
    :return: 존재하는 경우 _id, 없으면 None
    """
    query = {
        "query": {
            "match": {
                "text": text
            }
        },
        "size": 1,
    }
    response = es.search(index=index_name, body=query)
    if response['hits']['total']['value'] > 0:
        return es.search(index=index_name, body=query)["hits"]["hits"][0]["_id"]  # _id 반환
    else:
        return None  # None 반환


def delete_index_id(es, index_name, doc_id):
    """
    주어진 문서 ID를 사용하여 Elasticsearch 인덱스에서 문서를 삭제합니다.
    
    :param es: Elasticsearch 인스턴스
    :param index_name: 삭제할 인덱스 이름
    :param doc_id: 삭제할 문서 ID
    """
    try:
        if es.exists(index=index_name, id=doc_id):
            es.delete(index=index_name, id=doc_id)  # 문서 삭제
            print(f"✅ Document with ID '{doc_id}' deleted from index '{index_name}'.")
        else:
            print(f"⚠️ Document with ID '{doc_id}' does not exist in index '{index_name}'.")
    except Exception as e:
        print(f"❌ Failed to delete document: {e}")


def es_bm25_similarity_search_with_score(es, index, query, k):
    """
    주어진 쿼리로 Elasticsearch에서 BM25 유사도 검색을 수행합니다.
    
    :param es: Elasticsearch 인스턴스
    :param index: 검색할 인덱스 이름
    :param query: 검색할 쿼리
    :param k: 반환할 문서 수
    :return: 검색된 문서의 ID 목록
    """
    response = es.search(
        index=index,
        size=k,
        query={
            "match": {
                "text": query
            }
        },
    )

    result = [hit["_id"] for hit in response["hits"]["hits"]]  # 결과 ID 수집
    
    return result  # ID 반환  


def add_es_text(es, index_name, text):
    """
    주어진 텍스트를 Elasticsearch 인덱스에 추가합니다.
    
    :param es: Elasticsearch 인스턴스
    :param index_name: 추가할 인덱스 이름
    :param text: 추가할 텍스트
    """
    doc = {
        "text": text  # 문서에 텍스트 추가
    }
    # 현재 카운트를 기반으로 ID를 설정하여 인덱스에 추가
    es.index(index=index_name, id=es.count(index=index_name)['count'] + 1, body=doc)