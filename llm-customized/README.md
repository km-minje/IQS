# Hyundai RAG

## 📌 주요 기능

- Retrievers
- Consistency Check
- Evaluation

## 📁 프로젝트 구조

```bash
project/
├── configs/*.yaml          # 하이퍼파라미터 및 설정
├── datasets/               # 데이터 저장 공간 (RECOMMENDED)
├── embeddings/             # embed를 수행 가능한 embedding class 정의
├── kiwi_custom/            # customized kiwi
├── llms/                   # invoke를 수행 가능한 chatmodel class 정의
├── results/                # evaluation 수행 결과 json 형태로 저장
├── retrievers/             # similarity_search_with_score를 수행 가능한 retriever class 정의
├── template/               # llms 모델 조정을 위한 template
├── utils 
├── utils/
│   ├── consistency.py      # llm 모델 일관성 평가를 위한 스크립트
│   ├── load_retriever.py   # config를 통해 retriever 생성을 위한 스크립트
│   ├── metrics.py          # retriever, reranker 성능 평가를 위한 스크립트
│   └──preprocess.py       # 기타
├── configs/*.yaml       # 하이퍼파라미터 및 설정
├── requirements.txt     # 필요한 패키지 목록
├── README.md
└── ...
```

1. 설치

# miniconda env
```bash
conda create -n {env_name} python==3.12
```

# mecab 별도 설치
```bash
conda install -c conda-forge mecab-ko
pip install python-mecab-ko
```

# faiss 별도 설치
```bash
conda install -c conda-forge faiss-gpu 
```

# torch 별도 설치
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

# ollama 설치
https://ollama.com/download
```

# 의존성 설치
```bash
pip install -r requirements.txt
```

# jdk 별도 설치 (하기 이슈 대응)
```bash
jpype._jvmfinder.JVMNotFoundException: No JVM shared library file (jvm.dll) found. Try setting up the JAVA_HOME environment variable properly.
```
conda install conda-forge::openjdk

<!-- # vllm 별도 설치 (하기 이슈 대응)
```bash
error: could not create 'build\bdist.win-amd64\wheel\.\vllm\model_executor\layers\fused_moe\configs\E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json': No such file or directory
```
win+r -> regedit -> HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled -> value to 1
pip install vllm==0.8.5.post1 -->

2. config 설정

# 하드웨어
```bash
hardware:
  cuda_devices: "0, 1"
```
가용할 gpu 선택

# 데이터 경로
```bash
datasets:
  corpus: "D:\\project\\llm-develop\\datasets\\hyundai\\hyundai_corpus.jsonl"
  queries: "D:\\project\\llm-develop\\datasets\\hyundai\\hyundai_queries.jsonl"
  dev: "D:\\project\\llm-develop\\datasets\\hyundai\\hyundai_dev.jsonl"
```
dataset/ 폴더에 데이터셋을 넣거나, 별도 경로에 추가 후 경로 수정
corpus  : 검색을 진행할 documents
queries : 검색을 위한 queries
dev     : qid, cid 연결성 관리 

# 임베딩 모델 정의
```bash
embedding_models:
  finetune_bge_m3_3:
    name: "finetune BGE-M3"
    path: "D:\\project\\models\\bge-m3_finetune3"
    type: "flag"
```
임베딩 모델 불러올 경로와 불러오는 class 형태 정의

# LLM 모델 정의
```bash
llm_models:

  gemma3-4b:
    name: "gemma3-4b"
    path: "gemma3:4b"
    type: "ollamachatmodel"
    max_tokens: 512

  gemma3-4b:
    name: "gemma3-4b"
    path: "D:\\project\\models\\gemma3-4b"
    type: "customchatmodel" or "vllm"
    max_tokens: 512
```
LLM 모델 불러올 경로와 불러오는 class 형태 및 최대 응답 토큰 수 정의
ollama 경우 path는 serving되고 있는 NAME
* vllm은 linux 용도..

# 검색기, Retrievers 정의
```bash
  faiss_finetune3:
    name: "faiss_finetune3"
    type: "faiss"
    embedding_model: "finetune_bge_m3_3"
    search_type: "mmr"
    k: 5
    instruction: True # encode_queries
  
  bm25_mecab:
    name: "bm25_mecab"
    type: "bm25"
    tokenizer: "mecab"
    k: 5
    
  # Ensemble retrievers
  ensembles:

    bc_finetune_5_5:
      name: "bc_finetune_5_5"
      retrievers: ["bm25_mecab", "faiss_finetune3"]
      weights: [0.5, 0.5]
      search_type: "mmr"
      k: 5
      c: 60.0
```
검색기 타입 정의 - bm25 (sparse), faiss (dense)
bm25의 경우 tokenizer 필요 - retrievers/BaseRetriever.py 내 tokenizer 참조
faiss의 경우 embedding_model 필요 - config 내 정의되어 있는 임베딩 모델 호출 가능
faiss의 경우 instruction 설정 가능 - 모든 쿼리에 대한 
faiss-ollama의 경우 path는 serving되고 있는 이름 사용
* faiss는 FaissRetrieverWithScores, FaissIndexRetrieverWithScores 중 Index를 이용한 후자 사용 중
ensemble의 경우 retrievers, weights 필요 - config 내 정의되어 있는 retriever 호출 가능
k는 검색기 공통 요소로, 검색 개수 정의

# Reranker 정의
```bash
rerankers:
  rerank_bc_finetune_5_5:
    retriever: "bc_finetune_5_5"
    embedding_model: "reranker"
```
retriever, embedding_model 필요 - config 내 정의되어 있는 retriever, embedding_model(trained for reranker) 호출 가능

# Evalaution 
```bash
evaluation:
  k: 5
  k_rerank: 5
  metrics:
    - ndcg
    - recall
    - rr
    - precision
    - hit
```
retrievers, rerankers에 대해 위 5가지 항목 평가 가능
k, k_rearnk 분리 이유는 top_k 에 대해 뽑고 더 작은 밤위에 대해서 rerank 결과를 보고 싶을 경우 사용 가능하도록 설정

# Evalaution 
```bash
consistency:
  consistency_gemma3-4b:
    llm_model: gemma3-4b
    retriever: bc_finetune_5_5
    path_num: 5
    k: 5
    seed: 42
```
llm_model, retriever 필요 - config 내 정의되어 있는 llm_model, retriever 호출 가능


호출 config에 대한 설명들은
D:\project\llm-V\llm-v1.4\configs\retriever_short_config.yaml
를 참조

3. 실행

<!-- # ollama 구동
ollama serve  -->

# main 코드 호출 / ollama 연결 이슈로 fail
실행에 앞서 내부 모델 경로 등은 직접 입력 바람.

run.py 기준 실행 시 config와 상관없이 코드 내 index_name에 의해 데이터로드 및 검색기 저장 등이 진행되며 다음과 같이 추가 입력을 진행
local, vm serving, elasticsearch 중 모델 선택 (es는 vm에서만 가능)
실제 사용할 document를 만들기 위한 컬럼 선택 및 추가 enrich 선택 (tokenizer 기반 enrich, llm 기반 enrich, no enrich 중)
chunk 시 사이즈 선택 (데이터 전처리)
기존에 있는 dataset내 존재하는 쿼리만 쓸지 선택 (query 최적화, 새로운 쿼리 사용할 경우 n)
```bash
python run.py
python test.py --test {num} --dir {num}

```

4. 데이터셋

hyundai_qwen_11 : Benchmark Q&A sheet 에 존재하는 질문 / 답변(QwQ) 1:1 
hyundai_llm_1n : 19000개 이상 데이터를 corpus로 구성. query/corpus id 매칭은 Q&A sheet 정보 이용
hyundai_llm_1n_short : Q&A sheet 에 존재하는 id만을 corpus로 구성. query/corpus id 매칭은 Q&A sheet 정보 이용

5. 기타

- vm 서빙중인 모델들 사용 시 간혹 라지토큰 쿼리에 대한 임베딩이 잘 동작하지 않는 경우 존재함 (이유 확인 필요 -> 오래 걸릴 시 디버깅 해볼것 권장)
- -> 현재 batch_size를 try 해보면서 동작 batch_size를 찾아가도록 하였으나 vm 대비 적은 수의 batch_size로 동작함

- Elastic Search 항목들도 run.py 및 test.py 내 포함되어 있으나, 일부 ingest function 등은 vm 안에서가 아니면 진행 불가능.

- utils/compute_metrics 내 'nDCG 계산' 쪽 확인 시 계산 방법 두가지 존재함
- 1번 https://bge-model.com/tutorial/4_Evaluation/4.1.1.html 참조 (iDCG를 예측한 검색 문서에서 최대 값으로 계산)
- 2번 https://bge-model.com/tutorial/4_Evaluation/4.1.1.html 참조 (iDCG를 전체 코퍼스 상에서 나올 수 있는 최대 값으로 계산)