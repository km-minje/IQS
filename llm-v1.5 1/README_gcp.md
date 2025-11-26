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

# python 3.10

# create env
```bash
python -m venv test
source test/bin/activate
```

# 의존성 설치
```bash
pip install -r requirements.txt
```

# ollama 설치
```bash
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```

# llama cpp 설치
git clone https://github.com/ggerganov/llama.cpp.git (pws)
vm으로 해당 압축파일 이동 후 해제
폴더 내 디펜던시 설치 
```bash
pip install -r requirments.txt 
```

# ollama gguf 생성
하기 코드를 이용하여 hf 파일 gguf로 변경
```bash
python convert_hf_to_gguf.py ../gemma3-4b --outtype auto --outfile gemma3-4b.gguf
```

하기와 같이 Modelfile 만들고 gguf와 같은 경로에 배치
```bash
FROM ./gemma3-4b.gguf

# stop tokens 설정
PARAMETER stop ["<end_of_turn>"]

# temperature 설정 (1로)
PARAMETER temperature 1

# top_k 설정
PARAMETER top_k 64

# top_p 설정
PARAMETER top_p 0.95

# template
TEMPLATE """
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}
"""
```

gguf, Modelfile 경로에서 하기 명령어들 수행 확인.
ollama serve
ollama create gemma3-4b -f Modelfile
ollama list
ollama run gemma3-4b


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
  metrics:
    - ndcg
    - recall
    - rr
    - precision
    - hit
```
retrievers, rerankers에 대해 위 5가지 항목 평가 가능

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


3. 실행

<!-- # ollama 구동
ollama serve  -->

# main 코드 호출 / ollama 연결 이슈로 fail
```bash
python main.py --config configs/{file_name}.yaml
```