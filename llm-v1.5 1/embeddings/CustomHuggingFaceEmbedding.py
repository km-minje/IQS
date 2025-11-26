from transformers import AutoTokenizer, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import torch

from embeddings.BaseEmbedding import BaseEmbedding

# Custom HuggingFace Embeddings class
class CustomHuggingFaceEmbedding(BaseEmbedding):
    """
    CustomHuggingFaceEmbedding 클래스는 HuggingFace 모델을 기반으로 하여
    문서 및 쿼리 텍스트를 임베딩하는 기능을 제공합니다.
    """
    
    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정(config)을 사용하여 토크나이저와 모델을 초기화합니다.
        
        :param config: 모델과 토크나이저의 경로를 포함하는 설정 사전
        """
        super().__init__()
        path = config.get("path")  # 설정에서 모델 경로를 가져옴
        self.tokenizer = AutoTokenizer.from_pretrained(path)  # 토크나이저 초기화
        self.model = AutoModel.from_pretrained(path)  # 모델 초기화

        # Multi-GPU setup
        device_map = infer_auto_device_map(self.model, no_split_module_classes=["BertLayer", "TransformerLayer"])
        self.model = dispatch_model(self.model, device_map=device_map)  # 모델을 분산
        self.model.eval()  # 평가 모드로 설정

    def _mean_pooling(self, outputs, attention_mask):
        """
        주어진 출력과 어텐션 마스크를 사용하여 mean pooling을 수행합니다.
        
        :param outputs: 모델 출력
        :param attention_mask: 입력 시퀀스의 어텐션 마스크
        :return: pooling된 임베딩
        """
        token_embeddings = outputs.last_hidden_state  # 마지막 숨겨진 상태의 임베딩 가져오기
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())  # 마스크 확장
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)  # 가중합
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # 마스크 합산
        return sum_embeddings / sum_mask  # 평균 계산 후 반환

    def _embed(self, texts):
        """
        주어진 텍스트 리스트에 대해 임베딩을 생성합니다.
        
        :param texts: 임베딩할 텍스트 리스트
        :return: 각 텍스트당 생성된 임베딩의 리스트
        """
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # 입력 텍스트 토크나이즈
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # 모델 디바이스로 이동

            with torch.no_grad():  # 그래디언트 계산 비활성화
                outputs = self.model(**inputs)  # 모델을 통해 출력 얻기
                pooled = self._mean_pooling(outputs, inputs['attention_mask'])  # pooling 수행
                embeddings.append(pooled.squeeze().cpu().numpy())  # 리스트에 추가

        return embeddings  # 임베딩 리스트 반환

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        문서 텍스트 리스트를 임베딩합니다.
        
        :param texts: 임베딩할 문서 텍스트의 리스트
        :return: 각 문서에 대한 임베딩의 리스트
        """
        return self._embed(texts)  # 내부의 _embed 메서드 호출

    def embed_query(self, text: str) -> list[float]:
        """
        단일 쿼리 텍스트를 임베딩합니다.
        
        :param text: 임베딩할 쿼리 텍스트
        :return: 쿼리에 대한 임베딩
        """
        return self._embed([text])[0]  # 내부의 _embed 메서드 호출 후 첫 번째 결과 반환