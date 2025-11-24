from sklearn.metrics import ndcg_score
from sklearn.utils import check_array
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def tie_averaged_dcg(y_true, y_score, discount_cumsum):
    """
    Compute DCG by averaging over possible permutations of ties.

    The gain (`y_true`) of an index falling inside a tied group (in the order
    induced by `y_score`) is replaced by the average gain within this group.
    The discounted gain for a tied group is then the average `y_true` within
    this group times the sum of discounts of the corresponding ranks.

    This amounts to averaging scores for all possible orderings of the tied
    groups.

    (note in the case of dcg@k the discount is 0 after index k)

    Parameters
    ----------
    y_true : ndarray
        The true relevance scores.

    y_score : ndarray
        Predicted scores.

    discount_cumsum : ndarray
        Precomputed cumulative sum of the discounts.

    Returns
    -------
    discounted_cumulative_gain : float
        The discounted cumulative gain.

    References
    ----------
    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.
    """
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()


def dcg_sample_scores(y_true, y_score, k=None, log_base=2, ignore_ties=False):
    """Compute Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.

    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.

    log_base : float, default=2
        Base of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    Returns
    -------
    discounted_cumulative_gain : ndarray of shape (n_samples,)
        The DCG score for each sample.

    See Also
    --------
    ndcg_score : The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.
    """
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
    if k is not None:
        discount[k:] = 0
    if ignore_ties:
        ranking = np.argsort(y_score)[:, ::-1]
        ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
        cumulative_gains = discount.dot(ranked.T)
    else:
        discount_cumsum = np.cumsum(discount)
        cumulative_gains = [
            tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_score)
        ]
        cumulative_gains = np.asarray(cumulative_gains)
    return cumulative_gains


def calculate_rr(hits):
    """
    Reciprocal Rank (RR)를 계산합니다.

    :param hits: 검색 결과에서의 히트 여부 리스트
    :return: Reciprocal Rank 값
    """
    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:  # 히트가 있는 경우
            reciprocal_rank = 1.0 / rank  # RR 계산
            break

    return reciprocal_rank


# Evaluation metrics
def compute_metrics(pred_ids, true_ids, scores, k=10):
    """
    예측된 ID와 실제 ID를 기반으로 다양한 평가 메트릭을 계산합니다.

    :param pred_ids: 예측된 문서 ID 리스트
    :param true_ids: 실제 문서 ID 리스트
    :param scores: 예측된 문서의 점수 리스트
    :param k: 상위 k개의 문서 수
    :return: 계산된 메트릭 딕셔너리
    """
    topk_pred = pred_ids[:k]  # 상위 k개 예측문서
    scores = scores[:k]  # 상위 k개 점수
    if type(true_ids) == list and type(true_ids[0]) == list:
        true_ids = true_ids[0]  # 두 번째 리스트 처리
    hits = [1 if pid in true_ids else 0 for pid in topk_pred]  # 히트 계산

    len_true_ids = len(true_ids)
    # ideal_hits = [1] * sum(hits) + [0] * (k - sum(hits))  # 이상적인 히트 리스트
    len_true_ids = k if len_true_ids >= k else len_true_ids
    ideal_hits = [1] * len_true_ids + [0] * (k - len_true_ids)

    # nDCG 계산
    if scores:
        ndcg = ndcg_score([hits], [scores])  # 점수를 사용한 nDCG 계산

    else:
        y_true = check_array([hits], ensure_2d=True)
        y_score = check_array([scores], ensure_2d=True)
        gain = dcg_sample_scores(y_true, y_score)  # 점수를 사용한 nDCG 계산
        y_true = check_array([ideal_hits], ensure_2d=True)
        y_score = check_array([ideal_hits], ensure_2d=True)
        normalizing_gain = dcg_sample_scores(y_true, y_score)

        all_irrelevant = normalizing_gain == 0
        gain[all_irrelevant] = 0
        gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
        ndcg = np.average(gain)

    # Recall 계산
    recall = sum(hits) / len(true_ids) if true_ids else 0

    # Precision 계산
    precision = sum(hits) / k if k else 0

    # Hit 계산
    hit = 1.0 if any(pid in true_ids for pid in topk_pred) else 0.0

    # MRR 계산
    rr = calculate_rr(hits)

    return {
        "ndcg": ndcg,
        "recall": recall,
        "precision": precision,
        "hit": hit,
        "rr": rr,
    }


def evaluate_retrievers(retrievers, relevance_dict, query_dict, k=5, max_query=100):
    """
    주어진 검색기들을 평가하여 메트릭을 계산합니다.

    :param retrievers: 평가할 검색기 딕셔너리
    :param relevance_dict: 쿼리와 관련된 문서 ID를 매핑하는 딕셔너리
    :param query_dict: 쿼리 ID와 텍스트를 매핑하는 딕셔너리
    :param k: 상위 k개 문서 수
    :param max_query: 최대 쿼리 수
    :return: 모든 검색기에서 계산된 점수 딕셔너리
    """
    all_scores = {
        name: defaultdict(list) for name in retrievers
    }  # 메트릭 저장용 딕셔너리

    if max_query > len(relevance_dict):
        n_query = len(relevance_dict)  # 전체 쿼리 수
        keys = random.sample(list(relevance_dict.keys()), n_query)  # 랜덤 샘플링

    else:
        n_query = max_query
        keys = list(relevance_dict.keys())[:n_query]  # 쿼리 키 가져오기

    sampled_dict = {key: relevance_dict[key] for key in keys}  # 샘플 쿼리 생성

    for query_id, true_doc_ids in tqdm(sampled_dict.items(), desc="Evaluating"):
        query_text = query_dict.get(query_id)  # 쿼리 텍스트 얻기
        if not query_text:
            continue

        for name, retriever in retrievers.items():
            try:
                if (
                    hasattr(retriever, "_instruction")
                    and retriever._instruction
                    and not retriever._update_queries
                ):
                    queries = np.array(
                        [query_dict[key] for key in keys]
                    )  # 쿼리 배열 생성
                    retriever.update_querie_embeddings(queries)  # 쿼리 임베딩 업데이트
                    retrievers[name] = retriever
                docs_and_scores = retriever.similarity_search_with_score(
                    query=query_text, k=k
                )  # 유사도 검색 수행
                pred_ids = []
                scores = []
                for doc, score in docs_and_scores:
                    pred_ids.append(doc.metadata["_id"])  # 문서 ID 저장
                    scores.append(score)  # 점수 저장

                metrics = compute_metrics(
                    pred_ids, true_doc_ids, scores, k=k
                )  # 메트릭 계산
                for metric, val in metrics.items():
                    all_scores[name][metric].append(val)  # 점수 저장

            except Exception as e:
                print(f"❗ Error in retriever '{name}' with query '{query_id}': {e}")
                continue

    return all_scores  # 모든 검색기 메트릭 반환


def evaluate_rerankers(
    rerankers_config,
    retrievers,
    embedding_models,
    relevance_dict,
    query_dict,
    k=10,
    k_rerank=10,
    max_query=100,
    docs=None,  # 임시 테스트
):
    """
    주어진 리랭커를 평가하여 메트릭을 계산합니다.

    :param rerankers_config: 평가할 리랭커 설정
    :param retrievers: 각 리랭커와 연계된 검색기
    :param embedding_models: 사용할 임베딩 모델
    :param relevance_dict: 쿼리와 관련된 문서 ID 매핑
    :param query_dict: 쿼리 ID와 텍스트의 매핑
    :param k: 상위 k개의 문서 수
    :param k_rerank: 리랭크 시 사용할 문서 수
    :param max_query: 최대 쿼리 수
    :param docs: 임시 테스트에 사용할 문서 (선택적)
    :return: 계산된 메트릭 딕셔너리
    """
    all_scores = {
        name: defaultdict(list) for name in rerankers_config
    }  # 메트릭 저장용 딕셔너리

    if max_query > len(relevance_dict):
        n_query = len(relevance_dict)  # 전체 쿼리 수
        keys = random.sample(list(relevance_dict.keys()), n_query)

    else:
        n_query = max_query
        keys = list(relevance_dict.keys())[:n_query]  # 샘플 쿼리 키 가져오기

    sampled_dict = {key: relevance_dict[key] for key in keys}  # 샘플 쿼리 생성

    for query_id, true_doc_ids in tqdm(
        sampled_dict.items(), desc="Evaluating Rerankers"
    ):
        query_text = query_dict.get(query_id)  # 쿼리 텍스트 얻기
        if not query_text:
            continue

        for reranker_name, cfg in rerankers_config.items():
            try:
                retriever_name = cfg["retriever"]  # 연관된 검색기 이름
                embedder_name = cfg["embedding_model"]  # 연관된 임베딩 모델 이름

                retriever = retrievers[retriever_name]  # 검색기 가져오기
                embedder = embedding_models[embedder_name]  # 임베딩 모델 가져오기

                docs_and_scores = retriever.similarity_search_with_score(
                    query=query_text, k=k  # 검색 수행
                )

                pred_ids = []
                res_text = []
                scores = []
                new_scores = []
                doc_map = {}

                for doc, score in docs_and_scores:
                    _id = doc.metadata["_id"]  # 문서 ID 가져오기
                    text = doc.page_content  # 문서 내용 가져오기
                    res_text.append(text)  # 문서 내용 저장

                    # 옵션: 추가적인 텍스트를 붙일 수 있음
                    if docs:
                        add_text = docs[_id]  # 추가적인 문서 내용
                        text = text + "\n\n" + add_text  # 내용 결합

                    pred_ids.append(_id)  # 예측 ID 저장
                    scores.append(score)  # 점수 저장
                    doc_map[_id] = doc  # 문서 매핑

                    with tqdm(total=1, disable=True):
                        new_scores.append(
                            embedder.compute_score(query_text, text)[0]
                        )  # 리랭키에 사용할 새로운 점수 계산

                # 점수 정렬
                sorted_pairs = sorted(zip(new_scores, pred_ids), reverse=True)
                new_ids = [doc_id for score, doc_id in sorted_pairs]  # 정렬된 문서 ID
                new_scores_sorted = [
                    score for score, doc_id in sorted_pairs
                ]  # 정렬된 점수

                metrics = compute_metrics(
                    new_ids, true_doc_ids, new_scores_sorted, k=k_rerank
                )  # 메트릭 계산
                for metric, val in metrics.items():
                    all_scores[reranker_name][metric].append(val)  # 점수 저장

            except Exception as e:
                print(
                    f"❗ Error in reranker '{reranker_name}' with query '{query_id}': {e}"
                )
                continue

    return all_scores  # 모든 리랭커에서의 점수 반환
