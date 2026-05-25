import yaml  # type: ignore
from typing import Union, List, Any, Dict

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import numpy as np
import datetime
import pdfplumber  # 导入pdfplumber模块，用于处理PDF文件
from openai import OpenAI

import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
# from FlagEmbedding import FlagReranker
from es_api import es

device = config["device"]

EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}


问题：{#QUESTION#}
'''


def load_embdding_model(model_name: str, model_path: str) -> None:
    """
    加载编码模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    # sbert模型
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)


def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    加载重排序模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(device)


if config["rag"]["use_embedding"]:
    model_name = config["rag"]["embedding_model"]
    model_path = config["models"]["embedding_model"][model_name]["local_url"]

    print(f"Loading embedding model {model_name} from model_path...")
    load_embdding_model(model_name, model_path)

if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]

    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)


def split_text_with_overlap(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks


class RAG:
    def __init__(self):
        self.embedding_model = config["rag"]["embedding_model"]
        self.rerank_model = config["rag"]["rerank_model"]

        self.use_rerank = config["rag"]["use_rerank"]

        self.embedding_dims = config["models"]["embedding_model"][
            config["rag"]["embedding_model"]
        ]["dims"]

        self.chunk_size = config["rag"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunk_overlap"]
        self.chunk_candidate = config["rag"]["chunk_candidate"]

        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"]
        )
        self.llm_model = config["rag"]["llm_model"]

    def _extract_pdf_content(self, knowledge_id, document_id, title, file_path) -> bool:
        try:
            pdf = pdfplumber.open(file_path)
        except:
            print("打开文件失败")
            return False

        print(f"{file_path} pages: ", len(pdf.pages))  # 打印提示信息，显示PDF文件的页数

        abstract = ""

        for page_number in range(len(pdf.pages)):  # 每一页 提取
            current_page_text = pdf.pages[page_number].extract_text()  # 提取图片
            if page_number <= 3:
                abstract = abstract + '\n' + current_page_text

            # 每一页内容的内容
            embedding_vector = self.get_embedding(current_page_text)
            page_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "page_number": page_number,
                "chunk_id": 0,  # 先存储每一也所有内容
                "chunk_content": current_page_text,
                "chunk_images": [],
                "chunk_tables": [],
                "embedding_vector": [float(x) for x in list(embedding_vector)]
            }
            response = es.index(index="chunk_info", document=page_data)

            # 划分chunk
            page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
            embedding_vector = self.get_embedding(page_chunks)
            for chunk_idx in range(1, len(page_chunks) + 1):
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,
                    "chunk_content": page_chunks[chunk_idx - 1],
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": [float(x) for x in list(embedding_vector[chunk_idx - 1])]
                }
                response = es.index(index="chunk_info", document=page_data)

        document_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "document_name": title,
            "file_path": file_path,
            "abstract": abstract
        }
        response = es.index(index="document_meta", document=document_data)

    def _extract_word_content(self):
        pass

    def extract_content(self, knowledge_id, document_id, title, file_type, file_path):
        if "pdf" in file_type:
            self._extract_pdf_content(knowledge_id, document_id, title, file_path)
        elif "word" in file_type:
            pass

        print("提取完成", document_id, file_type, file_path)

    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码
        :param text: 待编码文本
        :return: 编码结果
        """
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)

        raise NotImplemented

    def get_rank(self, text_pair) -> np.ndarray:
        """
        对文本对进行重排序
        :param text_pair: 待排序文本
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, truncation=True,
                    return_tensors='pt', max_length=512,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        raise NotImplemented

    def query_document(self, query: str, knowledge_id: int) -> List[str]:
        # 全文检索，指定一个知识库检索，bm25打分
        word_search_response = es.search(index="chunk_info",
                                         body={
                                             "query": {
                                                 "bool": {
                                                     "must": [
                                                         {
                                                             "match": {
                                                                 "chunk_content": query
                                                             }
                                                         }
                                                     ],
                                                     "filter": [
                                                         {
                                                             "term": {
                                                                 "knowledge_id": knowledge_id
                                                             }
                                                         }
                                                     ]
                                                 }
                                             },
                                             "size": 50
                                         },
                                         fields=["chunk_id", "document_id", "knowledge_id", "page_number",
                                                 "chunk_content"],
                                         source=False,
                                         )

        # 语义检索
        embedding_vector = self.get_embedding(query)  # 编码
        knn_query = {
            "field": "embedding_vector",
            "query_vector": [float(x) for x in list(embedding_vector)],
            "k": 50,
            "num_candidates": 100,  # hnsw 检索检索初步计算得到top 100的待选文档， 筛选最相关的50个
            "filter": {
                "term": {
                    "knowledge_id": knowledge_id
                }
            }
        }
        vector_search_response = es.search(
            index="chunk_info", knn=knn_query,
            fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
            source=False,
        )

        # rrf
        # 检索1 ：[a， b， c]
        # 检索2 ：[b， e， a]
        # a 1/60    b 1/61    c 1/62
        # b 1/60    e 1/61    a 1/62

        k = 60
        fusion_score = {}
        search_id2record = {}
        for idx, record in enumerate(word_search_response['hits']['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)

            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        for idx, record in enumerate(vector_search_response['hits']['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)

            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
        sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
        sorted_content = [x["chunk_content"] for x in sorted_records]

        if self.use_rerank:
            text_pair = []
            for chunk_content in sorted_content:
                text_pair.append([query, chunk_content])
            rerank_score = self.rerank(text_pair)  # 重排序打分
            rerank_idx = np.argsort(rerank_score)[::-1]

            sorted_records = [sorted_records[x] for x in sorted_records]
            sorted_content = [sorted_content[x] for x in sorted_content]

        return sorted_records

    def chat_with_rag(
            self,
            knowledge_id: int,  # 知识库 哪一个知识库提问
            messages: List[Dict],
    ):
        # 用户的第一次提问用rag
        if len(messages) == 1:
            query = messages[0]["content"]
            related_records = self.query_document(query, knowledge_id)  # 检索到相关的文档
            print(related_records)
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                .replace("{#QUESTION#}", query) \
                .replace("{#RELATED_DOCUMENT#}", related_document)

            rag_response = self.chat(
                [{"role": "user", "content": rag_query}],
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": rag_response})
        # 后序提问 直接大模型回答
        else:
            normal_response = self.chat(
                messages,
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": normal_response})

        # messages.append({"role": "system", "content": rag_response})
        return messages

    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message

    def query_parse(self, query: str) -> str:
        return ""

    def query_rewrite(self, query: str) -> str:
        return ""
