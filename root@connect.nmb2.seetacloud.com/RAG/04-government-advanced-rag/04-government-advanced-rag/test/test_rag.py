import yaml  # type: ignore
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pytest
from rag_api import RAG


def test_embeddding_model():
    if config["rag"]["use_embedding"]:
        embedding = RAG().get_embedding("测试文本")
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
    else:
        assert 1==1


def test_rerank_model():
    if config["rag"]["use_rerank"]:
        test_pair = [["我今天很开心", "我今天很开心"], ["我今天很开心", "我今天很不开心"]]
        embedding = RAG().get_rank(test_pair)
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
        assert embedding[0] > embedding[1]
    else:
        assert 1==1

def test_llm():
    messages=[    
            {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},    
            {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事."} 
    ]
    assert RAG().chat(messages, 0.7, 0.9) != None

