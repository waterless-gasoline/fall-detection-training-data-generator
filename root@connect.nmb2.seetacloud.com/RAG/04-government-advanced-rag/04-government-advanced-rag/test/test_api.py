import json
import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_knowledge_base():
    new_data = {"category": "测试类别", "title": "测试标题"}
    response = client.post("/v1/knowledge_base/", json=new_data)
    assert response.status_code == 200
    assert response.json()["response_code"] == 200

    knowledge_id = response.json()["knowledge_id"]
    response = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    assert response.status_code == 200
    assert response.json()["title"] == "测试标题"

    response = client.delete(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    assert response.status_code == 200
    assert response.json()["response_msg"] == "知识库删除成功"

    response = client.delete(f"/v1/knowledge_base?knowledge_id={knowledge_id}&token=666")
    assert response.status_code == 200
    assert response.json()["response_msg"] == "知识库不存在"


def test_document():
    response = client.get(f"/v1/document?document_id=666&token=666")
    assert response.status_code == 200
    assert response.json()["response_msg"] == "文档不存在"

    response = client.delete(f"/v1/document?document_id=666&token=666")
    assert response.status_code == 200
    assert response.json()["response_msg"] == "文档不存在"