import pytest
import numpy as np

from es_api import es

def test_connect_es():
    assert es.ping() is not None

def test_init_es():
    assert es.indices.exists(index="document_meta"), "Index 'es_index' does not exist"
    assert es.indices.exists(index="chunk_info"), "Index 'chunk_info' does not exist"


def test_insert_document_meta():
    test_document = {
        "file_path": "test_file.txt",
        "file_name": "test_file.txt",
        "abstract": "This is a test abstract.",
        "full_content": "This is the full content of the test file."
    }
    response = es.index(index="document_meta", document=test_document)
    assert response['result'] == 'created'

    doc_id = response['_id']
    result = es.exists(index="document_meta", id=doc_id)
    assert result

    es.delete(index="document_meta", id=doc_id)

def test_query_document_meta():
    test_document = {
        "file_path": "query_test_file.txt",
        "file_name": "query_test_file.txt",
        "abstract": "This is an abstract for query testing.",
        "full_content": "Full content for the query test file."
    }
    
    insert_response = es.index(index="document_meta", document=test_document)
    assert insert_response['result'] == 'created'
    doc_id = insert_response['_id']
    
    search_response = es.search(index="document_meta", query={"match": {"file_name": "query_test_file.txt"}})
    assert search_response['hits']['total']['value'] > 0, "Query should return at least one document"
    
    retrieved_document = search_response['hits']['hits'][0]['_source']
    assert retrieved_document['file_name'] == test_document['file_name']
    assert retrieved_document['abstract'] == test_document['abstract']
    assert retrieved_document['full_content'] == test_document['full_content']

    es.delete(index="document_meta", id=doc_id)

def test_insert_chunk_info():
    test_chunk = {
        "chunk_id": 0,
        "knowledge_id": "knowledge_base_1",
        "document_id": "document_456",
        "page_number": 1,
        "chunk_content": "This is the content of chunk 123.",
        "chunk_images": ["/path/to/image1.jpg"],
        "chunk_tables": ["/path/to/table1.csv"],
        "embedding_vector": [0.1] * 512
    }
    
    response = es.index(index="chunk_info", document=test_chunk)
    assert response['result'] == 'created'

    doc_id = response['_id']

    result = es.exists(index="chunk_info", id=doc_id)
    assert result, "Document should exist in the chunk_info index after insertion"

    es.delete(index="chunk_info", id=doc_id)

def test_query_chunk_info():
    # Insert sample data for testing
    test_chunk = {
        "chunk_id": 0,
        "knowledge_id": "knowledge_base_1",
        "document_id": "document_456",
        "page_number": 1,
        "chunk_content": "This is the content of chunk 123.",
        "chunk_images": "/path/to/image1.jpg",
        "chunk_tables": "/path/to/table1.csv",
        "embedding_vector": np.random.rand(512).tolist()  # Random 512-dimensional vector for testing
    }

    # Insert the test chunk document into Elasticsearch
    response = es.index(index="chunk_info", document=test_chunk)
    assert response['result'] == 'created'

    doc_id = response['_id']

    # 1. Test query chunk_content
    query_content = "chunk 123"
    content_search_response = es.search(index="chunk_info", body={
        "query": {
            "match": {
                "chunk_content": query_content
            }
        }
    })
    assert content_search_response['hits']['total']['value'] > 0, "No matching documents found for chunk_content query"

    knn_query = {
        "field": "embedding_vector",
        "query_vector": [0.1] * 512,
        "k": 5,
        "num_candidates": 10,
    }

    vector_search_response = es.search(index="chunk_info", knn=knn_query)
    assert vector_search_response['hits']['total']['value'] > 0, "No similar chunks found in vector search"

    es.delete(index="chunk_info", id=doc_id)