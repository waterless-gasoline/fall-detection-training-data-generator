import yaml  # type: ignore
from elasticsearch import Elasticsearch  # type: ignore
import traceback

# 读取配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 从配置文件中提取ES的连接参数
es_host = config["elasticsearch"]["host"]
es_port = config["elasticsearch"]["port"]
es_scheme = config["elasticsearch"]["scheme"]
es_username = config["elasticsearch"]["username"]
es_password = config["elasticsearch"]["password"]

if es_username != "" and es_password != "":
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)
    )
else:
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]
]["dims"]


def init_es():
    """
    检查es环境配置
    :return: 环境是否配置成功
    """
    if not es.ping():
        print("Could not connect to Elasticsearch.")
        return False

    # es 非关系型数据库 -》 一个表 （index） 存储一类数据 -》 可以动态插入数据
    # 需要对字段进行检索， 需要定义索引
    # es 中 ik 类似jieba
    # doc1 {"content": "", "title": "", "page": 100}
    # doc2 {"content": "", "title": "", "author": "", "page": 20}

    document_meta_mapping = {
        "mappings":{
            'properties': {
                'file_name': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                'abstract': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                'full_content': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                }
            }
        }
    }
    try:
        # es.indices.delete(index='document_meta')
        if not es.indices.exists(index="document_meta"):
            es.indices.create(index='document_meta', body=document_meta_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of document_meta.")
        return False

    # 向量检索。 使用es 或 miluvs
    # 维度、索引方法
    chunk_info_mapping = {
        'mappings': {  # Add 'mappings' here
            'properties': {
                'chunk_content': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                "embedding_vector": {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": embedding_dims,
                    "index": True,
                    "index_options": {
                        "type": "int8_hnsw"
                    }
                }
            }
        }
    }

    try:
        # es.indices.delete(index='chunk_info')
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index='chunk_info', body=chunk_info_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of chunk_info.")
        return False

    print("Successfully connected to Elasticsearch!")
    return True

init_es()
