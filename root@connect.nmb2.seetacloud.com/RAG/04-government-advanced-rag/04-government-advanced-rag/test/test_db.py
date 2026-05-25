import pytest
from db_api import KnowledgeDatabase, Session

@pytest.fixture
def session():
    with Session() as session:
        yield session
        session.rollback()

def test_insert_knowledge_database(session):
    new_record = KnowledgeDatabase(title="test", category="category")
    session.add(new_record)
    session.commit()

    record = session.query(KnowledgeDatabase).filter_by(title="test").first()
    assert record is not None
    assert record.title == "test"
    assert record.category == "category"
    print("插入知识库成功")


def test_query_knowledge_database(session):
    session.add(KnowledgeDatabase(title="test", category="category"))
    session.commit()

    records = session.query(KnowledgeDatabase).filter_by(title="test").all()
    assert len(records) > 0
    assert records[0].title == "test"
    print("查询知识库成功")


def test_delete_knowledge_database(session):
    record_to_delete = KnowledgeDatabase(title="test", category="category")
    session.add(record_to_delete)
    session.commit()

    records_to_delete = session.query(KnowledgeDatabase).filter_by(title="test").all()
    for record in records_to_delete:
        session.delete(record)
        session.commit()

    records = session.query(KnowledgeDatabase).filter_by(title="test").all()
    assert len(records) == 0
    print("删除知识库成功")
