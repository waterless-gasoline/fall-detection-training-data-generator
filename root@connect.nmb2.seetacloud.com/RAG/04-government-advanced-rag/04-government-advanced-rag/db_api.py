from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

import yaml  # type: ignore

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

db_config = config['database']
db_type = db_config['engine']

if db_type == "sqlite":
    # SQLite 使用文件路径
    db_path = db_config.get('path', 'rag.db')
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
else:
    # MySQL 或其他数据库使用 host, port, username, password
    host = db_config.get('host', 'localhost')
    port = db_config.get('port', 3306)
    username = db_config.get('username', 'user')
    password = db_config.get('password', 'password')
    database = db_config.get('database', 'mydb')  # 数据库名

    engine = create_engine(
        f"{db_type}://{username}:{password}@{host}:{port}/{database}",
        echo=True
    )

# 创建 Base 类
Base = declarative_base()


# ORM
# 定义 knowledge_database 表
class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'

    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)  # 主键，自动递增
    title = Column(String)  # 名称
    category = Column(String)  # 类型
    create_dt = Column(DateTime, default=datetime.utcnow)  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新时间

    # 与 KnowledgeDocument 表的关系
    documents = relationship("KnowledgeDocument", back_populates="knowledge")

    def __str__(self):
        return (f"KnowledgeDatabase(knowledge_id={self.knowledge_id}, "
                f"title='{self.title}', category='{self.category}', "
                f"author_id={self.author_id}, create_dt={self.create_dt}, "
                f"update_dt={self.update_dt})")


# 定义 knowledge_document 表
class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'

    document_id = Column(Integer, primary_key=True, autoincrement=True)  # 文档主键，自动递增
    title = Column(String)  # 文档名称
    category = Column(String)  # 文档类型
    knowledge_id = Column(Integer, ForeignKey('knowledge_database.knowledge_id'))  # 知识库主键（外键）
    file_path = Column(String)  # 储存地址
    file_type = Column(String)  # 数据类型
    create_dt = Column(DateTime, default=datetime.utcnow)  # 创建时间
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新时间

    # 与 KnowledgeDatabase 表的关系
    knowledge = relationship("KnowledgeDatabase", back_populates="documents")


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
