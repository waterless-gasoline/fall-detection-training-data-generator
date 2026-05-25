# 数据接口，pydantic 定义
# data class， data model

import datetime

from pydantic import BaseModel, Field
from typing import Union, List, Any, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Form
from typing_extensions import Annotated


class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    token: str
    model: str


class EmbeddingResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[List[float]] = Field(description="文本对应的向量表示")
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


class RerankRequest(BaseModel):
    text_pair: List[Tuple[str, str]]
    token: str
    model: str


class RerankResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[float]
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


class KnowledgeRequest(BaseModel):
    category: str
    title: str


class KnowledgeResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    knowledge_id: int
    category: str
    title: str
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


class DocumentRequest(BaseModel):
    knowledge_id: int = Annotated[str, Form()],
    title: str = Annotated[str, Form()],
    category: str = Annotated[str, Form()],
    file: UploadFile = Annotated[str, File(...)]


class DocumentResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    document_id: int
    category: str
    title: str
    knowledge_id: int
    file_type: str
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


class RAGRequest(BaseModel):
    knowledge_id: int
    message: List[Dict]


class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")
