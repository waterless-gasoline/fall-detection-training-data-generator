## 实操步骤

- 步骤1: 数据库设计，关系型数据库设计，非关系型数据库设计
  - 哪些数据需要存储在关系型数据库（mysql / sqlite） -》 存储结构化信息？
    - 有哪些知识库？
      - 物业费知识库
      - 个人所得税知识库
      - 养老保险知识库
    - 知识库中有哪些文档？
      - 物业费知识库， 文档1 / 文档2 / 文档3
  - 哪些数据需要存储在es -》 文档存储和文档的检索？
    - 文档的信息
    - chunk的信息
  - db_api.py
  - es_api.py
- 步骤2: 前后端接口的定义 （工作量）
  - 模块进行划分
  - 接口划分，接口定义（前端什么时候调用，传入的参数，返回格式）
  - router_schema.py -》基础数据格式
  - main.py -》定义接口 传入的格式 + 返回格式
- 步骤3: 其他代码的实现，如接口实现细节
  - 推荐开发顺序：数据管理接口、基础算法接口、ai应用对话接口
  - 前端 -》 后端（校验、逻辑管理） -》 数据库
  - 例如：删除知识库接口
    - 基于 知识库id，在 knowledge_database 表中 删除记录
    - 基于 知识库id，找到文档id，在 knowledge_document 中 删除文档
    - ...
    - ...

## 数据库设计

### 元信息存储（mysql、sqlite）

- 知识库（knowledge_database）

| 字段      | 字段类型 | 字段含义         |
| --------- | -------- | ---------------- |
| knowledge_id        | bigint   | 知识库主键       |
| title     | varchar  | 名称             |
| category  | varchar  | 类型             |
| author_id | bigint   | 用户主键（外键） |
| create_dt | datetime | 创建时间         |
| update_dt | datetime | 更新时间         |

- 知识文档（knowledge_document）

| 字段         | 字段类型 | 字段含义           |
| ------------ | -------- | ------------------ |
| document_id           | bigint   | 文档主键           |
| title        | varchar  | 文档名称           |
| category     | varchar  | 文档类型           |
| knowledge_id | bigint   | 知识库主键（外键） |
| file_path    | varchar  | 储存地址           |
| file_type    | varchar  | 数据类型           |
| create_dt    | datetime | 创建时间           |
| update_dt    | datetime | 更新时间           |


### 文档全文与向量存储（es）

- **document_meta** 存储知识文档的元信息，包括文件路径、名称、摘要和全部内容等。

| 字段名           | 数据类型     | 字段说明               |
|------------------|--------------|------------------------|
| `document_id`    | `int`       | 文档元信息的唯一标识符   |
| `document_name`      | `text`       | 文档的名称              |
| `knowledge_id`   | `text`       | chunk 所属知识库的标识符 |
| `file_path`      | `text`       | 文档的存储路径          |
| `abstract`       | `text`       | 文档的摘要信息          |

- **chunk_info** 存储分块的文档内容，包含 chunk 的文字内容、图片、表格地址等。

| 字段名           | 数据类型     | 字段说明               |
|------------------|--------------|------------------------|
| `chunk_id`       | `text`       | chunk 的唯一标识符      |
| `document_id`    | `int`       | 文档元信息的唯一标识符   |
| `knowledge_id`   | `text`       | chunk 所属知识库的标识符 |
| `page_number`    | `integer`    | chunk 所在文档的页码    |
| `chunk_content`  | `text`       | chunk 的文字内容        |
| `chunk_images`   | `text`       | chunk 相关图片的路径     |
| `chunk_tables`   | `text`       | chunk 相关表格的路径     |
| `embedding_vector` | `array<float>` | chunk 的语义编码结果   |

## 开发资料

- https://elasticsearch-py.readthedocs.io/en/latest/

## 代码开发顺序
