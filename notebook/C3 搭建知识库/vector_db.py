import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import Chroma

_ = load_dotenv(find_dotenv())

# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890' 

file_paths = []
folder_path = 'data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.startswith('.'):
            continue
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(TextLoader(file_path))

texts = []
for loader in loaders: texts.extend(loader.load())
text = texts[1]
print(f"每一个元素的类型：{type(text)}.", f"该文档的描述性数据：{text.metadata}", f"查看该文档的内容:\n{text.page_content[0:]}", sep="\n------\n")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=80, separators=["\n## ", "\n# ", "\n\n", "\n", " "])
split_docs = text_splitter.split_documents(texts)
unique_docs = []
seen = set()
for d in split_docs:
    c = d.page_content.strip()
    if len(c) > 800:
        c = c[:800]
    key = (c, d.metadata.get("source"))
    if key in seen:
        continue
    seen.add(key)
    unique_docs.append(Document(page_content=c, metadata=d.metadata))
print(f"切分后的文件数量：{len(split_docs)}")
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

embedding = QianfanEmbeddingsEndpoint()
persist_directory = 'data_base/vector_db/chroma'
vectordb = Chroma.from_documents(documents=unique_docs, embedding=embedding, persist_directory=persist_directory)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

question = '什么是大语言模型'
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}") 
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
