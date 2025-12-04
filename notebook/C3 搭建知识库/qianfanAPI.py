import os
import re
from dotenv import load_dotenv, find_dotenv
import qianfan
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ = load_dotenv(find_dotenv())

CHUNK_SIZE = 500
OVERLAP_SIZE = 50

def _env_ready() -> bool:
    ak = (os.environ.get('QIANFAN_ACCESS_KEY') or '').strip()
    sk = (os.environ.get('QIANFAN_SECRET_KEY') or '').strip()
    return bool(ak and sk)

def wenxin_embedding(text: str, model: str = 'Embedding-V1') -> dict:
    emb = qianfan.Embedding()
    resp = emb.do(model=model, texts=[text])
    return resp

def data_sort_pdf(url: str):
    loader = PyMuPDFLoader(url)
    pdf_pages = loader.load()
    print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
    pdf_page = pdf_pages[1]
    # pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    # pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
    # pdf_page.page_content = pdf_page.page_content.replace('•', '').replace(' ', '')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
    text_splitter.split_text(pdf_page.page_content[0:1000])
    split_docs = text_splitter.split_documents(pdf_pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    print(f"每一个元素的类型：{type(pdf_page)}.", f"该文档的描述性数据：{pdf_page.metadata}", f"查看该文档的内容:\n{pdf_page.page_content}", sep="\n------\n")

def data_sort_md(url: str):
    loader = UnstructuredMarkdownLoader(url, encoding="utf-8")
    md_pages = loader.load()
    print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 PDF 一共包含 {len(md_pages)} 页")
    md_page = md_pages[0]
    md_page.page_content = md_page.page_content.replace('\n\n', '\n')
    print(f"每一个元素的类型：{type(md_page)}.", f"该文档的描述性数据：{md_page.metadata}", f"查看该文档的内容:\n{md_page.page_content[0:]}", sep="\n------\n")

data_sort_pdf("data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")
data_sort_md("data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")

text = '要生成 embedding 的输入文本，字符串形式。'
if not _env_ready():
    print('未配置完整的安全认证密钥对：请在 .env 设置。')
else:
    try:
        response = wenxin_embedding(text=text)
        print('本次embedding id为：{}'.format(response['id']))
        print('本次embedding产生时间戳为：{}'.format(response['created']))
        print('返回的embedding类型为:{}'.format(response['object']))
        print('embedding长度为：{}'.format(len(response['data'][0]['embedding'])))
        print('embedding（前10）为：{}'.format(response['data'][0]['embedding'][:10]))
    except Exception as e:
        print(f'调用失败：{e}')
