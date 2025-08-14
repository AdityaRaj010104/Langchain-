from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:\Data_mine\Projects\LangChain\langchain_text_split\dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ' '
)

result = splitter.split_documents(docs)

# print(result[1].page_content)
print(result)