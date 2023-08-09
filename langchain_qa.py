from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from simple_splitter import EmptyLineSplitter
from st_embedding import STEmbedding
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import sys
import argparse
import pathlib
import time

os.environ["OPENAI_API_KEY"] = open('corpus/.chatgpt').read().strip()
os.environ["TOKENIZERS_PARALLELISM"]="false"

'''
    # add a document to vectore store 1 with vectore file name
    # add a document to vectore store 2 without vectore file name
    # add another document to vectore store 2
    # start a conversation without new documents
'''
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--doc',dest="doc",type=pathlib.Path)
parser.add_argument('-v', '--vector',dest="vector",type=str)
args = parser.parse_args(sys.argv[1:])

embeddings_model = STEmbedding()
if args.vector:
    vec_file = args.vector
else:
    assert(args.doc)
    vec_file = os.path.basename(args.doc)+'.vecstore'
vectorstore = None
if os.path.exists(vec_file):
    print('doc store found, loading ...')
    vectorstore = FAISS.load_local(args.vector,embeddings_model)
if args.doc:
    print('new doc found')
    loader = TextLoader(args.doc)
    data = loader.load()
    #text_splitter = CharacterTextSplitter(r'\n\n.*?',is_separator_regex=True)
    text_splitter = EmptyLineSplitter()
    all_splits = text_splitter.split_documents(data)
    if vectorstore:
        print('append new doc to existing doc store ...')
        vectorstore.add_documents(all_splits)
    else:
        print('generate new doc store with new doc ...')
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings_model)
    print('save new doc store to file:'+vec_file)
    vectorstore.save_local(vec_file)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(search_type="mmr"),return_source_documents=True)
while 1:
    print('输入问题：\n')
    s = str(input())
    if s == 'exit':
        break
    t0 = time.time()
    result = qa_chain({"query": s})
    print('\n\n'.join([str(d) for d in result['source_documents']]))
    print("="*10)
    print(time.time()-t0,result["result"])

