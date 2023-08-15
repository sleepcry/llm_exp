import langchain
langchain.verbose = True
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
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.chains.question_answering import map_rerank_prompt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
import prompts
from langchain.output_parsers.regex import RegexParser

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

class MyMapRerankDocumentsChain(MapRerankDocumentsChain):

    def _process_results(
        self,
        docs: List[Document],
        results: Sequence[Union[str, List[str], Dict[str, str]]],
    ) -> Tuple[str, dict]:
        print(results)
        typed_results = cast(List[dict], results)
        sorted_res = sorted(
            zip(typed_results, docs), key=lambda x: -int(x[0][self.rank_key])
        )   
        output, document = sorted_res[0]
        extra_info = {}
        if self.metadata_keys is not None:
            for key in self.metadata_keys:
                extra_info[key] = document.metadata[key]
        if self.return_intermediate_steps:
            extra_info["intermediate_steps"] = results
        output["_doc_"] = document.page_content
        output["_doc_metadata_"] = document.metadata
        return output, extra_info

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# for complate question 
memory = ["","","","","",""]
def add_memory(question,answer):
    for i in range(5):
        memory[i] = memory[i+1]
    memory[-1] = "question:" + question

parser1 = RegexParser(
    regex=r"Question: (.*?)\n+Entity:(.*)",
    output_keys=["final_q",'reason'],
)
complete_llm_chain = LLMChain(
        llm=llm,
        prompt=prompts.COMPLETE_PROMPT,
        output_parser=parser1
    )
# for classifier
classifier_llm_chain = LLMChain(
        llm=llm,
        prompt=prompts.CLASSIFIER_PROMPT
    )
# for map-rerank
llm_chain = LLMChain(
        llm=llm,
        prompt=prompts.PROMPT
    )
#TODO: search and answer
# for direct llm
direct_llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["question"],template="请用心理学知识回答该问题：{question}")
    )
combine_documents_chain = MyMapRerankDocumentsChain(
        llm_chain=llm_chain,
        rank_key="score",
        answer_key="answer",
        document_variable_name="context",
    )
qa_chain = RetrievalQA(retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 3}),return_source_documents=True,combine_documents_chain=combine_documents_chain)
#qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(search_type="mmr"),return_source_documents=True,chain_type='map_rerank')
while 1:
    print('输入问题：\n')
    question = str(input())
    if question == 'exit':
        break
    res = complete_llm_chain({'context':'\n'.join(memory),'question':question})
    question = res['text']['final_q']
    print('问题补全结果:',question,'Entity:',res['text']['reason'])
    res = classifier_llm_chain(question)
    if res['text'].startswith('NO'):
        print(res['text'][3:])
        continue
    else:
        print('心理学问题校验结果：',res['text'])
    t0 = time.time()
    result = qa_chain({"query": question})
    #print('\n\n'.join([str(d) for d in result['source_documents']]))
    print("="*20,'答案','='*20)
    print("time cost:",time.time()-t0,'\n')
    print("问题：",question,'\n')
    print("答案：",result["result"]["answer"],'\n')
    print("评分(0-100)：",result["result"]["score"],'\n')
    print("对应的文档：",result["result"]["_doc_"],'\n')
    print("文档源头：",result["result"]["_doc_metadata_"],'\n')
    print("="*40)
    if(int(result["result"]["score"]) < 100):
        print('由于提供的文档不够充分，该回答可能不够完善。是否不基于文档作答？（y/n）：\n')
        ope = str(input())
        if not ope.lower().startswith('y'):
            add_memory(question,result["result"]['answer'])
            continue
        res = direct_llm_chain(question)
        print(res)
        add_memory(question,res['text'])
    else:
        add_memory(question,result["result"]['answer'])

