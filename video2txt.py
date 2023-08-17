import moviepy.editor as mp
from paddlespeech.cli.asr.infer import ASRExecutor
from langchain.text_splitter import SpacyTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from simple_splitter import EmptyLineSplitter
from st_embedding import STEmbedding
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
import sys
import os
import time
import prompts
import maprerank
from pydub import AudioSegment
from pydub.silence import split_on_silence
from langchain.chains.llm import LLMChain

os.environ["OPENAI_API_KEY"] = open('corpus/.chatgpt').read().strip()
os.environ["TOKENIZERS_PARALLELISM"]="false"


tmp_file = "111.wav"

print("从视频文件中提取音频文件...")
video = mp.VideoFileClip(sys.argv[1])
audio_file = video.audio
audio_file.write_audiofile(tmp_file)
print("音频文件提取完成:",tmp_file)
sound_file = AudioSegment.from_wav("out.wav")
print("切割音频文件...")
audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40 )

print("切割完成，chunk数量:",len(audio_chunks),",音频转文字...")
asr = ASRExecutor()
text = ""
"""
for i, chunk in enumerate(audio_chunks):
   out_file = "chunk{0}.wav".format(i)
   print("exporting", out_file)
   chunk.export(out_file, format="wav")
   text += asr(audio_file=out_file)
   """
loader = TextLoader("corpus/luoxiang.txt")
text = loader.load()
# Print the text
print("\n音频对应的文字是: \n")
print(text,'\n\n')
print("开始生成文字摘要...\n")

#text_splitter = SpacyTextSplitter(chunk_size=1000)
text_splitter = EmptyLineSplitter()
docs = text_splitter.split_documents(text)
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k")
chain = load_summarize_chain(llm, chain_type="stuff")
#chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(docs)
print("视频摘要：",summary,"\n")
print('='*40,'\n\n')

print("建立文档索引...","\n")
embeddings_model = STEmbedding()
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)
#vectorstore = FAISS.load_local("Q.vecstore",embeddings_model)
llm_chain = LLMChain(
        llm=llm,
        prompt=prompts.PROMPT
    )
combine_documents_chain = maprerank.MyMapRerankDocumentsChain(
        llm_chain=llm_chain,
        rank_key="score",
        answer_key="answer",
        document_variable_name="context",
    )       

qa_chain = RetrievalQA(retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 2}),return_source_documents=True,combine_documents_chain=combine_documents_chain)

while 1:
    print("针对该课程，你有任何疑问，都可以问我！\n")
    question = str(input()).strip()
    if question == 'exit':
        break
    t0 = time.time()    
    result = qa_chain({"query": question})
    print("="*20,'答案','='*20)
    print("time cost:",time.time()-t0,'\n')
    print("问题：",question,'\n')
    print("答案：",result["result"]["answer"],'\n')
    print("评分(0-100)：",result["result"]["score"],'\n')
    print("对应的文档：",result["result"]["_doc_"],'\n')
    print("文档源头：",result["result"]["_doc_metadata_"],'\n')
    print("="*40)

