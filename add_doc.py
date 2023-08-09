import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from st_embedding import STEmbedding 


embeddings_model = STEmbedding()

def add_emb(book,content,emb,outp):
    outp.write(f"《{book}》：{content}\n")
    outp.write(','.join([str(e) for e in emb]) + '\n')

def get_emb(content):
    return embeddings_model.embed_documents([content])[0]

book_name = sys.argv[1]    
with open(sys.argv[2]) as fp,open(sys.argv[3],'a') as op:
    paragraph = ''
    for row in fp:
        row = row.strip()
        if len(row) == 0: 
            if(len(paragraph)) > 0:
                emb = get_emb(paragraph)
                add_emb(book_name,paragraph,emb,op)
            paragraph = ''
            continue
        else:
            paragraph += row
