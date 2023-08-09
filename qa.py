import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from st_embedding import STEmbedding
import faiss

embeddings_model = STEmbedding()
embedding_size = 1024
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

def get_db():
    contents = []
    embs = []
    with open(sys.argv[1],'r') as fp:
        i = 0
        for row in fp:
            if i % 2 == 0:
                contents.append(row)
            else:
                embs.append([np.float32(e) for e in row.split(',')])
            i += 1
    return contents,np.asarray(embs)

if __name__ == '__main__':
    db_contents,db_embeddings = get_db()
    text_embedding_pairs = list(zip(db_contents, db_embeddings))
    vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings_model)
    while 1:
        print('输入问题：\n')
        s = str(input())
        if s == 'exit':
            break
        results = vectorstore.similarity_search_with_score(s,k=3)
        for r in results:
            print(r)
        '''emb = embeddings_model.embed_query(s)
        print(db_embeddings.shape,emb[None,:].shape)
        similiars = cosine_similarity(db_embeddings,emb[None,:]).flatten()
        idxs = np.argsort(similiars)[-3:]
        print(similiars[idxs],similiars.min(),similiars.max(),similiars.mean())
        for idx in idxs:
            print(idx,db_contents[idx])
        '''

