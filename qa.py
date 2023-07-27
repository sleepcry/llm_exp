import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

model = SentenceTransformer('intfloat/multilingual-e5-large')

def get_emb(content):
    return model.encode([content])[0]

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
    while 1:
        print('输入问题：\n')
        s = str(input())
        if s == 'exit':
            break
        emb = get_emb(s)
        print(db_embeddings.shape,emb[None,:].shape)
        similiars = cosine_similarity(db_embeddings,emb[None,:]).flatten()
        idxs = np.argsort(similiars)[-3:]
        print(similiars[idxs],similiars.min(),similiars.max(),similiars.mean())
        for idx in idxs:
            print(idx,db_contents[idx])

