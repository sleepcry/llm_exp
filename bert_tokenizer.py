import numpy as np
from sentence_transformers import SentenceTransformer
import sys


model = SentenceTransformer('intfloat/multilingual-e5-large')
MAX_PARAGRAPH = 300
MAX_SEGMENT = 500

def add_emb(content,emb,outp):
    outp.write(content + '\n')
    outp.write(','.join([str(e) for e in emb]) + '\n')

def get_emb(content):
    return model.encode(['passage:'+content],normalize_embeddings=True)[0]

    
with open(sys.argv[1]) as fp,open(sys.argv[2],'a') as op:
    paragraph = ''
    segment = ''
    for row in fp:
        row = row.strip()
        if len(row) == 0: 
            segment += paragraph
            paragraph = ''
            len_seg = len(segment)
            if len_seg >= MAX_PARAGRAPH:
                print(segment)
                print('='*10)
                pos = 0
                while pos < len_seg:
                    seg = segment[pos:pos+MAX_SEGMENT]
                    emb = get_emb(seg)
                    add_emb(seg,emb,op)
                    pos += MAX_SEGMENT
                segment = ''
                continue
        else:
            paragraph += row
