import tiktoken
import openai, numpy as np
#from openai.embeddings_utils import get_embedding, cosine_similarity
import sqlite3
import sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


openai.api_key = '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'
enc = tiktoken.get_encoding("cl100k_base")
MAX_PARAGRAPH = 200
MAX_SEGMENT = 1000

def add_emb(content,emb,outp):
    outp.write(content + '\n')
    outp.write(','.join([str(e) for e in emb]) + '\n')

def tokenizer(p):
    tokens = enc.encode(p)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_emb(content):
    resp = openai.Embedding.create(input=[content],engine='text-similarity-davinci-001')
    return resp['data'][0]['embedding']

    
with open(sys.argv[1]) as fp,open('embeddings.txt','a') as op:
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
