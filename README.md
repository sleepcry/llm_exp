## python environment
` miniconda + python3.9 `

## .pdf corpus to .txt 
`python pdf2txt.py [folder name]`

## .txt docment to embedding
`python bert_tokenizer.py [corpus path,*.txt] [output embedding path, *.txt]`

## demonstrate role play with langchain and LLm
`python stocker_agent_cn.py`

notes: 

1. 需要给命令行翻墙。以我的Mac为例:`export http_proxy=http://127.0.0.1:10887` `export https_proxy=http://127.0.0.1:10887`
2. 里面的使用的chatgpt-3.5，可能出现接口频次超限或者付费问题


## demonstrate query based paragraph extraction
`python qa.py [embedding path, *.txt]`


