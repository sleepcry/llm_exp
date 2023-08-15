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

## 心理学助教  

### 给助教添加相关文档并开始聊天  
`python langchain_qa.py -d [doc file] -v [store file]`

### 调用之前的文档直接开始聊天
`python langchain_qa.py -v [store file]`

### 已添加文档  
store文件名：psycho.vecstore
文档列表：  
1. 普通心理学第四版
2. 实验心理学
3. 动机与人格
4. 社会心理学
5. 心理学导论

