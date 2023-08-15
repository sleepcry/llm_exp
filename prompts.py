from langchain.output_parsers.regex import RegexParser
from langchain.prompts import PromptTemplate


parser2 = RegexParser(
    regex=r"Score: (.*?)\n.*?Answer:(.*)",
    output_keys=["score", "answer"],
)
#''切记：Score为必填字段，无论任何情况都需要返回Score。
prompt_template = """阅读"Context"中的信息然后用中文回答"Question"描述的问题，你的答案必须源自于Context中的内容。
除了问题的答案，你必须评估一个分数"Score"来表示你多大程度上回答了用户的问题。如果你无法获取答案，回答你不知道并评估分数，不要试图编造答案。
按照如下格式进行回答：

--------
Score: [你评估的分数，分数为区间[0,100]的整数值，最小为0，最大为100]
Helpful Answer: [你的答案]
--------

如何评分：
- 分数Score越高代表回答越完整。
- Score=100,代表你认为答案准确地回答了"Question"中的问题，并且包含了充分的层次和细节。
- Score=0,表示你认为文档中的内容中完全找不到"Question"中问题的答案。
- 认真评估问题和答案的相关性，不要轻易给满分。

Example #1

Context:
---------
苹果是红色的。
---------
Question: 苹果是什么颜色的？
Score: 100
Helpful Answer: 红色 

Example #2

Context:
---------
当时是夜晚并且证人忘掉戴眼睛，因此他不是很确定那是一辆运动型轿车还是一辆SUV。
---------
Question: 那是一辆什么类型的车？
Score: 60
Helpful Answer: 运动型轿车或者SUV

Example #3

Context:
---------
梨子是要么是红色要么是橘色的
---------
Question: 苹果是什么颜色的？
Score: 0
Helpful Answer: 这篇文档没有对应答案。

现在开始吧！

Context:
---------
{context}
---------
Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=parser2
)

prompt_template = """ 评估问题是否为“心理学”相关问题，回答必须以“YES”或者“NO”开头，并说明原因。
如果该问题是“心理学”相关问题，回答必须以“YES”开头，否则，以“NO”开头。用中文作答！
比如：
Example #1

Question: 世界上最大的湖泊是什么？
Answer: NO，该问题是地理问题，并非心理学相关问题。

Example #2
Question: 什么是人脑发育的关键期和可塑性？
Answer: YES，该问题是心理学相关问题。
下面为正式问题：
Question: {question}
Answer: 
"""
CLASSIFIER_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)

prompt_template = """阅读“Context”中描述的QA上下文，评估“Question”是否为一个完整句子。如果是，不做任何改动，否则，基于上下文补全句子并返回，并详细说明你的判断的理由。
请按照下面格式回答：
Final Question: [补全后的问题]
Reason: [判断的理由]

Context:
--------
{context}
--------
Question:{question}
"""
prompt_template = """Find the entity inside the question "Qeustion", if you're not sure about the entity, find the entity inside "Context", and rewrite the question with this entity.
DO NOT add anything to change the original meaning of the question.
请按照下面格式回答：
--------
Final Question: [rewritten question]
Entity: [entity]
--------

开始吧！
Context:
--------
{context}
--------
Question:{question}
"""
COMPLETE_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"],
)
