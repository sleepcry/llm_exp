import langchain
#langchain.verbose = True
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers.regex import RegexParser
import time
import os
import tiktoken
import interview_prompts
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

os.environ["OPENAI_API_KEY"] = open('corpus/.chatgpt').read().strip()
os.environ["TOKENIZERS_PARALLELISM"]="false"
enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKEN = 4000
class InterViewer:
    def __init__(self,sys_tmpl,company,job):
        assistant_sys_msg = sys_tmpl.format_messages(
            company=company,
            job=job,
        )[0]
        self.system_message = assistant_sys_msg
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        self.init_messages()

    def reset(self):
        self.init_messages()
        return self.stored_messages

    def init_messages(self):
        self.stored_messages = [self.system_message]

    def filter_old_msg(self):
        token_cnt = [len(enc.encode(s.content)) for s in self.stored_messages]
        total_token = token_cnt[0]
        msg_len = 0 
        for l in token_cnt[1:][::-1]:
            if total_token + l > MAX_TOKEN:
                break
            total_token += l
            msg_len += 1
        self.stored_messages = [self.stored_messages[0]] + self.stored_messages[-msg_len:]

    def update_messages(self, message):
        self.stored_messages.append(message)
        len_msg = [len(enc.encode(s.content)) for s in self.stored_messages]
        self.filter_old_msg()
        return self.stored_messages

    def step(self):
        output_message = self.model(self.stored_messages)
        self.update_messages(output_message) 
        #print('-'*20,'\n\n','\n\n'.join([s.content for s in self.stored_messages]),'\n\n','-'*20)
        return output_message



class ReViewer:
    def __init__(self,prompt,company,job):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        parser = RegexParser(
            regex=r"Score: (.*?)\n.*?Comment:(.*)",
            output_keys=["score", "comments"],
        )
        self.model = LLMChain(llm=llm,prompt=prompt,output_parser=parser)
        self.company = company
        self.job = job
        self.init_score()

    def init_score(self):
        self.stored_messages = []

    def review(self, question, answer):
        res = self.model({"company":self.company,"job":self.job,"question":question,"answer":answer})
        score = res['text']['score']
        comments = res['text']['comments']
        self.stored_messages.append((question,answer,comments,score))
        return comments,score

    def summary(self):
        total_score = 0
        for q,a,c,s in self.stored_messages:
            print("问题:",q,'\n')
            print("答案：",a,'\n')
            print("点评：",c,'\n')
            print("得分:",s,'\n')
            total_score += int(s)
        print('--'*20,'\n',"综合得分：%.2f%%"%(total_score*1.0/len(self.stored_messages)))

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
another_candidate = LLMChain(
        llm=llm,
        prompt=interview_prompts.ANOTHER_CANDIDATE_PROMPT
    )

#company = '心理健康诊所'
#job = '心理咨询师'
company = '互联网公司'
job = 'JAVA后端程序员'
interviewer = InterViewer(interview_prompts.INTERVIEWER_PROMPT,company,job)
reviewer = ReViewer(interview_prompts.REVIEWER_PROMPT,company,job)
print('面试官：你好，感谢你前来应聘"%s"岗位，我是这家%s的面试官，下面我们开始面试！\n'%(job,company))
user_hi = "我已经准备好了，请开始提问吧！"
print("求职者：",user_hi,'\n')
interviewer.update_messages(HumanMessage(content=(user_hi)))
while 1:
    question = interviewer.step().content
    print("面试官：",question,'\n\n****TIPS:****\n输入pass可以跳过该问题。输入support可以让系统帮你回答。输入over查看成绩并结束面试。\n')
    answer = str(input()).strip()
    if answer == 'pass':
        interviewer.update_messages(HumanMessage(content=("这个问题我不会，换下一个问题。")))
        continue
    elif answer == 'support':
        res = another_candidate({"company":company,"job":job,"question":question})
        reply = res["text"]
        print("\n\n",reply)
    elif answer == 'over':
        reviewer.summary()
        break
    else:
        reply = answer
    interviewer.update_messages(HumanMessage(content=(reply)))
    comments,score = reviewer.review(question,reply)
    print("面试得分：%s/100"%(score,))
    print("面试点评：",comments)
    print('\n\n',"="*40,'\n\n')
