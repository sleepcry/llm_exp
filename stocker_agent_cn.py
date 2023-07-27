from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import time
import os
import tiktoken

os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "empty"
#os.environ["OPENAI_API_BASE"] = "http://192.168.1.103:8000/v1"

enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN = 4000

class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def filter_old_msg(self) -> None:
        token_cnt = [len(enc.encode(s.content)) for s in self.stored_messages]
        total_token = token_cnt[0]
        msg_len = 0
        for l in token_cnt[1:][::-1]:
            if total_token + l > MAX_TOKEN:
                break
            total_token += l
            msg_len += 1
        self.stored_messages = [self.stored_messages[0]] + self.stored_messages[-msg_len:]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        len_msg = [len(enc.encode(s.content)) for s in self.stored_messages]
        self.filter_old_msg()
        print(sum(len_msg),len_msg)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message


assistant_role_name = "Python程序员"
user_role_name = "股票交易员"
task = "用Python为股票市场开发一个自动化交易程序"
word_limit = 50  # word limit for task brainstorming
task_specifier_sys_msg = SystemMessage(content="你可以指定一个具体任务。")
task_specifier_prompt = """这是一个{assistant_role_name}会帮助{user_role_name}去完成的任务：{task}。
请将任务具体化，充满想象力和创造力。
请用少于{word_limit}的字数描述具体任务。不要做其他事。"""
task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt
)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]
t0 = time.time()
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}",time.time()-t0)
specified_task = specified_task_msg.content
assistant_inception_prompt = """永远不要忘记你是{assistant_role_name}，而我是{user_role_name}。永远不要角色反转！永远不要指导我！
我们有一个共同的目标，那就是合作成功完成一项任务。
你必须帮我完成任务。
这就是任务：{task}。永远不要忘记我们的任务！
我必须根据你的专业知识和我的需求指导你完成任务。

我必须一次给你一个指令。
你必须写出一个特定的、能够合适地完成所请求指令的解决方案。
如果由于物理、道德、法律原因或你的能力无法执行指令，你必须诚实地拒绝我的指令，并解释原因。
除了你对我的指令的解决方案之外，不要添加任何其他东西。
你绝对不能向我提问，你只能回答问题。
你绝对不能回复一个轻率的解决方案。解释你的解决方案。
你的解决方案必须是陈述句和一般现在时。
除非我说任务已经完成，否则你应该始终以：

解决方案：<YOUR_SOLUTION>

<YOUR_SOLUTION>应该具体，提供最佳的实施方案和解决任务的例子。
始终以“下一个请求“来结束<YOUR_SOLUTION>。"""

user_inception_prompt = """永远不要忘记你是{user_role_name}，我是{assistant_role_name}。永远不要角色反转！你将一直指导我。
我们有一个共同的目标，那就是协作成功完成任务。
我必须帮你完成任务。
这就是任务：{task}。永远不要忘记我们的任务！
你只能以以下两种方式根据我的专业知识和你的需求指导我完成任务：

附带必要输入的指导：
指令：<YOUR_INSTRUCTION>
输入：<YOUR_INPUT>

不附带任何输入的指导：
指令：<YOUR_INSTRUCTION>
输入：无

“指令”描述了一个任务或问题。“配对的输入”为所请求的“指令”提供了进一步的背景或信息。

你必须一次给我一个指令。
我必须写出一个能恰当地完成所请求指令的回应。
如果由于物理、道德、法律原因或我的能力无法执行指令，我必须诚实地拒绝你的指令，并解释原因。
你应该指导我，而不是问我问题。
现在你必须开始使用上述两种方式指导我。
除你的指令及可选的相应输入之外，不要添加任何其他东西！
继续给我指令和必要的输入，直到你认为任务已经完成。
当任务完成时，你只需回复一个单词<CAMEL_TASK_DONE>。
除非我的回应已经解决了你的任务，否则永远不要说<CAMEL_TASK_DONE>。"""

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg

assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task
)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}. "
        "现在开始逐条给我指令。"
        "仅仅回复'指令'和'输入'。"
    )
)

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
t0 = time.time()
user_msg = assistant_agent.step(user_msg)
print(f"Original task prompt:\n{task}\n",time.time()-t0)
print(f"Specified task prompt:\n{specified_task}\n")

chat_turn_limit, n = 30, 0
while n < chat_turn_limit:
    n += 1
    t0 = time.time()
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n",time.time()-t0)

    t0 = time.time()
    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n",time.time()-t0)
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
