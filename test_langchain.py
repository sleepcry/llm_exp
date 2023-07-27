from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents import AgentType, initialize_agent, load_tools


llm = OpenAI(openai_api_key="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",temperature=0.9)

#print(llm.predict("What would be a good company name for a company that makes colorful socks?"))
#print(llm.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")]))

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")

