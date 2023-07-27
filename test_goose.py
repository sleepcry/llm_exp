from typing import List
from langchain.chat_models import ChatOpenAI
import openai
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
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "empty"
#os.environ["OPENAI_API_BASE"] = "http://192.168.1.103:8000/v1"
openai.api_key = "sk-fwqJpKxvQJC00QTvoLlcFmv16zEObRF0beXRADug8iufttF3"
openai.api_base = "https://api.goose.ai/v1"

# List Engines (Models)
engines = openai.Engine.list()
# Print all engines IDs
for engine in engines.data:
  print(engine.id)

# Create a completion, return results streaming as they are generated. Run with `python3 -u` to ensure unbuffered output.
completion = openai.Completion.create(
  engine="gpt-neo-20b",
  prompt="很久很久之前，森林里住着一只大鹅,",
  max_tokens=160,
  stream=True)

# Print each token as it is returned
for c in completion:
  print (c.choices[0].text, end = '')

print("")
