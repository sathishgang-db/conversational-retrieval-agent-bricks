# Databricks notebook source
# MAGIC %pip install faiss-gpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Like before, start with the retriever tool again...

# COMMAND ----------

from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool

loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/84/pg84.txt")
#LOAD. SPLIT. EMBED. DEFINE TOOL
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()
retreive_tool = create_retriever_tool(
    retriever, 
    "frankenstein_or_the_modern_prometheus",
    "Searches and returns documents regarding the frankenstein."
)

# COMMAND ----------

# MAGIC %md Define the LLM. Note, in a real app, you're better off using GPT-4

# COMMAND ----------

import os
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd", "openai_api")
llm = ChatOpenAI(temperature = 0)

# COMMAND ----------

# MAGIC %md This time around, let's add another tool to the mix. We'll use our deployed model and set it up as a tool for the agent to use.
# MAGIC This is an important pattern, because this means you can build any and all tools and make it a part of the toolkit that the agent can use. 

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv()
import requests
import json


def score_model(payload):
    """Use the deployed model to score and fetch responses"""
    url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/mistral-7b-parser-sg/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    json_payload = json.dumps(payload)
    response = requests.request(method='POST', headers=headers, url=url, data=json_payload)
    return response


def build_prompt_payload(query):
    """compose payload to send to the deployed model using the prompts"""
    prompt = f"""Extract in JSON the category of the user text (other, frankenstein, banter, political, violence, hate), the reason for the category, and an action (gently avoid the question if the category is political, violence, hate, other; else engage) based on the user text from the message below.user text: {query} """
    payload = {"dataframe_split": {"columns": ["prompt"], "data": [[prompt]]}}
    return payload

# COMMAND ----------

test_payload = build_prompt_payload("who is george w. bush?")
res = score_model(test_payload)
json.dumps(res.json()['predictions']['candidates'][0])

# COMMAND ----------

from langchain.tools import tool
@tool("gater")
def check_query(query: str) -> str:
    """Check all user queries and determines if agent should engage or not."""
    payload = build_prompt_payload(query)
    result = score_model(test_payload)
    return json.dumps(res.json()['predictions']['candidates'][0])

# COMMAND ----------

tools = [retreive_tool, check_query] #now we have 2 tools in our toolkit

# COMMAND ----------

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
# This is needed for both the memory and the prompt
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

# COMMAND ----------

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
system_message = SystemMessage(
        content=(
            "Always start with the gater tool and follow its advice on action on how to handle user questions. Use other tools as needed"
            "to provide relevant information." 
            "DO NOT entertain political, hurtful or violent questions."
        )
)
prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

# COMMAND ----------

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
#define the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,return_intermediate_steps=True)

# COMMAND ----------

user_query = agent_executor({"input": "hi, im Sathish"})

# COMMAND ----------

user_query = agent_executor({"input": "What can you tell me about trump?"})
user_query["output"]

# COMMAND ----------

result = agent_executor({"input": "I think all politicians are bad. do you think so?"})
result['output']

# COMMAND ----------

result = agent_executor({"input": "whats the moral of Frankenstein's story?"})
result["output"]

# COMMAND ----------

# MAGIC %md
# MAGIC You can log the agent into MLflow just like everything else and serve the whole agent as an API.
# MAGIC [See here for details](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html?highlight=agent%20logging#mlflow.langchain.log_model)

# COMMAND ----------


