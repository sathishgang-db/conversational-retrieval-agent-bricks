# Databricks notebook source
# MAGIC %md Let's start with a toy example.
# MAGIC We're a ficticious start up - building chat systems for books. Publishers are our customers. 
# MAGIC
# MAGIC Because, its halloween season 🎃, we'll use an example where we take the <b>Frankenstein</b> novel and show how to build a **conversational retrieval agent** around it.
# MAGIC <img src="files/tables/sathish.gangichetty@databricks.com/frankenstein.png" alt="Example Image" width="400" height="300"/>

# COMMAND ----------

# MAGIC %md
# MAGIC In the rest of this notebook, we'll explore
# MAGIC - Everything thats needed to set up a proper retrieval system based on Semantic Search
# MAGIC - Quickly go through document loading, text splitting/chunking, embedding generation and loading them into a local vector store FAISS
# MAGIC - Then set this up as a tool for the agent, set up the agent to interact with this tool
# MAGIC - Identify opportunities of using an agent, highlight possible problems that need to be fixed

# COMMAND ----------

# MAGIC %pip install faiss-gpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Load the book
from langchain.document_loaders import GutenbergLoader
loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/84/pg84.txt")
data = loader.load()

# COMMAND ----------

# Set up the local vector store, Use local HuggingFace Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# COMMAND ----------

# Set up the vector store as a retriever
retriever = db.as_retriever()

# COMMAND ----------

# https://blog.langchain.dev/conversational-retrieval-agents/
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
import os
from langchain.chat_models import ChatOpenAI
# Use Databricks Secrets API to fetch the OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd", "openai_api")

tool = create_retriever_tool(
    retriever, 
    "frankenstein_or_the_modern_prometheus",
    "Searches and returns documents regarding the frankenstein."
)
tools = [tool]
llm = ChatOpenAI(temperature = 0) # define llm
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True) # attach llm and tools to the agent
result = agent_executor({"input": "hi, im sathish"}) # run the agent

# COMMAND ----------

result = agent_executor({"input": "whats my name?"}) # agent has a memory

# COMMAND ----------

result = agent_executor({"input": "whats the moral of Frankenstein's story?"}) # can use tools
print(result['output'])

# COMMAND ----------

result = agent_executor({"input": "oh is victor frankenstien the same as viktor frankl?"}) # can reason about the world without tools -> inherent intelligence

# COMMAND ----------

result = agent_executor({"input": "what is my name?"}) # memory doesn't fail

# COMMAND ----------

result = agent_executor({"input": "remember this info. I like dogs"})

# COMMAND ----------

result = agent_executor({"input": "no, who are the main characters in the Frankenstein?"})
print(result['output'])

# COMMAND ----------

agent_executor({"input": "do I like cats?"}) #careful what you unleash

# COMMAND ----------

result = agent_executor({"input": "summarize the story of frankenstein?"}) # Retriever is kinda useless for summarization
print(result['output'])

# COMMAND ----------

result = agent_executor({"input": "what do you know about George Bush?"}) # do we really want to engage with these types of questions
print(result['output'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### So the behavior has to be altered. How do we go about doing this?

# COMMAND ----------

# MAGIC %md Our Options:
# MAGIC 1. Change the prompts and hope it works - Expensive & UnReliable.
# MAGIC 2. Do we truly want to send **EVERYTHING** what users say to another 3rd party service, especially the **worst questions**?
# MAGIC 3. Can we use a great open source model to reduce the expense, while being efficient? Get more done with less $.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="files/tables/sathish.gangichetty@databricks.com/Hf_ceo.jpg" alt="Example Image" width="600" height="600"/>
