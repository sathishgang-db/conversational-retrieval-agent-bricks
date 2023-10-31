# conversational-retrieval-agent-bricks

This repository provides a set of Python scripts that demonstrate the process of creating tools and agents, specificially, conversational retrieval agent (CRAG) on Databricks, as presented in the Phoenix Databricks User Group. 
The idea is to show how one can equip Databricks endpoints as tools that an agent can use when answering user queries. The example here assumes that you're a fictitious start up building a CRAG - as a book companion for Frankenstein.

This is based on this blog --> https://blog.langchain.dev/conversational-retrieval-agents/

## Installation

To set up the necessary environment:

1. Clone the repository into Databricks repos:
   ```bash
   git clone https://github.com/your-username/conversational-retrieval-agent-bricks.git
   cd conversational-retrieval-agent-bricks

## Scripts Overview

1.basicWorkflow.py

Purpose: Demonstrates the simplest way of using a conversational retrieval agent. Simple user query processing and agent response generation.

2.structuredOutputs.py

Purpose: Illustrates the process of converting a user query or prompt into structured JSON format.
- Conversion of user query to structured JSON.
- Deployment of the model as an endpoint on Databricks Serverless GPU endpoints.
- Highlights the benefits of a small-sih LLM on model serving: Quick, Cheap, Efficient, and Performant.

3.finetuningAlternative.py

Purpose: A guide on fine-tuning the model for structured outputs using state-of-the-art open-source models. Detailed steps on model fine-tuning on Databricks using QLoRA and NEFTune.

4.advanceWorkflow.py

Purpose: Explores the advanced usage of a Databricks Endpoint as a tool within a conversational agent. Demonstrates the concept that every endpoint can, in fact, be a tool that an agent can use, showcasing the flexibility and extensibility of the system.
