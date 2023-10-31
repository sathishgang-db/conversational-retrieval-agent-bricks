# Databricks notebook source
# MAGIC %md
# MAGIC ###### Reference -> https://github.com/databricks/databricks-ml-examples

# COMMAND ----------

# MAGIC %pip install jsonformer
# MAGIC %pip install --upgrade git+https://github.com/huggingface/transformers
# MAGIC %pip install auto-gptq
# MAGIC %pip install optimum

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from jsonformer import Jsonformer
# https://github.com/1rgs/jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Set the warning action to ignore
warnings.filterwarnings('ignore')
# https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ
model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
system_message = "You are a helpful assistant who parses inputs to JSON."
prompt = """Extract in JSON the category of the user text (other, frankenstein, banter, political, violence, hate), 
the reason for the category and an action (gently avoid question if category is political, violence, hate, other. else engage) 
based on the user text from this message below.
user text:
Who is frankenstein"""
prompt_template=f'''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
json_schema = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "reason": {"type": "string"},
        "action": {"type": "string"},
        }
    }
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
generated_data = jsonformer()
print(generated_data)

# COMMAND ----------

from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id=model_name_or_path)

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

class Mistral7BJSONParser(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Method to initialize the model and tokenizer."""
        # model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
        self.model = AutoModelForCausalLM.from_pretrained(
          context.artifacts['repository'],
            device_map="auto",
            trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'], use_fast=True)
        self.model.eval()
        self.system_message = "You are a helpful assistant who parses inputs to JSON."
        self.json_schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "reason": {"type": "string"},
                "action": {"type": "string"},
            }
        }
        # self.jsonformer = Jsonformer(self.model, self.tokenizer, json_schema, system_message)
    def _build_prompt(self, instruction):
        """this method is used to build prompt for a single input"""
        return f'''<|im_start|>system
        {self.system_message}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        '''
    
    def _generate_response(self, prompt):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)
        # Call JSONFormer
        jsonformer = Jsonformer(self.model, self.tokenizer, json_schema, prompt)
        generated_response = jsonformer()
        return generated_response

    def predict(self, context, model_input):
        """Method to generate predictions for the given input."""
        outputs = []
        for i in range(len(model_input)):
            prompt = model_input["prompt"][i]
            generated_data = self._generate_response(prompt)
            outputs.append(generated_data)
        return {"candidates": outputs}

# Define input and output schema for the model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

input_schema = Schema([ColSpec(DataType.string, "prompt")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame({
    "prompt":["""Extract in JSON the category of the user text (other, frankenstein, banter, political, violence, hate), 
the reason for the category and an action (gently avoid question if category is political, violence, hate, other. else engage) 
based on the user text from this message below.
user text:
Who is frankenstein?"""],
})


# COMMAND ----------

# Log the model using MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=Mistral7BJSONParser(),
        artifacts={'repository' : snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1","transformers==4.34.1", "jsonformer==0.12.0","accelerate==0.21.0","torchvision==0.15.2","auto-gptq","optimum"],
        signature=signature
    )

# COMMAND ----------

import mlflow
#register model to UC
mlflow.set_registry_uri("databricks-uc")
registered_name = "main.sgfs.mistral_7b_jsonparse" 
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
# Annotate the model as "CHAMPION".
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)
# Load it back from UC
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": ["""Extract in JSON the category of the user text (other, frankenstein, banter, political, violence, hate), 
the reason for the category and an action (gently avoid question if category is political, violence, hate, other. else engage) 
based on the user text from this message below.
user text:
Who is frankenstein?"""],
    }
)

# COMMAND ----------

# MAGIC %md Deploy as API Endpoint

# COMMAND ----------

endpoint_name = 'mistral-7b-parser-sg'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_SMALL"

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": served_name,
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": workload_type,
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')
# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# COMMAND ----------

print(deploy_response.json())
