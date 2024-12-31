# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Lab 2: Prepare your chat model
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-2.png?raw=true" style="display: block; margin: 25px auto"  width="900px;">
# MAGIC
# MAGIC Now that a knowledge base is set up and vector search is ready, let's create a RAG application using LangChain , a framework for developing applications powered by foundation models. You use LangChain to:
# MAGIC
# MAGIC * Build a complete chain supporting conversation history
# MAGIC * Add a filter to only answer questions about a specific topic
# MAGIC * Compute embeddings within our chain to query the Vector Search Index
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=4148934117703857&notebook=%2F02-advanced%2F02-Advanced-Chatbot-Chain&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-advanced%2F02-Advanced-Chatbot-Chain&version=1">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Install dependencies and run helper code
# MAGIC
# MAGIC To get started, let's run the following code blocks within your new notebook. This installs dependencies, set environment variables, and run the helper code loaded into our Databricks Workspace. 
# MAGIC
# MAGIC *Please be patient: It may take up to 3-5 minutes for packages to install.*

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2 sqlalchemy==2.0.30
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper_code/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Working with Large Language Models (LLMs)
# MAGIC
# MAGIC Let's explore how you can access leading Large Language Models (LLMs) with Databricks and Amazon Bedrock. 
# MAGIC
# MAGIC First, let's import the necessary libaries you will require. Don't worry, you will learn about these libraries later on. 

# COMMAND ----------

import warnings

import io
import re


from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c




# COMMAND ----------

# Import utility libraries
import cloudpickle
import json
from operator import itemgetter

# Import LangChain libaries for embeddings and vector search
from langchain.embeddings import DatabricksEmbeddings
from langchain.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient

# Import LangChain libaries for your chat model
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatDatabricks

# Import MLFlow for model deployment
import mlflow.deployments
from mlflow.models import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Claude 3 on Amazon Bedrock
# MAGIC
# MAGIC Let's try an [external model](https://docs.databricks.com/en/generative-ai/external-models/index.html) `claude-3-sonnet-20240229-v1:0` from Amazon Bedrock. To deploy, you use AWS Credentials securely configured in [secret scopes](https://docs.databricks.com/en/security/secrets/secret-scopes.html).

# COMMAND ----------

bedrock_chat_model_endpoint_name

# COMMAND ----------

client = mlflow.deployments.get_deploy_client("databricks")

if bedrock_chat_model_endpoint_name not in [endpoints['name'] for endpoints in client.list_endpoints()]:
    client.create_endpoint(
        name=bedrock_chat_model_endpoint_name,
        config={
            "served_entities": [
                {
                    "external_model": {
                        "name": "claude-3-5-sonnet-20240620-v1:0",
                        "provider": "amazon-bedrock",
                        "task": "llm/v1/chat",
                        "amazon_bedrock_config": {
                            "aws_region": "us-west-2",
                            "aws_access_key_id": "{{secrets/" + scope_name + "/aws_access_key_id}}",
                            "aws_secret_access_key": "{{secrets/" + scope_name + "/aws_secret_access_key}}",
                            "bedrock_provider": "anthropic",
                        },
                    }
                }
            ]
        },
    )
else:
    print("model already created")


# COMMAND ----------

# MAGIC %md
# MAGIC Now that the model serving endpoint is available, let's try it. 
# MAGIC
# MAGIC *Note that you can use [system prompts with Anthropic Claude on Amazon Bedrock](https://community.aws/content/2dJmYpKlFNh6NOeC71GIZWZkfST/system-prompts-with-anthropic-claude-on-amazon-aws-bedrock).*
# MAGIC  

# COMMAND ----------

messages = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(content="Tell me What the HDFC Cancer Care plan does not cover?"),
]

claude_model = ChatDatabricks(endpoint=bedrock_chat_model_endpoint_name, max_tokens=500)

claude_model.invoke(messages, stop=["Human:"])

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we'll use [PromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/quick_start#prompttemplate) and send a query.

# COMMAND ----------

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an expert chatbot designed to provide information strictly from two specific documents: HDFC-Life-Click-2-Wealth-Policy-Bond and HDFC-Life-Cancer-Care-Plan. When answering questions, ensure your response is based solely on these documents. If you cannot find relevant information in these documents, indicate that you do not know. Give a short answer to this question: {question}. "
)

chain_claude = (
    prompt
    | claude_model
    | StrOutputParser()
)

print(chain_claude.invoke({"question": "How does Databricks work on AWS?"}))

# COMMAND ----------

print(chain_claude.invoke({"question": "What are the benifits avaialble for hdfc cancer care Silver plan?"}))

# COMMAND ----------

print(chain_claude.invoke({"question": "What are the benifits of hdfc life click-2-Wealth policy?" }))

# COMMAND ----------

print(chain_claude.invoke({"question": "What are the Golden Years Benefit Option in hdfc life  click-2-wealth policy?"}))



# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's select our model of choice:

# COMMAND ----------

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an assistant. Give a short answer to this question: {question}"
)

dbllm_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=500)


chain = (
    prompt
    | dbllm_model
    | StrOutputParser()
)

print(chain.invoke({"question": "How does Databricks work on AWS?"}))

# COMMAND ----------


aws_chat_model = claude_model
#aws_chat_model = dbllm_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Working with Langchain
# MAGIC
# MAGIC ### Adding conversation history to the prompt
# MAGIC
# MAGIC Now, we enhance our prompt by including conversation history along with the question. When invoking, we pass the conversation history as a list, specifying whether each message was sent by a user or the assistant. For example:

# COMMAND ----------

prompt_with_history_str = """
You are an expert chatbot designed to provide information strictly from two specific documents: HDFC-Life-Click-2-Wealth-Policy-Bond and HDFC-Life-Cancer-Care-Plan. When answering questions, ensure your response is based solely on these documents. If you cannot find relevant information in these documents, indicate that you do not know. 

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md
# MAGIC When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. 
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"},
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

# The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

# The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | aws_chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "do you have access to hdfc life click-2-Wealth policy document?"},
        {"role": "assistant", "content": "yes I have access to HDFC Life Click-2-Wealth-Policy document "},
        {"role": "user", "content": "what is Golden Years Benefit Option?"}
    ]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filtering for questions on a specific topic
# MAGIC Our chatbot should be professional and only answer questions related to a specific topic, for example, Databricks-related questions. One approach is to create a `is_question_about_databricks_prompt` prompt template for a classification step.
# MAGIC
# MAGIC *Note: this is a fairly naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

is_question_about_relevant_str = """
You are classifying documents to determine if a question is related to HDFC-Life-Click2-Wealth-Policy-Bond or HDFC-Life-Cancer-Care-Policy-Plan or  related to completely different subject. Also, respond with 'no' if the question is not about HDFC Policy.

Here are some examples:

Question: Given the previous discussion about HDFC click-2-wealth Policy, classify this question: Can you explain more about Invest Plus Option for For Single Pay Policy?
Expected Response: Yes

Question: After discussing LLMs, classify this question: Can you write a poem for me?
Expected Response: No

Only answer with "yes" or "no".

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=is_question_about_relevant_str
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we define the LLM that we use, and create our simple chain. This chain will execute multiple steps: extract question and chat history, classify the question to determine relevance, and finally parse the output a user-friendly format.
# MAGIC
# MAGIC

# COMMAND ----------

is_about_llm_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | aws_chat_model
    | StrOutputParser()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test our chain with a question which matches the desired topic.

# COMMAND ----------

# Returns "Yes" as this is about LLMs:
print(is_about_llm_chain.invoke({
    "messages": [
        {"role": "user", "content": "do you have access to hdfc life click-2-Wealth policy document?"},
        {"role": "assistant", "content": "yes I have access to HDFC Life Click-2-Wealth-Policy document "},
        {"role": "user", "content": "what is Golden Years Benefit Option?"}
    ]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we'll test our chain with a question which is **not** about the desired topic.

# COMMAND ----------

# Return "no" as this isn't about the desired topic
print(is_about_llm_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use LangChain to retrieve documents from the vector store
# MAGIC
# MAGIC Let's add our LangChain [retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/).
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="display: block; margin: 25px auto;" width="900px">
# MAGIC
# MAGIC [Databricks LangChain wrapper](https://python.langchain.com/docs/integrations/providers/databricks) makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test if our secrets and permissions are properly setup. In a new cell, run the following code to test the same.

# COMMAND ----------

index_name = f"{catalog}.{db}.llm_pdf_documentation_self_managed_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(workspace_url, secret_scope=scope_name, secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-gte-large-en")

# COMMAND ----------

index_name

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test the index with predict function from notebook 1, remove later 

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# import databricks.sdk.service.catalog as c
# from databricks.vector_search.client import VectorSearchClient
# vsc = VectorSearchClient(disable_notice=True)

# deploy_client = get_deploy_client("databricks")

# # The table we'd like to index
# source_table_fullname = f"{catalog}.{db}.llm_pdf_documentation"
# # Where we want to store our index
# vs_index_fullname = f"{catalog}.{db}.llm_pdf_documentation_self_managed_vs_index"

# question = "Suicide Exclusions"

# response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
# embeddings = [e['embedding'] for e in response.data]

# results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(query_vector=embeddings[0], columns=["url", "content"], num_results=1)

# docs = results.get('result', {}).get('data_array', [])
# pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's add a retriever to the chain. This retriever is responsible for finding the most relevant documents to our prompt. In a new cell, run the following code.

# COMMAND ----------

# from langchain.chains import RetrievalQA
embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope_name, "rag_sp_token")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages")
    | RunnableLambda(extract_question)
    | retriever
)

# COMMAND ----------

# print("Query 1: What is Invest Plus Option For Single Pay Policy ?")
# result = retrieve_document_chain.invoke(
#     {"messages": [{"role": "user", "content": "What is Invest Plus Option For Single Pay Policy "}]})

# for document in result:
#     print(document.metadata)

#print("Query 2: Explain Golden Years Benefit Option?")
result = retrieve_document_chain.invoke(
    {"messages": [{"role": "user", "content": "Suicide Exclusions"}]})
    
#"what is the percentage of Applicable Sum Insured in Silver policy coverage for Major Cancer?"

for document in result:
    print(document.metadata)

# COMMAND ----------


for document in result:
    print(document)
    print("****************")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improving our document retrieval process
# MAGIC We need to retrieve documents related the the last question but also the conversation history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain.

# COMMAND ----------

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  # Augment query only when there is a chat history
        (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt |
         aws_chat_model | StrOutputParser()),
        (lambda x: not x["chat_history"],
            RunnableLambda(lambda x: x["question"])),
        RunnableLambda(lambda x: x["question"])
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test our new chain.

# COMMAND ----------

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the policy for suicidal benifits exclusion? "}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    # "messages": [
    #     {"role": "user", "content": "What is Invest Plus Option For Single Pay Policy "},
    #     {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
    #     {"role": "user", "content": "Does it support streaming?"}
    # ]
    "messages": [
        {"role": "user", "content": "do you know about hdfc life click2-wealth policy?"},
        {"role": "assistant", "content": "yesa, i know about HDFC LifeClick-2-Wealth policy."},
        {"role": "user", "content": "What is the policy for suicidal benifits exclusion? "}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Merging the retriever into our full chain
# MAGIC
# MAGIC Let's now merge the retriever and the full LangChain chain. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="display: block; margin: 25px auto;" width="900px">
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer. Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.

# COMMAND ----------

question_with_history_and_context_str = """
You are an expert chatbot designed to provide information from HDFC-Life-Click-2-Wealth-Policy document and HDFC-Life-Cancer-Care-Plan document. When answering questions, ensure your response is based solely on these documents. If you cannot find relevant information in any of these documents, indicate that you do not know.
If a question falls outside your knowledge, you'll honestly say you don't know. In the chat, you are identified as "system" and the user as "user." Remember to read the conversation history  before responding.

Discussion: {chat_history}

Here's some context which to help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this  context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
    RunnablePassthrough() |
    {
        "relevant_docs": generate_query_to_retrieve_context_prompt | aws_chat_model | StrOutputParser() | retriever,
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question")
    }
    |
    {
        "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
        "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question")
    }
    |
    {
        "prompt": question_with_history_and_context_prompt,
        "sources": itemgetter("sources")
    }
    |
    {
        "result": itemgetter("prompt") | aws_chat_model | StrOutputParser(),
        "sources": itemgetter("sources")
    }
)

irrelevant_question_chain = (
    RunnableLambda(lambda x: {
                   "result": 'I cannot answer questions that are not about HDFC Policy.', "sources": []})
)

branch_node = RunnableBranch(
    (lambda x: "yes" in x["question_is_relevant"].lower(),
     relevant_question_chain),
    (lambda x: "no" in x["question_is_relevant"].lower(),
        irrelevant_question_chain),
    irrelevant_question_chain
)

full_chain = (
    {
        "question_is_relevant": is_about_llm_chain,
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | branch_node
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try our full chain. Let's start by asking an out-of-scope question. In a new cell, run the following code.

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
# non_relevant_dialog = {
#     "messages": [
#         {"role": "user", "content": "What is Apache Spark?"},
#         {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
#         {"role": "user", "content": "Why is the sky blue?"}
#     ]
# }
# print(f'Testing with a non relevant question...')
# response = full_chain.invoke(non_relevant_dialog)
# display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's ask a relevant question. In a new cell, run the following code.

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "do you know about hdfc life Click-2-Wealth-Policy?"},
        {"role": "assistant", "content": "yes i know about hdfc life Click-2-Wealth-Policy?"},
        {"role": "user", "content": "What is the policy for suicidal benifits exclusion?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Asking a relevant question
dialog = {
    "messages": [
        {"role": "user", "content": "Does HDFC has any cancer care policy?"},
        {"role": "assistant", "content": "Yes HDFC has HDFC-Life-Cancer-Care-Plan."},
        {"role": "user", "content": "what is the percentage of Applicable Sum Insured in Silver policy coverage for Major Cancer?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "Does HDFC has any cancer care policy?"},
        {"role": "assistant", "content": "Yes HDFC has Cancer Care policy."},
        {"role": "user", "content": "what is the percentage of Applicable Sum Insured in Silver policy coverage for Major Cancer?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Register the chatbot model to Unity Catalog
# MAGIC
# MAGIC Databricks provides a fully-managed [ML lifecycle management using MLFlow](https://docs.databricks.com/en/mlflow/index.html), an open source platform for managing the end-to-end machine learning lifecycle including tracking experiments, model deployment and registry, and model serving.
# MAGIC
# MAGIC In a new cell, run the following code.

# COMMAND ----------

init_experiment_for_batch("sar_test_bedrock_west", "experiment_" + aws_account_id)

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, run the following code:

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.chatbot_bedrock_west"

with mlflow.start_run(run_name="chatbot_rag") as run:
    # Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle==" + cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True
    )

# COMMAND ----------

# MAGIC %md Let's try loading our model. In a new cell, run the following code:

# COMMAND ----------

model_info.model_uri

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's register the latest version of our model with the `prod` alias. In a new cell, run the following code:

# COMMAND ----------

client = MlflowClient()
model_version_to_evaluate = get_latest_model_version(model_name)
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this Lab, you learned how to improve a chatbot, adding capabilities such as handling conversation history and retrieval using vector search. As you add even more capabilities and tune your prompts, it may become more difficult to evaluate your model performance in a repeatable way. Your new prompt may work well for what you tried to fix, but could also have impact on other questions.
# MAGIC
# MAGIC Next, open [03-Deploy-Model]($./03-Deploy-Model) to deploy your model endpoint.

# COMMAND ----------


