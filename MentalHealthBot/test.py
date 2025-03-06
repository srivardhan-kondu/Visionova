# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

response = llm.invoke("Hello, how are you?")
print(response)
