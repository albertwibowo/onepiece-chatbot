from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from src.prompts.prompts import ONEPIECE_EXPERT_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Any 

def get_onepiece_universe_chatbot_chain(llm:ChatOllama) -> Any:
    onepiece_universe_chatbot_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', ONEPIECE_EXPERT_PROMPT)
        ]
    )

    onepiece_universe_chatbot_chain = (
        {'context': RunnablePassthrough(), 'query': RunnablePassthrough()}
        | onepiece_universe_chatbot_prompt
        | llm 
        | StrOutputParser()
    )

    return onepiece_universe_chatbot_chain