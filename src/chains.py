from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from src.prompts.prompts import (
    ONEPIECE_EXPERT_PROMPT, 
    CHARACTER_MATCHER_PROMPT,
    USER_INFO_SUMMARISER_PROMPT
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from typing import Any 

class OnePieceCharacterMatch(BaseModel):
    character: str
    explanation: list[str]

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

def get_onepiece_character_chain(llm:ChatOllama) -> Any:
    onepiece_character_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', CHARACTER_MATCHER_PROMPT)
        ]
    )

    onepiece_character_chain = (
        {'context': RunnablePassthrough(), 'information': RunnablePassthrough()}
        | onepiece_character_prompt
        | llm
        | StrOutputParser()
    )

    return onepiece_character_chain

def get_user_info_summariser_chain(llm:ChatOllama) -> Any:
    user_info_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', USER_INFO_SUMMARISER_PROMPT)
        ]
    )

    user_info_prompt_chain = (
        {'information': RunnablePassthrough()}
        | user_info_prompt
        | llm 
        | StrOutputParser()
    )

    return user_info_prompt_chain