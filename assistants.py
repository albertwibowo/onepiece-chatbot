import chainlit as cl
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.utils.etl import ChromaVectorDatabase
from src.utils.memory import USER_INFORMATION_QUESTION_BANKS
from src.chains import (
    get_onepiece_universe_chatbot_chain, 
    get_onepiece_character_chain,
    get_user_info_summariser_chain
)

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma:latest"

# Different chatbot profiles
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="One piece universe chatbot",
            icon="/public/comments.svg",
            markdown_description="Talk anything about the one piece universe!",
            starters=[
                cl.Starter(
                    label="Conversation starter",
                    message="What is one piece manga all about?",
                    icon="/public/bell.svg"
                ),
            ]
        ),
        cl.ChatProfile(
            name="One piece character finder",
            icon="/public/book.svg",
            markdown_description="Find a one piece character that matches your profile",
            starters=[
                cl.Starter(
                    label="Conversation starter",
                    message="Hello!",
                    icon="/public/bell.svg"
                ),
            ]
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        top_k=5,
        top_p=0.05,
        num_predict=128,
    )
    cl.user_session.set("llm", llm)
    vectordb = ChromaVectorDatabase(collection_name='onepiece', embedders=OllamaEmbeddings(
                model=EMBEDDING_MODEL
            ))
    cl.user_session.set("vectordb", vectordb)
    
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the selected profile
    selected_profile = cl.user_session.get("chat_profile") # important
    session_id = cl.user_session.get("id")
    vectordb = cl.user_session.get("vectordb")
    llm = cl.user_session.get("llm")
    cb = cl.AsyncLangchainCallbackHandler()

    if selected_profile == 'One piece universe chatbot':
        context = vectordb.generate_context(query=message.content, n_results=3)
        chain = get_onepiece_universe_chatbot_chain(llm=llm)
        res = await chain.ainvoke({"context": f"{context}", "query": f"{message.content}"},
                                  config={"configurable": {"session_id": session_id}},
                                  callbacks=[cb]
                                  )
        await cl.Message(content=res).send()

    elif selected_profile == 'One piece character finder':
        user_info = {}
        await cl.Message(content="Hello, I need a few information from you to get started.").send()
        for key, qns in USER_INFORMATION_QUESTION_BANKS.items():
            res = await cl.AskUserMessage(content=qns, timeout=300).send()
            if res:
                user_info[key] = res['output']
        if len(user_info) == len(USER_INFORMATION_QUESTION_BANKS):
            await cl.Message(content=f"The following information has been recorded: {user_info}").send()
            await cl.Message(content="Fetching the most similar character ......").send()
            summariser_chain = get_user_info_summariser_chain(llm=llm)
            user_info_summary = await summariser_chain.ainvoke({"information": f"{user_info}"},
                                  config={"configurable": {"session_id": session_id}},
                                  callbacks=[cb]
                                  )
            character_matcher_chain = get_onepiece_character_chain(llm=llm)
            context = vectordb.generate_context(query=user_info_summary, n_results=3)
            res = await character_matcher_chain.ainvoke({"context": f"{context}", "information": f"{user_info_summary}"},
                                  config={"configurable": {"session_id": session_id}},
                                  callbacks=[cb]
                                  )
            # answer = {'character': res.character,
            #           'explanation': res.explanation}
            await cl.Message(content=f"{res}").send()

            
