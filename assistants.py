import chainlit as cl
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.utils.etl import ChromaVectorDatabase
from src.chains import get_onepiece_universe_chatbot_chain

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
                    message="What is one piece?",
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
                    message="Hello! Can you please tell me about yourself?",
                    icon="/public/bell.svg"
                ),
            ]
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    llm = ChatOllama(
        model="phi3.5:latest",
        temperature=0,
    )
    cl.user_session.set("llm", llm)
    vectordb = ChromaVectorDatabase(collection_name='onepiece', embedders=OllamaEmbeddings(
                model="nomic-embed-text:latest"
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
        context = vectordb.generate_context(query=message.content)
        chain = get_onepiece_universe_chatbot_chain(llm=llm)
        res = await chain.ainvoke({"context": f"{context}", "query": f"{message.content}"},
                                  config={"configurable": {"session_id": session_id}},
                                  callbacks=[cb]
                                  )
        await cl.Message(content=res).send()

    elif selected_profile == 'One piece character finder':
        # res = await chain.ainvoke({"input": f"{message.content}"},
        #                             config={"configurable": {"session_id": session_id}},
        #                             callbacks=[cb]
        #                             )
        await cl.Message(content='hello one piece character finder').send()