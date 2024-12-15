from langchain_core.chat_history import (
    BaseChatMessageHistory, 
    InMemoryChatMessageHistory
)


# in memory user session cache 
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


USER_INFORMATION_QUESTION_BANKS = {
    "name": "What is your name?",
    "dob": "Can you please tell me your date of birth?",
    "height": "What about your height in cm?",
    "favourite_food": "Any favourite food?",
    "gender": "How about gender? Which gender do you identify with?",
    "occupation": "A few more things I promise. What occupation do you have?",
    "blood_type": "May I know your blood type as well?",
    "personality": "Finally, please tell me your personality. The more detailed the better!"
}