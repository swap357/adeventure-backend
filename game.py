from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory

from utils import get_embedding
from flask import jsonify
from config import *
from flask import current_app
import tiktoken
from config import *
import openai
TOP_K = 10

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
# memory = ConversationBufferMemory(return_messages=True)
memory = ConversationBufferWindowMemory(k=2, return_messages=True)
# memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=1000)
long_system_prompt = "The following is a text adventure game with context. \
                    You are an highly intelligent Game Co-ordinator for Adventure Land. \
                    Your qualities are - intelligence, creativity, humor, zest. \
                    You walk the player through tailored step-by-step guidance based on their input.\
                    The gameplay should not be asking player with choices, rather more human conversational way of getting input using self as character and wingman. \
                    You can use emojis to express emotions in narration and dialogues. \
                    Strictly use only 1 JSON object with key for response with this exact key- messages(list), where each message contain these 4 exact keys  - type, content, characterName, characterRole \
                    type has (bot,character), content has (narration-content,dialogue-content), characterName has (name), characterRole has (role)"

system_prompt = "You are the Game Coordinator for Adventure Land, possessing intelligence, creativity, humor, and zest. \
                Guide players through a text adventure using immersive, conversational interactions as a character and wingman—no bulleted choices. \
                Emojis may enhance narration and dialogue. Responses must be a JSON object with the key 'messages' (a list), where each message has these keys: type (bot or character), content (narration or dialogue), characterName (name), and characterRole (role). \
                Let the grand adventure unfold!"

prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
# Dictionary to store ConversationChain instances for each session ID
session_chains = {}
def get_response(question):
    logging.info(f"Getting answer for question: {question}")
    
    try:  
        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)  
        # print("conversation: ", conversation.memory.buffer)
        answer = conversation.predict(input=question)
        logging.info(f"answer: {answer}")
        # initialize tokenizer
        # tokenizer = tiktoken.encoding_for_model('text-davinci-003')
        # show number of tokens for the memory used by each memory type
        # logging.info(f"memory tokens: {tokenizer.encode(memory.get_memory())}")
    except Exception as e:
        logging.error(f"error: {e}")
        return "someting went wrong"
        
    return answer

context = ""
def get_answer(question):
    openai.api_key = OPENAI_API_KEY
    global context
    global long_system_prompt
    if context == "":
        context = f" System: {long_system_prompt}"
    prompt = f"{context}\n Player: {question}"
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    logging.info(f"long_system_prompt: {long_system_prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=256,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0
    )
    
    bot_response = response.choices[0].message.content
    context = f"{context}\n Player: {question}\n  Bot: {bot_response}" 

    return response.choices[0].message.content