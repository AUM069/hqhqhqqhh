!pip install transformers datasets
!pip install -q -U google-generativeai langchain-google-genai langchain_groq langchain faiss-cpu transformers datasets

import os
import textwrap
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import display, Markdown
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Set the API keys directly
groq_api_key = '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'
google_api_key = 'AIzaSyAwsTppA0gKyg6eaQs6fbB2AxjULN_drDI'
huggingface_api_key = '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'

# Initialize Groq Langchain chat object
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

# Configure Google Generative AI
genai.configure(api_key=google_api_key)
model_name = next((m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods), None)
if not model_name:
    raise ValueError("No suitable model found.")

llm_google = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key)

# Load the Hugging Face model and tokenizer
model_name = "gpt2"  # Replace with the actual model name you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("sujantkumarkv/indian_legal_corpus")


# Define the prompts
groq_system_prompt = """
You are an expert in Indian law, capable of providing detailed legal advice that rivals the expertise of the best lawyers.
Your explanations should be thorough, covering each term and concept in-depth, supported by relevant examples from Indian laws, rules, and cases.
"""

google_system_prompt = """
You are an Indian law expert named Nivan. Your legal advice should be detailed and comprehensive, explaining each term and concept clearly.
You should provide examples from Indian laws, rules, and cases to illustrate your points. Remember to give practical advice and explanations that would
make it easy for anyone to understand the complexities of the law.
"""

# Helper function to format the text as markdown
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Define the prompt template for Groq
groq_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=groq_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

# Initialize conversational memory
memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)

# Create a conversation chain for Groq
groq_conversation = LLMChain(
    llm=groq_chat,
    prompt=groq_prompt_template,
    verbose=False,  # Disable verbose logging
    memory=memory,
)

# Function to ask a question to Groq API
def ask_groq(question, chat_history):
    # Clear the memory at the start of each new question
    memory.clear()
    for message in chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )
    try:
        response = groq_conversation.predict(human_input=question)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to ask a question to Google Generative AI
def ask_google(question, conversation_history):
    prompt = f"""
    {google_system_prompt}
    Conversation History:
    {conversation_history}
    Question: {question}

    *Additional Information:*

    * For questions related to specific acts or laws, please mention them (e.g., Indian Contract Act, 1872).
    * If you need help with legal procedures, I can provide general guidance on the steps involved but can provide specific legal advice.
    """
    result = llm_google.invoke(prompt)
    return result.content

# Function to ask a question to Hugging Face model
def ask_huggingface(question, conversation_history):
    input_text = f"{conversation_history}\nUser: {question}\nNivan:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=2500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Nivan:")[-1].strip()

# Integrated chatbot function
def chat_with_bot():
    print("Chat with Nivan-law in your pocket. Type 'exit' to end the chat.")
    conversation_history = ""
    chat_history = []
    api_toggle = 0  # 0 for Groq, 1 for Google, 2 for Hugging Face

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Nivan: Goodbye!,see yaaa")
            break

        if api_toggle == 0:
            response = ask_groq(user_input, chat_history)
        elif api_toggle == 1:
            response = ask_google(user_input, conversation_history)
        else:
            response = ask_huggingface(user_input, conversation_history)

        api_toggle = (api_toggle + 1) % 3  # Switch to the next API

        chat_history.append({'human': user_input, 'AI': response})
        conversation_history += f"User: {user_input}\nNivan: {response}\n"
        display(to_markdown(f"Nivan: {response}"))

# Run the chatbot
chat_with_bot()
