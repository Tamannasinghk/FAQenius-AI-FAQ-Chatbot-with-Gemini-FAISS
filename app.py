from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("Final_faqs.csv")
df["question_plus_answer"] = "Question : " + df["Question"] + " , Answer : " + df["Answer"]

def generate_embeddings(text):
  embed = model.encode(text)
  embed = embed / np.linalg.norm(embed)
  return embed

embeddings = df['question_plus_answer'].apply(generate_embeddings).tolist()
embeddings_array = np.array(embeddings).astype(np.float32)
dimensions = embeddings_array.shape[1]

index = faiss.IndexFlatIP(dimensions)
index.add(embeddings_array)

faiss.write_index(index, "question_answer_embeddings_cosine.index")
loaded_index = faiss.read_index("question_answer_embeddings_cosine.index")

def search_similar_questions(query, k=3):

    # Generate and normalize query embedding
    query_embedding = generate_embeddings(query)
    query_embedding = np.array([query_embedding]).astype('float32')

    # Search in FAISS index (results will be in cosine similarity order)
    similarities, indices = loaded_index.search(query_embedding, k)

    # Return results (similarities will be between 0 and 1)
    return df['question_plus_answer'][indices[0][0]]


api_key = "Please put your own Gemini Api key here."
gemini_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash' , google_api_key=api_key)

parser = StrOutputParser()

from langchain.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and concise assistant trained on a set of Frequently Asked Questions (FAQs).
Your goal is to answer user questions based on the most relevant FAQ or let the user know when no direct match is found.

Rules:
1. If the user greets (e.g., "hi", "hello", "hey", "buddy", "bro", "how are you", etc.), reply with a warm, friendly message and DO NOT reference FAQ data.
2. If a relevant FAQ is found, use it to answer the user’s question directly but not exactly same as retrieved query , add something by yourself.
3. If no FAQ matches the query then show the closest retrieved content.

{retrieved_context}
"""),

    ("human", """User Query:
{query}

Your Response:""")
])
chain = chat_template | gemini_model | parser


def chatbot_fn(message, history):
    if message.lower().strip() in ["hi", "hello", "hey", "how are you", "bro", "buddy"]:
        return "Hey there! 👋 I'm your friendly assistant. Ask me anything from the FAQs!"
    
    retrieved = search_similar_questions(message)
    response = chain.invoke({'retrieved_context': retrieved, 'query': message})
    
    return f"📄 **Retrieved FAQ Snippet:**\n{retrieved}\n\n🤖 **Answer:**\n{response}"

# Launch real chatbot UI
chatbot = gr.ChatInterface(
    fn=chatbot_fn,
    title="🤖 FAQ Chatbot",
    theme="soft",
    examples=["tell me something about meesho .", "What is the price of i-phone 15 pro 128 gb?", "Can I open the FiftyOne App in a browser?.", "Hey !"],
    chatbot=gr.Chatbot(show_label=False, avatar_images=("🧑", "🤖")),
    description="Ask your questions and get instant answers based on FAQs. Try saying 'hi' or ask about a process!",
)

if __name__ == "__main__":
    chatbot.launch(debug=True)
