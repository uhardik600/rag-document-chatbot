import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful document assistant. 
Answer the user's question based strictly on the provided document context.
If the answer is not found in the context, say "I couldn't find that information in the uploaded document."
Be concise, accurate, and cite relevant parts of the context where appropriate."""


def generate_response(question: str, context_chunks: list[str], history: list[dict]) -> str:
    """
    Generate a context-aware response using retrieved chunks and chat history.
    """
    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Document context:\n\n{context}")
    ]

    # Append previous conversation history
    for msg in history[-6:]:  # Keep last 3 turns (6 messages) for context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content
