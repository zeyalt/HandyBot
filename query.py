import argparse
import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAI

from get_embeddings_function import get_embeddings_function

load_dotenv()

client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are HandyBot, an expert LLM that has deep knowledge about all electrical appliances in 
the household. You are able to provide clear and concise instructions, in no more than 200 words, 
on any questions relating to the user's electrical appliances. 
---

Now, here's the question: {question}

Please answer this question based only on the following context:

{context}

Ensure that your response is clear, concise and uses simple language. Use bullet points if necessary
to make your response coherent. 
"""

def query_rag(query_text: str):
    embedding_function = get_embeddings_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # model = OllamaLLM(model="phi3.5")
    model = OpenAI()
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # return formatted_response
    return response_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    llm_response = query_rag(query_text)
    print(llm_response, "\n")
