from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
import gzip
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import  RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langgraph.graph import END, StateGraph
from openai import OpenAI
import markdown
import pdfplumber 

import pdfplumber  # or fitz from PyMuPDF

from langchain.embeddings import OpenAIEmbeddings



# Load environment variables from the .env file
load_dotenv()

# Access your variables
langchain_key = os.getenv("LANGCHAIN_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_key)

# LLM for detecting greetings and farewells
llmg = ChatOpenAI(model="gpt-4o", temperature=0)
# Prompt template for detector
SYS_PROMPT_G = """You are an agent that detects if user query is exactly a pure greeting or pure farewell, in such a case, you just answer yes or no.
                You are able to identify if user query is a greeting even when that contains typos or mistakes.
                if the query contains a greeting or farewell with a question or different context, that's not considered a pure greeting.
                 - pure greeting examples: hi, hello, good morning, good afternon, good evening.
                 - pure farewell examples: good bye, bye bye, see ya, see you later, good night, so long.
                 -non-greeting examples: Hello, who was Nikola Tesla?, Hi, what time is it in USA?, good morning, I want to know about Italy.
             """
greetings_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT_G),
        ("human", """Here is the initial question:
                     {question}
                     Define ifthe user query is precisely a pure greting or farewell.
                  """,
        ),
    ]
)
# Create greetings chain
question_greetings = (greetings_prompt
                        |
                       llmg
                        |
                     StrOutputParser())


# LLM to response  in case of greetings or farewells
llmg_r = ChatOpenAI(model="gpt-4o", temperature=1)
# Prompt template for detector
SYS_PROMPT_R = """You are a polite and friendly assistant. 
If the user greets you (e.g., "hi", "hello", "good morning"), respond warmly with a greeting and offer your help as a research assistant. 
If the user says goodbye (e.g., "bye", "see you", "take care", "so long"), reply kindly with a farewell.
Keep your response natural and engaging, like a human conversation."""

greetings_response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT_R),
        ("human", """Here is the initial question:
                     {question}
                     respond gently and polite to user.
                  """,
        ),
    ]
)

# Create greetings chain
question_greetings_r = (greetings_response_prompt
                        |
                       llmg_r
                        |
                     StrOutputParser())


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]
    


# --- File Paths ---
pdf_paths = [
    'F:/Lucio/Latex/Copernicus_LaTeX_Package/Copernicus_LaTeX_Package/statistical_co2_paper.pdf',
    'F:/Lucio/Documents/Lucho/harmonic_resonance_mode.pdf'
    # Add more file paths here as needed
]

# --- Extract and Wrap ---
documents = []
for path in pdf_paths:
    full_text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    filename = os.path.basename(path)
    doc = Document(page_content=full_text, metadata={"source": filename})
    documents.append(doc)

# --- Split into Chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = []
for doc in documents:
    chunks = splitter.split_documents([doc])
    chunked_docs.extend(chunks)

# --- Embedding Model ---
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-large')

# --- Create FAISS Index ---
dim = len(openai_embed_model.embed_query("hello world"))
index = faiss.IndexFlatL2(dim)

vector_store = FAISS(
    embedding_function=openai_embed_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# --- Add Documents with UUIDs ---
uuids = [str(uuid4()) for _ in range(len(chunked_docs))]
_ = vector_store.add_documents(documents=chunked_docs, ids=uuids)


# #embedding model
# openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-large')

# wikipedia_filepath = 'F:/Lucio/Descargas/simplewiki-2020-11-01.jsonl.gz'
# docs = []
# with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
#     for line in fIn:
#         data = json.loads(line.strip())
#         #Add documents
#         docs.append({
#                         'metadata': {
#                                         'title': data.get('title'),
#                                         'article_id': data.get('id')
#                         },
#                         'data': ' '.join(data.get('paragraphs')[0:3]) 
#         # restrict data to first 3 paragraphs to run later modules faster
#         })
# # We subset our data to use a subset of wikipedia documents to run things faster
# docs = [doc for doc in docs for x in ['india']
#               if x in doc['data'].lower().split()]
# # Create docs
# docs = [Document(page_content=doc['data'],
#                  metadata=doc['metadata']) for doc in docs]
# # Chunk docs
# splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
# chunked_docs = splitter.split_documents(docs)

# #index to initialize vector database
# index = faiss.IndexFlatL2(len(openai_embed_model.embed_query("hello world")))

# #FAISS vector database
# vector_store = FAISS(
#     embedding_function=openai_embed_model,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

# #document IDs to alocate random tags to nuew documents added to the database
# uuids = [str(uuid4()) for _ in range(len(chunked_docs))]


# #initial documents as a reference
# _ = vector_store.add_documents(documents=chunked_docs, ids=uuids)

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM for grading 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}
                     User question:
                     {question}
                  """),
    ]
)
# Build grader chain
doc_grader = (grade_prompt
                  |
              structured_llm_grader)


# Create RAG prompt for response generation
prompt = """You are an polite and kind assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)
# Initialize connection with GPT-4o
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=1)
# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context')
                        |
                    RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
      |
    prompt_template
      |
    chatgpt
      |
    StrOutputParser()
)

# LLM for question rewriting
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Prompt template for rewriting
SYS_PROMPT = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
             """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
                     {question}
                     Formulate an improved question.
                  """,
        ),
    ]
)

# Create rephraser chain
question_rewriter = (re_write_prompt
                        |
                       llm
                        |
                     StrOutputParser())

#retrieval method to consider relevant documents
similarity_threshold_retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                       search_kwargs={"k": 3,                                                                       
                       "score_threshold": 0.3})

#Tavily search engine to get new docuemnets in case of retrieval strategies don't find a suitable context for the answe
tv_search = TavilySearchResults(max_results=10, search_depth='advanced',max_tokens=10000)

#retrieval function
def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    question = state["question"]
    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question}

#function to assign relevance score to new docuemnts
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                web_search_needed = "Yes"
                continue
    else:
        web_search_needed = "Yes"
    return {"documents": filtered_docs, "question": question, 
            "web_search_needed": web_search_needed}
    
#block to take user queries and improve them to generate better answers  
def rewrite_query(state):
    """
    Rewrite the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

#optional block to search for new documents to compelment the answer, and add possible docuements to the vector dataase
def web_search(state):
    """
    Web search based on the re-written question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]
    documents = state["documents"]
    
    # Web search
    docs = tv_search.invoke(question)
    web_results_text = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results_text)
    
    # Chunk the Document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300
    )
    # Pass a list containing the Document to be split
    chunked_docs = splitter.split_documents([web_results])
    
    # Generate new UUIDs for each chunk
    uuids = [str(uuid4()) for _ in range(len(chunked_docs))]
    
    # Add the chunks to the vector store
    _ = vector_store.add_documents(documents=chunked_docs, ids=uuids)
    
    # Instead of appending a list, extend the documents list with individual chunks
    documents.extend(chunked_docs)
    
    return {"documents": documents, "question": question}

#function to read the final context and generate the suitable answer
def generate_answer(state):
    """
    Generate answer from context document using LLM
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
  
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, 
            "generation": generation}
    
#evaluatior function to define when to answer or modify queries and search for new information
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    web_search_needed = state["web_search_needed"]
    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        return "generate_answer"
    
    
#langgraph structure to build agent chain
agentic_rag = StateGraph(GraphState)
# Define the nodes
agentic_rag.add_node("retrieve", retrieve)  # retrieve
agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
agentic_rag.add_node("web_search", web_search)  # web search
agentic_rag.add_node("generate_answer", generate_answer)  # generate answer
# Build graph
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("generate_answer", END)
# Compile
agentic_rag = agentic_rag.compile()


#function to invoke agent model
def agent(query: str) -> str:
    response = agentic_rag.invoke({"question": query})
    return response['generation']


#function to detect greetings and farewells
def greeting_detector(question: str) -> str:
    """
    detect if user query is a greeting or a farewell
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): updates the nature of user query, classifying it as a greeting or farewell (yes), or none of them (no)
    """
    # Retrieval
    generation = question_greetings.invoke(question)
    return generation

#function to answer in case of greetings or farewells
def greeting_responder(question: str) -> str:
    """
    answer as a polite and gentle assistant
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): ends the chain with the final answer.
    """
    # Retrieval
    generation = question_greetings_r.invoke(question)
    return generation


#moderator function in case of inapropriate words
def moderate_response( question: str) -> bool:
        response =client.moderations.create(
        model="omni-moderation-latest",
        input=question,
        )
        return response.results[0].flagged