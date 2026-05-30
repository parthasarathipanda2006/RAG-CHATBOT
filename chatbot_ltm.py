from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun,tool
from langgraph.prebuilt import ToolNode,tools_condition
import sqlite3
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever
from groq import APIStatusError
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import uuid

#=========================================MODEL=================================================================================
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
llm2 = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#=========================================RAG=================================================================================

VECTORSTORE={}
STORE_DIR="vectorstore"

def get_dir_path(thread_id:str)->str:
    return os.path.join(STORE_DIR,thread_id)

def load_vectorstore(thread_id: str):
    """Load from disk if exists, else return None."""
    path = get_dir_path(thread_id)
    if os.path.exists(path):
        VECTORSTORE[thread_id] = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

def fetch_retriever(vectorstore):
    retrieve = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
    return retrieve


def ingest(file_bytes,file_name:Optional[str], thread_id:str):

    os.makedirs(STORE_DIR, exist_ok=True)
    temp_path = None
    try:
        os.makedirs("temp", exist_ok=True)
        # FILE PATH
        temp_path = f"temp/{thread_id}_{file_name}"
        # SAVE PDF
        with open(temp_path, "wb") as f:

            f.write(file_bytes)
        # LOAD PDF
        loader = PyPDFLoader(temp_path)
        document=loader.load()

        text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
        chunks=text_splitter.split_documents(document)
        path=get_dir_path(thread_id)
        if thread_id  in VECTORSTORE: 
            VECTORSTORE[thread_id].add_documents(chunks)
        elif os.path.exists(path):
            VECTORSTORE[thread_id] = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            VECTORSTORE[thread_id].add_documents(chunks)
        
        else:
            VECTORSTORE[thread_id] = FAISS.from_documents(chunks,embeddings)

        VECTORSTORE[thread_id].save_local(path)
    except Exception as e:
        return e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
#=========================================TOOLS=================================================================================

@tool
def rag_tool(query,config:RunnableConfig)->dict:
    """
    retrieve relevent information from the pdf document.
    use this tool when the user asks factual or conceptual Question
    that might be answered from stored document
    
    """
    thread_id=None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id not in VECTORSTORE:
        load_vectorstore(thread_id)

    if thread_id not in VECTORSTORE:

        return {
        "error": "No document uploaded for this chat."
        }
    retrieve=fetch_retriever(VECTORSTORE[thread_id])
    relevent_docs=retrieve.invoke(query)
    context=[doc.page_content for doc in relevent_docs]
    metadata=[doc.metadata for doc in relevent_docs]

    return {
        "query":query,
        "context":context,
        "metadata":metadata
    }

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def stock_price(symbol:str)->dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    ALPHA_API_KEY=os.getenv("ALPHA_URL")
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_API_KEY}"
    result=requests.get(url)
    return result.json()
    

tools=[search_tool,stock_price,rag_tool]
llm_with_tool=llm.bind_tools(tools)

#=========================================MEMORY=================================================================================


conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
DB_URI = os.getenv("DATABASE_URL")

_store_cm = PostgresStore.from_conn_string(DB_URI)
store = _store_cm.__enter__()
store.setup()


class memoryitem(BaseModel):
    is_new:bool=Field(description="is the memory new")
    text:str
class MemoryDecision(BaseModel):
    should_write: bool
    memories: list[memoryitem] = Field(default_factory=list)

#===============================================NODE=============================================================================
extractor=llm.with_structured_output(MemoryDecision)

def remember(state:MessagesState,config:RunnableConfig,store:BaseStore):

    user_id=config["configurable"]["user_id"]
    ns=("user",user_id,"details")
    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"
   
    recent=state["messages"][-1].content
    MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

                        CURRENT USER DETAILS (existing memories):
                        {user_details_content}

                        TASK:
                        - Review the user's latest message.
                        - Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
                        - For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
                        - If it is basically the same meaning as something already present, set is_new=false.
                        - Keep each memory as a short atomic sentence.
                        - No speculation; only facts stated by the user.
                        - If there is nothing memory-worthy, return should_write=false and an empty list.
                        """
    response=extractor.invoke(
        [
            SystemMessage(
                content=(
                    MEMORY_PROMPT.format(user_details_content=existing)
                )
            ),

            {"role": "user", "content": recent}
        ]
    )
    if response.should_write:
        for mem in response.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}
def chat(state: MessagesState, config:RunnableConfig,store:BaseStore):

    """LLM node that may answer or request a tool call."""
    thread_id = None
    user_id=None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
        user_id=config.get("configurable",{}).get("user_id")
    ns=("user",user_id,"details")
    items = store.search(ns)
    user_details_content = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    path = get_dir_path(thread_id)
    if thread_id in VECTORSTORE or os.path.exists(path):
        doc_status = "✅ STATUS: A PDF document IS CURRENTLY UPLOADED and ready. You CAN and SHOULD use the `rag_tool` to answer questions about it."
    else:
        doc_status = "❌ STATUS: No document is currently uploaded. Ask the user to upload a PDF."

    system_message = SystemMessage(
        content=(
                       f"""You are a helpful assistant with memory capabilities.
            If user-specific memory is available, use it to personalize 
            your responses based on what you know about the user.

            Your goal is to provide relevant, friendly, and tailored 
            assistance that reflects the user’s preferences, context, and past interactions.

            If the user’s name or relevant personal context is available, always personalize your responses by:
                – Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
                – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
                – Adjusting the tone to feel friendly, natural, and directly aimed at the user

            Avoid generic phrasing when personalization is possible.

            Use personalization especially in:
                – Greetings and transitions
                – Help or guidance tailored to tools and frameworks the user uses
                – Follow-up messages that continue from past context

            Always ensure that personalization is based only on known user details and not assumed.

            In the end suggest 3 relevant further questions based on the current response and user profile

            The user’s memory (which may be empty) is provided as: {user_details_content}
        
           {doc_status}
            For questions about the uploaded PDF, call the `rag_tool` with the Query. 
            You can also use the web search, stock price, and calculator tools when helpful."""
        )
    )
    messages = [system_message, *state["messages"]]

    try:
        response = llm_with_tool.invoke(messages,config=config)
        return {"messages": [response]}
    
    except APIStatusError as e:
        if "413" in str(e) or "Request too large" in str(e):
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=(
                "Your request is too large for me to process at once. "
                "Please try asking one question at a time."
            ))]}
        raise e
#===============================================CHATBOT=========================================================
tool_node=ToolNode(tools)
graph=StateGraph(MessagesState)
graph.add_node("remember",remember)
graph.add_node("chat",chat)
graph.add_node("tools",tool_node)

graph.add_edge(START,"remember")
graph.add_edge("remember","chat")
graph.add_conditional_edges('chat',tools_condition)
graph.add_edge("tools", "chat")


chatbot = graph.compile(store=store, checkpointer=checkpointer)
#config={'configurable':{"thread_id":"p","user_id":"u3"}}
#chatbot.invoke({"messages": [{"role": "user", "content": "whaat is attention"}]}, config)


def extract_pointer():
    emp_set=set()
    for pointer in checkpointer.list(None):
        emp_set.add(pointer.config["configurable"]["thread_id"])
    list_pointers=list(emp_set)
    return list_pointers