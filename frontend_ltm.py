import streamlit as st
from chatbot_ltm import chatbot,extract_pointer,ingest
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid

st.title(":rainbow[ADVANCED RAG CHATBOT]")
if "thread_id" not in st.session_state:
    st.session_state["thread_id"]=extract_pointer()
    st.session_state["thread_id"].append(str(uuid.uuid4()))
if 'history' not in st.session_state:
    st.session_state['history']=[]
if 'key' not in st.session_state:
    st.session_state['key']=[-1]
if "processed_file" not in st.session_state:
     st.session_state["processed_file"]=set()

def load_conv(thread_id):
    state = chatbot.get_state(config={'configurable': {"thread_id": thread_id,"user_id":"u3"}})
    return state.values.get("messages", [])



#=================================SIDEBAR=======================================================================
st.sidebar.title("CHATBOT")


if st.sidebar.button(":blue[New chat]"):
    st.session_state["thread_id"].append(str(uuid.uuid4()))
    st.session_state['key'].append(-1)
    st.session_state['history']=[]

st.sidebar.header(":green[My conversation]")

for id in st.session_state["thread_id"]:
    if st.sidebar.button(id) :
        st.session_state['key'].append(st.session_state["thread_id"].index(id))

thread_id=st.session_state["thread_id"][st.session_state['key'][-1]]
st.sidebar.markdown(f"current thread :green[{thread_id}]")
#===========================================LOAD_CONVERSATION================================================
messages=load_conv(thread_id)
if messages:
    temp_messages=[]
    for msg in messages:
        if isinstance(msg,HumanMessage):
            temp_messages.append({'role':'user','content':msg.content})
        elif isinstance(msg,AIMessage) and msg.content: 
            temp_messages.append({'role':'ai','content':msg.content})
        else: continue
            
       
    st.session_state['history']=temp_messages

    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])


config={'configurable':{"thread_id":thread_id,"user_id":"u3"}}

#===========================================INVOKE===========================================================
user_input=st.chat_input(
    "type here...",accept_file=True, 
    file_type=["pdf"],
    key=f"chat_input_{thread_id}"
    )

if user_input: 
    if user_input.files:
        uploaded_file = user_input.files[0] # Grab the attached document
        file_key = f"{thread_id}_{uploaded_file.name}"
        
        # Deduplicate and process exactly like your old sidebar logic
        if file_key not in st.session_state["processed_file"]:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                file_bytes = uploaded_file.read()
                ingest(file_bytes, uploaded_file.name, thread_id)
                st.session_state["processed_file"].add(file_key)
                st.toast(f"Processed {uploaded_file.name} successfully!", icon="📄")

    if user_input.text.strip():
        user_query = user_input.text
    else:
        user_query = f"Uploaded and attached: {user_input.files[0].name}"

    st.session_state['history'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.text(user_query)

    if user_query.strip().lower() in ["exit", "end", "bye"]:
        st.write("Goodbye!")
    else:
    
        with st.chat_message("ai"):
                status_holder = {"box": None}
            
                def stream():
                    for message_chunk, metadata in chatbot.stream(
                        {
                            "messages": [HumanMessage(content=user_query)]
                        },
                        stream_mode="messages",
                        config=config
                    ):
                        # 1. Handle AI deciding to use a tool (Streaming Tool Calls)
                        if isinstance(message_chunk, AIMessage):
                            # Check if the AI is currently outputting a tool call chunk
                            if hasattr(message_chunk, 'tool_call_chunks') and message_chunk.tool_call_chunks:
                                tool_name = message_chunk.tool_call_chunks[0].get("name")
                                if tool_name and tool_name != "MemoryDecision": # Update UI when the tool name is yielded
                                    if status_holder["box"] is None:
                                        status_holder["box"] = st.status(f"🤖 Preparing `{tool_name}`...", expanded=True)
                                    else:
                                        status_holder["box"].update(label=f"🤖 Preparing `{tool_name}`...", state="running")
                            
                            # Check if it is a standard text response and NOT empty
                            elif message_chunk.content:
                                yield message_chunk.content

                        # 2. Handle the Result of the ToolExecution
                        elif isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Used `{tool_name}`", state="complete", expanded=False
                                )
                            else:
                                # Collapse the box once the tool is done
                                status_holder["box"].update(
                                    label=f"✅ Finished `{tool_name}`",
                                    state="complete",
                                    expanded=False,
                                )
                                
                # Streamlit safely consumes only the valid text chunks
                ai_message = st.write_stream(stream)

        st.session_state['history'].append({"role":"ai","content":ai_message})