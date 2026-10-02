import gradio as gr
import os
import shutil
from rag_engine import answer, build_index_for_docs
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def process_uploads(files, state):
    if not files: return state, gr.update()
    
    if "docs" not in state:
        state["docs"] = {}
    
    for f in files:
        filename = os.path.basename(f.name)
        if filename in state["docs"]:
            continue # already uploaded
            
        ext = filename.split(".")[-1].lower()
        if ext == "pdf":
            docs = PyPDFLoader(f.name).load()
        elif ext == "txt":
            docs = TextLoader(f.name, encoding="utf-8").load()
        else:
            continue
            
        if not docs: continue
        
        # Override source metadata to be just the filename for cleaner display
        for d in docs:
            d.metadata["source"] = filename
            
        db, bm25, chunks = build_index_for_docs(docs)
        
        state["docs"][filename] = {
            "db": db,
            "bm25": bm25,
            "chunks": chunks,
            "type": ext
        }
        
        # Set active if it's the first one
        if state.get("active") is None:
            state["active"] = filename
            
    choices = list(state["docs"].keys())
    return state, gr.update(choices=choices, value=state.get("active"))

def select_doc(selected, state):
    state["active"] = selected
    return state

def delete_doc(state):
    active = state.get("active")
    if not active or "docs" not in state:
        return state, gr.update()
        
    del state["docs"][active]
    
    choices = list(state["docs"].keys())
    if choices:
        state["active"] = choices[0]
    else:
        state["active"] = None
        
    return state, gr.update(choices=choices, value=state.get("active"))

def clear_all(state):
    state["docs"] = {}
    state["active"] = None
    return state, gr.update(choices=[], value=None)

def respond(user_query, history, state):
    if not state.get("active") or state["active"] not in state.get("docs", {}):
        response = "Please select a document first."
    else:
        active_doc = state["active"]
        doc_data = state["docs"][active_doc]
        
        # Get answer using custom index
        ans = answer(user_query, 
                     custom_db=doc_data["db"], 
                     custom_bm25=doc_data["bm25"], 
                     custom_docs=doc_data["chunks"])
                     
        response = f"{ans}\n\n*Source document: {active_doc}*"
        
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response})
    return "", history

with gr.Blocks(title="RAG Chatbot") as ui:
    # Session state
    session_state = gr.State({"docs": {}, "active": None})
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Uploaded Documents")
            
            doc_selector = gr.Radio(choices=[], label="Active Document")
            
            file_upload = gr.File(file_count="multiple", label="Upload New Documents")
            
            with gr.Row():
                delete_btn = gr.Button("Delete Active", variant="secondary")
                clear_btn = gr.Button("Clear All", variant="stop")
                
        with gr.Column(scale=3):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(label="RAG Assistant", height=600)
            with gr.Row():
                msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...", scale=4)
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

    # Event handlers
    file_upload.upload(process_uploads, inputs=[file_upload, session_state], outputs=[session_state, doc_selector])
    
    doc_selector.change(select_doc, inputs=[doc_selector, session_state], outputs=[session_state])
    
    delete_btn.click(delete_doc, inputs=[session_state], outputs=[session_state, doc_selector])
    clear_btn.click(clear_all, inputs=[session_state], outputs=[session_state, doc_selector])
    
    msg.submit(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])
    submit_btn.click(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])
    clear_chat_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    ui.launch()
