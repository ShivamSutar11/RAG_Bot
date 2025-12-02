# ui_app.py

import gradio as gr
from rag_engine import answer

def respond(user_query):
    return answer(user_query)  # ONLY final answer, no prompt garbage

ui = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Ask something"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot",
    description="Ask questions based ONLY on your uploaded documents."
)

ui.launch()
