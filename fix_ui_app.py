import re

with open("ui_App.py", "r") as f:
    code = f.read()

# Fix Chatbot type
code = code.replace(
    'chatbot = gr.Chatbot(label="RAG Assistant", height=600)',
    'chatbot = gr.Chatbot(label="RAG Assistant", height=600, type="messages")'
)

# Fix respond function history append
old_respond = """        response = f"{ans}\\n\\n*Source document: {active_doc}*"
        
    history.append((user_query, response))
    return "", history"""

new_respond = """        response = f"{ans}\\n\\n*Source document: {active_doc}*"
        
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response})
    return "", history"""

code = code.replace(old_respond, new_respond)

# Add clear chat button
old_chat_col = """            with gr.Row():
                msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...", scale=4)
                submit_btn = gr.Button("Send", variant="primary", scale=1)"""

new_chat_col = """            with gr.Row():
                msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...", scale=4)
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary", scale=1)"""

code = code.replace(old_chat_col, new_chat_col)

# Add clear chat click handler
old_submit = """    msg.submit(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])
    submit_btn.click(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])"""

new_submit = """    msg.submit(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])
    submit_btn.click(respond, inputs=[msg, chatbot, session_state], outputs=[msg, chatbot])
    clear_chat_btn.click(lambda: [], None, chatbot)"""

code = code.replace(old_submit, new_submit)

with open("ui_App.py", "w") as f:
    f.write(code)

print("ui_App.py updated with messages format.")
