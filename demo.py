"""
Run this app based on one of the following criteria:
    (1) "python3 -m pdb demo.py" if debugging code
    (2) "gradio demo.py"
        (a) demo.launch(debug=, share=False) => get a local URL
        (a) demo.launch(debug=, share=True) => get a public URL that is good for 72 hours
    (3) "gradio deploy" => get a free permanent hosting and GPU upgrades to deploy to Spaces (https://huggingface.co/spaces)
"""
import gradio as gr
from Pipeline import Pipeline
from typing import List, Dict, Any, Tuple
from json import loads

pipeline = Pipeline()


def respond(message: str, chat_history: List[List[str]]):
    # chat_history: [["msg", "prevTrnUserOut"], ...]
    if not len(message):
        return "", chat_history
    if chat_history:
        try:
            # convert dict-in-a-string to dict
            prevTrnUserOut: Dict[str, List[str]] = loads(
                chat_history[-1][1].replace("'", "\""))
            if len(prevTrnUserOut) != 6:
                prevTrnUserOut = {}
        except Exception:
            prevTrnUserOut = {}
    else:
        prevTrnUserOut = {}

    sessionId = 98
    userOut = pipeline.input(sessionId, message, prevTrnUserOut)
    if isinstance(userOut, dict):
        chat_history.append([message, str(userOut)])
    elif isinstance(userOut, str):
        chat_history.append([message, userOut])
    else:
        assert False, f"unknown userOut={userOut}"

    print("-----------------------------------------------------")
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Dialogue Prompt")
    examples = gr.Examples(
        examples=[
            "2022 - 2024 white olatinum tei-coat metaloic vf 9 vinfast less than $32000 5000 miles or less",
            "landriver dedender 90", "remove $32000 5000 miles",
            "40,000 dillars or lesa", "renove", "more than 8000 miles", "clear"
        ],
        inputs=[msg],
        label=
        "Familiarize yourself with the interface by running the following examples by clicking on them one-by-one from left-to-right:",
    )
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
