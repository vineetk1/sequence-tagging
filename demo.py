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

pipeline = Pipeline()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Dialogue Prompt")
    nnOut_userOut = gr.State()
    examples = gr.Examples(
        examples=[
            "2022 - 2024 white olatinum tei-coat metaloic vf 9 vinfast less than $32000 5000 miles or less",
            "landriver dedender 90", "remove $32000 5000 miles",
            "40,000 dillars or lesa", "renove", "more than 8000 miles",
            "clear"
        ],
        inputs=[msg],
        label="Familiarize yourself with the interface by running the following examples by clicking on them one-by-one from left-to-right:",
    )

    def respond(message, chat_history):
        global nnOut_userOut
        if not len(message):
            # nnOut_userOut is unchanged
            return "", chat_history
        elif len(message) > 400:    # no need to do this; Tokenizer truncates
            if not chat_history:
                nnOut_userOut = 0
            chat_history.append(
                [message, "Shorten your text; it must be less than 101 words"])
            return "", chat_history

        if not chat_history:
            prevTrnUserOut = 0
        else:
            prevTrnUserOut = nnOut_userOut

        sessionId = 98
        nnOut_userOut_temp = pipeline.input(sessionId, message,
                                                prevTrnUserOut)
        if isinstance(nnOut_userOut_temp, str):
            if not chat_history:
                nnOut_userOut = 0
            chat_history.append([message, nnOut_userOut_temp])
            return "", chat_history
        else:
            # nnOut_userOut stores prevTrnUserOut
            nnOut_userOut = nnOut_userOut_temp

        bot_msg = []
        for key, values in nnOut_userOut.items():
            bot_msg.append(f'{key}: {values}')
        bot_msg = ',\t'.join(bot_msg)
        chat_history.append([message, bot_msg])

        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
    #demo.launch(debug=True, share=True)
    # demo.launch()
