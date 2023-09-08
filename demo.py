import gradio as gr
from Inference import Inference

inference = Inference()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Dialogue Prompt")
    nnOut_userOut = gr.State()
    examples = gr.Examples(
        examples=[
            "2022 - 2024 red vf 9 vinfast less than $32000 5000 miles or less", "black",
            "less than 8000 miles", "remove red", "remove $32000 8000 miles",
        ],
        inputs=[msg],
        label="Familiarize yourself with the interface by running the following examples by clicking on them one-by-one from left-to-right:",
    )

    def respond(message, chat_history):
        global nnOut_userOut
        if not len(message):
            # nnOut_userOut is unchanged
            return "", chat_history
        elif len(message) > 400:
            if not chat_history:
                nnOut_userOut = 0
            chat_history.append(
                [message, "Shorten your text; it must be less than 100 words"])
            return "", chat_history

        if not chat_history:
            prevTrnUserOut = 0
        else:
            prevTrnUserOut = nnOut_userOut

        sessionId = 98
        nnOut_userOut_temp = inference.batching(sessionId, message,
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

demo.launch(debug=True, share=False)
