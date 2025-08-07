import gradio as gr
from app.model import run_inference

QUESTION = "Extract key fields: vendor, date, total, tax, and items"

def extract_data_ui(image):
    return run_inference(image, QUESTION)

def launch_ui():
    demo = gr.Interface(
        fn=extract_data_ui,
        inputs=gr.Image(type="pil", label="Upload Invoice or Receipt"),
        outputs=gr.Textbox(label="Extracted Data"),
        title="🧾 Invoice & Receipt Data Extractor",
        description="Extracts structured info using Hugging Face Donut model.",
    )
    demo.launch()
