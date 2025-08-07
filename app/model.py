from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"

# Load model and processor once
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def run_inference(image, question: str):
    image = image.convert("RGB")
    task_prompt = f"<s_docvqa>{question}</s_docvqa>"

    inputs = processor(image, task_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=512, num_beams=3)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result.strip()
