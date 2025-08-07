from app.model import run_inference
from PIL import Image

def test_invoice_sample():
    img = Image.open("assets/sample_invoice.jpg")
    output = run_inference(img, "Extract key fields: vendor, date, total")
    assert "vendor" in output.lower()
