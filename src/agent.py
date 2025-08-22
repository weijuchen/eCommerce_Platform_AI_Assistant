from audio_to_text import transcribe_audio
from image_ocr import extract_text
from image_classify import classify_image
from faq_rag import ask_question


def multimodal_agent(input_type: str, data: str):
    if input_type == "text":
        return ask_question(data)
    elif input_type == "audio":
        text = transcribe_audio(data)
        return ask_question(text)
    elif input_type == "image":
        text = extract_text(data)
        category = classify_image(data)
        return f"OCR: {text}, 商品分類: {category}"
    else:
        return "Unsupported input type."
