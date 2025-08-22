import paddleocr

ocr = paddleocr.OCR()


def extract_text(image_path: str):
    result = ocr.ocr(image_path)
    return " ".join([line[1][0] for line in result[0]])


if __name__ == "__main__":
    print(extract_text("../data/product.png"))
