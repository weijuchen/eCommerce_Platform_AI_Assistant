from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")  # 或 "cuda" 如果有 GPU


def transcribe_audio(audio_path: str) -> str:
    segments, info = model.transcribe(audio_path, language="zh")
    text = "".join([segment.text for segment in segments])
    return text


if __name__ == "__main__":
    print(transcribe_audio("../data/sample.wav"))


# import whisper

# model = whisper.load_model("base")


# def transcribe_audio(audio_path: str) -> str:
#     result = model.transcribe(audio_path, language="zh")
#     return result["text"]


# if __name__ == "__main__":
#     print(transcribe_audio("../data/sample.wav"))
