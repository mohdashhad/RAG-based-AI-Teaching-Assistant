import whisper
import json
import os
# import warnings
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# warnings.filterwarnings("ignore", category=UserWarning)

print("Loading Whisper model...")
# model = whisper.load_model("large-v2", device="cpu")
model = whisper.load_model("medium", device="cuda")
print("Model loaded")

audios = os.listdir("audios")

for audio in audios:
    if "_" in audio and audio.endswith(".mp3"):
        number, title = audio.split("_", 1)
        title = title.replace(".mp3","")

        print(f"\nStarting file: {number} {title}")

        # result = model.transcribe(
        #     audio=f"audios/{audio}",
        #     language="hi",
        #     task="translate",
        #     word_timestamps=False,
        #     fp16=False
        # )
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            fp16=True
        )


        print("Transcription done, saving JSON...")

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        os.makedirs("jsons", exist_ok=True)
        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": chunks, "text": result["text"]},
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved: jsons/{audio}.json")
