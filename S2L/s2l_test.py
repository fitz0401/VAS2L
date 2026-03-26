import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# ===== 参数 =====
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ckpts", "vosk-model-en-us-0.22")
SAMPLE_RATE = 16000

# ===== 初始化 =====
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# ===== 启动麦克风 =====
with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=8000,
    dtype='int16',
    channels=1,
    callback=callback,
    # device=8
):
    print("🎤 Start speaking... (Ctrl+C to stop)")

    while True:
        data = q.get()

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            print("✅ FINAL:", result.get("text", ""))
        else:
            partial = json.loads(recognizer.PartialResult())
            print("…", partial.get("partial", ""), end="\r")