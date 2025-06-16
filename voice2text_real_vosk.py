import pyaudio
import queue
import threading
import time
import wave
import vosk
import warnings
import os
import json  # Replace eval() with json.loads() for safety

# Suppress warnings
warnings.filterwarnings("ignore")

# Audio stream parameters
CHUNK = 160
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioStream:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = False
        self.p = pyaudio.PyAudio()
        # Define model path (update this if model is not in script directory)
        model_path = "vosk-model-en-us-0.22-lgraph" #"vosk-model-en-us-0.22-lgraph" #"vosk-model-en-us-0.22" 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist. Please check the path or download the model.")
        try:
            self.model = vosk.Model(model_path)
        except Exception as e:
            raise Exception(f"Failed to load Vosk model from '{model_path}': {e}")
        self.recognizer = vosk.KaldiRecognizer(self.model, RATE)
        
    def audio_callback(self):
        """Continuously collect audio data and put it in queue"""
        stream = self.p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        while self.running:
            data = stream.read(CHUNK, exception_on_overflow=False)
            self.audio_queue.put(data)
        stream.stop_stream()
        stream.close()

    def process_audio(self):
        """Process audio from queue and transcribe with Vosk"""
        while self.running or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                if self.recognizer.AcceptWaveform(audio_data):
                    result = self.recognizer.Result()
                    if result:
                        text = json.loads(result)["text"]  # Use json.loads instead of eval for safety
                        if text:
                            print(f"Transcription: '{text}'")
                        else:
                            print("No speech detected")
                    else:
                        print("No result from recognizer")
                else:
                    partial = self.recognizer.PartialResult()
                    if partial:
                        partial_text = json.loads(partial)["partial"]
                        if partial_text:
                            print(f"Partial: '{partial_text}'")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
            ##time.sleep(0.1)

    def start(self):
        if not self.running:
            self.running = True
            self.audio_thread = threading.Thread(target=self.audio_callback)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            print("Started real-time offline transcription with Vosk (vosk-model-small-en-us-0.15). Speak now...")
            
    def stop(self):
        self.running = False
        self.audio_thread.join()
        self.process_thread.join()
        self.p.terminate()
        print("Stopped transcription")

def main():
    try:
        audio_stream = AudioStream()
        audio_stream.start()
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        audio_stream.stop()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()