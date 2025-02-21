import os
import json
import base64
import logging
import asyncio
from flask import Flask, request
from flask_socketio import SocketIO, emit
from google.cloud import speech
from openai import AsyncOpenAI
from six.moves import queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

# Google Cloud Speech client setup
client = speech.SpeechClient()

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms
LANGUAGE_CODE = "en-IN"

# Transcription config
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=LANGUAGE_CODE,
    enable_speaker_diarization=True,
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config, interim_results=True
)

class Stream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self.buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        self.buff.put(None)

    def fill_buffer(self, in_data):
        self.buff.put(in_data)

    def generator(self):
        while True:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

@socketio.on('connect')
def handle_connect():
    app.logger.info("WebSocket connected")

@socketio.on('disconnect')
def handle_disconnect():
    app.logger.info("WebSocket disconnected")

@socketio.on('media')
async def handle_media(data):
    payload = data['media']['payload']
    chunk = base64.b64decode(payload)
    stream.fill_buffer(chunk)
    await process_transcription()

async def process_transcription():
    audio_generator = stream.generator()
    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in audio_generator
    )
    responses = client.streaming_recognize(streaming_config, requests)
    await handle_responses(responses)

async def handle_responses(responses):
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        if result.is_final:
            openai_response = await get_openai_response(transcript)
            await send_openai_response(openai_response)

async def get_openai_response(transcript):
    try:
        response = await oai_client.completions.create(
            model="gpt-4", prompt=transcript, max_tokens=150, temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        app.logger.error(f"Error with OpenAI API: {e}")
        return "Error processing request."

async def send_openai_response(response_text):
    await socketio.emit('response', {'text': response_text})

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    stream = Stream(RATE, CHUNK)
    socketio.run(app, host='0.0.0.0', port=5000)
