import json
import base64
import logging
import asyncio
import openai
from flask import Flask, request
from flask_socketio import SocketIO, emit
from google.cloud import speech_v1p1beta1 as speech
from six.moves import queue

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")  # Use eventlet or gevent for async handling

# Google Cloud Speech client setup
client = speech.SpeechClient()

# OpenAI API setup (Ensure you have your OpenAI API key)
openai.api_key = "your-openai-api-key"

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms
language_code = "en-IN"  # Language Code for Speech-to-Text

# Transcription config
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=language_code,
    enable_speaker_diarization=True,
)

streaming_config = speech.StreamingRecognitionConfig(
    config=config, interim_results=True
)

class Stream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self.buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        self.buff.put(None)

    def fill_buffer(self, in_data):
        """Collect data from audio stream into the buffer."""
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

# WebSocket event handling
@socketio.on('connect')
def handle_connect():
    app.logger.info("WebSocket connection established")

@socketio.on('disconnect')
def handle_disconnect():
    app.logger.info("WebSocket connection closed")

@socketio.on('media')
async def handle_media(data):
    """Handle audio data sent from WebSocket and process it."""
    app.logger.info(f"Received media data: {data}")
    payload = data['media']['payload']
    chunk = base64.b64decode(payload)
    stream.fill_buffer(chunk)
    
    # Streaming the transcription and sending it back in real-time
    await stream_transcript()

async def stream_transcript():
    """Transcribe the audio stream and send it to OpenAI for response."""
    audio_generator = stream.generator()
    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in audio_generator
    )
    responses = client.streaming_recognize(streaming_config, requests)
    await listen_print_loop(responses)

async def listen_print_loop(responses):
    """Print transcription and use OpenAI's GPT to generate responses."""
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        if not result.is_final:
            print(transcript, end='\r')  # Overwrite last line for interim result
        else:
            print(transcript)
            # If transcription is final, use OpenAI to generate a response
            if transcript:
                openai_response = await get_openai_response(transcript)
                await send_openai_response(openai_response)

async def get_openai_response(transcript):
    """Send the transcription to OpenAI's API and get the response."""
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # or "gpt-4" based on your use case
            prompt=transcript,
            max_tokens=150,
            temperature=0.7,
        )
        return response['choices'][0]['text']
    except Exception as e:
        app.logger.error(f"Error communicating with OpenAI: {e}")
        return "Sorry, I couldn't process that."

async def send_openai_response(response_text):
    """Send OpenAI's response back through WebSocket."""
    await socketio.emit('response', {'text': response_text})

# Main entry point
if __name__ == '__main__':
    # Configure logging
    app.logger.setLevel(logging.DEBUG)
    
    # Initialize stream
    stream = Stream(RATE, CHUNK)

    # Start Flask server with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000)
