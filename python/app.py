import os
import json
import base64
import logging
import asyncio
from flask import Flask, request
from flask_socketio import SocketIO, emit
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

# Initialize OpenAI Realtime client
from examplerealtimecode import RealtimeClient, TurnDetectionMode

client = RealtimeClient(
    api_key=OPENAI_API_KEY,
    model="gpt-4-turbo-preview",
    voice="alloy",
    instructions="You are a helpful assistant",
    temperature=0.7,
    turn_detection_mode=TurnDetectionMode.SERVER_VAD
)

@socketio.on('connect')
async def handle_connect():
    app.logger.info("WebSocket connected")
    await client.connect()

@socketio.on('disconnect')
async def handle_disconnect():
    app.logger.info("WebSocket disconnected")
    await client.close()

@socketio.on('media')
async def handle_media(data):
    payload = data['media']['payload']
    audio_chunk = base64.b64decode(payload)
    await client.stream_audio(audio_chunk)

# Set up callbacks for OpenAI responses
@client.on_text_delta
def handle_text(delta):
    socketio.emit('response', {'text': delta})

@client.on_audio_delta
def handle_audio(audio_bytes):
    socketio.emit('audio', {'data': base64.b64encode(audio_bytes).decode()})

@client.on_input_transcript
def handle_input_transcript(transcript):
    socketio.emit('transcript', {'type': 'input', 'text': transcript})

@client.on_output_transcript
def handle_output_transcript(transcript):
    socketio.emit('transcript', {'type': 'output', 'text': transcript})

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    socketio.run(app, host='0.0.0.0', port=5000)
