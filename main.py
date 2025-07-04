import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import wave

import numpy as np
import requests
import sounddevice as sd

from constants import (BASE_TRANSCRIPT_FOLDER, BLOCK_SIZE, CHANNELS,
                       DEFAULT_AUDIO_FILE_PATH, DEFAULT_TRANSCRIPT_FILE_PATH,
                       DTYPE, LLM_URL, MODEL_NAME, SAMPLE_RATE)
from prompts import (SP_ANALYSE_SPEECH, SP_INCIDENT_LEARNING_DISCUSSION,
                     UP_ANALYSE_SPEECH, UP_INCIDENT_LEARNING_DISCUSSION)

# Thread-safe queue to store audio chunks
audio_queue = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def record_audio():
    global recording
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
        dtype=DTYPE
):
        print("🎙️ Recording... Press Ctrl+C to stop.")
        while recording:
            sd.sleep(100)

def save_audio_to_wav(filename):
    print("📝 Saving audio...")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(np.dtype(DTYPE).itemsize)
    wf.setframerate(SAMPLE_RATE)

    while not audio_queue.empty():
        wf.writeframes(audio_queue.get())
    wf.close()
    print(f"✅ Audio saved to {filename}")
    return filename


def signal_handler(sig, frame):
    global recording
    print("\n🛑 Interrupt received. Stopping recording...")
    recording = False

def transcribe_text(audio_file_path, output_file=None):

    file_path = os.path.abspath(audio_file_path)
    print(f"📜 Transcribing audio from {file_path}...")

    # Paths
    python_path = "/Users/akashpatki/Documents/kash/code/moon/stt/mlx/.venv/bin/python"
    model = "mlx-community/parakeet-tdt-0.6b-v2"
    output_file = output_file or os.path.join(BASE_TRANSCRIPT_FOLDER, "transcript.txt")
    output_file = os.path.abspath(output_file)
    output_file = os.path.splitext(output_file)[0]  # Ensure the output file has a valid extension
    print(f"📂 Output will be saved to {output_file}")
    # Build and execute the command
    cmd = [
        python_path,
        "-m", "mlx_audio.stt.generate",
        "--model", model,
        "--audio", file_path,
        "--output", output_file
    ]

    print("🔄 Running transcription command...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Transcription complete. Output saved to {output_file}")
        # Optionally print the transcription
        try:
            with open(f"{output_file}.txt", 'r') as f:
                print(f"📝 Transcription: {f.read().strip()}")
                return f.read().strip()
        except Exception as e:
            print(f"Could not read transcription file: {e}")
    else:
        print(f"❌ Transcription failed: {result.stderr}")


def process_transcription(transcript, task="speech"):
    print(f"🔍 Processing transcript for task: {task}...")

    # Get appropriate prompts based on task
    system_prompt, user_prompt = get_prompts_for_task(task)

    payload = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system",
                    "name": "string"
                },
                {
                    "content": user_prompt.format(transcript=transcript),
                    "role": "user",
                    "name": "string"
                }
            ],
            "max_tokens": 8192,
            "stream": True,
            "user": "string",
            "tools": [],
            "tool_choice": "string",
            "additionalProp1": {}
        }
    )
    print("📤 Sending request to LLM Gateway...")

    out = "" # Accumulate the response
    try:
        with requests.post(LLM_URL, headers={'Content-Type': 'application/json'}, data=payload, stream=True) as response:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
                if chunk:
                    try:
                        # Decode bytes to string
                        chunk_str = chunk.decode('utf-8')
                        lines = chunk_str.strip().split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                json_str = line[6:]  # Remove 'data: ' prefix
                                if json_str.strip() == '[DONE]':
                                    break
                                if json_str.strip():  # Only process non-empty JSON strings
                                    json_chunk = json.loads(json_str)
                                    if 'choices' in json_chunk and len(json_chunk['choices']) > 0:
                                        content = json_chunk['choices'][0]['delta'].get('content',"")
                                        if content:
                                            out += content
                                            print(content, end='', flush=True)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON chunk: {e}")
                        print(f"Problematic chunk: {chunk}")
                    except UnicodeDecodeError as e:
                        print(f"Error decoding bytes to string: {e}")
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", flush=True)
        raise e

    return out.strip()  # Return the accumulated content

def render_markdown(text):
    from rich.console import Console
    from rich.markdown import Markdown

    md = Markdown(text)
    Console().print(md)

def read_file(file_path):
    with open(file_path, 'r') as fh:
        return fh.read().strip()

def get_prompts_for_task(task):
    """Return appropriate prompts based on the task"""
    if task == "speech":
        return SP_ANALYSE_SPEECH, UP_ANALYSE_SPEECH
    elif task == "incident":
        return SP_INCIDENT_LEARNING_DISCUSSION, UP_INCIDENT_LEARNING_DISCUSSION
    else:
        print(f"⚠️  Unknown task '{task}', using default speech prompts")
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Audio recording and transcription analysis tool')
    parser.add_argument(
        '--task',
        choices=['speech', 'incident'],
        default='speech',
        help='Type of analysis to perform on the transcript (default: speech)'
    )
    parser.add_argument(
        '--file',
        help='Path to existing transcription file to analyze'
    )

    args = parser.parse_args()

    try:
        if args.file:
            # from constants import SAMPLE_CHAT
            # transcript = SAMPLE_CHAT
            transcript = read_file(args.file)
            print(f"📖 Read transcription from file: {args.file}")
            print(transcript)
        else:
            signal.signal(signal.SIGINT, signal_handler)
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()
            record_thread.join()
            audio_file_path = save_audio_to_wav(DEFAULT_AUDIO_FILE_PATH)

            transcript = transcribe_text(audio_file_path, output_file=DEFAULT_TRANSCRIPT_FILE_PATH)
            print("📖 Read Transcribed text from default file")

        print(f"📋 Analyzing transcript using '{args.task}' prompts...")
        response = process_transcription(transcript, task=args.task)

        # Render the response in markdown format
        render_markdown(response)

    except Exception as e:
        print(f"Error: {e}")
        raise e  # Re-raise the exception for further handling if needed