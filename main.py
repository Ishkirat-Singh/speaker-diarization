import os
import time
import json
import torch
import warnings
from pydub import AudioSegment
from dotenv import load_dotenv
# from huggingface_hub import login
from pyannote.audio import Pipeline
from requests.exceptions import ReadTimeout

warnings.filterwarnings("ignore")

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
# login(token=HF_API_KEY)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_diarization_pipeline(retries=3, timeout=30):
    for i in range(retries):
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_API_KEY)
            return pipeline
        except ReadTimeout:
            if i < retries - 1:
                print(f"Timeout occurred. Retrying... ({i+1}/{retries})")
                time.sleep(5)  # Wait before retrying
            else:
                raise
        except Exception as e:
            print(f"Error: {e}")
            raise

diarization_pipeline = load_diarization_pipeline()

def diarize_with_huggingface(audio_path):
    diarization = diarization_pipeline(audio_path)
    audio = AudioSegment.from_wav(audio_path)
    
    diarization_result = {
        "segments": []
    }
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_info = {
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        }
        diarization_result["segments"].append(segment_info)
    
    if not os.path.exists("speaker_segments"):
        os.makedirs("speaker_segments")
    
    combined_segments = {}
    
    for segment in diarization_result["segments"]:
        speaker = segment["speaker"]
        start_time = segment["start"] * 1000
        end_time = segment["end"] * 1000
        audio_segment = audio[start_time:end_time]
        
        if speaker not in combined_segments:
            combined_segments[speaker] = audio_segment
        else:
            combined_segments[speaker] += audio_segment
    
    for speaker, combined_segment in combined_segments.items():
        output_path = os.path.join("speaker_segments", f"{speaker}.wav")
        combined_segment.export(output_path, format="wav")
        print(f"Extracted and combined segment for {speaker}")

audio_path = "interview3.wav"
start_time = time.time()
diarization_result = diarize_with_huggingface(audio_path)
print(f"Time taken: {time.time() - start_time:.2f} seconds")
