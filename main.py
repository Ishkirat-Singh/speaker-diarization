import os
import torch
import warnings
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from huggingface_hub import login
# from langchain_community import clear_cache

warnings.filterwarnings("ignore")
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
login(token=HF_API_KEY)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_API_KEY)

def diarize_with_huggingface(audio_path):
    diarization = diarization_pipeline(audio_path)
    # diarization_result = {
    #     "segments": [
    #         {
    #             "speaker": turn.label,
    #             "start": turn.start,
    #             "end": turn.end,
    #         }
    #         for turn, _, speaker in diarization.itertracks(yield_label=True)
    #     ]
    # }
    print(diarization)
    return diarization

def print_diarization_results(diarization_result):
    for segment in diarization_result['segments']:
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        print(f"Speaker {speaker} from {start:.1f}s to {end:.1f}s")

audio_path = "interview.wav"
diarization_result = diarize_with_huggingface(audio_path)
# print_diarization_results(diarization_result)
