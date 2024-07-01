import os
import torch
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

whisper_model = whisper.load_model("base", device=device)

diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_API_KEY"))

def transcribe_and_diarize(audio_path):
    # Transcribe the audio using Whisper
    result = whisper_model.transcribe(audio_path)
    transcription = result['text']
    
    # Print transcription
    print("Transcription:\n", transcription)

    # Perform speaker diarization
    diarization = diarization_pipeline(audio_path)
    
    # Print diarization results
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker} from {turn.start:.1f}s to {turn.end:.1f}s")
    
    # Combine transcription and diarization results (optional)
    # Here we would combine the time segments and the transcription text
    # This is a simplified example and may need adjustments based on specific requirements
    diarized_transcription = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_transcription = result['segments'][turn.start:turn.end]['text']
        diarized_transcription.append(f"Speaker {speaker}: {segment_transcription}")
    
    return diarized_transcription

# Example usage
audio_path = "path/to/your/audio/file.wav"
diarized_transcription = transcribe_and_diarize(audio_path)
for line in diarized_transcription:
    print(line)
