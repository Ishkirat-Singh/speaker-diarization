import os
import json
from pydub import AudioSegment

# Load the diarization results
diarization_results = {
    "segments": [
        {
            "speaker": "SPEAKER_01",
            "start": 0.03096875,
            "end": 10.20659375
        },
        {
            "speaker": "SPEAKER_00",
            "start": 9.26159375,
            "end": 23.824718750000002
        },
        {
            "speaker": "SPEAKER_00",
            "start": 24.516593750000002,
            "end": 29.96721875
        }
    ]
}

audio_path = "interview_short.wav"
audio = AudioSegment.from_wav(audio_path)

def extract_and_combine_speaker_segments(diarization_results, audio, output_dir="speaker_segments"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    combined_segments = {}
    
    for segment in diarization_results["segments"]:
        speaker = segment["speaker"]
        start_time = segment["start"] * 1000
        end_time = segment["end"] * 1000
        audio_segment = audio[start_time:end_time]
        
        if speaker not in combined_segments:
            combined_segments[speaker] = audio_segment
        else:
            combined_segments[speaker] += audio_segment
    
    for speaker, combined_segment in combined_segments.items():
        output_path = os.path.join(output_dir, f"{speaker}.wav")
        combined_segment.export(output_path, format="wav")
        print(f"Extracted and combined segment for {speaker}")

extract_and_combine_speaker_segments(diarization_results, audio)
