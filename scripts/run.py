import os

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment

# Speaker Diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

with ProgressHook() as hook:
    diarization = pipeline(
        "test_data/data.wav",
        hook=hook,
        num_speakers=2,
    )

rttm_file_path = "audio.rttm"
with open(rttm_file_path, "w") as rttm:
    diarization.write_rttm(rttm)


# Audio Segmentation
def parse_rttm(rttm_path):
    """Parse the RTTM file to extract speaker segments."""
    speaker_segments = {}

    with open(rttm_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            speaker_id = parts[7]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration

            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []

            speaker_segments[speaker_id].append((start_time, end_time))

    return speaker_segments


def segment_audio(audio_path, speaker_segments, output_dir):
    """Segment the audio file into separate files for each speaker."""
    audio = AudioSegment.from_mp3(audio_path)

    for speaker, segments in speaker_segments.items():
        speaker_audio = AudioSegment.empty()

        for start, end in segments:
            segment_audio = audio[
                start * 1000 : end * 1000
            ]  # Convert seconds to milliseconds
            speaker_audio += segment_audio

        speaker_file = os.path.join(output_dir, f"speaker_{speaker}.mp3")
        speaker_audio.export(speaker_file, format="mp3")


def main(rttm_path, audio_path, output_dir):
    """Main function to parse RTTM and split audio file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    speaker_segments = parse_rttm(rttm_path)
    segment_audio(audio_path, speaker_segments, output_dir)


# Example usage:
audio_file_path = "test_data/data.wav"
output_directory = "output_speakers"

main(rttm_file_path, audio_file_path, output_directory)
