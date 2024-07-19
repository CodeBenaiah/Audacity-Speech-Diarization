import os

from pydub import AudioSegment


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
rttm_file_path = "scripts/audio.rttm"
audio_file_path = "test_data/Test_File.mp3"
output_directory = "output_speakers"

main(rttm_file_path, audio_file_path, output_directory)
