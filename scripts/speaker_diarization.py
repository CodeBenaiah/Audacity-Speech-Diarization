import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

from pyannote.audio.pipelines.utils.hook import ProgressHook

with ProgressHook() as hook:
    diarization = pipeline(
        "test_data/test_file.wav",
        hook=hook,
        num_speakers=2,
    )

with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
