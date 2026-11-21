# Encrypted speech to text

Whisper base.en turns a spoken clip of up to 8 s into text fully encrypted, in 19.2 s for the audio and then 1.39 s
per token of the transcript on one GPU. The transcripts below were produced this way from LibriSpeech test-clean
clips.

| Clip | Encrypted transcript |
|---|---|
| 1221-135767-0024 | Pearl, seeing the rose bushes, began to cry for a red rose and would not be pacified. |
| 4077-13754-0000 | The army found the people in poverty and left them in comparative wealth. |
| 1580-141084-0045 | Suddenly he heard him at the very door. There was no possible escape. |
| 2830-3980-0062 | Not for a crown or a kingdom or our goodness, but for our sins. |

```bash
perseus-asr clip.wav
perseus-asr first.wav second.wav
```
