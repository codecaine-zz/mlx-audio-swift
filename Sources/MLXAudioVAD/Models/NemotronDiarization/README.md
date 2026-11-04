# NVIDIA Nemotron 3 Diarization

MLX Swift implementation of [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization): a 31-layer RoPE Transformer with 8x feature stacking, an 8-speaker 10 ms output head with subpixel upsampling, and Arrival-Order Speaker Cache (AOSC) compression.

## Supported Models

| Precision | Hugging Face Repository |
| --- | --- |
| BF16 | [mlx-community/Nemotron-3-Diarization](https://huggingface.co/mlx-community/Nemotron-3-Diarization) |
| 8-bit | [mlx-community/Nemotron-3-Diarization-8bit](https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit) |

## Usage

```swift
import MLXAudioCore
import MLXAudioVAD

let model = try await NemotronDiarizationModel.fromPretrained("mlx-community/Nemotron-3-Diarization")
let (sampleRate, audio) = try loadAudioArray(from: audioURL)
let output = try model.generate(audio: audio, sampleRate: sampleRate, threshold: 0.5)
print(output.text) // RTTM
```
