# NVIDIA NemotronLabs RE-USE

[nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) is a time–frequency Mamba speech enhancement and bandwidth extension model supporting any integer sample rate from 8 kHz to 48 kHz.

## Usage

```swift
import MLXAudioCore
import MLXAudioSTS

let model = try await NemotronReuseModel.fromPretrained("nvidia/NVIDIA-NemotronLabs-RE-USE")
let (sampleRate, audio) = try loadAudioArray(from: inputURL)
let output = try model.generate(
    audio: audio,
    sampleRate: sampleRate,
    targetSampleRate: 48_000
)
```
