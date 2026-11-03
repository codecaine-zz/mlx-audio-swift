//
//  MelSpectrogram.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX

// MARK: - Mel Spectrogram Computation

private struct WindowCacheKey: Hashable {
    let kind: UInt8 // 0 = hanning, 1 = hamming
    let size: Int
    let periodic: Bool
}

private struct MelFilterCacheKey: Hashable {
    let sampleRate: Int
    let nFft: Int
    let nMels: Int
    let fMinBits: UInt32
    let fMaxBits: UInt32
    let norm: String?
    let isSlaneyScale: Bool
}

private enum DSPCache {
    static let lock = NSLock()
    nonisolated(unsafe) static var windows: [WindowCacheKey: MLXArray] = [:]
    nonisolated(unsafe) static var melFilters: [MelFilterCacheKey: MLXArray] = [:]
    nonisolated(unsafe) static var reverseIndices: [Int: MLXArray] = [:]
}

/// Create a Hanning window of given size.
/// `periodic: true` matches `torch.hann_window` (librosa/Whisper default);
/// the previous symmetric-only form is kept as default for compatibility.
public func hanningWindow(size: Int, periodic: Bool = false) -> MLXArray {
    guard size > 0 else { return MLXArray.zeros([0], type: Float.self) }
    if size == 1 { return MLXArray([Float(1.0)]) }

    let key = WindowCacheKey(kind: 0, size: size, periodic: periodic)
    DSPCache.lock.lock()
    if let cached = DSPCache.windows[key] {
        DSPCache.lock.unlock()
        return cached
    }
    DSPCache.lock.unlock()

    let effectiveSize = periodic ? size + 1 : size
    let invDenom = (2.0 * Float.pi) / Float(effectiveSize - 1)
    let window = [Float](unsafeUninitializedCapacity: size) { buffer, initializedCount in
        for n in 0..<size {
            buffer[n] = 0.5 * (1.0 - cos(Float(n) * invDenom))
        }
        initializedCount = size
    }
    let array = MLXArray(window)
    eval(array)

    DSPCache.lock.lock()
    DSPCache.windows[key] = array
    DSPCache.lock.unlock()
    return array
}

/// Create a Hamming window of given size.
public func hammingWindow(size: Int, periodic: Bool = true) -> MLXArray {
    guard size > 0 else { return MLXArray.zeros([0], type: Float.self) }
    if size == 1 { return MLXArray([Float(1.0)]) }

    let key = WindowCacheKey(kind: 1, size: size, periodic: periodic)
    DSPCache.lock.lock()
    if let cached = DSPCache.windows[key] {
        DSPCache.lock.unlock()
        return cached
    }
    DSPCache.lock.unlock()

    let effectiveSize = periodic ? size + 1 : size
    let invDenom = (2.0 * Float.pi) / Float(effectiveSize - 1)
    let values = [Float](unsafeUninitializedCapacity: size) { buffer, initializedCount in
        for n in 0..<size {
            buffer[n] = 0.54 - 0.46 * cos(Float(n) * invDenom)
        }
        initializedCount = size
    }
    let array = MLXArray(values)
    eval(array)

    DSPCache.lock.lock()
    DSPCache.windows[key] = array
    DSPCache.lock.unlock()
    return array
}

/// Mel scale variants for filterbank computation.
public enum MelScale: Hashable, Sendable {
    /// HTK formula: mel = 2595 * log10(1 + f/700)
    case htk
    /// Slaney (Auditory Toolbox): linear below 1000 Hz, logarithmic above
    case slaney
}

/// Padding mode for STFT.
public enum PadMode: Hashable, Sendable {
    /// Reflect padding (default, used by Whisper)
    case reflect
    /// Zero (constant) padding (used by NeMo/Sortformer)
    case constant
}

/// Convert a power spectrogram to decibels with optional dynamic range clipping.
public func powerToDB(
    _ spectrogram: MLXArray,
    amin: Float = 1e-10,
    topDB: Float? = nil
) -> MLXArray {
    var db = 10.0 * MLX.log10(MLX.maximum(spectrogram, MLXArray(amin)))

    if let topDB {
        db = MLX.maximum(db, db.max() - MLXArray(topDB))
    }

    return db
}

/// Create mel filterbank matrix.
public func melFilters(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = "slaney",
    melScale: MelScale = .htk
) -> MLXArray {
    let fMaxVal = fMax ?? Float(sampleRate) / 2.0
    let key = MelFilterCacheKey(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
        fMinBits: fMin.bitPattern,
        fMaxBits: fMaxVal.bitPattern,
        norm: norm,
        isSlaneyScale: melScale == .slaney
    )

    DSPCache.lock.lock()
    if let cached = DSPCache.melFilters[key] {
        DSPCache.lock.unlock()
        return cached
    }
    DSPCache.lock.unlock()

    let nFreqs = nFft / 2 + 1

    // Generate frequency points
    let freqScale = Float(sampleRate) / Float(nFft)
    let allFreqs = [Float](unsafeUninitializedCapacity: nFreqs) { buffer, initializedCount in
        for i in 0..<nFreqs {
            buffer[i] = Float(i) * freqScale
        }
        initializedCount = nFreqs
    }

    // Mel scale conversion functions
    let hzToMel: (Float) -> Float
    let melToHz: (Float) -> Float

    switch melScale {
    case .htk:
        hzToMel = { freq in 2595.0 * log10(1.0 + freq / 700.0) }
        melToHz = { mel in 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

    case .slaney:
        // Slaney (Auditory Toolbox) piecewise linear/log scale
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep = log(Float(6.4)) / 27.0

        hzToMel = { freq in
            if freq < minLogHz {
                return (freq - fMin) / fSp
            } else {
                return minLogMel + log(freq / minLogHz) / logStep
            }
        }
        melToHz = { mel in
            if mel < minLogMel {
                return fMin + fSp * mel
            } else {
                return minLogHz * exp(logStep * (mel - minLogMel))
            }
        }
    }

    // Convert to mel scale and back
    let mMin = hzToMel(fMin)
    let mMax = hzToMel(fMaxVal)
    let mStep = (mMax - mMin) / Float(nMels + 1)

    let fPts = [Float](unsafeUninitializedCapacity: nMels + 2) { buffer, initializedCount in
        for i in 0..<(nMels + 2) {
            buffer[i] = melToHz(mMin + Float(i) * mStep)
        }
        initializedCount = nMels + 2
    }

    // Compute flat filterbank [nFreqs, nMels] in a single allocation with fused normalization
    let useSlaneyNorm = (norm == "slaney")
    var flatFilters = [Float](repeating: 0, count: nFreqs * nMels)
    flatFilters.withUnsafeMutableBufferPointer { buf in
        for j in 0..<nMels {
            let low = fPts[j]
            let center = fPts[j + 1]
            let high = fPts[j + 2]
            let enorm: Float = useSlaneyNorm ? (2.0 / (high - low)) : 1.0
            let invLower = center > low ? (enorm / (center - low)) : 0
            let invUpper = high > center ? (enorm / (high - center)) : 0

            for i in 0..<nFreqs {
                let freq = allFreqs[i]
                if freq < low { continue }
                if freq > high { break }
                if freq < center {
                    buf[i * nMels + j] = (freq - low) * invLower
                } else {
                    buf[i * nMels + j] = (high - freq) * invUpper
                }
            }
        }
    }

    let result = MLXArray(flatFilters).reshaped([nFreqs, nMels])
    eval(result)

    DSPCache.lock.lock()
    DSPCache.melFilters[key] = result
    DSPCache.lock.unlock()
    return result
}

/// Reverse an array along the first axis using cached GPU index gather.
private func reverseArray(_ arr: MLXArray) -> MLXArray {
    let len = arr.shape[0]
    guard len > 1 else { return arr }

    if len <= 4096 {
        DSPCache.lock.lock()
        if let cachedIdx = DSPCache.reverseIndices[len] {
            DSPCache.lock.unlock()
            return arr[cachedIdx]
        }
        DSPCache.lock.unlock()

        let indices = [Int32](unsafeUninitializedCapacity: len) { buffer, initializedCount in
            for i in 0..<len {
                buffer[i] = Int32(len - 1 - i)
            }
            initializedCount = len
        }
        let idxArray = MLXArray(indices)
        eval(idxArray)

        DSPCache.lock.lock()
        DSPCache.reverseIndices[len] = idxArray
        DSPCache.lock.unlock()
        return arr[idxArray]
    }

    let indices = [Int32](unsafeUninitializedCapacity: len) { buffer, initializedCount in
        for i in 0..<len {
            buffer[i] = Int32(len - 1 - i)
        }
        initializedCount = len
    }
    return arr[MLXArray(indices)]
}

/// Short-time Fourier Transform.
public func stft(
    audio: MLXArray,
    window: MLXArray,
    nFft: Int,
    hopLength: Int,
    padMode: PadMode = .reflect
) -> MLXArray {
    // Pad audio for centering
    let padding = nFft / 2
    let audioLen = audio.shape[0]

    let padded: MLXArray
    switch padMode {
    case .reflect where audioLen > 1:
        // Reflect padding: reverse slices at both ends
        let prefixEnd = min(padding + 1, audioLen)
        let prefixSlice = audio[1..<prefixEnd]
        let prefix = reverseArray(prefixSlice)

        let suffixStart = max(0, audioLen - padding - 1)
        let suffixEnd = max(suffixStart + 1, audioLen - 1)
        let suffixSlice = audio[suffixStart..<suffixEnd]
        let suffix = reverseArray(suffixSlice)

        padded = MLX.concatenated([prefix, audio, suffix])

    case .reflect, .constant:
        // Zero padding (or fallback when audioLen <= 1)
        let prefix = MLXArray.zeros([padding])
        let suffix = MLXArray.zeros([padding])
        padded = MLX.concatenated([prefix, audio, suffix])
    }

    // Calculate number of frames
    let paddedLen = padded.shape[0]
    let numFrames = max(1, 1 + (paddedLen - nFft) / hopLength)

    // Create frames using strided view — zero-copy, no per-frame loop
    let framesStacked = asStrided(padded, [numFrames, nFft], strides: [hopLength, 1], offset: 0)

    // Apply window
    let windowed = framesStacked * window

    // Compute FFT (real FFT)
    let fft = MLXFFT.rfft(windowed, axis: 1)  // [numFrames, nFft/2 + 1]

    return fft
}

/// Compute mel spectrogram from audio waveform.
/// `melScale`/`hannPeriodic` defaults preserve legacy behavior; Whisper-style
/// front-ends (Qwen3-ASR) must pass `.slaney` + `true` to match
/// transformers' WhisperFeatureExtractor.
public func computeMelSpectrogram(
    audio: MLXArray,
    sampleRate: Int,
    nFft: Int,
    hopLength: Int,
    nMels: Int,
    melScale: MelScale = .htk,
    hannPeriodic: Bool = false
) -> MLXArray {
    // If audio is 1D, compute proper mel spectrogram
    if audio.ndim == 1 {
        // Create cached Hanning window
        let window = hanningWindow(size: nFft, periodic: hannPeriodic)

        // Compute STFT
        let freqs = stft(audio: audio, window: window, nFft: nFft, hopLength: hopLength)

        // Compute magnitude squared (power spectrum). Drop the last centred
        // frame, as WhisperFeatureExtractor and Python mlx-audio do, so the
        // frame count (and the max used by the -8 clamp below) matches theirs.
        let numFrames = freqs.dim(0)
        let kept = numFrames > 1 ? freqs[0..<(numFrames - 1)] : freqs
        let magnitudes = MLX.abs(kept).square()

        // Retrieve cached mel filterbank [nFreqs, nMels]
        let filters = melFilters(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: nMels,
            norm: "slaney",
            melScale: melScale
        )

        // Apply mel filterbank: [numFrames, nFreqs] @ [nFreqs, nMels] = [numFrames, nMels]
        var melSpec = MLX.matmul(magnitudes, filters)

        // Apply log scaling with clamping (Whisper-style normalization)
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        melSpec = MLX.log10(melSpec)
        let maxVal = melSpec.max()
        melSpec = MLX.maximum(melSpec, maxVal - MLXArray(Float(8.0)))
        melSpec = (melSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        return melSpec
    }

    // If already 2D, assume it's a spectrogram
    return audio
}
