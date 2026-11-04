// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
// Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.

import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFFT
import MLXNN

public enum NemotronReuseError: Error, LocalizedError, Equatable {
    case invalidRepositoryID(String)
    case invalidSampleRate(Int)
    case invalidAudio
    case invalidChunkConfiguration(String)
    case sampleRateMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let repo):
            return "Invalid Nemotron RE-USE repository ID: \(repo)"
        case .invalidSampleRate(let rate):
            return "Sample rate must be an integer between 8000 and 48000 Hz (got \(rate))"
        case .invalidAudio:
            return "Expected nonempty, finite audio with shape [samples] or [samples, channels]"
        case .invalidChunkConfiguration(let reason):
            return reason
        case .sampleRateMismatch(let expected, let actual):
            return "sampleRate (\(expected)) disagrees with the audio file (\(actual))"
        }
    }
}

public struct NemotronReuseOutput: @unchecked Sendable {
    public let audio: MLXArray
    public let sampleRate: Int

    public init(audio: MLXArray, sampleRate: Int) {
        self.audio = audio
        self.sampleRate = sampleRate
    }
}

public final class NemotronReuseModel: Module, STSModel, @unchecked Sendable {
    public static let defaultRepo = "nvidia/NVIDIA-NemotronLabs-RE-USE"
    public static let outputSuffix = "_enhanced"

    public let config: NemotronReuseConfig
    public var sampleRate: Int { 48_000 }

    @ModuleInfo(key: "dense_encoder") var denseEncoder: NemotronReuseDenseEncoder
    @ModuleInfo(key: "TSMamba") var tsMamba: [NemotronReuseTFMambaBlock]
    @ModuleInfo(key: "mask_decoder") var maskDecoder: NemotronReuseDecoder
    @ModuleInfo(key: "phase_decoder") var phaseDecoder: NemotronReuseDecoder

    /// Optional testing hook to override STFT spectrum generation.
    var stftOverride: ((MLXArray, Int, Int, Int, MLXArray) -> MLXArray)?
    /// Optional testing hook to override per-chunk enhancement.
    var chunkEnhanceOverride: ((MLXArray, Int) throws -> MLXArray)?

    public init(_ config: NemotronReuseConfig = NemotronReuseConfig()) {
        self.config = config
        self._denseEncoder.wrappedValue = NemotronReuseDenseEncoder(config)
        self._tsMamba.wrappedValue = (0..<config.numTfmamba).map { _ in
            NemotronReuseTFMambaBlock(config)
        }
        self._maskDecoder.wrappedValue = NemotronReuseDecoder(config, phase: false)
        self._phaseDecoder.wrappedValue = NemotronReuseDecoder(config, phase: true)
    }

    public func callAsFunction(_ magnitude: MLXArray, phase: MLXArray) -> (MLXArray, MLXArray) {
        let freq = magnitude.dim(-2)
        let time = magnitude.dim(-1)
        var x = MLX.stacked([magnitude, phase], axis: -1).transposed(0, 2, 1, 3)
        x = MLX.padded(
            x,
            widths: [.init((0, 0)), .init((0, 2)), .init((0, 2)), .init((0, 0))]
        )
        x = denseEncoder(x)
        for block in tsMamba {
            x = block(x)
        }
        let outMag = maskDecoder(x)[0..., 0..<time, 0..<freq, 0].transposed(0, 2, 1)
        let outPhase = phaseDecoder(x)[0..., 0..<time, 0..<freq, 0].transposed(0, 2, 1)
        return (outMag, outPhase)
    }

    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let key = "dense_encoder.dense_conv_1.0.weight"
        let expectedShape = (denseEncoder.denseConv1[0] as! Conv2d).weight.shape
        if let existing = weights[key], existing.shape == expectedShape {
            return weights
        }
        guard weights[key] != nil else { return weights }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (name, value) in weights {
            if value.ndim == 4 {
                sanitized[name] = value.transposed(0, 2, 3, 1)
            } else if name.hasSuffix("conv1d.weight"), value.ndim == 3 {
                sanitized[name] = value.transposed(0, 2, 1)
            } else {
                sanitized[name] = value
            }
        }
        return sanitized
    }

    public static func validateSampleRate(_ rate: Int) throws {
        guard (8_000...48_000).contains(rate) else {
            throw NemotronReuseError.invalidSampleRate(rate)
        }
    }

    static func periodicHannWindow(_ size: Int) -> MLXArray {
        let indices = MLXArray(0..<size).asType(.float32)
        return 0.5 - 0.5 * MLX.cos((2.0 * Float.pi * indices) / Float(size))
    }

    static func istft(
        _ spectrum: MLXArray,
        hopLength: Int,
        winLength: Int,
        window: MLXArray,
        normalized: Bool = true
    ) -> MLXArray {
        let nFft = (spectrum.dim(0) - 1) * 2
        var w = window
        if w.dim(0) < nFft {
            let padding = nFft - w.dim(0)
            w = MLX.padded(w, widths: [.init((padding / 2, padding - padding / 2))])
        }

        let numFrames = spectrum.dim(1)
        let totalLen = (numFrames - 1) * hopLength + nFft

        var reconstructed = MLXArray.zeros([totalLen], dtype: .float32)
        var windowSum = MLXArray.zeros([totalLen], dtype: .float32)

        let framesTime = MLXFFT.irfft(spectrum, axis: 0).transposed(1, 0)
        let frameOffsets = MLXArray(0..<numFrames) * Int32(hopLength)
        let indices = (
            frameOffsets.expandedDimensions(axis: 1)
            + MLXArray(0..<nFft).expandedDimensions(axis: 0)
        ).reshaped([-1])

        let updatesReconstructed = (framesTime * w).reshaped([-1])
        let windowNorm = normalized ? (w * w) : w
        let updatesWindow = MLX.tiled(windowNorm, repetitions: [numFrames]).reshaped([-1])

        reconstructed = reconstructed.at[indices].add(updatesReconstructed)
        windowSum = windowSum.at[indices].add(updatesWindow)

        let denominator = MLX.where(windowSum .> 1e-10, windowSum, MLXArray(Float(1.0)))
        reconstructed = reconstructed / denominator

        let start = nFft / 2
        let end = totalLen - start
        return reconstructed[start..<end]
    }

    func enhanceChunk(_ audio: MLXArray, sampleRate: Int) throws -> MLXArray {
        if let chunkEnhanceOverride {
            return try chunkEnhanceOverride(audio, sampleRate)
        }

        let length = audio.dim(0)
        let (nFft, hop, win) = config.stftSizes(sampleRate: sampleRate)

        var working = audio
        if length <= nFft / 2 {
            working = MLX.padded(working, widths: [.init((0, nFft / 2 + 1 - length))])
        }

        let window = Self.periodicHannWindow(win)
        let rawSpectrum: MLXArray
        if let stftOverride {
            rawSpectrum = stftOverride(working, nFft, hop, win, window)
        } else {
            rawSpectrum = MLXAudioCore.stft(
                audio: working,
                window: window,
                nFft: nFft,
                hopLength: hop,
                padMode: .reflect
            )
        }
        let spectrum = rawSpectrum.transposed(1, 0)
        var magnitude = MLX.abs(spectrum)

        switch config.compressFactor {
        case .log1p, .reluLog1p, .signedLog1p:
            magnitude = MLX.log1p(magnitude)
        case .power(let p):
            magnitude = MLX.pow(magnitude, p)
        }

        // Real FFT endpoints can contain -0j in MLX. Force +0.0 so atan2(-1, 0) == +pi.
        let specReal = spectrum.realPart()
        let specImag = spectrum.imaginaryPart()
        let imaginary = MLX.where(specImag .== 0, MLXArray(Float(0.0)), specImag)
        let phase = MLX.atan2(imaginary, specReal)

        let (predMagBatch, predPhaseBatch) = self(
            magnitude.expandedDimensions(axis: 0),
            phase: phase.expandedDimensions(axis: 0)
        )
        var predMag = predMagBatch[0]
        let predPhase = predPhaseBatch[0]

        let silent = MLX.mean((predMag .<= 0).asType(.float32), axis: 0) .> 0.5
        predMag = MLX.where(silent.expandedDimensions(axis: 0), MLXArray(Float(0.0)), predMag)

        switch config.compressFactor {
        case .reluLog1p:
            predMag = MLX.expm1(relu(predMag))
        case .log1p:
            predMag = MLX.expm1(predMag)
        case .signedLog1p:
            predMag = MLX.sign(predMag) * MLX.expm1(MLX.abs(predMag))
        case .power(let p):
            predMag = MLX.pow(relu(predMag), 1.0 / p)
        }

        let realOut = predMag * MLX.cos(predPhase)
        let imagOut = predMag * MLX.sin(predPhase)
        let outSpectrum = realOut + MLXArray(real: Float(0), imaginary: Float(1)) * imagOut
        var enhanced = Self.istft(outSpectrum, hopLength: hop, winLength: win, window: window, normalized: true)

        if enhanced.dim(0) < length {
            enhanced = MLX.padded(
                enhanced,
                widths: [.init((0, length - enhanced.dim(0)))],
                value: MLXArray(Float(1e-8))
            )
        }
        return enhanced[0..<length]
    }

    public func enhance(
        _ audio: MLXArray,
        sampleRate: Int = 48_000,
        targetSampleRate: Int? = nil,
        chunkSeconds: Float? = 5.0,
        overlapSeconds: Float? = nil
    ) throws -> MLXArray {
        try generate(
            audio: audio,
            sampleRate: sampleRate,
            targetSampleRate: targetSampleRate,
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds
        ).audio
    }

    public func generate(
        audioURL: URL,
        sampleRate: Int? = nil,
        targetSampleRate: Int? = nil,
        chunkSeconds: Float? = 5.0,
        overlapSeconds: Float? = nil
    ) throws -> NemotronReuseOutput {
        let (fileSampleRate, audio) = try loadAudioArray(from: audioURL)
        if let sampleRate, sampleRate != fileSampleRate {
            throw NemotronReuseError.sampleRateMismatch(expected: sampleRate, actual: fileSampleRate)
        }
        return try generate(
            audio: audio,
            sampleRate: fileSampleRate,
            targetSampleRate: targetSampleRate,
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds
        )
    }

    public func generate(
        audio: MLXArray,
        sampleRate: Int,
        targetSampleRate: Int? = nil,
        chunkSeconds: Float? = 5.0,
        overlapSeconds: Float? = nil
    ) throws -> NemotronReuseOutput {
        try Self.validateSampleRate(sampleRate)
        let input = audio.asType(.float32)
        guard (input.ndim == 1 || input.ndim == 2),
              input.size > 0,
              all(isFinite(input)).item(Bool.self) else {
            throw NemotronReuseError.invalidAudio
        }

        let isMono = input.ndim == 1
        var working = isMono ? input.expandedDimensions(axis: 1) : input
        var activeSampleRate = sampleRate

        if let targetSampleRate {
            try Self.validateSampleRate(targetSampleRate)
            if targetSampleRate != sampleRate {
                var resampledChannels: [MLXArray] = []
                for ch in 0..<working.dim(1) {
                    let resampled = try resampleAudio(
                        working[0..., ch],
                        from: sampleRate,
                        to: targetSampleRate
                    )
                    resampledChannels.append(resampled)
                }
                working = MLX.stacked(resampledChannels, axis: 1)
            }
            activeSampleRate = targetSampleRate
        }

        let length = working.dim(0)
        let numChannels = working.dim(1)
        let chunk: Int
        let overlap: Int

        if let chunkSeconds {
            guard chunkSeconds.isFinite, chunkSeconds > 0 else {
                throw NemotronReuseError.invalidChunkConfiguration("chunkSeconds must be positive or nil")
            }
            let resolvedOverlap = overlapSeconds ?? (chunkSeconds / 2.0)
            guard resolvedOverlap.isFinite, resolvedOverlap >= 0, resolvedOverlap < chunkSeconds else {
                throw NemotronReuseError.invalidChunkConfiguration(
                    "overlapSeconds must be nonnegative and less than chunkSeconds"
                )
            }
            chunk = max(1, Int(chunkSeconds * Float(activeSampleRate)))
            overlap = Int(resolvedOverlap * Float(activeSampleRate))
            guard overlap < chunk else {
                throw NemotronReuseError.invalidChunkConfiguration("Chunk hop must be at least one sample")
            }
        } else {
            chunk = length
            overlap = 0
        }

        let enhanced: MLXArray
        if length <= chunk {
            var channels: [MLXArray] = []
            for ch in 0..<numChannels {
                channels.append(try enhanceChunk(working[0..., ch], sampleRate: activeSampleRate))
            }
            enhanced = MLX.stacked(channels, axis: -1)
        } else {
            let hop = chunk - overlap
            var starts = Array(stride(from: 0, to: length - chunk, by: hop))
            starts.append(length - chunk)

            let window = MLX.maximum(Self.periodicHannWindow(chunk), MLXArray(Float(1e-7)))
            var channelOutputs = [MLXArray](
                repeating: MLXArray.zeros([length], dtype: .float32),
                count: numChannels
            )
            var total = MLXArray.zeros([length], dtype: .float32)

            for start in starts {
                let indices = MLXArray(Int32(start)..<Int32(start + chunk))
                for ch in 0..<numChannels {
                    let chunkAudio = working[start..<(start + chunk), ch]
                    let result = try enhanceChunk(chunkAudio, sampleRate: activeSampleRate)
                    channelOutputs[ch] = channelOutputs[ch].at[indices].add(result * window)
                }
                total = total.at[indices].add(window)
            }

            let stacked = MLX.stacked(channelOutputs, axis: -1)
            enhanced = stacked / total.expandedDimensions(axis: -1)
        }

        eval(enhanced)
        let finalAudio = isMono ? enhanced[0..., 0] : enhanced
        return NemotronReuseOutput(audio: finalAudio, sampleRate: activeSampleRate)
    }

    public static func fromDirectory(_ modelDirectory: URL, strict: Bool = true) throws -> NemotronReuseModel {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(NemotronReuseConfig.self, from: configData)

        let model = NemotronReuseModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        let safetensorFiles = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var rawWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let shard = try MLX.loadArrays(url: file)
            rawWeights.merge(shard) { _, new in new }
        }

        let weights = model.sanitize(rawWeights)
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: strict ? .all : .none
        )
        eval(model)
        return model
    }

    public static func fromPretrained(
        _ modelRepo: String = defaultRepo,
        hfToken: String? = nil,
        cache: HubCache = .default,
        strict: Bool = true
    ) async throws -> NemotronReuseModel {
        let fm = FileManager.default
        let localURL = URL(fileURLWithPath: (modelRepo as NSString).expandingTildeInPath)
        if fm.fileExists(atPath: localURL.path) {
            return try fromDirectory(localURL, strict: strict)
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NemotronReuseError.invalidRepositoryID(modelRepo)
        }
        let modelDirectory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try fromDirectory(modelDirectory, strict: strict)
    }
}
