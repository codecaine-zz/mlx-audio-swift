import Accelerate
import AVFoundation
import Foundation
import MLX

public class AudioUtils {
    public enum AudioUtilsErrors: Error, LocalizedError {
        case cannotCreateAVAudioFormat
        case cannotCreateAudioBuffer
        case cannotReadFloatChannelData
        case invalidSampleRate(Int)
        case resamplingFailed

        public var errorDescription: String? {
            switch self {
            case .cannotCreateAVAudioFormat:
                "Failed to create AVAudioFormat."
            case .cannotCreateAudioBuffer:
                "Failed to create audio buffer."
            case .cannotReadFloatChannelData:
                "Failed to access float channel data."
            case .invalidSampleRate(let sampleRate):
                "Sample rate must be positive, got \(sampleRate)."
            case .resamplingFailed:
                "Audio resampling failed."
            }
        }
    }

    private init() {}

    public static func writeWavFile(samples: [Float], sampleRate: Int, fileURL: URL) throws {
        try writeWavFile(samples: samples, sampleRate: Double(sampleRate), fileURL: fileURL)
    }

    public static func writeWavFile(samples: [Float], sampleRate: Double, fileURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)

        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
        else {
            throw AudioUtilsErrors.cannotCreateAVAudioFormat
        }

        buffer.frameLength = frameCount
        if let channelData = buffer.floatChannelData, !samples.isEmpty {
            samples.withUnsafeBufferPointer { src in
                guard let base = src.baseAddress else { return }
                channelData[0].update(from: base, count: samples.count)
            }
        }

        let audioFile = try AVAudioFile(
            forWriting: fileURL,
            settings: format.settings,
            commonFormat: format.commonFormat,
            interleaved: format.isInterleaved
        )

        try audioFile.write(from: buffer)
    }
}

/// Load audio from any audio or video file and return the sample rate and audio data.
/// Supports WAV, MP3, M4A, M4B, FLAC, AIFF, CAF, AAC, OGG, OPUS, WMA, APE, WV, AMR, AC3,
/// and video containers (MP4, MOV, M4V, MKV, WEBM, AVI, FLV, WMV, TS, 3GP) via native
/// AVAudioFile -> AVAssetReader -> afconvert -> ffmpeg fallback.
/// The returned tensor is an immutable snapshot that can be shared across actors.
public func loadAudioArray(from url: URL, sampleRate: Int? = nil) throws -> (Int, MaterializedArray) {
    if let (sr, samples) = try? readSamplesViaAVAudioFile(from: url), !samples.isEmpty {
        return try finalizeLoadedSamples(samples, sourceSampleRate: sr, targetSampleRate: sampleRate)
    }

    if let (sr, samples) = try? readSamplesViaAVAssetReader(from: url), !samples.isEmpty {
        return try finalizeLoadedSamples(samples, sourceSampleRate: sr, targetSampleRate: sampleRate)
    }

    #if os(macOS)
    if let tempWav = transcodeToTemporaryWAV(from: url, targetSampleRate: sampleRate) {
        defer { try? FileManager.default.removeItem(at: tempWav) }
        if let (sr, samples) = try? readSamplesViaAVAudioFile(from: tempWav), !samples.isEmpty {
            return try finalizeLoadedSamples(samples, sourceSampleRate: sr, targetSampleRate: sampleRate)
        }
    }
    #endif

    // Final attempt with AVAudioFile so the original error is surfaced if all fallbacks failed
    let (sr, samples) = try readSamplesViaAVAudioFile(from: url)
    return try finalizeLoadedSamples(samples, sourceSampleRate: sr, targetSampleRate: sampleRate)
}

private func finalizeLoadedSamples(
    _ samples: [Float],
    sourceSampleRate: Int,
    targetSampleRate: Int?
) throws -> (Int, MaterializedArray) {
    let targetSR = targetSampleRate ?? sourceSampleRate
    if targetSR <= 0 {
        throw AudioUtils.AudioUtilsErrors.invalidSampleRate(targetSR)
    }
    if targetSR == sourceSampleRate {
        return (sourceSampleRate, MLXArray(samples).materialized())
    }
    let resampled = try resampleAudio(
        samples,
        from: sourceSampleRate,
        to: targetSR
    )
    return (targetSR, MLXArray(resampled).materialized())
}

private func readSamplesViaAVAudioFile(from url: URL) throws -> (Int, [Float]) {
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    let capacity = max(1, min(frameCount, 1_048_576))
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }

    let channelCount = Int(format.channelCount)
    var samples: [Float] = []
    samples.reserveCapacity(Int(frameCount))
    while audioFile.framePosition < audioFile.length {
        let previousPosition = audioFile.framePosition
        try audioFile.read(into: buffer)
        guard buffer.frameLength > 0 else { break }
        guard let floatChannelData = buffer.floatChannelData else {
            throw AudioUtils.AudioUtilsErrors.cannotReadFloatChannelData
        }
        let start = samples.count
        let frames = Int(buffer.frameLength)
        samples.append(contentsOf: UnsafeBufferPointer(start: floatChannelData[0], count: frames))
        if channelCount > 1 {
            var invChannels = Float(channelCount)
            samples.withUnsafeMutableBufferPointer { dstBuf in
                guard let dstBase = dstBuf.baseAddress?.advanced(by: start) else { return }
                for channel in 1..<channelCount {
                    vDSP_vadd(dstBase, 1, floatChannelData[channel], 1, dstBase, 1, vDSP_Length(frames))
                }
                vDSP_vsdiv(dstBase, 1, &invChannels, dstBase, 1, vDSP_Length(frames))
            }
        }
        if audioFile.framePosition <= previousPosition { break }
    }

    return (Int(format.sampleRate), samples)
}

private func readSamplesViaAVAssetReader(from url: URL) throws -> (Int, [Float]) {
    let asset = AVURLAsset(url: url)
    let tracks = asset.tracks(withMediaType: .audio)
    guard let track = tracks.first else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAVAudioFormat
    }

    var detectedSampleRate = 24_000
    if let desc = track.formatDescriptions.first {
        let cfDesc = desc as! CMAudioFormatDescription
        if let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(cfDesc)?.pointee,
           asbd.mSampleRate > 0 {
            detectedSampleRate = Int(asbd.mSampleRate.rounded())
        }
    }

    let outputSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
        AVNumberOfChannelsKey: 1,
        AVSampleRateKey: Double(detectedSampleRate)
    ]

    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }
    reader.add(output)
    guard reader.startReading() else {
        throw reader.error ?? AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }

    var samples: [Float] = []
    while reader.status == .reading {
        guard let sampleBuffer = output.copyNextSampleBuffer() else { break }
        if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
            let length = CMBlockBufferGetDataLength(blockBuffer)
            let floatCount = length / MemoryLayout<Float>.size
            if floatCount > 0 {
                var chunk = [Float](repeating: 0, count: floatCount)
                chunk.withUnsafeMutableBytes { rawPtr in
                    if let base = rawPtr.baseAddress {
                        _ = CMBlockBufferCopyDataBytes(blockBuffer, atOffset: 0, dataLength: length, destination: base)
                    }
                }
                samples.append(contentsOf: chunk)
            }
        }
        CMSampleBufferInvalidate(sampleBuffer)
    }

    guard !samples.isEmpty else {
        throw AudioUtils.AudioUtilsErrors.cannotReadFloatChannelData
    }
    return (detectedSampleRate, samples)
}

#if os(macOS)
public func transcodeToTemporaryWAV(from url: URL, targetSampleRate: Int? = nil) -> URL? {
    let tempWAV = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlx_audio_transcode_\(UUID().uuidString).wav")

    // 1. Try ffmpeg first for broad container support (.ogg, .opus, .mkv, .webm, .wma, .mp4, .mov, .avi, .flv, .ape, .wv, .m4b)
    let ffmpegCandidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/bin/ffmpeg"
    ]
    if let ffmpegPath = ffmpegCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpegPath)
        var args = ["-y", "-v", "error", "-i", url.path, "-vn", "-ac", "1"]
        if let sr = targetSampleRate, sr > 0 {
            args += ["-ar", "\(sr)"]
        }
        args += ["-acodec", "pcm_f32le", tempWAV.path]
        proc.arguments = args
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus == 0,
           FileManager.default.fileExists(atPath: tempWAV.path),
           let attrs = try? FileManager.default.attributesOfItem(atPath: tempWAV.path),
           (attrs[.size] as? UInt64 ?? 0) > 64 {
            return tempWAV
        }
        try? FileManager.default.removeItem(at: tempWAV)
    }

    // 2. Fallback to native macOS /usr/bin/afconvert
    let afconvertPath = "/usr/bin/afconvert"
    if FileManager.default.isExecutableFile(atPath: afconvertPath) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: afconvertPath)
        let dataFmt = (targetSampleRate != nil && targetSampleRate! > 0) ? "LEF32@\(targetSampleRate!)" : "LEF32"
        proc.arguments = ["-f", "WAVE", "-d", dataFmt, "-c", "1", url.path, tempWAV.path]
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus == 0,
           FileManager.default.fileExists(atPath: tempWAV.path),
           let attrs = try? FileManager.default.attributesOfItem(atPath: tempWAV.path),
           (attrs[.size] as? UInt64 ?? 0) > 64 {
            return tempWAV
        }
        try? FileManager.default.removeItem(at: tempWAV)
    }

    return nil
}
#endif

/// Save audio data to a WAV file.
func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let audioFile = try AVAudioFile(forWriting: url, settings: format.settings)

    let frameCount = AVAudioFrameCount(samples.count)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw AudioUtils.AudioUtilsErrors.cannotCreateAudioBuffer
    }

    buffer.frameLength = frameCount

    if let channelData = buffer.floatChannelData, !samples.isEmpty {
        samples.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            channelData[0].update(from: base, count: samples.count)
        }
    }

    try audioFile.write(from: buffer)
}

private final class AudioConverterInputProvider: @unchecked Sendable {
    let inputBuffer: AVAudioPCMBuffer
    var consumedInput = false

    init(inputBuffer: AVAudioPCMBuffer) {
        self.inputBuffer = inputBuffer
    }
}

/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: [Float],
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> [Float] {
    if samples.isEmpty || sourceSampleRate == targetSampleRate {
        return samples
    }
    guard sourceSampleRate > 0 else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }
    guard targetSampleRate > 0 else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard let inputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sourceSampleRate),
        channels: 1,
        interleaved: false
    ), let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(targetSampleRate),
        channels: 1,
        interleaved: false
    ) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let inputFrameCount = AVAudioFrameCount(samples.count)
    guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }
    inputBuffer.frameLength = inputFrameCount
    samples.withUnsafeBufferPointer { ptr in
        guard let src = ptr.baseAddress else { return }
        memcpy(
            inputBuffer.floatChannelData![0],
            src,
            samples.count * MemoryLayout<Float>.size
        )
    }

    let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
    let estimatedFrames = max(1, Int(ceil(Double(samples.count) * ratio)) + 64)
    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: AVAudioFrameCount(estimatedFrames)
    ) else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let provider = AudioConverterInputProvider(inputBuffer: inputBuffer)
    var conversionError: NSError?
    let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
        if provider.consumedInput {
            outStatus.pointee = .endOfStream
            return nil
        }
        provider.consumedInput = true
        outStatus.pointee = .haveData
        return provider.inputBuffer
    }

    if let conversionError {
        _ = conversionError
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    guard status == .haveData || status == .endOfStream || status == .inputRanDry else {
        throw AudioUtils.AudioUtilsErrors.resamplingFailed
    }

    let outCount = Int(outputBuffer.frameLength)
    guard let outData = outputBuffer.floatChannelData?[0], outCount > 0 else {
        return []
    }
    return Array(UnsafeBufferPointer(start: outData, count: outCount))
}

/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: MLXArray,
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> MLXArray {
    let input = samples.asArray(Float.self)
    let resampled = try resampleAudio(
        input,
        from: sourceSampleRate,
        to: targetSampleRate
    )
    return MLXArray(resampled)
}

/// A streaming WAV writer that allows writing audio chunks incrementally to a file.
/// This is more memory efficient than collecting all audio in memory before writing.
public class StreamingWAVWriter {
    private let url: URL
    private let sampleRate: Double
    private var audioFile: AVAudioFile?
    private let format: AVAudioFormat
    private var reusableBuffer: AVAudioPCMBuffer?
    private var reusableCapacity: Int = 0
    public private(set) var framesWritten: Int = 0

    public init(url: URL, sampleRate: Double) throws {
        self.url = url
        self.sampleRate = sampleRate

        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"]
            )
        }
        self.format = format
        self.audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
    }

    /// Write a chunk of audio samples to the file.
    public func writeChunk(_ samples: [Float]) throws {
        guard let audioFile else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Audio file not initialized or already finalized"]
            )
        }
        let sampleCount = samples.count
        guard sampleCount > 0 else { return }

        let frameCount = AVAudioFrameCount(sampleCount)
        let buffer = try getReusableBuffer(minFrameCapacity: sampleCount)
        buffer.frameLength = frameCount

        guard let channelData = buffer.floatChannelData else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Failed to access output channel data"]
            )
        }
        samples.withUnsafeBufferPointer { src in
            if let srcBase = src.baseAddress {
                channelData[0].update(from: srcBase, count: sampleCount)
            }
        }

        try audioFile.write(from: buffer)
        framesWritten += sampleCount
    }

    /// Write a chunk of audio samples from an MLX array directly.
    /// Expects a 1D float32 tensor.
    public func writeChunk(_ samples: MLXArray) throws {
        let f32 = samples.dtype == .float32 ? samples : samples.asType(.float32)
        let sampleCount = f32.size
        guard sampleCount > 0 else { return }
        let data = f32.asData(access: .noCopyIfContiguous).data
        try writeChunkData(data, sampleCount: sampleCount)
    }

    /// Write a chunk of float32 samples from a byte buffer.
    public func writeChunkData(_ samplesData: Data, sampleCount: Int) throws {
        guard let audioFile else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Audio file not initialized or already finalized"]
            )
        }
        guard sampleCount >= 0 else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Invalid negative sample count: \(sampleCount)"]
            )
        }
        guard sampleCount > 0 else { return }

        let expectedBytes = sampleCount * MemoryLayout<Float>.size
        guard samplesData.count >= expectedBytes else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Insufficient sample data bytes (\(samplesData.count) < \(expectedBytes))"]
            )
        }

        let frameCount = AVAudioFrameCount(sampleCount)
        let buffer = try getReusableBuffer(minFrameCapacity: sampleCount)
        buffer.frameLength = frameCount

        guard let channelData = buffer.floatChannelData else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Failed to access output channel data"]
            )
        }
        samplesData.withUnsafeBytes { src in
            if let srcBase = src.baseAddress {
                memcpy(channelData[0], srcBase, expectedBytes)
            }
        }

        try audioFile.write(from: buffer)
        framesWritten += sampleCount
    }

    /// Finalize the WAV file and return the URL.
    /// After calling this method, no more chunks can be written.
    public func finalize() -> URL {
        audioFile = nil // Close the file by releasing the reference
        reusableBuffer = nil
        reusableCapacity = 0
        return url
    }

    private func getReusableBuffer(minFrameCapacity: Int) throws -> AVAudioPCMBuffer {
        if let reusableBuffer, reusableCapacity >= minFrameCapacity {
            return reusableBuffer
        }
        guard let newBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(minFrameCapacity)
        ) else {
            throw NSError(
                domain: "StreamingWAVWriter",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"]
            )
        }
        reusableBuffer = newBuffer
        reusableCapacity = minFrameCapacity
        return newBuffer
    }
}
