import AVFoundation
import Combine
import CryptoKit
import Foundation
import MLX
import MLXAudioCore
import MLXAudioTTS
import SwiftUI
#if os(macOS)
import AppKit
#endif

/// Codecaine File Manager Synthesis Scope for instant 1-second EPUB/Book start, unlimited reading & 2s export
enum SynthesisScope: String, CaseIterable, Identifiable {
    case fullDocument = "fullDocument"
    case fastSection = "fastSection"
    case activeChunk = "activeChunk"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fullDocument:
            return "Unlimited Continuous Stream (1s Start · Reads Entire Book)"
        case .fastSection:
            return "Fast 4-Chunk Section (~2s Start & Quick Section Export)"
        case .activeChunk:
            return "Active Chunk Only (~1.5s Instant Synthesis & Export)"
        }
    }

    var shortBadge: String {
        switch self {
        case .fullDocument: return "♾️ Unlimited (1s)"
        case .fastSection: return "⚡ 4-Chunk Section"
        case .activeChunk: return "🎯 1 Chunk (1.5s)"
        }
    }
}

/// Writes an IEEE Float32 RIFF/WAVE file incrementally and updates its 44-byte header in place after every chunk
/// so `audioURL` is a valid, playable, exportable WAV file on disk within ~1.0–1.5 seconds (after Chunk #1).
final class LivePlayableWAVWriter: @unchecked Sendable {
    let url: URL
    let sampleRate: UInt32
    private let lock = NSLock()
    private var fileHandle: FileHandle?
    private(set) var framesWritten: Int = 0

    init(url: URL, sampleRate: Double) throws {
        self.url = url
        self.sampleRate = UInt32(max(8000, min(192_000, Int(sampleRate.rounded()))))
        let header = Self.makeWAVHeader(sampleRate: self.sampleRate, dataBytes: 0)
        try header.write(to: url, options: .atomic)
        self.fileHandle = try FileHandle(forUpdating: url)
    }

    func writeChunk(_ samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        guard let fh = fileHandle else { return }
        try fh.seekToEnd()
        let pcmData = samples.withUnsafeBytes { Data($0) }
        try fh.write(contentsOf: pcmData)
        framesWritten += samples.count
        let dataBytes = UInt32(clamping: framesWritten * MemoryLayout<Float>.size)
        let header = Self.makeWAVHeader(sampleRate: sampleRate, dataBytes: dataBytes)
        try fh.seek(toOffset: 0)
        try fh.write(contentsOf: header)
    }

    @discardableResult
    func finalize() -> URL {
        lock.lock()
        defer { lock.unlock() }
        try? fileHandle?.synchronize()
        try? fileHandle?.close()
        fileHandle = nil
        return url
    }

    deinit {
        try? fileHandle?.close()
    }

    private static func makeWAVHeader(sampleRate: UInt32, dataBytes: UInt32) -> Data {
        var data = Data(capacity: 44)
        let riffSize = UInt32(clamping: UInt64(36) + UInt64(dataBytes))
        let byteRate = sampleRate * 4
        let blockAlign: UInt16 = 4
        let bitsPerSample: UInt16 = 32
        let formatTag: UInt16 = 3 // WAVE_FORMAT_IEEE_FLOAT
        let channels: UInt16 = 1
        let fmtChunkSize: UInt32 = 16

        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        withUnsafeBytes(of: riffSize.littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
        withUnsafeBytes(of: fmtChunkSize.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: formatTag.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: channels.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: sampleRate.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
        withUnsafeBytes(of: dataBytes.littleEndian) { data.append(contentsOf: $0) }
        return data
    }
}

/// Codecaine File Manager In-Memory + Disk Chunk Audio Cache (`fmPhonon.ts` `chunkAudioCache`)
enum PhononChunkCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var memoryCache: [String: [Float]] = [:]
    nonisolated(unsafe) private static var lruKeys: [String] = []
    private static let maxMemoryEntries = 256
    private static let diskWriteQueue = DispatchQueue(label: "VoicesApp.PhononChunkCache.DiskWrite", qos: .utility)

    private static var cacheDirectory: URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("VoicesAppPhononChunkCache", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func cacheKey(modelId: String, voiceParam: String?, refText: String?, pitch: Float, text: String) -> String {
        let raw = "v3|\(modelId)|\(voiceParam ?? "default")|\(refText ?? "")|\(String(format: "%.2f", pitch))|\(text)"
        let digest = SHA256.hash(data: Data(raw.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func get(key: String) -> [Float]? {
        lock.lock()
        if let hit = memoryCache[key], !hit.isEmpty {
            lock.unlock()
            return hit
        }
        lock.unlock()

        let fileURL = cacheDirectory.appendingPathComponent("\(key).pcm")
        guard let data = try? Data(contentsOf: fileURL),
              data.count >= MemoryLayout<Float>.size,
              data.count % MemoryLayout<Float>.size == 0 else {
            return nil
        }
        let count = data.count / MemoryLayout<Float>.size
        let samples: [Float] = data.withUnsafeBytes { raw in
            guard let base = raw.bindMemory(to: Float.self).baseAddress else { return [] }
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
        guard !samples.isEmpty else { return nil }
        lock.lock()
        memoryCache[key] = samples
        lruKeys.removeAll { $0 == key }
        lruKeys.append(key)
        lock.unlock()
        return samples
    }

    static func put(key: String, samples: [Float]) {
        guard !samples.isEmpty else { return }
        lock.lock()
        memoryCache[key] = samples
        lruKeys.removeAll { $0 == key }
        lruKeys.append(key)
        while lruKeys.count > maxMemoryEntries {
            let oldest = lruKeys.removeFirst()
            memoryCache.removeValue(forKey: oldest)
        }
        lock.unlock()

        let fileURL = cacheDirectory.appendingPathComponent("\(key).pcm")
        diskWriteQueue.async {
            let data = samples.withUnsafeBytes { Data($0) }
            try? data.write(to: fileURL, options: .atomic)
        }
    }
}

@MainActor
@Observable
class TTSViewModel {
    private static let defaultModelId = "nativ-community/phonon"
    static let presetModels: [(name: String, id: String)] = [
        ("Phonon MLX (Default · Freya/Harper/Sterling/Toby)", "nativ-community/phonon"),
        ("Phonon 40M Multilingual", "gradium/phonon-40m-multilingual"),
        ("Phonon 40M English", "gradium/phonon-40m-en"),
        ("Phonon 31M English", "gradium/phonon-31m-en"),
        ("Phonon 40M French", "gradium/phonon-40m-fr"),
        ("Phonon 40M German", "gradium/phonon-40m-de"),
        ("Phonon 40M Spanish", "gradium/phonon-40m-es"),
        ("Phonon 40M Portuguese", "gradium/phonon-40m-pt"),
        ("Maya-2 Native (20 Built-in Voices + Cloning)", "maya-research/maya2-native"),
        ("Audio8 TTS Realistic 0.6B (BF16)", "nativ-community/Audio8-TTS-Realistic-0.6b-bf16"),
        ("Audio8 TTS Realistic 0.6B (8-bit)", "nativ-community/Audio8-TTS-Realistic-0.6b-8bit"),
        ("Audio8 TTS Realistic 0.6B (4-bit)", "nativ-community/Audio8-TTS-Realistic-0.6b-4bit"),
        ("Audio8 0.1B Preview (BF16)", "mlx-community/Audio8-TTS-Preview-0.1b-bf16"),
        ("Audio8 0.6B Preview (BF16)", "mlx-community/Audio8-TTS-Preview-0.6b-bf16"),
        ("Fish Audio S2 Pro (8-bit)", "mlx-community/fish-audio-s2-pro-8bit"),
        ("Fish Audio S2 Pro (BF16)", "mlx-community/fish-audio-s2-pro-bf16"),
        ("Pocket TTS", "mlx-community/pocket-tts"),
        ("Qwen3-TTS 0.6B Base (8-bit)", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"),
        ("Qwen3-TTS 1.7B Base (BF16)", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
        ("Qwen3-TTS 0.6B CustomVoice (BF16)", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"),
        ("Qwen3-TTS 1.7B CustomVoice (BF16)", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"),
        ("Qwen3-TTS 1.7B VoiceDesign (BF16)", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"),
        ("Marvis TTS 250M (8-bit)", "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"),
        ("Kokoro 82M (BF16)", "mlx-community/Kokoro-82M-bf16"),
        ("KittenTTS Mini 0.8 (80M)", "mlx-community/kitten-tts-mini-0.8"),
        ("KittenTTS Micro 0.8 (40M)", "mlx-community/kitten-tts-micro-0.8"),
        ("KittenTTS Nano 0.8 (15M · Int8)", "KittenML/kitten-tts-nano-0.8-int8"),
        ("Orpheus 3B (BF16)", "mlx-community/orpheus-3b-0.1-ft-bf16"),
        ("Soprano 1.1 80M (BF16)", "mlx-community/Soprano-1.1-80M-bf16"),
        ("Soprano 80M (BF16)", "mlx-community/Soprano-80M-bf16"),
        ("VibeVoice Realtime 0.5B (8-bit)", "mlx-community/VibeVoice-Realtime-0.5B-8bit"),
        ("VibeVoice TTS 1.5B", "microsoft/VibeVoice-1.5B"),
        ("KugelAudio 7B Open (VibeVoice 7B)", "kugelaudio/kugelaudio-0-open"),
        ("Voxtral 4B TTS (4-bit)", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"),
        ("Voxtral 4B TTS (BF16)", "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"),
        ("Chatterbox TTS Multilingual (FP16)", "mlx-community/Chatterbox-TTS-fp16"),
        ("Chatterbox Turbo (FP16)", "mlx-community/chatterbox-turbo-fp16"),
        ("Chatterbox Turbo (8-bit)", "mlx-community/chatterbox-turbo-8bit"),
        ("Chatterbox Turbo (4-bit)", "mlx-community/chatterbox-turbo-4bit"),
        ("Echo TTS Base (44.1kHz)", "mlx-community/echo-tts-base"),
        ("VoxCPM2 2B 48kHz (BF16)", "mlx-community/VoxCPM2-bf16"),
        ("VoxCPM2 2B 48kHz (8-bit)", "mlx-community/VoxCPM2-8bit"),
        ("VoxCPM2 2B 48kHz (4-bit)", "mlx-community/VoxCPM2-4bit"),
        ("VoxCPM 1.5 (4-bit)", "mlx-community/VoxCPM1.5-4bit"),
        ("VoxCPM 1.5 (8-bit)", "mlx-community/VoxCPM1.5-8bit"),
        ("MeloTTS English", "mlx-community/MeloTTS-English-MLX"),
        ("Ming-3.0 TTS Tiny (4-bit)", "mlx-community/Ming-3.0-tts-tiny-4bit"),
        ("Ming-3.0 TTS 0.5B (BF16)", "mlx-community/Ming-3.0-tts-0.5B-bf16"),
        ("Ming Omni 16.8B-A3B (4-bit)", "mlx-community/Ming-omni-tts-16.8B-A3B-4bit"),
        ("Bark Small", "mlx-community/bark-small"),
        ("Confucius4-TTS (int8)", "mlx-community/Confucius4-TTS-mlx-int8"),
        ("Higgs Audio v2 3B (q8)", "mlx-community/higgs-audio-v2-3B-mlx-q8"),
        ("Higgs Audio v3 TTS 4B", "bosonai/higgs-tts-3-4b"),
        ("Breeze TTS 2 (4-bit)", "mlx-community/Breeze-TTS-2-mlx-4bit"),
        ("OuteTTS 1.0 0.6B (FP16)", "mlx-community/OuteTTS-1.0-0.6B-fp16"),
        ("OmniVoice (BF16)", "mlx-community/OmniVoice-bf16"),
        ("OmniVoice", "mlx-community/OmniVoice"),
        ("TADA 1B", "HumeAI/mlx-tada-1b"),
        ("TADA 3B", "HumeAI/mlx-tada-3b"),
        ("ZONOS2 (4-bit)", "amal-david/Zyphra-ZONOS2-4bit"),
        ("VyvoTTS EN Beta (4-bit)", "mlx-community/VyvoTTS-EN-Beta-4bit"),
        ("LongCat-AudioDiT 1B (4-bit)", "mlx-community/LongCat-AudioDiT-1B-4bit"),
        ("LongCat-AudioDiT 1B (BF16)", "mlx-community/LongCat-AudioDiT-1B-bf16"),
        ("MOSS-TTS Nano 100M", "mlx-community/MOSS-TTS-Nano-100M"),
        ("MOSS-TTSD v1.0 (Dialogue)", "OpenMOSS-Team/MOSS-TTSD-v1.0"),
        ("MOSS-TTS Local Transformer", "OpenMOSS-Team/MOSS-TTS-Local-Transformer"),
        ("MOSS-TTS", "OpenMOSS-Team/MOSS-TTS"),
        ("IndexTTS 1.5", "mlx-community/IndexTTS-1.5"),
        ("Irodori TTS v4.1 Small MeanFlow (8-bit)", "mlx-community/Irodori-TTS-v4.1-Small-MF-8bit"),
        ("Irodori TTS v4.1 Small (8-bit)", "mlx-community/Irodori-TTS-v4.1-Small-8bit"),
        ("Irodori TTS 600M v3 VoiceDesign (8-bit)", "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"),
        ("Spark-TTS 0.5B (BF16)", "mlx-community/Spark-TTS-0.5B-bf16"),
        ("Rumik OSS 1 (8-bit)", "rumik-ai/rumik-oss-1-mlx-8bit"),
        ("Dia 1.6B", "mlx-community/Dia-1.6B"),
    ]

    static var categorizedPresetModels: [(category: String, icon: String, models: [(name: String, id: String)])] {
        var fastDefault: [(name: String, id: String)] = []
        var voiceCloning: [(name: String, id: String)] = []
        var expressive: [(name: String, id: String)] = []
        var multilingual: [(name: String, id: String)] = []
        var lightweight: [(name: String, id: String)] = []

        for item in presetModels {
            let l = item.id.lowercased()
            if l.contains("phonon") {
                fastDefault.append(item)
            } else if l.contains("maya") || l.contains("audio8") || l.contains("fish-audio") || l.contains("chatterbox") || l.contains("echo-tts") || l.contains("voxcpm") || l.contains("indextts") || l.contains("spark-tts") || l.contains("pocket-tts") || l.contains("outetts") {
                voiceCloning.append(item)
            } else if l.contains("qwen3-tts") || l.contains("orpheus") || l.contains("dia-") || l.contains("moss-tts") || l.contains("irodori") || l.contains("higgs") || l.contains("tada") || l.contains("zonos") || l.contains("bark") || l.contains("kugelaudio") {
                expressive.append(item)
            } else if l.contains("voxtral") || l.contains("ming") || l.contains("omnivoice") || l.contains("breeze") || l.contains("confucius") || l.contains("melotts") {
                multilingual.append(item)
            } else {
                lightweight.append(item)
            }
        }

        return [
            ("Fast Default & Phonon (31M–40M)", "bolt.fill", fastDefault),
            ("Zero-Shot Voice Cloning & Realistic", "person.wave.2.fill", voiceCloning),
            ("Expressive, VoiceDesign & Dialogue", "theatermasks.fill", expressive),
            ("Multilingual & Omni TTS", "globe", multilingual),
            ("Ultra-Lightweight & Realtime", "hare.fill", lightweight),
        ]
    }

    func radCodeSnippet(language: String, text: String, port: UInt16 = 8080) -> String {
        let voiceId = selectedVoice?.voiceId ?? selectedVoice?.name ?? "freya"
        let rawSample = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine = rawSample.components(separatedBy: .newlines).first(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) ?? "Welcome to Codecaine MLX RAD AI Audio Studio running locally on Apple Silicon!"
        let sampleText = String(firstLine.prefix(200))
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let rateStr = String(format: "%.2f", playbackRate)

        switch language.lowercased() {
        case "python":
            return """
            from openai import OpenAI

            client = OpenAI(base_url="http://127.0.0.1:\(port)/v1", api_key="mlx-local")
            response = client.audio.speech.create(
                model="\(modelId)",
                voice="\(voiceId)",
                input="\(sampleText)",
                speed=\(rateStr),
                response_format="wav"
            )
            response.stream_to_file("speech_output.wav")
            print("Saved speech_output.wav")
            """
        case "typescript", "javascript", "ts", "js":
            return """
            import fs from "node:fs/promises";

            const res = await fetch("http://127.0.0.1:\(port)/v1/audio/speech", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model: "\(modelId)",
                voice: "\(voiceId)",
                input: "\(sampleText)",
                speed: \(rateStr),
                response_format: "wav"
              })
            });
            await fs.writeFile("speech_output.wav", Buffer.from(await res.arrayBuffer()));
            """
        case "swift":
            return """
            import Foundation

            let url = URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")!
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = #\"\"\"
            {"model":"\(modelId)","voice":"\(voiceId)","input":"\(sampleText)","speed":\(rateStr),"response_format":"wav"}
            \"\"\"#.data(using: .utf8)

            let (wavData, _) = try await URLSession.shared.data(for: req)
            try wavData.write(to: URL(fileURLWithPath: "speech_output.wav"))
            """
        default:
            return """
            curl -s http://127.0.0.1:\(port)/v1/audio/speech \\
              -H "Content-Type: application/json" \\
              -d '{"model":"\(modelId)","voice":"\(voiceId)","input":"\(sampleText)","speed":\(rateStr),"response_format":"wav"}' \\
              --output speech_output.wav && afplay speech_output.wav
            """
        }
    }
    private static let modelIdStorageKey = "VoicesApp.TTSViewModel.modelId"
    private static let maxTokensStorageKey = "VoicesApp.TTSViewModel.maxTokens"
    private static let temperatureStorageKey = "VoicesApp.TTSViewModel.temperature"
    private static let topPStorageKey = "VoicesApp.TTSViewModel.topP"
    private static let repetitionPenaltyStorageKey = "VoicesApp.TTSViewModel.repetitionPenalty"
    private static let voiceDescriptionStorageKey = "VoicesApp.TTSViewModel.voiceDescription"
    private static let useVoiceDesignStorageKey = "VoicesApp.TTSViewModel.useVoiceDesign"
    private static let enableChunkingStorageKey = "VoicesApp.TTSViewModel.enableChunking"
    private static let maxChunkLengthStorageKey = "VoicesApp.TTSViewModel.maxChunkLength"
    private static let splitPatternStorageKey = "VoicesApp.TTSViewModel.splitPattern"
    private static let streamingPlaybackStorageKey = "VoicesApp.TTSViewModel.streamingPlayback"

    var isLoading = false
    private var loadRequestToken = UUID()
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var tokensPerSecond: Double = 0

    // Generation parameters: model defaults + optional UI overrides
    private var defaultGenerationParameters = GenerateParameters(
        maxTokens: 4096,
        temperature: 0.6,
        topP: 0.8,
        repetitionPenalty: 1.3,
        repetitionContextSize: 20
    )
    private var maxTokensOverride: Int? = UserDefaults.standard.object(forKey: TTSViewModel.maxTokensStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: TTSViewModel.maxTokensStorageKey)
    }
    private var temperatureOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.temperatureStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.temperatureStorageKey)
    }
    private var topPOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.topPStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.topPStorageKey)
    }
    private var repetitionPenaltyOverride: Float? = UserDefaults.standard.object(forKey: TTSViewModel.repetitionPenaltyStorageKey).map { _ in
        UserDefaults.standard.float(forKey: TTSViewModel.repetitionPenaltyStorageKey)
    }

    var maxTokens: Int {
        get { maxTokensOverride ?? defaultMaxTokens }
        set {
            maxTokensOverride = (newValue == defaultMaxTokens) ? nil : newValue
            if let maxTokensOverride {
                UserDefaults.standard.set(maxTokensOverride, forKey: TTSViewModel.maxTokensStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.maxTokensStorageKey)
            }
        }
    }

    var temperature: Float {
        get { temperatureOverride ?? defaultGenerationParameters.temperature }
        set {
            let defaultValue = defaultGenerationParameters.temperature
            temperatureOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let temperatureOverride {
                UserDefaults.standard.set(temperatureOverride, forKey: TTSViewModel.temperatureStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.temperatureStorageKey)
            }
        }
    }

    var topP: Float {
        get { topPOverride ?? defaultGenerationParameters.topP }
        set {
            let defaultValue = defaultGenerationParameters.topP
            topPOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let topPOverride {
                UserDefaults.standard.set(topPOverride, forKey: TTSViewModel.topPStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.topPStorageKey)
            }
        }
    }
    
    var repetitionPenalty: Float {
        get { repetitionPenaltyOverride ?? defaultGenerationParameters.repetitionPenalty ?? 1.3 }
        set {
            let defaultValue = defaultGenerationParameters.repetitionPenalty ?? 1.3
            repetitionPenaltyOverride = abs(newValue - defaultValue) < 0.0001 ? nil : newValue
            if let repetitionPenaltyOverride {
                UserDefaults.standard.set(repetitionPenaltyOverride, forKey: TTSViewModel.repetitionPenaltyStorageKey)
            } else {
                UserDefaults.standard.removeObject(forKey: TTSViewModel.repetitionPenaltyStorageKey)
            }
        }
    }
    
    // Voice Design (for Qwen3-TTS VoiceDesign models)
    var voiceDescription: String = UserDefaults.standard.string(forKey: TTSViewModel.voiceDescriptionStorageKey) ?? "" {
        didSet {
            UserDefaults.standard.set(voiceDescription, forKey: TTSViewModel.voiceDescriptionStorageKey)
        }
    }
    var useVoiceDesign: Bool = UserDefaults.standard.bool(forKey: TTSViewModel.useVoiceDesignStorageKey) {
        didSet {
            UserDefaults.standard.set(useVoiceDesign, forKey: TTSViewModel.useVoiceDesignStorageKey)
        }
    }

    // Text chunking
    var enableChunking: Bool = UserDefaults.standard.object(forKey: TTSViewModel.enableChunkingStorageKey).map { _ in
        UserDefaults.standard.bool(forKey: TTSViewModel.enableChunkingStorageKey)
    } ?? true {
        didSet {
            UserDefaults.standard.set(enableChunking, forKey: TTSViewModel.enableChunkingStorageKey)
        }
    }
    var maxChunkLength: Int = UserDefaults.standard.object(forKey: TTSViewModel.maxChunkLengthStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: TTSViewModel.maxChunkLengthStorageKey)
    } ?? 520 {
        didSet {
            UserDefaults.standard.set(maxChunkLength, forKey: TTSViewModel.maxChunkLengthStorageKey)
        }
    }
    var synthesisScope: SynthesisScope = SynthesisScope(
        rawValue: UserDefaults.standard.string(forKey: "VoicesApp.TTSViewModel.synthesisScope") ?? ""
    ) ?? .fullDocument {
        didSet {
            UserDefaults.standard.set(synthesisScope.rawValue, forKey: "VoicesApp.TTSViewModel.synthesisScope")
        }
    }
    var splitPattern: String = UserDefaults.standard.string(forKey: TTSViewModel.splitPatternStorageKey) ?? "\n" {
        didSet {
            UserDefaults.standard.set(splitPattern, forKey: TTSViewModel.splitPatternStorageKey)
        }
    } // Can be regex like "\\n" or "[.!?]\\s+"

    // Streaming playback
    var streamingPlayback: Bool = UserDefaults.standard.object(forKey: TTSViewModel.streamingPlaybackStorageKey).map { _ in
        UserDefaults.standard.bool(forKey: TTSViewModel.streamingPlaybackStorageKey)
    } ?? true {
        didSet {
            UserDefaults.standard.set(streamingPlayback, forKey: TTSViewModel.streamingPlaybackStorageKey)
        }
    } // Play audio as chunks are generated

    // Model configuration
    var modelId: String = UserDefaults.standard.string(forKey: TTSViewModel.modelIdStorageKey) ?? TTSViewModel.defaultModelId {
        didSet {
            UserDefaults.standard.set(modelId, forKey: TTSViewModel.modelIdStorageKey)
            StudioUserPreferencesManager.shared.recordRecentTTSModel(modelId)
            if oldValue != modelId {
                refreshVoicesForCurrentModel(forceSelectFirst: true)
            }
        }
    }
    private var loadedModelId: String?

    // Auto-populated built-in voices and currently selected voice for the active model
    var builtInVoices: [Voice] = []
    private var isRestoringVoiceSelection: Bool = false
    var selectedVoice: Voice? {
        didSet {
            guard !isRestoringVoiceSelection, let v = selectedVoice else { return }
            StudioUserPreferencesManager.shared.saveSelectedVoice(v, for: modelId)
        }
    }

    // Session generation history & power-user controls
    var history: [GeneratedTake] = []
    var hfToken: String = ModelCacheHelper.hfToken {
        didSet {
            ModelCacheHelper.hfToken = hfToken
        }
    }
    var playbackRate: Float = UserDefaults.standard.object(forKey: "VoicesApp.TTSViewModel.playbackRate").map { _ in
        max(0.25, min(3.0, UserDefaults.standard.float(forKey: "VoicesApp.TTSViewModel.playbackRate")))
    } ?? 1.0 {
        didSet {
            audioPlayer.playbackRate = playbackRate
            UserDefaults.standard.set(playbackRate, forKey: "VoicesApp.TTSViewModel.playbackRate")
        }
    }

    // Codecaine File Manager Prosody, Chunking & Read-Along Karaoke Controls
    var pitchSemitones: Float = UserDefaults.standard.object(forKey: "VoicesApp.TTSViewModel.pitchSemitones").map { _ in
        UserDefaults.standard.float(forKey: "VoicesApp.TTSViewModel.pitchSemitones")
    } ?? 0.0 {
        didSet { UserDefaults.standard.set(pitchSemitones, forKey: "VoicesApp.TTSViewModel.pitchSemitones") }
    }
    var paragraphPauseMs: Int = UserDefaults.standard.object(forKey: "VoicesApp.TTSViewModel.paragraphPauseMs").map { _ in
        UserDefaults.standard.integer(forKey: "VoicesApp.TTSViewModel.paragraphPauseMs")
    } ?? 0 {
        didSet { UserDefaults.standard.set(paragraphPauseMs, forKey: "VoicesApp.TTSViewModel.paragraphPauseMs") }
    }
    var wordSyncOffsetMs: Double = UserDefaults.standard.object(forKey: "VoicesApp.TTSViewModel.wordSyncOffsetMs").map { _ in
        UserDefaults.standard.double(forKey: "VoicesApp.TTSViewModel.wordSyncOffsetMs")
    } ?? 0.0 {
        didSet { UserDefaults.standard.set(wordSyncOffsetMs, forKey: "VoicesApp.TTSViewModel.wordSyncOffsetMs") }
    }
    var readAlongTheme: ReadAlongTheme = ReadAlongTheme(
        rawValue: UserDefaults.standard.string(forKey: "VoicesApp.TTSViewModel.readAlongTheme") ?? ""
    ) ?? .midnight {
        didSet { UserDefaults.standard.set(readAlongTheme.rawValue, forKey: "VoicesApp.TTSViewModel.readAlongTheme") }
    }
    var autoReadAlong: Bool = UserDefaults.standard.object(forKey: "VoicesApp.TTSViewModel.autoReadAlong").map { _ in
        UserDefaults.standard.bool(forKey: "VoicesApp.TTSViewModel.autoReadAlong")
    } ?? true {
        didSet { UserDefaults.standard.set(autoReadAlong, forKey: "VoicesApp.TTSViewModel.autoReadAlong") }
    }
    var activeChunkIndex: Int = 0
    var synthesizingChunkIndex: Int = 0
    var totalChunkCount: Int = 0
    var synthesizedStartChunkIndex: Int = 0
    var synthesizedEndChunkIndex: Int? = nil
    var synthesizedWordTimings: [WordTiming] = []
    var synthesizedChunkFrameRanges: [Int: Range<Int>] = [:]
    var lastSynthesizedText: String = ""

    // Audio player state (manually synced from AudioPlayerManager)
    var isPlaying: Bool = false
    var isSpeaking: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: SpeechGenerationModel?
    private let audioPlayer = AudioPlayer()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    private var defaultMaxTokens: Int {
        max(defaultGenerationParameters.maxTokens ?? 4096, 4096)
    }

    var isModelLoaded: Bool {
        model != nil && loadedModelId == modelId
    }

    init() {
        if !UserDefaults.standard.bool(forKey: "VoicesApp.TTSViewModel.migratedDefaultPhononV2") {
            UserDefaults.standard.set(true, forKey: "VoicesApp.TTSViewModel.migratedDefaultPhononV2")
            modelId = Self.defaultModelId
        } else if !ModelCacheHelper.isModelCached(modelId),
                  let firstCached = Self.presetModels.first(where: { ModelCacheHelper.isModelCached($0.id) }) {
            modelId = firstCached.id
        }
        if !UserDefaults.standard.bool(forKey: "VoicesApp.TTSViewModel.migratedChunk520V1") {
            UserDefaults.standard.set(true, forKey: "VoicesApp.TTSViewModel.migratedChunk520V1")
            if maxChunkLength <= 300 {
                maxChunkLength = 520
            }
        }
        if !UserDefaults.standard.bool(forKey: "VoicesApp.TTSViewModel.migratedFastPhononEngineV5") {
            UserDefaults.standard.set(true, forKey: "VoicesApp.TTSViewModel.migratedFastPhononEngineV5")
            if ModelCacheHelper.isModelCached(Self.defaultModelId) {
                modelId = Self.defaultModelId
            }
            synthesisScope = .fullDocument
            paragraphPauseMs = 0
        }
        setupAudioPlayerObservers()
        audioPlayer.playbackRate = playbackRate
        refreshVoicesForCurrentModel(forceSelectFirst: true)
        let persistedTakes = StudioDocumentSessionManager.shared.loadPersistedTakes()
        if !persistedTakes.isEmpty {
            self.history = persistedTakes.map { item in
                GeneratedTake(
                    id: item.record.id,
                    text: item.record.textSnippet,
                    modelId: item.record.modelName,
                    voiceName: item.record.voiceName,
                    audioURL: item.url,
                    duration: item.record.duration,
                    tokensPerSecond: item.record.rtf,
                    createdAt: item.record.createdAt
                )
            }
        }
        NotificationCenter.default.publisher(for: .modelCacheDidChange)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self,
                      let deletedId = note.userInfo?["deletedModelId"] as? String else { return }
                if self.loadedModelId == deletedId {
                    self.model = nil
                    self.loadedModelId = nil
                    Memory.clearCache()
                }
            }
            .store(in: &cancellables)
    }

    func refreshVoicesForCurrentModel(forceSelectFirst: Bool = false) {
        var dynamicNames: [String] = []
        if let loadedModel = model {
            if let kokoro = loadedModel as? KokoroModel {
                dynamicNames = kokoro.availableVoices()
            } else if let vibe = loadedModel as? VibeVoiceRealtimeModel {
                dynamicNames = vibe.availableVoices
            } else if let voxtral = loadedModel as? VoxtralTTSModel {
                dynamicNames = voxtral.availableVoices
            } else if let rumik = loadedModel as? RumikModel {
                dynamicNames = rumik.voices
            } else if let maya2 = loadedModel as? Maya2Model {
                dynamicNames = maya2.voices
            }
        }
        let voices = Voice.builtInVoices(for: modelId, dynamicVoiceNames: dynamicNames)
        builtInVoices = voices
        let customVoices = Voice.loadCustomVoices()
        let currentStillValid: Bool = {
            guard let cur = selectedVoice else { return false }
            if cur.isCustom && customVoices.contains(where: { $0.voiceKey == cur.voiceKey }) {
                return true
            }
            return voices.contains(where: { $0.name == cur.name && $0.voiceId == cur.voiceId })
        }()

        if forceSelectFirst || !currentStillValid {
            isRestoringVoiceSelection = true
            defer { isRestoringVoiceSelection = false }
            if let restored = StudioUserPreferencesManager.shared.resolveSavedVoice(for: modelId, availableVoices: voices) {
                selectedVoice = restored
            } else {
                selectedVoice = voices.first
            }
        }
    }

    func applyStudioProfile(_ profile: StudioUserProfile) {
        if modelId != profile.modelId {
            modelId = profile.modelId
        }
        let customVoices = Voice.loadCustomVoices()
        let allCandidates = customVoices + builtInVoices
        if let matched = allCandidates.first(where: {
            $0.name.caseInsensitiveCompare(profile.voiceName) == .orderedSame ||
            (($0.voiceId ?? "").caseInsensitiveCompare(profile.voiceId ?? "") == .orderedSame && !(profile.voiceId ?? "").isEmpty)
        }) {
            selectedVoice = matched
        }
        playbackRate = profile.playbackRate
        pitchSemitones = profile.pitchSemitones
        paragraphPauseMs = profile.paragraphPauseMs
        maxChunkLength = profile.maxChunkLength
        if let theme = ReadAlongTheme(rawValue: profile.themeRaw) {
            readAlongTheme = theme
        }
        if let mastering = MasteringPreset(rawValue: profile.masteringPresetRaw) {
            StudioMasteringManager.shared.applyPreset(mastering)
        }
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$isSpeaking
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isSpeaking = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                guard let self else { return }
                self.currentTime = value
                if value > 0.02, !self.synthesizedWordTimings.isEmpty,
                   let wid = WordTimingEngine.activeWordIndex(
                       in: self.synthesizedWordTimings,
                       at: value,
                       offsetMs: self.wordSyncOffsetMs
                   ),
                   wid >= 0, wid < self.synthesizedWordTimings.count {
                    let spokenChunk = self.synthesizedWordTimings[wid].chunkIndex
                    if self.activeChunkIndex != spokenChunk {
                        self.activeChunkIndex = spokenChunk
                    }
                }
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.duration = value
            }
            .store(in: &cancellables)
    }

    func loadModel() async {
        // Skip if same model already loaded
        let requestedModelId = modelId
        guard model == nil || loadedModelId != requestedModelId else {
            errorMessage = nil
            return
        }

        let requestToken = UUID()
        loadRequestToken = requestToken
        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            let loadedModel = try await TTS.loadModel(
                modelRepo: requestedModelId,
                hfToken: ModelCacheHelper.effectiveHFToken
            )
            guard loadRequestToken == requestToken, modelId == requestedModelId else { return }
            model = loadedModel
            defaultGenerationParameters = loadedModel.defaultGenerationParameters
            loadedModelId = requestedModelId
            errorMessage = nil
            DownloadedModelsManager.shared.refresh()
            refreshVoicesForCurrentModel(forceSelectFirst: false)
            generationProgress = "" // Clear progress on success
        } catch {
            guard loadRequestToken == requestToken, !Task.isCancelled else { return }
            if model == nil || loadedModelId != requestedModelId {
                errorMessage = "Failed to load model: \(error.localizedDescription)"
            }
            generationProgress = ""
        }

        if loadRequestToken == requestToken {
            isLoading = false
        }
    }

    func reloadModel() async {
        // Unload current model and clear GPU memory
        model = nil
        loadedModelId = nil
        Memory.clearCache()

        await loadModel()
    }

    func resetGenerationParameterOverrides() {
        maxTokensOverride = nil
        temperatureOverride = nil
        topPOverride = nil
        repetitionPenaltyOverride = nil
        UserDefaults.standard.removeObject(forKey: TTSViewModel.maxTokensStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.temperatureStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.topPStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.repetitionPenaltyStorageKey)
    }

    func resetSettingsToDefaults() {
        modelId = TTSViewModel.defaultModelId
        resetGenerationParameterOverrides()
        useVoiceDesign = false
        voiceDescription = ""
        enableChunking = true
        maxChunkLength = 520
        synthesisScope = .fullDocument
        splitPattern = "\n"
        streamingPlayback = true
        pitchSemitones = 0.0
        paragraphPauseMs = 0
        wordSyncOffsetMs = 0.0
        readAlongTheme = .midnight
        autoReadAlong = true

        UserDefaults.standard.removeObject(forKey: TTSViewModel.modelIdStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.voiceDescriptionStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.useVoiceDesignStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.enableChunkingStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.maxChunkLengthStorageKey)
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.synthesisScope")
        UserDefaults.standard.removeObject(forKey: TTSViewModel.splitPatternStorageKey)
        UserDefaults.standard.removeObject(forKey: TTSViewModel.streamingPlaybackStorageKey)
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.pitchSemitones")
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.paragraphPauseMs")
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.wordSyncOffsetMs")
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.readAlongTheme")
        UserDefaults.standard.removeObject(forKey: "VoicesApp.TTSViewModel.autoReadAlong")
        if loadedModelId != modelId {
            Task { await loadModel() }
        }
    }

    private func effectiveGenerationParameters() -> GenerateParameters {
        var parameters = defaultGenerationParameters
        if let maxTokensOverride {
            parameters.maxTokens = maxTokensOverride
        } else {
            parameters.maxTokens = max(parameters.maxTokens ?? 4096, 4096)
        }
        if let temperatureOverride {
            parameters.temperature = temperatureOverride
        }
        if let topPOverride {
            parameters.topP = topPOverride
        }
        if let repetitionPenaltyOverride {
            parameters.repetitionPenalty = repetitionPenaltyOverride
        }
        return parameters
    }

    /// Split arbitrarily large text into chunks with paragraph metadata using Codecaine File Manager's HierarchicalChunker
    func chunkTextWithMetadata(_ text: String, maxCharsOverride: Int? = nil) -> [PhononTextChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard enableChunking else {
            let words = trimmed.split { $0.isWhitespace || $0.isNewline }.count
            return [PhononTextChunk(id: 0, index: 0, paragraphIndex: 0, text: trimmed, charCount: trimmed.count, wordCount: words)]
        }
        let effectiveMax = max(180, maxCharsOverride ?? maxChunkLength)
        var normalizedText = text
        let trimmedPattern = splitPattern.trimmingCharacters(in: .whitespaces)
        if !trimmedPattern.isEmpty && trimmedPattern != "\n" && trimmedPattern != "\\n" {
            if let regex = try? NSRegularExpression(pattern: trimmedPattern) {
                let nsRange = NSRange(normalizedText.startIndex..., in: normalizedText)
                normalizedText = regex.stringByReplacingMatches(in: normalizedText, range: nsRange, withTemplate: "\n\n")
            }
        }
        let phononChunks = HierarchicalChunker.splitText(normalizedText, maxChars: effectiveMax, mergeShortParagraphs: true)
        if phononChunks.isEmpty {
            let words = trimmed.split { $0.isWhitespace || $0.isNewline }.count
            return [PhononTextChunk(id: 0, index: 0, paragraphIndex: 0, text: trimmed, charCount: trimmed.count, wordCount: words)]
        }
        return phononChunks
    }

    /// Split arbitrarily large text into chunks using Codecaine File Manager's HierarchicalChunker
    func chunkText(_ text: String) -> [String] {
        return chunkTextWithMetadata(text).map(\.text)
    }

    /// Start synthesis in a cancellable task (optionally starting from a specific chunk index or scope override)
    func startSynthesis(
        text: String,
        voice: Voice? = nil,
        startChunkIndex: Int = 0,
        scopeOverride: SynthesisScope? = nil
    ) {
        generationTask?.cancel()
        audioPlayer.stop()
        generationTask = Task {
            await synthesize(
                text: text,
                voice: voice,
                startChunkIndex: startChunkIndex,
                scopeOverride: scopeOverride
            )
        }
    }

    /// Codecaine File Manager 2-Second Silent Synthesis & Export (no sound played, works without pressing Generate Speech)
    func quickExportActiveChunk(
        text: String,
        voice: Voice?,
        chunkIndex: Int? = nil,
        format: AudioExportFormat = .wav,
        scope: SynthesisScope = .activeChunk
    ) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else {
            errorMessage = "Enter or import text before exporting audio"
            return
        }
        let targetChunk = chunkIndex ?? activeChunkIndex
        let rawDocStem = URL(fileURLWithPath: StudioDocumentSessionManager.shared.currentDocumentTitle)
            .deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: "[^A-Za-z0-9_\\-]+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let docPrefix = rawDocStem.isEmpty ? "studio_doc" : String(rawDocStem.prefix(28)).lowercased()
        let baseName = scope == .fullDocument
            ? "\(docPrefix)_full"
            : (scope == .fastSection ? "\(docPrefix)_section_from_\(targetChunk + 1)" : "\(docPrefix)_chunk_\(targetChunk + 1)")
        saveAudioInFormat(
            format,
            from: nil,
            baseName: baseName,
            fallbackText: cleaned,
            voice: voice ?? selectedVoice,
            chunkIndex: targetChunk,
            scopeOverride: scope
        )
    }

    func synthesize(
        text: String,
        voice: Voice? = nil,
        startChunkIndex: Int = 0,
        playAudio: Bool = true,
        recordHistory: Bool = true,
        scopeOverride: SynthesisScope? = nil
    ) async {
        guard let model else {
            errorMessage = "Model not loaded"
            return
        }

        let cleanedInput = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedInput.isEmpty else {
            errorMessage = "Please enter text to synthesize"
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = "Starting generation..."
        tokensPerSecond = 0

        do {
            // Load reference audio if this is a cloned voice and the model supports reference audio
            var referenceAudio: MaterializedArray?
            var refText: String?
            var fishPreset: Voice.FishS2VoicePreset?

            let isFishModel = modelId.lowercased().contains("fish-audio")
                || modelId.lowercased().contains("fish_speech")
                || modelId.lowercased().contains("fishspeech")

            if let voice, voice.isClonedVoice,
               Voice.supportsReferenceCloning(for: modelId),
               let audioURL = voice.audioFileURL,
               let transcription = voice.transcription {
                generationProgress = "Loading reference audio..."
                let (_, audioData) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
                referenceAudio = audioData
                refText = transcription
            } else if isFishModel, let voice,
                      let preset = Voice.fishS2Preset(for: voice.voiceId ?? voice.name) {
                fishPreset = preset
                if let resourceName = preset.resourceName,
                   let refURL = Bundle.main.url(forResource: resourceName, withExtension: "wav") {
                    let (_, audioData) = try loadAudioArray(from: refURL, sampleRate: model.sampleRate)
                    referenceAudio = audioData
                    refText = preset.refText
                }
            }

            // Split text into chunks with paragraph metadata (unlimited length hierarchical chunker)
            let allChunks = chunkTextWithMetadata(text)
            let safeStart = min(max(0, startChunkIndex), max(0, allChunks.count - 1))
            let effectiveScope = scopeOverride ?? synthesisScope
            let maxWindowCount: Int
            switch effectiveScope {
            case .activeChunk:
                maxWindowCount = 1
            case .fastSection:
                maxWindowCount = allChunks.count > 4 ? 4 : allChunks.count
            case .fullDocument:
                // Unlimited length reading: synthesize Chunk #1 in ~1s and stream all remaining chunks continuously!
                maxWindowCount = allChunks.count
            }

            let chunks = Array(allChunks.dropFirst(safeStart).prefix(maxWindowCount))
            let plannedEndChunk = safeStart + max(0, chunks.count - 1)
            totalChunkCount = allChunks.count
            activeChunkIndex = safeStart
            synthesizingChunkIndex = safeStart
            synthesizedStartChunkIndex = safeStart
            synthesizedEndChunkIndex = plannedEndChunk
            lastSynthesizedText = cleanedInput
            let sampleRate = Double(model.sampleRate)

            // Create LivePlayableWAVWriter — updates WAV header in place after every chunk so `audioURL` is ready after Chunk #1 (~1.0s)
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            let wavWriter = try LivePlayableWAVWriter(url: tempURL, sampleRate: sampleRate)

            // Streaming playback — model yields audio chunks progressively with 0ms gapless queueing
            let useStreaming = streamingPlayback && playAudio
            if useStreaming {
                audioPlayer.startStreaming(sampleRate: sampleRate)
            }

            let totalPitchShift = pitchSemitones + (fishPreset?.pitchSemitones ?? 0)
            let pauseFrames = max(0, Int((Double(paragraphPauseMs) / 1000.0) * sampleRate))

            // Initialize live waveform-aligned word timings across all chunks so Read-Along tracks exact spoken audio speed
            let masterWords = WordTimingEngine.extractMasterWords(
                text: text,
                maxChunkChars: max(180, maxChunkLength),
                enableChunking: enableChunking
            )
            let masterWordsByChunk = Dictionary(grouping: masterWords, by: \.chunkIndex)
            var workingTimings = masterWords.map {
                WordTiming(id: $0.id, word: $0.word, start: 0, end: 0, chunkIndex: $0.chunkIndex, paragraphIndex: $0.paragraphIndex)
            }
            synthesizedWordTimings = workingTimings
            synthesizedChunkFrameRanges = [:]
            var cumulativeFramesWritten = 0
            let masteringSnapshot = StudioMasteringManager.shared.snapshot

            // Determine voice parameter once outside the hot loop
            let voiceParam: String?
            if useVoiceDesign && !voiceDescription.isEmpty {
                voiceParam = voiceDescription
            } else if let voice {
                if voice.isClonedVoice {
                    voiceParam = nil
                } else if voice.isBuiltIn {
                    voiceParam = voice.voiceId
                } else {
                    voiceParam = voice.voiceId ?? voice.name
                }
            } else {
                voiceParam = nil
            }
            let activeModelId = modelId
            let generationParameters = effectiveGenerationParameters()
            let inlineTag = fishPreset?.inlineTag
            let voiceCacheId = voiceParam ?? voice?.name

            Memory.cacheLimit = 512 * 1024 * 1024 // 512MB cache limit

            for (offset, chunkMeta) in chunks.enumerated() {
                let chunk = chunkMeta.text
                let globalIndex = safeStart + offset
                synthesizingChunkIndex = globalIndex
                if offset == 0 {
                    activeChunkIndex = safeStart
                }
                try Task.checkCancellation()

                let chunkWords = masterWordsByChunk[globalIndex] ?? []
                let chunkStartSec = Double(cumulativeFramesWritten) / sampleRate
                let isParagraphChangeToNext: Bool = {
                    guard offset + 1 < chunks.count else { return false }
                    return chunks[offset + 1].paragraphIndex != chunkMeta.paragraphIndex
                }()
                let interChunkPauseFrames = isParagraphChangeToNext ? pauseFrames : 0

                let castResolved = CharacterCastingManager.shared.resolveChunkCast(
                    chunk,
                    defaultVoiceParam: voiceParam,
                    defaultPitch: totalPitchShift
                )
                let chunkVoiceParam = castResolved.voiceParam
                let chunkPitchShift = castResolved.pitch
                let effectiveChunk: String
                let spokenChunk = SpeechPronunciationLexicon.shared.apply(to: castResolved.spokenText)
                if let tag = inlineTag, !spokenChunk.trimmingCharacters(in: .whitespaces).hasPrefix("[") {
                    effectiveChunk = "\(tag) \(spokenChunk)"
                } else {
                    effectiveChunk = spokenChunk
                }

                let cacheKey = PhononChunkCache.cacheKey(
                    modelId: activeModelId,
                    voiceParam: chunkVoiceParam ?? voiceCacheId,
                    refText: refText,
                    pitch: chunkPitchShift,
                    text: effectiveChunk
                )

                if allChunks.count > 1 {
                    let castBadge = castResolved.characterName.map { " [\($0)]" } ?? ""
                    if offset == 0 {
                        generationProgress = "Synthesizing §\(globalIndex + 1) of \(allChunks.count)\(castBadge) (instant 1s start & export)..."
                    } else {
                        generationProgress = "Speaking §\(activeChunkIndex + 1) · Prefetching §\(globalIndex + 1)/\(plannedEndChunk + 1)\(castBadge) (gapless)..."
                    }
                }

                // Yield to keep @MainActor (AudioPlayer callbacks, timers, 120Hz Teleprompter UI) responsive
                await Task.yield()

                var rawSamples: [Float] = []
                if let cachedSamples = PhononChunkCache.get(key: cacheKey) {
                    rawSamples = cachedSamples
                } else {
                    var tokenTick = 0
                    for try await event in model.generateStream(
                        text: effectiveChunk,
                        voice: chunkVoiceParam,
                        refAudio: referenceAudio,
                        refText: refText,
                        language: nil,
                        generationParameters: generationParameters,
                        streamingInterval: 60.0
                    ) {
                        try Task.checkCancellation()
                        switch event {
                        case .token:
                            tokenTick += 1
                            if tokenTick % 20 == 0 {
                                await Task.yield()
                            }
                        case .progress:
                            break
                        case .info(let info):
                            tokensPerSecond = info.tokensPerSecond
                        case .audio(let audioData):
                            autoreleasepool {
                                let samples = audioData.asArray(Float.self)
                                rawSamples.append(contentsOf: samples)
                            }
                        }
                    }

                    // Offload pitch shift, silence trimming, and cache write to a background thread
                    if !rawSamples.isEmpty {
                        let unshifted = rawSamples
                        rawSamples = await Task.detached(priority: .userInitiated) {
                            var processed = unshifted
                            if abs(chunkPitchShift) > 0.05 {
                                processed = Voice.applyPitchShift(processed, semitones: chunkPitchShift, sampleRate: sampleRate)
                            }
                            processed = SpeechAudioTrimmer.trimSilence(processed, sampleRate: sampleRate)
                            PhononChunkCache.put(key: cacheKey, samples: processed)
                            return processed
                        }.value
                    }
                }

                // Offload waveform DP word alignment, Master Bus DSP, and WAV file writing off @MainActor
                let finalChunkSamples = rawSamples
                let chunkResult: (samplesToWrite: [Float], alignedTimings: [WordTiming]) = await Task.detached(priority: .userInitiated) {
                    guard !finalChunkSamples.isEmpty else {
                        return ([], [])
                    }
                    let aligned: [WordTiming]
                    if !chunkWords.isEmpty {
                        aligned = WordTimingEngine.alignWordsToPCM(
                            words: chunkWords,
                            samples: finalChunkSamples,
                            sampleRate: sampleRate,
                            startOffsetSec: chunkStartSec
                        )
                    } else {
                        aligned = []
                    }

                    var samplesToWrite = StudioMasteringManager.processSamples(
                        finalChunkSamples,
                        sampleRate: sampleRate,
                        config: masteringSnapshot,
                        preserveExactLength: true
                    )
                    if interChunkPauseFrames > 0 {
                        samplesToWrite.append(contentsOf: repeatElement(Float(0), count: interChunkPauseFrames))
                    }
                    try? wavWriter.writeChunk(samplesToWrite)
                    return (samplesToWrite, aligned)
                }.value

                if !chunkResult.samplesToWrite.isEmpty {
                    for t in chunkResult.alignedTimings where t.id >= 0 && t.id < workingTimings.count {
                        workingTimings[t.id] = t
                    }
                    self.synthesizedWordTimings = workingTimings
                    let chunkStartFrame = cumulativeFramesWritten
                    cumulativeFramesWritten += chunkResult.samplesToWrite.count
                    self.synthesizedChunkFrameRanges[globalIndex] = chunkStartFrame..<cumulativeFramesWritten
                    self.synthesizedEndChunkIndex = globalIndex

                    // Publish playable & exportable WAV URL immediately after Chunk #1 (~1.0s!)
                    let liveDuration = Double(wavWriter.framesWritten) / sampleRate
                    if self.audioURL != tempURL {
                        self.audioURL = tempURL
                    }
                    if self.duration < liveDuration {
                        self.duration = liveDuration
                    }

                    if useStreaming {
                        audioPlayer.scheduleAudioChunk(chunkResult.samplesToWrite, withCrossfade: true)
                    }
                }

                // Only clear GPU cache periodically so Metal buffers stay warm for instant chunk-to-chunk synthesis
                if (offset + 1) % 10 == 0 {
                    Memory.clearCache()
                }
            }

            // Finalize the WAV file
            let finalURL = wavWriter.finalize()

            guard wavWriter.framesWritten > 0 else {
                throw NSError(
                    domain: "TTSViewModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No audio generated"]
                )
            }

            Memory.clearCache()

            let clipDuration = Double(wavWriter.framesWritten) / sampleRate
            audioURL = finalURL
            duration = clipDuration
            generationProgress = "" // Clear progress

            Task.detached(priority: .utility) {
                if let report = StudioMasteringManager.analyzeWAVFile(at: finalURL) {
                    await MainActor.run {
                        StudioMasteringManager.shared.lastReport = report
                    }
                }
            }

            // Record in session take history (and persist to Application Support so takes survive app restarts)
            if recordHistory {
                let takeId = UUID()
                let takeText = chunks.map(\.text).joined(separator: "\n\n")
                let persistedURL = StudioDocumentSessionManager.shared.persistTake(
                    id: takeId,
                    textSnippet: String(takeText.prefix(600)),
                    voiceName: voice?.name ?? "Default",
                    modelName: modelId,
                    duration: clipDuration,
                    rtf: tokensPerSecond,
                    sourceAudioURL: finalURL
                )
                let take = GeneratedTake(
                    id: takeId,
                    text: takeText,
                    modelId: modelId,
                    voiceName: voice?.name ?? "Default",
                    audioURL: persistedURL,
                    duration: clipDuration,
                    tokensPerSecond: tokensPerSecond
                )
                history.insert(take, at: 0)
            }

            // Finalize playback state once generation completes.
            if playAudio {
                if useStreaming {
                    audioPlayer.onDidFinishStreaming = { [weak self] in
                        guard let self else { return }
                        self.audioPlayer.loadAudio(from: finalURL)
                    }
                    audioPlayer.finishStreamingInput()
                } else {
                    audioPlayer.loadAudio(from: finalURL)
                    audioPlayer.play()
                }
            }

        } catch is CancellationError {
            // User cancelled - clean up silently
            audioPlayer.stop()
            Memory.clearCache()
            generationProgress = ""
        } catch {
            if !Task.isCancelled && (error as NSError).code != NSURLErrorCancelled {
                errorMessage = "Generation failed: \(error.localizedDescription)"
            }
            generationProgress = ""
        }

        isGenerating = false
    }

    /// Applies the active Studio Master Bus DSP chain (and optional ACX head/tail room-tone floor) to the current WAV file in place.
    func applyMasterBusInPlaceToCurrentAudio() {
        let sourceURL: URL? = audioURL ?? Voice.bundledStudioSamples().first?.url
        guard let targetURL = sourceURL else {
            errorMessage = "Synthesize or load an audio take before applying the Master Bus"
            return
        }
        let config = StudioMasteringManager.shared.snapshot
        let hasWordTimings = !synthesizedWordTimings.isEmpty
        generationProgress = "Mastering audio (\(config.preset.shortBadge))..."

        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let (rawSamples, sr) = try StudioMasteringManager.readPCM16WAVSamples(from: targetURL)
                let mastered = StudioMasteringManager.processSamples(
                    rawSamples,
                    sampleRate: Double(sr),
                    config: config,
                    preserveExactLength: hasWordTimings && !config.acxHeadTailRoomTone
                )
                let outURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mastered_\(UUID().uuidString).wav")
                try StudioMasteringManager.writePCM16WAVSamples(mastered, sampleRate: sr, to: outURL)
                let report = StudioMasteringManager.analyzeSamples(mastered, sampleRate: Double(sr))

                await MainActor.run {
                    guard let self else { return }
                    if !hasWordTimings || !config.acxHeadTailRoomTone {
                        // Timings unchanged
                    } else if config.preset == .acxAudiobook && config.acxHeadTailRoomTone {
                        // Shift word timings forward by +0.65s ACX head room tone
                        self.synthesizedWordTimings = self.synthesizedWordTimings.map { t in
                            WordTiming(
                                id: t.id,
                                word: t.word,
                                start: t.start + 0.65,
                                end: t.end + 0.65,
                                chunkIndex: t.chunkIndex,
                                paragraphIndex: t.paragraphIndex
                            )
                        }
                    }
                    self.audioURL = outURL
                    self.duration = report.durationSec
                    self.audioPlayer.loadAudio(from: outURL)
                    StudioMasteringManager.shared.lastReport = report
                    StudioMasteringManager.shared.statusMessage = "✅ Mastered (\(String(format: "%.1f", report.integratedLUFS)) LUFS · \(String(format: "%.1f", report.rmsDBFS)) dB RMS · Peak \(String(format: "%.1f", report.truePeakDBTP)) dBTP)"
                    self.generationProgress = "Master Bus Applied (\(config.preset.shortBadge))"
                    Task { @MainActor in
                        try? await Task.sleep(for: .seconds(2.2))
                        if self.generationProgress.hasPrefix("Master Bus Applied") {
                            self.generationProgress = ""
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    self?.errorMessage = "Mastering failed: \(error.localizedDescription)"
                    self?.generationProgress = ""
                }
            }
        }
    }

    /// Surgically re-synthesizes a single chunk (`chunkIndex`) and splices it into the existing master WAV in-place with 12ms cosine crossfades, shifting downstream word timings automatically.
    func retakeAndSpliceChunk(fullText: String, chunkIndex: Int, voice: Voice?) {
        let allChunks = chunkTextWithMetadata(fullText)
        guard !allChunks.isEmpty else { return }
        let safeIndex = min(max(0, chunkIndex), allChunks.count - 1)

        // If we don't have a multi-chunk master WAV or frame range yet, run a direct synthesis of the active chunk
        guard let existingURL = audioURL,
              let oldRange = synthesizedChunkFrameRanges[safeIndex],
              let model = model else {
            startSynthesis(text: fullText, voice: voice, startChunkIndex: safeIndex, scopeOverride: .activeChunk)
            return
        }

        generationTask?.cancel()
        audioPlayer.pause()
        isGenerating = true
        generationProgress = "Surgical Retake & Splice for §\(safeIndex + 1)..."

        generationTask = Task {
            do {
                let sampleRate = Double(model.sampleRate)
                let chunkMeta = allChunks[safeIndex]
                var referenceAudio: MaterializedArray?
                var refText: String?
                var fishPreset: Voice.FishS2VoicePreset?

                let isFishModel = modelId.lowercased().contains("fish-audio")
                    || modelId.lowercased().contains("fish_speech")
                    || modelId.lowercased().contains("fishspeech")

                if let voice, voice.isClonedVoice,
                   Voice.supportsReferenceCloning(for: modelId),
                   let audioURL = voice.audioFileURL,
                   let transcription = voice.transcription {
                    let (_, audioData) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
                    referenceAudio = audioData
                    refText = transcription
                } else if isFishModel, let voice,
                          let preset = Voice.fishS2Preset(for: voice.voiceId ?? voice.name) {
                    fishPreset = preset
                    if let resourceName = preset.resourceName,
                       let refURL = Bundle.main.url(forResource: resourceName, withExtension: "wav") {
                        let (_, audioData) = try loadAudioArray(from: refURL, sampleRate: model.sampleRate)
                        referenceAudio = audioData
                        refText = preset.refText
                    }
                }
                let totalPitchShift: Float = pitchSemitones + (fishPreset?.pitchSemitones ?? 0)

                let defaultVoiceParam: String? = {
                    if useVoiceDesign && !voiceDescription.isEmpty { return voiceDescription }
                    guard let voice else { return nil }
                    if voice.isClonedVoice { return nil }
                    return voice.voiceId ?? voice.name
                }()

                let castResolved = CharacterCastingManager.shared.resolveChunkCast(
                    chunkMeta.text,
                    defaultVoiceParam: defaultVoiceParam,
                    defaultPitch: totalPitchShift
                )
                let spokenChunk = SpeechPronunciationLexicon.shared.apply(to: castResolved.spokenText)
                let effectiveChunk: String
                if let tag = fishPreset?.inlineTag, !spokenChunk.trimmingCharacters(in: .whitespaces).hasPrefix("[") {
                    effectiveChunk = "\(tag) \(spokenChunk)"
                } else {
                    effectiveChunk = spokenChunk
                }

                // Slightly jitter temperature by +0.03 on retake so neural prosody produces a fresh inflection
                var retakeParams = effectiveGenerationParameters()
                let baseTemp = retakeParams.temperature ?? 0.65
                retakeParams.temperature = min(1.0, baseTemp + 0.03)

                var rawSamples: [Float] = []
                for try await event in model.generateStream(
                    text: effectiveChunk,
                    voice: castResolved.voiceParam,
                    refAudio: referenceAudio,
                    refText: refText,
                    language: nil,
                    generationParameters: retakeParams,
                    streamingInterval: 60.0
                ) {
                    try Task.checkCancellation()
                    if case .audio(let audioData) = event {
                        autoreleasepool {
                            rawSamples.append(contentsOf: audioData.asArray(Float.self))
                        }
                    }
                }

                guard !rawSamples.isEmpty else {
                    throw NSError(domain: "TTSViewModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Retake produced empty audio"])
                }

                let masterSnap = StudioMasteringManager.shared.snapshot
                let chunkPitch = castResolved.pitch
                let newChunkSamples: [Float] = await Task.detached(priority: .userInitiated) {
                    var p = rawSamples
                    if abs(chunkPitch) > 0.05 {
                        p = Voice.applyPitchShift(p, semitones: chunkPitch, sampleRate: sampleRate)
                    }
                    p = SpeechAudioTrimmer.trimSilence(p, sampleRate: sampleRate)
                    p = StudioMasteringManager.processSamples(p, sampleRate: sampleRate, config: masterSnap, preserveExactLength: true)
                    return p
                }.value

                // Read existing master WAV and splice newChunkSamples in place of oldRange with 12ms cosine crossfades
                let (existingSamples, srUInt) = try StudioMasteringManager.readPCM16WAVSamples(from: existingURL)
                let clampedStart = min(max(0, oldRange.lowerBound), existingSamples.count)
                let clampedEnd = min(max(clampedStart, oldRange.upperBound), existingSamples.count)
                let headSlice = Array(existingSamples[0..<clampedStart])
                let tailSlice = Array(existingSamples[clampedEnd..<existingSamples.count])

                let fadeFrames = min(Int(sampleRate * 0.012), newChunkSamples.count / 4)
                var smoothedChunk = newChunkSamples
                if fadeFrames > 4 {
                    for i in 0..<fadeFrames {
                        let w = Float(0.5 * (1.0 - cos(Double.pi * Double(i) / Double(fadeFrames))))
                        smoothedChunk[i] *= w
                        smoothedChunk[smoothedChunk.count - 1 - i] *= w
                    }
                }

                var splicedSamples: [Float] = []
                splicedSamples.reserveCapacity(headSlice.count + smoothedChunk.count + tailSlice.count)
                splicedSamples.append(contentsOf: headSlice)
                splicedSamples.append(contentsOf: smoothedChunk)
                splicedSamples.append(contentsOf: tailSlice)

                let newWAVURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("spliced_\(UUID().uuidString).wav")
                try StudioMasteringManager.writePCM16WAVSamples(splicedSamples, sampleRate: srUInt, to: newWAVURL)

                let oldFrameCount = clampedEnd - clampedStart
                let deltaFrames = smoothedChunk.count - oldFrameCount
                let deltaSec = Double(deltaFrames) / sampleRate
                let chunkStartSec = Double(clampedStart) / sampleRate

                // Re-align words inside `safeIndex` and shift downstream chunks by `deltaSec`
                let masterWords = WordTimingEngine.extractMasterWords(
                    text: fullText,
                    maxChunkChars: max(180, maxChunkLength),
                    enableChunking: enableChunking
                )
                let retakeWords = masterWords.filter { $0.chunkIndex == safeIndex }
                let realigned = WordTimingEngine.alignWordsToPCM(
                    words: retakeWords,
                    samples: smoothedChunk,
                    sampleRate: sampleRate,
                    startOffsetSec: chunkStartSec
                )
                let realignedById = Dictionary(uniqueKeysWithValues: realigned.map { ($0.id, $0) })

                self.synthesizedWordTimings = self.synthesizedWordTimings.map { t in
                    if t.chunkIndex == safeIndex, let updated = realignedById[t.id] {
                        return updated
                    } else if t.chunkIndex > safeIndex {
                        return WordTiming(
                            id: t.id,
                            word: t.word,
                            start: max(0, t.start + deltaSec),
                            end: max(0, t.end + deltaSec),
                            chunkIndex: t.chunkIndex,
                            paragraphIndex: t.paragraphIndex
                        )
                    } else {
                        return t
                    }
                }

                // Update frame ranges for safeIndex and all subsequent chunks
                var updatedRanges = self.synthesizedChunkFrameRanges
                updatedRanges[safeIndex] = clampedStart..<(clampedStart + smoothedChunk.count)
                for (k, r) in self.synthesizedChunkFrameRanges where k > safeIndex {
                    updatedRanges[k] = max(0, r.lowerBound + deltaFrames)..<max(0, r.upperBound + deltaFrames)
                }
                self.synthesizedChunkFrameRanges = updatedRanges

                let newDuration = Double(splicedSamples.count) / sampleRate
                self.audioURL = newWAVURL
                self.duration = newDuration
                self.audioPlayer.loadAudio(from: newWAVURL)
                self.audioPlayer.seek(to: chunkStartSec)
                self.audioPlayer.play()
                self.generationProgress = "Spliced Retake §\(safeIndex + 1) (\(String(format: "%+.2fs", deltaSec)))"
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(2.2))
                    if self.generationProgress.hasPrefix("Spliced Retake") {
                        self.generationProgress = ""
                    }
                }
            } catch is CancellationError {
                generationProgress = ""
            } catch {
                errorMessage = "Retake failed: \(error.localizedDescription)"
                generationProgress = ""
            }
            isGenerating = false
        }
    }

    func play() {
        audioPlayer.play()
    }

    func pause() {
        audioPlayer.pause()
    }

    func togglePlayPause() {
        audioPlayer.togglePlayPause()
    }

    func stop() {
        // Cancel any ongoing generation
        generationTask?.cancel()
        generationTask = nil

        // Stop audio playback
        audioPlayer.stop()

        // Reset state
        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func playTake(_ take: GeneratedTake) {
        synthesizedStartChunkIndex = 0
        synthesizedEndChunkIndex = nil
        synthesizedWordTimings = []
        audioURL = take.audioURL
        audioPlayer.loadAudio(from: take.audioURL)
        audioPlayer.play()
    }

    func deleteTake(_ take: GeneratedTake) {
        if audioURL == take.audioURL {
            audioPlayer.unloadAudio()
            audioURL = nil
        }
        history.removeAll { $0.id == take.id }
    }

    func clearHistory() {
        if let activeURL = audioURL, history.contains(where: { $0.audioURL == activeURL }) {
            audioPlayer.unloadAudio()
            audioURL = nil
        }
        history.removeAll()
        StudioDocumentSessionManager.shared.clearPersistedTakes()
    }

    func unloadModelAndPurgeVRAM() {
        model = nil
        loadedModelId = nil
        Memory.clearCache()
        generationProgress = "Model unloaded & Metal GPU cache purged (keep_alive: 0)"
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.5))
            if self.generationProgress == "Model unloaded & Metal GPU cache purged (keep_alive: 0)" {
                self.generationProgress = ""
            }
        }
    }

    func saveAudioInFormat(
        _ format: AudioExportFormat,
        from sourceURL: URL? = nil,
        baseName: String = "generated_audio",
        fallbackText: String = "",
        voice: Voice? = nil,
        chunkIndex: Int? = nil,
        scopeOverride: SynthesisScope? = nil
    ) {
        let cleanedFallback = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
        // Reuse already-synthesized audio immediately (~0.2s export!) whenever `sourceURL` is passed
        // or when `audioURL` already exists for the current session and no explicit sub-chunk scope override was requested
        let canReuseExistingAudio: Bool = {
            if sourceURL != nil { return true }
            guard let currentURL = audioURL, FileManager.default.fileExists(atPath: currentURL.path) else { return false }
            if !cleanedFallback.isEmpty && !lastSynthesizedText.isEmpty && lastSynthesizedText != cleanedFallback {
                return false
            }
            if scopeOverride == nil { return true }
            if cleanedFallback.isEmpty { return true }
            if lastSynthesizedText == cleanedFallback && scopeOverride == synthesisScope { return true }
            return false
        }()
        let existingURL = sourceURL ?? (canReuseExistingAudio ? audioURL : nil)
        guard existingURL != nil || !cleanedFallback.isEmpty else {
            errorMessage = "Enter or import text before exporting audio"
            return
        }
        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = existingURL != nil
            ? "Instant Export Audio as \(format.label) (<1s)"
            : "Export Audio as \(format.label) (Silent Synthesis — No Playback Needed)"
        savePanel.nameFieldStringValue = "\(baseName).\(format.rawValue)"

        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }

            // Fast path: if audio is already on disk, snapshot-copy and convert in ~0.2s without stopping live playback!
            if let readyWAV = existingURL, FileManager.default.fileExists(atPath: readyWAV.path) {
                self.generationProgress = "Instant exporting \(format.label)..."
                let snapshotWAV = FileManager.default.temporaryDirectory
                    .appendingPathComponent("export_snap_\(UUID().uuidString).wav")
                try? FileManager.default.copyItem(at: readyWAV, to: snapshotWAV)
                let srcForConvert = FileManager.default.fileExists(atPath: snapshotWAV.path) ? snapshotWAV : readyWAV
                Task.detached(priority: .userInitiated) {
                    defer {
                        if srcForConvert == snapshotWAV {
                            try? FileManager.default.removeItem(at: snapshotWAV)
                        }
                    }
                    do {
                        try AudioFormatExporter.convert(sourceWAV: srcForConvert, to: dest, format: format)
                        await MainActor.run {
                            self.generationProgress = "Exported \(dest.lastPathComponent) (<1s instant export)"
                            Task { @MainActor in
                                try? await Task.sleep(for: .seconds(2.2))
                                if self.generationProgress.hasPrefix("Exported \(dest.lastPathComponent)") {
                                    self.generationProgress = ""
                                }
                            }
                        }
                    } catch {
                        await MainActor.run {
                            self.generationProgress = ""
                            self.errorMessage = "Audio export failed: \(error.localizedDescription)"
                        }
                    }
                }
                return
            }

            // Silent synthesis path when no audio has been generated yet:
            // Default to fastSection (or explicit scopeOverride) so un-synthesized exports finish in a couple of seconds!
            let effectiveExportScope = scopeOverride ?? .fastSection
            self.generationTask?.cancel()
            self.generationTask = Task { @MainActor in
                if !self.isModelLoaded {
                    await self.loadModel()
                }
                self.generationProgress = "Synthesizing silently for \(format.label) export..."
                await self.synthesize(
                    text: cleanedFallback,
                    voice: voice ?? self.selectedVoice,
                    startChunkIndex: chunkIndex ?? self.activeChunkIndex,
                    playAudio: false,
                    recordHistory: true,
                    scopeOverride: effectiveExportScope
                )
                guard let finalWAV = self.audioURL else { return }
                self.generationProgress = "Exporting \(format.label)..."
                Task.detached(priority: .userInitiated) {
                    do {
                        try AudioFormatExporter.convert(sourceWAV: finalWAV, to: dest, format: format)
                        await MainActor.run {
                            self.generationProgress = "Exported \(dest.lastPathComponent) (silent export complete)"
                            Task { @MainActor in
                                try? await Task.sleep(for: .seconds(2.2))
                                if self.generationProgress.hasPrefix("Exported \(dest.lastPathComponent)") {
                                    self.generationProgress = ""
                                }
                            }
                        }
                    } catch {
                        await MainActor.run {
                            self.generationProgress = ""
                            self.errorMessage = "Audio export failed: \(error.localizedDescription)"
                        }
                    }
                }
            }
        }
        #endif
    }

    func exportChapterizedM4B(
        text: String,
        bookTitle: String,
        author: String,
        voice: Voice?,
        customCoverURL: URL? = nil,
        scope: SynthesisScope = .fullDocument
    ) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard audioURL != nil || !cleaned.isEmpty else {
            errorMessage = "Enter or import text before exporting an M4B Audiobook"
            return
        }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Chapterized .M4B Audiobook (Chapters + Cover Art)"
        let safeStem = bookTitle
            .replacingOccurrences(of: "[^a-zA-Z0-9 _-]", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: " ", with: "_")
        savePanel.nameFieldStringValue = "\(safeStem.isEmpty ? "Studio_Audiobook" : safeStem).m4b"

        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }

            Task { @MainActor in
                if self.audioURL == nil && !cleaned.isEmpty {
                    self.generationTask?.cancel()
                    if !self.isModelLoaded {
                        await self.loadModel()
                    }
                    self.generationProgress = "Synthesizing chapters for .M4B Audiobook export..."
                    await self.synthesize(
                        text: cleaned,
                        voice: voice ?? self.selectedVoice,
                        startChunkIndex: 0,
                        playAudio: false,
                        recordHistory: true,
                        scopeOverride: scope
                    )
                }
                guard let urlToSave = self.audioURL else { return }
                let snapshotWAV = FileManager.default.temporaryDirectory
                    .appendingPathComponent("m4b_snap_\(UUID().uuidString).wav")
                try? FileManager.default.copyItem(at: urlToSave, to: snapshotWAV)
                let wavForM4B = FileManager.default.fileExists(atPath: snapshotWAV.path) ? snapshotWAV : urlToSave

                let allChunks = self.chunkTextWithMetadata(cleaned)
                let estDuration = self.duration > 0 ? self.duration : max(2.0, Double(cleaned.count) / 14.0)
                let timings = self.synthesizedWordTimings.filter { $0.end > $0.start }
                let chapters = M4BAudiobookExporter.buildChapters(
                    chunks: allChunks,
                    timings: timings,
                    totalDurationSec: estDuration
                )
                let resolvedVoiceName = (voice ?? self.selectedVoice)?.name ?? "Freya"
                self.generationProgress = "Packaging .M4B Audiobook (\(chapters.count) chapters + cover art)..."

                Task.detached(priority: .userInitiated) {
                    defer {
                        if wavForM4B == snapshotWAV {
                            try? FileManager.default.removeItem(at: snapshotWAV)
                        }
                    }
                    do {
                        try M4BAudiobookExporter.exportM4B(
                            sourceWAV: wavForM4B,
                            to: dest,
                            bookTitle: bookTitle.isEmpty ? "Codecaine Studio Audiobook" : bookTitle,
                            author: author.isEmpty ? "Codecaine MLX Audio Studio" : author,
                            voiceName: resolvedVoiceName,
                            chapters: chapters,
                            customCoverURL: customCoverURL
                        )
                        await MainActor.run {
                            self.generationProgress = "Exported M4B Audiobook: \(dest.lastPathComponent) (\(chapters.count) chapters)"
                            Task { @MainActor in
                                try? await Task.sleep(for: .seconds(2.8))
                                if self.generationProgress.hasPrefix("Exported M4B Audiobook:") {
                                    self.generationProgress = ""
                                }
                            }
                        }
                    } catch {
                        await MainActor.run {
                            self.generationProgress = ""
                            self.errorMessage = "M4B Audiobook export failed: \(error.localizedDescription)"
                        }
                    }
                }
            }
        }
        #endif
    }

    func exportReadAlongVideo(text: String, theme: ReadAlongTheme, aspect: ReadAlongVideoAspect) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard audioURL != nil || !cleaned.isEmpty else {
            errorMessage = "Enter or import text before exporting Read-Along Video"
            return
        }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Read-Along Karaoke Video (\(aspect.label) · \(theme.label))"
        savePanel.nameFieldStringValue = "read_along_\(theme.rawValue)_\(aspect.rawValue.replacingOccurrences(of: ":", with: "x")).mp4"
        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }

            Task { @MainActor in
                if self.audioURL == nil && !cleaned.isEmpty {
                    self.generationTask?.cancel()
                    if !self.isModelLoaded {
                        await self.loadModel()
                    }
                    self.generationProgress = "Synthesizing speech silently for Read-Along MP4 export..."
                    await self.synthesize(
                        text: cleaned,
                        voice: self.selectedVoice,
                        startChunkIndex: self.activeChunkIndex,
                        playAudio: false,
                        recordHistory: true,
                        scopeOverride: .fastSection
                    )
                }
                guard let urlToSave = self.audioURL else { return }
                let snapshotWAV = FileManager.default.temporaryDirectory
                    .appendingPathComponent("video_snap_\(UUID().uuidString).wav")
                try? FileManager.default.copyItem(at: urlToSave, to: snapshotWAV)
                let wavForVideo = FileManager.default.fileExists(atPath: snapshotWAV.path) ? snapshotWAV : urlToSave

                let estDuration = self.duration > 0 ? self.duration : max(1.5, Double(cleaned.count) / 14.0)
                let activeDirect = self.synthesizedWordTimings.filter { $0.end > $0.start }
                let rawTimings: [WordTiming]
                if !activeDirect.isEmpty {
                    rawTimings = activeDirect
                } else {
                    rawTimings = WordTimingEngine.computeWordTimings(
                        text: cleaned,
                        audioURL: wavForVideo,
                        duration: estDuration,
                        maxChunkChars: max(180, self.maxChunkLength),
                        enableChunking: self.enableChunking,
                        startChunkIndex: self.synthesizedStartChunkIndex,
                        endChunkIndex: self.synthesizedEndChunkIndex,
                        paragraphPauseMs: self.paragraphPauseMs
                    ).filter { $0.end > $0.start }
                }
                let timings = WordTimingEngine.shiftedTimings(rawTimings, offsetMs: self.wordSyncOffsetMs)
                self.generationProgress = "Rendering Read-Along MP4 Video (\(aspect.rawValue))..."
                Task.detached(priority: .userInitiated) {
                    defer {
                        if wavForVideo == snapshotWAV {
                            try? FileManager.default.removeItem(at: snapshotWAV)
                        }
                    }
                    do {
                        try ReadAlongVideoExporter.exportKaraokeVideo(
                            sourceWAV: wavForVideo,
                            timings: timings,
                            theme: theme,
                            aspect: aspect,
                            to: dest
                        )
                        await MainActor.run {
                            self.generationProgress = "Exported Read-Along Video: \(dest.lastPathComponent)"
                            Task { @MainActor in
                                try? await Task.sleep(for: .seconds(2.2))
                                if self.generationProgress.hasPrefix("Exported Read-Along Video:") {
                                    self.generationProgress = ""
                                }
                            }
                        }
                    } catch {
                        await MainActor.run {
                            self.generationProgress = ""
                            self.errorMessage = "Read-Along Video export failed: \(error.localizedDescription)"
                        }
                    }
                }
            }
        }
        #endif
    }

    func exportSubtitles(text: String, format: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else {
            errorMessage = "Enter or synthesize text before exporting subtitles"
            return
        }
        let estDuration = duration > 0 ? duration : max(1.5, Double(cleaned.count) / 14.0)
        let activeDirect = synthesizedWordTimings.filter { $0.end > $0.start }
        let rawTimings: [WordTiming]
        if !activeDirect.isEmpty {
            rawTimings = activeDirect
        } else {
            rawTimings = WordTimingEngine.computeWordTimings(
                text: cleaned,
                audioURL: audioURL,
                duration: estDuration,
                maxChunkChars: max(180, maxChunkLength),
                enableChunking: enableChunking,
                startChunkIndex: synthesizedStartChunkIndex,
                endChunkIndex: synthesizedEndChunkIndex,
                paragraphPauseMs: paragraphPauseMs
            ).filter { $0.end > $0.start }
        }
        let timings = WordTimingEngine.shiftedTimings(rawTimings, offsetMs: wordSyncOffsetMs)
        let content: String
        let ext: String
        switch format.lowercased() {
        case "vtt", "webvtt":
            content = SubtitleExporter.formatVTT(timings: timings)
            ext = "vtt"
        case "ass", "ssa":
            content = SubtitleExporter.formatASSKaraoke(timings: timings, theme: readAlongTheme)
            ext = "ass"
        case "lrc":
            content = SubtitleExporter.formatLRC(timings: timings)
            ext = "lrc"
        case "ttml", "dfxp":
            content = SubtitleExporter.formatTTML(timings: timings)
            ext = "ttml"
        case "sbv":
            content = SubtitleExporter.formatSBV(timings: timings)
            ext = "sbv"
        case "json":
            let rows: [[String: Any]] = timings.map { w in
                [
                    "word": w.word,
                    "start": round(w.start * 1000) / 1000,
                    "end": round(w.end * 1000) / 1000,
                    "chunk_index": w.chunkIndex
                ]
            }
            if let data = try? JSONSerialization.data(withJSONObject: ["timings": rows, "text": cleaned], options: [.prettyPrinted, .sortedKeys]),
               let jsonStr = String(data: data, encoding: .utf8) {
                content = jsonStr
            } else {
                content = "{\"text\": \"\"}"
            }
            ext = "json"
        case "txt":
            content = cleaned
            ext = "txt"
        case "md", "markdown":
            let docTitle = StudioDocumentSessionManager.shared.currentDocumentTitle
            content = "# \(docTitle)\n\n\(cleaned)\n"
            ext = "md"
        default:
            content = SubtitleExporter.formatSRT(timings: timings)
            ext = "srt"
        }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Subtitles (.\(ext))"
        savePanel.nameFieldStringValue = "speech_subtitles.\(ext)"
        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }
            do {
                try content.write(to: dest, atomically: true, encoding: .utf8)
                self.generationProgress = "Exported \(dest.lastPathComponent)"
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(2.0))
                    if self.generationProgress == "Exported \(dest.lastPathComponent)" {
                        self.generationProgress = ""
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Subtitle export failed: \(error.localizedDescription)"
                }
            }
        }
        #endif
    }

    func saveAudioFile(from sourceURL: URL? = nil, suggestedName: String = "generated_audio.wav") {
        guard let urlToSave = sourceURL ?? audioURL else { return }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.wav]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Save Audio File"
        savePanel.nameFieldStringValue = suggestedName

        savePanel.begin { response in
            if response == .OK, let destinationURL = savePanel.url {
                do {
                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try FileManager.default.removeItem(at: destinationURL)
                    }
                    try FileManager.default.copyItem(at: urlToSave, to: destinationURL)
                } catch {
                    DispatchQueue.main.async {
                        self.errorMessage = "Failed to save file: \(error.localizedDescription)"
                    }
                }
            }
        }
        #else
        errorMessage = "Save functionality is available on macOS"
        #endif
    }

    func batchExportLines(text: String, voice: Voice?) {
        guard isModelLoaded else {
            errorMessage = "Wait for the TTS model to finish loading before batch exporting"
            return
        }
        let lines = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else {
            errorMessage = "Enter at least one line of text to batch export"
            return
        }

        #if os(macOS)
        let openPanel = NSOpenPanel()
        openPanel.canChooseFiles = false
        openPanel.canChooseDirectories = true
        openPanel.canCreateDirectories = true
        openPanel.prompt = "Export \(lines.count) Line(s) Here"
        openPanel.title = "Choose Folder for Batch Audio Export"

        openPanel.begin { [weak self] response in
            guard let self, response == .OK, let folderURL = openPanel.url else { return }
            self.generationTask?.cancel()
            self.audioPlayer.stop()
            self.generationTask = Task {
                for (idx, line) in lines.enumerated() {
                    if Task.isCancelled { break }
                    self.generationProgress = "Batch exporting line \(idx + 1)/\(lines.count)..."
                    await self.synthesize(text: line, voice: voice, startChunkIndex: 0, playAudio: false, recordHistory: false)
                    if let generatedURL = self.audioURL {
                        let rawPrefix = String(
                            line.prefix(24)
                                .replacingOccurrences(of: "[^a-zA-Z0-9_-]", with: "_", options: .regularExpression)
                                .replacingOccurrences(of: "^_+|_+$", with: "", options: .regularExpression)
                        )
                        let fileName = String(format: "%02d_%@.wav", idx + 1, rawPrefix.isEmpty ? "line" : rawPrefix)
                        let destURL = folderURL.appendingPathComponent(fileName)
                        try? FileManager.default.removeItem(at: destURL)
                        try? FileManager.default.copyItem(at: generatedURL, to: destURL)
                    }
                }
                self.generationProgress = "Batch exported \(lines.count) WAV files to \(folderURL.lastPathComponent)"
                try? await Task.sleep(for: .seconds(2.2))
                if self.generationProgress.hasPrefix("Batch exported") {
                    self.generationProgress = ""
                }
            }
        }
        #endif
    }
}
