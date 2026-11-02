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

/// Codecaine File Manager Synthesis Scope for instant 2-second EPUB/Book start & export
enum SynthesisScope: String, CaseIterable, Identifiable {
    case fastSection = "fastSection"
    case activeChunk = "activeChunk"
    case fullDocument = "fullDocument"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fastSection:
            return "Smart Lookahead (2s Start + 4-Chunk Window)"
        case .activeChunk:
            return "Active Chunk Only (~2s Instant Synthesis & Export)"
        case .fullDocument:
            return "Full Book / Document Batch (All Chunks)"
        }
    }

    var shortBadge: String {
        switch self {
        case .fastSection: return "⚡ Smart 4-Chunk (2s)"
        case .activeChunk: return "🎯 1 Chunk (2s)"
        case .fullDocument: return "📚 Full Book"
        }
    }
}

/// Writes an IEEE Float32 RIFF/WAVE file incrementally and updates its 44-byte header in place after every chunk
/// so `audioURL` is a valid, playable, exportable WAV file on disk within ~1.5–2 seconds (after Chunk #1).
final class LivePlayableWAVWriter {
    let url: URL
    let sampleRate: UInt32
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
        guard !samples.isEmpty, let fh = fileHandle else { return }
        try fh.seekToEnd()
        let pcmData = samples.withUnsafeBytes { Data($0) }
        try fh.write(contentsOf: pcmData)
        framesWritten += samples.count
        let dataBytes = UInt32(clamping: framesWritten * MemoryLayout<Float>.size)
        let header = Self.makeWAVHeader(sampleRate: sampleRate, dataBytes: dataBytes)
        try fh.seek(toOffset: 0)
        try fh.write(contentsOf: header)
        try? fh.synchronize()
    }

    @discardableResult
    func finalize() -> URL {
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
    private static let maxMemoryEntries = 96

    private static var cacheDirectory: URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("VoicesAppPhononChunkCache", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func cacheKey(modelId: String, voiceParam: String?, refText: String?, pitch: Float, text: String) -> String {
        let raw = "\(modelId)|\(voiceParam ?? "default")|\(refText ?? "")|\(String(format: "%.2f", pitch))|\(text)"
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
        let data = samples.withUnsafeBytes { Data($0) }
        try? data.write(to: fileURL, options: .atomic)
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
        ("Orpheus 3B (BF16)", "mlx-community/orpheus-3b-0.1-ft-bf16"),
        ("Soprano 80M (BF16)", "mlx-community/Soprano-80M-bf16"),
        ("VibeVoice Realtime 0.5B (8-bit)", "mlx-community/VibeVoice-Realtime-0.5B-8bit"),
        ("VibeVoice TTS 1.5B", "microsoft/VibeVoice-1.5B"),
        ("Voxtral 4B TTS (4-bit)", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"),
        ("Voxtral 4B TTS (BF16)", "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"),
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
        ("OmniVoice", "mlx-community/OmniVoice"),
        ("Audio8 0.1B Preview (BF16)", "mlx-community/Audio8-TTS-Preview-0.1b-bf16"),
        ("Audio8 0.6B Preview (BF16)", "mlx-community/Audio8-TTS-Preview-0.6b-bf16"),
        ("TADA 1B", "HumeAI/mlx-tada-1b"),
        ("TADA 3B", "HumeAI/mlx-tada-3b"),
        ("VoxCPM 1.5 (4-bit)", "mlx-community/VoxCPM1.5-4bit"),
        ("VoxCPM 1.5 (8-bit)", "mlx-community/VoxCPM1.5-8bit"),
        ("ZONOS2 (4-bit)", "amal-david/Zyphra-ZONOS2-4bit"),
        ("VyvoTTS EN Beta (4-bit)", "mlx-community/VyvoTTS-EN-Beta-4bit"),
        ("LongCat-AudioDiT 1B (4-bit)", "mlx-community/LongCat-AudioDiT-1B-4bit"),
        ("LongCat-AudioDiT 1B (BF16)", "mlx-community/LongCat-AudioDiT-1B-bf16"),
        ("MOSS-TTS", "OpenMOSS-Team/MOSS-TTS"),
        ("IndexTTS 1.5", "mlx-community/IndexTTS-1.5"),
        ("Irodori TTS v4.1 Small (8-bit)", "mlx-community/Irodori-TTS-v4.1-Small-8bit"),
        ("Irodori TTS 600M v3 VoiceDesign (8-bit)", "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"),
        ("Spark-TTS 0.5B (BF16)", "mlx-community/Spark-TTS-0.5B-bf16"),
        ("Rumik OSS 1 (8-bit)", "rumik-ai/rumik-oss-1-mlx-8bit"),
        ("Dia 1.6B", "mlx-community/Dia-1.6B"),
    ]
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
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var tokensPerSecond: Double = 0

    // Generation parameters: model defaults + optional UI overrides
    private var defaultGenerationParameters = GenerateParameters(
        maxTokens: 1200,
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
    ) ?? .fastSection {
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
            if oldValue != modelId {
                refreshVoicesForCurrentModel(forceSelectFirst: true)
            }
        }
    }
    private var loadedModelId: String?

    // Auto-populated built-in voices and currently selected voice for the active model
    var builtInVoices: [Voice] = []
    var selectedVoice: Voice?

    // Session generation history & power-user controls
    var history: [GeneratedTake] = []
    var hfToken: String = ModelCacheHelper.hfToken {
        didSet {
            ModelCacheHelper.hfToken = hfToken
        }
    }
    var playbackRate: Float = 1.0 {
        didSet {
            audioPlayer.playbackRate = playbackRate
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
    } ?? 350 {
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
        defaultGenerationParameters.maxTokens ?? 1200
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
        setupAudioPlayerObservers()
        refreshVoicesForCurrentModel(forceSelectFirst: true)
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
            }
        }
        let voices = Voice.builtInVoices(for: modelId, dynamicVoiceNames: dynamicNames)
        builtInVoices = voices
        if forceSelectFirst || selectedVoice == nil || !voices.contains(where: { $0.name == selectedVoice?.name && $0.voiceId == selectedVoice?.voiceId }) {
            selectedVoice = voices.first
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
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            let loadedModel = try await TTS.loadModel(
                modelRepo: modelId,
                hfToken: ModelCacheHelper.effectiveHFToken
            )
            model = loadedModel
            defaultGenerationParameters = loadedModel.defaultGenerationParameters
            loadedModelId = modelId
            refreshVoicesForCurrentModel(forceSelectFirst: false)
            generationProgress = "" // Clear progress on success
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
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
        synthesisScope = .fastSection
        splitPattern = "\n"
        streamingPlayback = true
        pitchSemitones = 0.0
        paragraphPauseMs = 350
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

    /// Split arbitrarily large text into chunks using Codecaine File Manager's HierarchicalChunker
    func chunkText(_ text: String) -> [String] {
        guard enableChunking else {
            return [text]
        }
        let effectiveMax = max(180, maxChunkLength)
        guard text.count > effectiveMax else {
            return [text]
        }
        var normalizedText = text
        let trimmedPattern = splitPattern.trimmingCharacters(in: .whitespaces)
        if !trimmedPattern.isEmpty && trimmedPattern != "\n" && trimmedPattern != "\\n" {
            if let regex = try? NSRegularExpression(pattern: trimmedPattern) {
                let nsRange = NSRange(normalizedText.startIndex..., in: normalizedText)
                normalizedText = regex.stringByReplacingMatches(in: normalizedText, range: nsRange, withTemplate: "\n\n")
            }
        }
        let phononChunks = HierarchicalChunker.splitText(normalizedText, maxChars: effectiveMax, mergeShortParagraphs: true)
        return phononChunks.isEmpty ? [text] : phononChunks.map(\.text)
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
        let baseName = scope == .fullDocument
            ? "audiobook_full"
            : (scope == .fastSection ? "epub_section_from_\(targetChunk + 1)" : "epub_chunk_\(targetChunk + 1)")
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

        guard !text.isEmpty else {
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

            // Split text into chunks (unlimited length hierarchical chunker)
            let allChunks = chunkText(text)
            let safeStart = min(max(0, startChunkIndex), max(0, allChunks.count - 1))
            let effectiveScope = scopeOverride ?? synthesisScope
            let maxWindowCount: Int
            switch effectiveScope {
            case .activeChunk:
                maxWindowCount = 1
            case .fastSection:
                // Codecaine File Manager fast lookahead window: synthesize Chunk #1 in ~1.5s + 3 lookahead chunks
                maxWindowCount = allChunks.count > 4 ? 4 : allChunks.count
            case .fullDocument:
                maxWindowCount = allChunks.count
            }

            let chunks = Array(allChunks.dropFirst(safeStart).prefix(maxWindowCount))
            let plannedEndChunk = safeStart + max(0, chunks.count - 1)
            totalChunkCount = allChunks.count
            activeChunkIndex = safeStart
            synthesizingChunkIndex = safeStart
            synthesizedStartChunkIndex = safeStart
            synthesizedEndChunkIndex = plannedEndChunk
            let sampleRate = Double(model.sampleRate)

            // Create LivePlayableWAVWriter — updates WAV header in place after every chunk so `audioURL` is ready after Chunk #1 (~1.5s)
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            let wavWriter = try LivePlayableWAVWriter(url: tempURL, sampleRate: sampleRate)

            var totalTokenCount = 0

            // Streaming playback — model yields audio chunks progressively
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
            var cumulativeFramesWritten = 0

            for (offset, chunk) in chunks.enumerated() {
                let globalIndex = safeStart + offset
                synthesizingChunkIndex = globalIndex
                if offset == 0 {
                    activeChunkIndex = safeStart
                }
                // Check for cancellation between chunks
                try Task.checkCancellation()

                var chunkTokenCount = 0
                let chunkWords = masterWordsByChunk[globalIndex] ?? []
                let chunkStartSec = Double(cumulativeFramesWritten) / sampleRate
                var chunkSamples: [Float] = []

                // Set cache limit for this chunk
                Memory.cacheLimit = 512 * 1024 * 1024 // 512MB cache limit
                let generationParameters = effectiveGenerationParameters()

                // Determine voice parameter (VoiceDesign description, built-in voiceId, or nil for cloned voice)
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

                let effectiveChunk: String
                if let tag = fishPreset?.inlineTag, !chunk.trimmingCharacters(in: .whitespaces).hasPrefix("[") {
                    effectiveChunk = "\(tag) \(chunk)"
                } else {
                    effectiveChunk = chunk
                }

                let cacheKey = PhononChunkCache.cacheKey(
                    modelId: modelId,
                    voiceParam: voiceParam ?? voice?.name,
                    refText: refText,
                    pitch: totalPitchShift,
                    text: effectiveChunk
                )

                if let cachedSamples = PhononChunkCache.get(key: cacheKey) {
                    chunkSamples = cachedSamples
                    generationProgress = "Loaded chunk §\(globalIndex + 1)/\(allChunks.count) from instant cache (0ms)..."
                } else {
                    if allChunks.count > 1 {
                        if offset == 0 {
                            generationProgress = "Synthesizing §\(globalIndex + 1) of \(allChunks.count) (instant 2s start & export)..."
                        } else {
                            generationProgress = "Speaking §\(activeChunkIndex + 1) · Prefetching §\(globalIndex + 1)/\(plannedEndChunk + 1)..."
                        }
                    }

                    // Each chunk needs a fresh generation.
                    // Pass streamingInterval: 60.0 so models that support sub-chunk streaming synthesize each sentence chunk cohesively
                    // without mid-sentence buffering gaps or partial-waveform word timing distortion.
                    for try await event in model.generateStream(
                        text: effectiveChunk,
                        voice: voiceParam,
                        refAudio: referenceAudio,
                        refText: refText,
                        language: nil,
                        generationParameters: generationParameters,
                        streamingInterval: 60.0
                    ) {
                        // Throw if cancelled - this will exit the loop and be caught below
                        try Task.checkCancellation()

                        switch event {
                        case .token:
                            chunkTokenCount += 1
                            totalTokenCount += 1
                            if chunkTokenCount % 50 == 0 {
                                if allChunks.count > 1 {
                                    generationProgress = "Chunk §\(globalIndex + 1)/\(plannedEndChunk + 1): \(chunkTokenCount) tokens..."
                                } else {
                                    generationProgress = "Generated \(chunkTokenCount) tokens..."
                                }
                            }
                        case .progress(let fraction):
                            let percent = Int((fraction * 100).rounded())
                            if allChunks.count > 1 {
                                generationProgress = "Chunk §\(globalIndex + 1)/\(plannedEndChunk + 1): \(percent)%"
                            } else {
                                generationProgress = "Generating... \(percent)%"
                            }
                        case .info(let info):
                            tokensPerSecond = info.tokensPerSecond
                        case .audio(let audioData):
                            autoreleasepool {
                                let samples = audioData.asArray(Float.self)
                                chunkSamples.append(contentsOf: samples)
                            }
                        }
                    }

                    if !chunkSamples.isEmpty && abs(totalPitchShift) > 0.05 {
                        chunkSamples = Voice.applyPitchShift(chunkSamples, semitones: totalPitchShift, sampleRate: sampleRate)
                    }
                    if !chunkSamples.isEmpty {
                        PhononChunkCache.put(key: cacheKey, samples: chunkSamples)
                    }
                }

                if !chunkSamples.isEmpty {
                    // Align chunk words against the complete synthesized waveform of this chunk before playback begins
                    if !chunkWords.isEmpty {
                        let aligned = WordTimingEngine.alignWordsToPCM(
                            words: chunkWords,
                            samples: chunkSamples,
                            sampleRate: sampleRate,
                            startOffsetSec: chunkStartSec
                        )
                        for t in aligned where t.id >= 0 && t.id < workingTimings.count {
                            workingTimings[t.id] = t
                        }
                        self.synthesizedWordTimings = workingTimings
                    }

                    var samplesToWrite = chunkSamples
                    if offset + 1 < chunks.count && pauseFrames > 0 {
                        samplesToWrite.append(contentsOf: repeatElement(Float(0), count: pauseFrames))
                    }

                    try? wavWriter.writeChunk(samplesToWrite)
                    cumulativeFramesWritten += samplesToWrite.count
                    self.synthesizedEndChunkIndex = globalIndex

                    // Publish playable & exportable WAV URL immediately after Chunk #1 (~1.5–2.0 seconds!)
                    let liveDuration = Double(wavWriter.framesWritten) / sampleRate
                    if self.audioURL != tempURL {
                        self.audioURL = tempURL
                    }
                    if self.duration < liveDuration {
                        self.duration = liveDuration
                    }

                    if useStreaming {
                        audioPlayer.scheduleAudioChunk(samplesToWrite, withCrossfade: true)
                    }
                }

                // Clear GPU cache after each chunk so arbitrarily large books never accumulate VRAM
                Memory.clearCache()
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

            // Record in session take history
            if recordHistory {
                let takeText = chunks.joined(separator: "\n\n")
                let take = GeneratedTake(
                    text: takeText,
                    modelId: modelId,
                    voiceName: voice?.name ?? "Default",
                    audioURL: finalURL,
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
            errorMessage = "Generation failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
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
        let existingURL = sourceURL ?? ((chunkIndex == nil && scopeOverride == nil) ? audioURL : nil)
        let cleanedFallback = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard existingURL != nil || !cleanedFallback.isEmpty else {
            errorMessage = "Enter or import text before exporting audio"
            return
        }
        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Audio as \(format.label) (Silent Synthesis — No Playback Needed)"
        savePanel.nameFieldStringValue = "\(baseName).\(format.rawValue)"

        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }
            self.generationTask?.cancel()
            self.generationTask = Task { @MainActor in
                var wavToConvert = existingURL
                if wavToConvert == nil {
                    if !self.isModelLoaded {
                        await self.loadModel()
                    }
                    self.generationProgress = "Synthesizing silently for \(format.label) export (no audio playback)..."
                    await self.synthesize(
                        text: cleanedFallback,
                        voice: voice ?? self.selectedVoice,
                        startChunkIndex: chunkIndex ?? self.activeChunkIndex,
                        playAudio: false,
                        recordHistory: true,
                        scopeOverride: scopeOverride ?? self.synthesisScope
                    )
                    wavToConvert = self.audioURL
                }
                guard let finalWAV = wavToConvert else { return }
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
            self.generationTask?.cancel()
            self.generationTask = Task { @MainActor in
                if self.audioURL == nil && !cleaned.isEmpty {
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
                        scopeOverride: self.synthesisScope
                    )
                }
                guard let urlToSave = self.audioURL else { return }
                let estDuration = self.duration > 0 ? self.duration : max(1.5, Double(cleaned.count) / 14.0)
                let activeDirect = self.synthesizedWordTimings.filter { $0.end > $0.start }
                let rawTimings: [WordTiming]
                if !activeDirect.isEmpty {
                    rawTimings = activeDirect
                } else {
                    rawTimings = WordTimingEngine.computeWordTimings(
                        text: cleaned,
                        audioURL: urlToSave,
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
                    do {
                        try ReadAlongVideoExporter.exportKaraokeVideo(
                            sourceWAV: urlToSave,
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
        case "vtt":
            content = SubtitleExporter.formatVTT(timings: timings)
            ext = "vtt"
        case "ass":
            content = SubtitleExporter.formatASSKaraoke(timings: timings, theme: readAlongTheme)
            ext = "ass"
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
