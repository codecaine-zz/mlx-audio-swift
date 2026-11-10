import Foundation
import SwiftUI
import MLXAudioSTT
import MLXAudioCore
import MLX
@preconcurrency import AVFoundation
import Combine
#if os(macOS)
import AppKit
#endif

@MainActor
@Observable
class STTViewModel {
    private static let defaultModelId = "FermionResearch/Phonon-2.5-English"
    static let presetModels: [(name: String, id: String)] = [
        ("Phonon 2.5 English (Default)", "FermionResearch/Phonon-2.5-English"),
        ("Phonon 2.5 Multilingual", "FermionResearch/Phonon-2.5"),
        ("Phonon 2 (MLX)", "mlx-community/Phonon-2"),
        ("Qwen3-ASR 0.6B (4-bit)", "mlx-community/Qwen3-ASR-0.6B-4bit"),
        ("Qwen3-ASR 1.7B (BF16)", "mlx-community/Qwen3-ASR-1.7B-bf16"),
        ("Parakeet TDT 0.6B v3", "mlx-community/parakeet-tdt-0.6b-v3"),
        ("Parakeet TDT 1.1B", "mlx-community/parakeet-tdt-1.1b"),
        ("Parakeet Redux", "moondream/parakeet-redux"),
        ("Parakeet Ultra", "moondream/parakeet-ultra"),
        ("Orukeet", "oruk/orukeet"),
        ("Whisper Large v3 Turbo", "mlx-community/whisper-large-v3-turbo"),
        ("SenseVoice Small", "mlx-community/SenseVoiceSmall"),
        ("FireRedASR2 AED (MLX)", "mlx-community/FireRedASR2-AED-mlx"),
        ("Cohere Transcribe (FP16)", "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"),
        ("GLM-ASR Nano (4-bit)", "mlx-community/GLM-ASR-Nano-2512-4bit"),
        ("Voxtral Mini 3B (BF16)", "mlx-community/Voxtral-Mini-3B-2507-bf16"),
        ("Voxtral Mini 4B Realtime (FP16)", "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16"),
        ("Voxtral Mini 4B Realtime (4-bit)", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"),
        ("VibeVoice ASR Long-Form (4-bit)", "mlx-community/VibeVoice-ASR-4bit"),
        ("VibeVoice ASR Streaming 1.5B", "microsoft/VibeVoice-ASR-Streaming-1.5B"),
        ("VibeVoice ASR Streaming 7B", "microsoft/VibeVoice-ASR-Streaming-7B"),
        ("Higgs-Audio v3 STT", "bosonai/higgs-audio-v3-stt"),
        ("Mega-ASR (BF16)", "mlx-community/Mega-ASR-bf16"),
        ("Qwen2-Audio 7B Instruct (4-bit)", "mlx-community/Qwen2-Audio-7B-Instruct-4bit"),
        ("MOSS-Transcribe-Diarize", "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
        ("MOSS-Music 8B Instruct (4-bit)", "rtikw/MOSS-Music-8B-Instruct-4bit"),
        ("MOSS-Music 8B Thinking (4-bit)", "mlx-community/MOSS-Music-8B-Thinking-4bit"),
        ("Ming-3.0 ASR Tiny (4-bit)", "mlx-community/Ming-3.0-asr-tiny-4bit"),
        ("MiMo-V2.5-ASR", "mlx-community/MiMo-V2.5-ASR-MLX"),
        ("Nemotron Unified ASR 0.6B (8-bit)", "mlx-community/nemotron-labs-unified-asr-0.6b-8bit"),
        ("Nemotron 3.5 ASR Streaming 0.6B (8-bit)", "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"),
        ("Nemotron Speech Streaming EN 0.6B (8-bit)", "animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit"),
        ("Nemotron 3.5 Transcribe 30B-A4B (4-bit)", "nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit"),
        ("Nemotron 3.5 Transcribe 30B-A4B (NVFP4)", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"),
        ("Fun-ASR-Nano", "mlx-community/Fun-ASR-Nano-2512"),
        ("Granite Speech 5.0 2B", "ibm-granite/granite-speech-5.0-2b"),
        ("Granite Speech 5.0 470M TurboCTC", "ibm-granite/granite-speech-5.0-470m-turboctc"),
        ("Granite Speech 4.1 2B Plus", "ibm-granite/granite-speech-4.1-2b-plus"),
        ("Granite Speech 4.1 2B NAR (5-bit)", "mlx-community/granite-speech-4.1-2b-nar-mlx-5bit"),
        ("Canary 1B v2 (q8)", "Mediform/canary-1b-v2-mlx-q8"),
        ("Moonshine Tiny", "UsefulSensors/moonshine-tiny"),
        ("Wav2Vec2 Base 960h CTC", "facebook/wav2vec2-base-960h"),
        ("Meta MMS 1B FL102 (Multilingual CTC)", "facebook/mms-1b-fl102"),
    ]

    static var categorizedPresetModels: [(category: String, icon: String, models: [(name: String, id: String)])] {
        var featured: [(name: String, id: String)] = []
        var streamingDiarize: [(name: String, id: String)] = []
        var multilingual: [(name: String, id: String)] = []
        var audioUnderstanding: [(name: String, id: String)] = []

        for item in presetModels {
            let l = item.id.lowercased()
            if l.contains("phonon") || l.contains("qwen3-asr") || l.contains("parakeet") || l.contains("orukeet") {
                featured.append(item)
            } else if l.contains("realtime") || l.contains("streaming") || l.contains("diarize") || l.contains("vibevoice") || l.contains("cohere") || l.contains("nemotron") {
                streamingDiarize.append(item)
            } else if l.contains("music") || l.contains("qwen2-audio") || l.contains("higgs") || l.contains("mega-asr") || l.contains("mimo") {
                audioUnderstanding.append(item)
            } else {
                multilingual.append(item)
            }
        }

        return [
            ("Featured & Fast Default (Phonon 2.5 / Qwen3 / Parakeet)", "bolt.fill", featured),
            ("Realtime Streaming & Speaker Diarization", "waveform.badge.mic", streamingDiarize),
            ("Multilingual ASR (Whisper / SenseVoice / Granite / MMS)", "globe", multilingual),
            ("Music, Lyrics & Multimodal Audio Understanding", "music.note.list", audioUnderstanding),
        ]
    }

    func radCodeSnippet(language: String, audioPath: String? = nil, port: UInt16 = 8080) -> String {
        let pathStr = (audioPath ?? "sample.wav")
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        switch language.lowercased() {
        case "python":
            return """
            import requests

            resp = requests.post(
                "http://127.0.0.1:\(port)/v1/audio/transcriptions",
                json={"model": "\(modelId)", "audio_path": "\(pathStr)"}
            )
            print(resp.json()["text"])
            """
        case "typescript", "javascript", "ts", "js":
            return """
            const res = await fetch("http://127.0.0.1:\(port)/v1/audio/transcriptions", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ model: "\(modelId)", audio_path: "\(pathStr)" })
            });
            const data = await res.json();
            console.log(data.text);
            """
        case "swift":
            return """
            import Foundation

            let url = URL(string: "http://127.0.0.1:\(port)/v1/audio/transcriptions")!
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = #\"\"\"
            {"model":"\(modelId)","audio_path":"\(pathStr)"}
            \"\"\"#.data(using: .utf8)

            let (data, _) = try await URLSession.shared.data(for: req)
            print(String(data: data, encoding: .utf8) ?? "")
            """
        default:
            return """
            curl -s http://127.0.0.1:\(port)/v1/audio/transcriptions \\
              -H "Content-Type: application/json" \\
              -d '{"model":"\(modelId)","audio_path":"\(pathStr)"}' | jq .
            """
        }
    }
    private static let defaultMaxTokens = 8192
    private static let defaultTemperature: Float = 0.0
    private static let defaultLanguage = "English"
    private static let defaultChunkDuration: Float = 24.0
    private static let defaultStreamingDelayMs = 480
    private static let modelIdStorageKey = "VoicesApp.STTViewModel.modelId"
    private static let maxTokensStorageKey = "VoicesApp.STTViewModel.maxTokens"
    private static let temperatureStorageKey = "VoicesApp.STTViewModel.temperature"
    private static let languageStorageKey = "VoicesApp.STTViewModel.language"
    private static let chunkDurationStorageKey = "VoicesApp.STTViewModel.chunkDuration"
    private static let streamingDelayMsStorageKey = "VoicesApp.STTViewModel.streamingDelayMs"

    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var transcriptionText: String = UserDefaults.standard.string(forKey: "VoicesApp.STTViewModel.lastTranscriptionText") ?? "" {
        didSet {
            UserDefaults.standard.set(transcriptionText, forKey: "VoicesApp.STTViewModel.lastTranscriptionText")
        }
    }
    var tokensPerSecond: Double = 0
    var peakMemory: Double = 0

    // Generation parameters
    var maxTokens: Int = UserDefaults.standard.object(forKey: STTViewModel.maxTokensStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: STTViewModel.maxTokensStorageKey)
    } ?? STTViewModel.defaultMaxTokens {
        didSet {
            UserDefaults.standard.set(maxTokens, forKey: STTViewModel.maxTokensStorageKey)
        }
    }
    var temperature: Float = UserDefaults.standard.object(forKey: STTViewModel.temperatureStorageKey).map { _ in
        UserDefaults.standard.float(forKey: STTViewModel.temperatureStorageKey)
    } ?? STTViewModel.defaultTemperature {
        didSet {
            UserDefaults.standard.set(temperature, forKey: STTViewModel.temperatureStorageKey)
        }
    }
    var language: String = UserDefaults.standard.string(forKey: STTViewModel.languageStorageKey) ?? STTViewModel.defaultLanguage {
        didSet {
            UserDefaults.standard.set(language, forKey: STTViewModel.languageStorageKey)
        }
    }
    var chunkDuration: Float = UserDefaults.standard.object(forKey: STTViewModel.chunkDurationStorageKey).map { _ in
        UserDefaults.standard.float(forKey: STTViewModel.chunkDurationStorageKey)
    } ?? STTViewModel.defaultChunkDuration {
        didSet {
            UserDefaults.standard.set(chunkDuration, forKey: STTViewModel.chunkDurationStorageKey)
        }
    }

    // Streaming parameters
    var streamingDelayMs: Int = UserDefaults.standard.object(forKey: STTViewModel.streamingDelayMsStorageKey).map { _ in
        UserDefaults.standard.integer(forKey: STTViewModel.streamingDelayMsStorageKey)
    } ?? STTViewModel.defaultStreamingDelayMs {
        didSet {
            UserDefaults.standard.set(streamingDelayMs, forKey: STTViewModel.streamingDelayMsStorageKey)
        }
    } // .agent default

    // Model configuration
    var modelId: String = UserDefaults.standard.string(forKey: STTViewModel.modelIdStorageKey) ?? STTViewModel.defaultModelId {
        didSet {
            UserDefaults.standard.set(modelId, forKey: STTViewModel.modelIdStorageKey)
        }
    }
    private var loadedModelId: String?

    private var languageHint: String? {
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // Audio file
    var selectedAudioURL: URL?
    var originalAudioURL: URL?
    var audioFileName: String?

    // Embedded / Sidecar Song Lyrics state
    var rawEmbeddedLyrics: String?
    var embeddedLyricsSource: String?
    var usingEmbeddedLyrics: Bool = false

    var hasEmbeddedLyrics: Bool {
        guard let raw = rawEmbeddedLyrics?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return false
        }
        return !raw.isEmpty
    }

    // Audio player state
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    // Recording state
    var isRecording: Bool { recorder.isRecording }
    var recordingDuration: TimeInterval { recorder.recordingDuration }
    var audioLevel: Float { recorder.audioLevel }

    private var model: (any STTGenerationModel)?
    private let audioPlayer = AudioPlayer()
    private let recorder = AudioRecorderManager()
    private var cancellables = Set<AnyCancellable>()
    private var generationTask: Task<Void, Never>?

    var isModelLoaded: Bool {
        model != nil && loadedModelId == modelId
    }

    var canRecord: Bool {
        isRecording || (isModelLoaded && !isLoading && !isGenerating)
    }

    var supportsRealtimeRecording: Bool {
        model is Qwen3ASRModel || model is CohereTranscribeModel || model is MossTranscribeDiarizeModel
    }

    var usesRealtimeRecording: Bool {
        isRecording && streamingSession != nil
    }

    init() {
        let migrationKey = "VoicesApp.STTViewModel.migratedDefaultPhonon25V2"
        if !UserDefaults.standard.bool(forKey: migrationKey) {
            UserDefaults.standard.set(true, forKey: migrationKey)
            modelId = STTViewModel.defaultModelId
        }
        if maxTokens < 4096 {
            maxTokens = STTViewModel.defaultMaxTokens
        }
        setupAudioPlayerObservers()
        if let autoPath = UserDefaults.standard.string(forKey: "VoicesApp.STTViewModel.autoSelectAudioPath"),
           !autoPath.isEmpty,
           FileManager.default.fileExists(atPath: autoPath) {
            UserDefaults.standard.removeObject(forKey: "VoicesApp.STTViewModel.autoSelectAudioPath")
            selectAudioFile(URL(fileURLWithPath: autoPath))
        }
    }

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.isPlaying = value
            }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                self?.currentTime = value
            }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in
                guard let self else { return }
                self.duration = value
                if self.usingEmbeddedLyrics,
                   let raw = self.rawEmbeddedLyrics,
                   !raw.isEmpty,
                   value > 0 {
                    self.transcriptionText = SongLyricsExtractor.formatForTranscriptView(raw, duration: value)
                }
            }
            .store(in: &cancellables)

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

    func loadModel() async {
        guard model == nil || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading model..."

        do {
            let loadedModel = try await loadSTTModel(modelId)
            model = loadedModel
            loadedModelId = modelId
            DownloadedModelsManager.shared.refresh()
            generationProgress = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            generationProgress = ""
        }

        isLoading = false
    }

    func reloadModel() async {
        model = nil
        loadedModelId = nil
        Memory.clearCache()
        await loadModel()
    }

    func resetSettingsToDefaults() {
        modelId = STTViewModel.defaultModelId
        maxTokens = STTViewModel.defaultMaxTokens
        temperature = STTViewModel.defaultTemperature
        language = STTViewModel.defaultLanguage
        chunkDuration = STTViewModel.defaultChunkDuration
        streamingDelayMs = STTViewModel.defaultStreamingDelayMs

        UserDefaults.standard.removeObject(forKey: STTViewModel.modelIdStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.maxTokensStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.temperatureStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.languageStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.chunkDurationStorageKey)
        UserDefaults.standard.removeObject(forKey: STTViewModel.streamingDelayMsStorageKey)

        if loadedModelId != modelId {
            Task { await reloadModel() }
        }
    }

    private var lyricsExtractionTask: Task<Void, Never>?

    func selectAudioFile(_ url: URL) {
        lyricsExtractionTask?.cancel()
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        let tempDest = FileManager.default.temporaryDirectory
            .appendingPathComponent("stt_input_\(UUID().uuidString)_\(url.lastPathComponent)")
        let effectiveURL: URL
        // Prefer O(1) hard-link first so multi-hundred-MB .wav files never stall the UI thread copying PCM data
        if (try? FileManager.default.linkItem(at: url, to: tempDest)) != nil {
            effectiveURL = tempDest
        } else if (try? FileManager.default.copyItem(at: url, to: tempDest)) != nil {
            effectiveURL = tempDest
        } else {
            effectiveURL = url
        }

        originalAudioURL = url
        selectedAudioURL = effectiveURL
        audioFileName = url.lastPathComponent
        errorMessage = nil
        rawEmbeddedLyrics = nil
        embeddedLyricsSource = nil
        usingEmbeddedLyrics = false
        audioPlayer.loadAudio(from: effectiveURL)

        // Extract embedded or sidecar song lyrics asynchronously off the main actor so importing .wav/.mp3 never freezes UI
        let origURL = url
        let fileURL = effectiveURL
        lyricsExtractionTask = Task.detached(priority: .userInitiated) { [weak self] in
            let extracted = SongLyricsExtractor.extractLyrics(from: origURL, fallbackURL: fileURL)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.selectedAudioURL == fileURL else { return }
                if let extracted {
                    self.rawEmbeddedLyrics = extracted.lyrics
                    self.embeddedLyricsSource = extracted.source
                    self.usingEmbeddedLyrics = true
                    self.transcriptionText = SongLyricsExtractor.formatForTranscriptView(
                        extracted.lyrics,
                        duration: self.duration
                    )
                    self.generationProgress = ""
                }
            }
        }
    }

    func removeAudioFile() {
        guard !isGenerating && !isRecording else { return }

        lyricsExtractionTask?.cancel()
        audioPlayer.unloadAudio()
        selectedAudioURL = nil
        originalAudioURL = nil
        audioFileName = nil
        rawEmbeddedLyrics = nil
        embeddedLyricsSource = nil
        usingEmbeddedLyrics = false
        currentTime = 0
        duration = 0
        isPlaying = false
    }

    func applyEmbeddedLyricsIfAvailable() -> Bool {
        if let raw = rawEmbeddedLyrics, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            usingEmbeddedLyrics = true
            errorMessage = nil
            generationProgress = ""
            transcriptionText = SongLyricsExtractor.formatForTranscriptView(raw, duration: duration)
            return true
        }
        return false
    }

    func startTranscription(forceAI: Bool = false) {
        guard let audioURL = selectedAudioURL else {
            errorMessage = "No audio file selected"
            return
        }

        // If the song file has readable lyrics already detected, use the real lyrics instead of AI transcription
        if !forceAI && applyEmbeddedLyricsIfAvailable() {
            return
        }

        usingEmbeddedLyrics = false
        generationTask?.cancel()
        generationTask = Task {
            await transcribe(audioURL: audioURL, forceAI: forceAI)
        }
    }

    func transcribe(audioURL: URL, forceAI: Bool = false) async {
        // Check for readable song lyrics off the main thread before invoking AI ASR
        if !forceAI {
            if applyEmbeddedLyricsIfAvailable() {
                return
            }
            let origURL = originalAudioURL ?? audioURL
            let extracted = await Task.detached(priority: .userInitiated) {
                SongLyricsExtractor.extractLyrics(from: origURL, fallbackURL: audioURL)
            }.value
            if let extracted {
                rawEmbeddedLyrics = extracted.lyrics
                embeddedLyricsSource = extracted.source
                usingEmbeddedLyrics = true
                errorMessage = nil
                generationProgress = ""
                transcriptionText = SongLyricsExtractor.formatForTranscriptView(extracted.lyrics, duration: duration)
                return
            }
        }

        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        usingEmbeddedLyrics = false
        errorMessage = nil
        transcriptionText = ""
        generationProgress = "Loading audio..."
        resetGenerationStats()

        do {
            let (sampleRate, audioData) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
            if sampleRate != 16_000 {
                generationProgress = "Resampling \(sampleRate)Hz → 16000Hz..."
            }

            try await transcribe(audioData: audioData, with: model)

            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    private func transcribeRecording(audioData: MLXArray, clearExistingText: Bool = true) async {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        if clearExistingText {
            transcriptionText = ""
        }
        generationProgress = "Transcribing recording..."
        resetGenerationStats()

        do {
            try await transcribe(audioData: audioData, with: model)
            generationProgress = ""
        } catch is CancellationError {
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
            generationProgress = ""
        }

        isGenerating = false
    }

    static func cleanTranscriptForVoiceStudio(_ raw: String) -> String {
        let lines = raw.components(separatedBy: .newlines)
        let cleanedLines: [String] = lines.compactMap { line in
            var s = line
            // Strip leading/inline [0.00][S1] or [0.00] or [S1] or [SPEAKER_00] tags
            s = s.replacingOccurrences(
                of: #"\[\d+(?:[\.,]\d+)?\](?:\[[A-Za-z0-9_ -]{1,16}\])?"#,
                with: "",
                options: .regularExpression
            )
            s = s.replacingOccurrences(
                of: #"^\[(?:S\d{1,3}|SPEAKER_\d+|Speaker\s*\d+)\]\s*"#,
                with: "",
                options: [.regularExpression, .caseInsensitive]
            )
            // Strip Qwen3-ASR language prefix if present
            s = s.replacingOccurrences(
                of: #"^language\s+[A-Za-z]+\s*<asr_text>"#,
                with: "",
                options: [.regularExpression, .caseInsensitive]
            )
            // Strip Whisper special tokens if present
            s = s.replacingOccurrences(
                of: #"<\|[^|]+\|>"#,
                with: "",
                options: .regularExpression
            )
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
            return s.isEmpty ? nil : s
        }
        let joined = cleanedLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? raw.trimmingCharacters(in: .whitespacesAndNewlines) : joined
    }

    private static func formatClockTime(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        let m = total / 60
        let s = total % 60
        return String(format: "%d:%02d", m, s)
    }

    private func transcribe(audioData: MLXArray, with model: any STTGenerationModel) async throws {
        generationProgress = "Transcribing..."

        let parameters = generationParameters(for: model)
        let sampleRate = 16_000
        let monoAudio = audioData.ndim > 1 ? audioData.mean(axis: -1) : audioData
        let totalSamples = monoAudio.dim(0)
        let totalAudioSec = Float(totalSamples) / Float(sampleRate)

        // Split multi-minute songs and recordings at low-energy boundaries into acoustic windows
        // so every verse, chorus, and bridge from 0:00 to the end of the file is transcribed without truncation.
        let windowSec = min(max(chunkDuration, 15.0), 120.0)
        let windows: [(MLXArray, Float)]
        if totalAudioSec > max(24.0, windowSec + 2.0) {
            windows = splitAudioIntoChunks(
                monoAudio,
                sampleRate: sampleRate,
                chunkDuration: windowSec,
                minChunkDuration: 1.0,
                searchExpandSec: 3.0
            )
        } else {
            windows = [(monoAudio, 0.0)]
        }

        var committedLines: [String] = []
        var totalTokenCount = 0
        let overallStart = Date()

        for (windowIdx, (windowAudio, windowOffsetSec)) in windows.enumerated() {
            try Task.checkCancellation()
            let windowDurSec = Double(windowAudio.dim(0)) / Double(sampleRate)
            let startSec = Double(windowOffsetSec)
            let endSec = startSec + windowDurSec

            let windowPrefix: String
            if windows.count > 1 {
                windowPrefix = "Transcribing part \(windowIdx + 1)/\(windows.count) (\(Self.formatClockTime(startSec))–\(Self.formatClockTime(endSec)))"
            } else {
                windowPrefix = "Transcribing"
            }
            generationProgress = "\(windowPrefix)..."

            var windowStreamedText = ""
            var windowOutput: STTOutput?

            let windowParams = STTGenerateParameters(
                maxTokens: parameters.maxTokens,
                temperature: parameters.temperature,
                topP: parameters.topP,
                topK: parameters.topK,
                verbose: parameters.verbose,
                language: parameters.language,
                chunkDuration: max(windowSec + 10.0, 30.0),
                minChunkDuration: parameters.minChunkDuration,
                repetitionPenalty: parameters.repetitionPenalty,
                repetitionContextSize: parameters.repetitionContextSize
            )

            for try await event in model.generateStream(audio: windowAudio, generationParameters: windowParams) {
                try Task.checkCancellation()

                switch event {
                case .token(let token):
                    windowStreamedText += token
                    totalTokenCount += 1
                    let cleanPartial = Self.cleanTranscriptForVoiceStudio(windowStreamedText)
                    if !cleanPartial.isEmpty {
                        let liveLine = String(format: "[%.2f][S1] %@", startSec, cleanPartial)
                        if committedLines.isEmpty {
                            transcriptionText = liveLine
                        } else {
                            transcriptionText = (committedLines + [liveLine]).joined(separator: "\n")
                        }
                    }
                    generationProgress = "\(windowPrefix)... \(totalTokenCount) tokens"
                case .info(let info):
                    tokensPerSecond = info.tokensPerSecond
                    peakMemory = max(peakMemory, info.peakMemoryUsage)
                case .result(let output):
                    windowOutput = output
                    if output.generationTps > 0 {
                        tokensPerSecond = output.generationTps
                    }
                    peakMemory = max(peakMemory, output.peakMemoryUsage)
                }
            }

            if let output = windowOutput {
                var addedFromSegments = false
                if let segs = output.segments, !segs.isEmpty {
                    let formattedSegs = segs.compactMap { seg -> String? in
                        guard let rawText = seg["text"] as? String else { return nil }
                        let cleanSegText = Self.cleanTranscriptForVoiceStudio(rawText)
                        guard !cleanSegText.isEmpty else { return nil }
                        let relStart = (seg["start"] as? Double) ?? (seg["start"] as? Float).map(Double.init) ?? 0.0
                        let relEnd = (seg["end"] as? Double) ?? (seg["end"] as? Float).map(Double.init)
                        let absStart = (relStart >= startSec - 0.1 && startSec > 1.0) ? relStart : (startSec + relStart)
                        let speaker = (seg["speaker"] as? String) ?? "S1"
                        if let relEnd {
                            let absEnd = (relEnd >= startSec && startSec > 1.0) ? relEnd : (startSec + max(relEnd, relStart + 0.2))
                            return String(format: "[%.2f][%@] %@ [%.2f]", absStart, speaker, cleanSegText, absEnd)
                        } else {
                            return String(format: "[%.2f][%@] %@ [%.2f]", absStart, speaker, cleanSegText, endSec)
                        }
                    }
                    if !formattedSegs.isEmpty {
                        committedLines.append(contentsOf: formattedSegs)
                        addedFromSegments = true
                    }
                }
                if !addedFromSegments {
                    let fallbackRaw = output.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        ? windowStreamedText
                        : output.text
                    let cleanFallback = Self.cleanTranscriptForVoiceStudio(fallbackRaw)
                    if !cleanFallback.isEmpty {
                        committedLines.append(
                            String(format: "[%.2f][S1] %@ [%.2f]", startSec, cleanFallback, endSec)
                        )
                    }
                }
            } else {
                let cleanFallback = Self.cleanTranscriptForVoiceStudio(windowStreamedText)
                if !cleanFallback.isEmpty {
                    committedLines.append(
                        String(format: "[%.2f][S1] %@ [%.2f]", startSec, cleanFallback, endSec)
                    )
                }
            }

            if !committedLines.isEmpty {
                transcriptionText = committedLines.joined(separator: "\n")
            }
        }

        let elapsed = Date().timeIntervalSince(overallStart)
        if tokensPerSecond <= 0 && elapsed > 0 && totalTokenCount > 0 {
            tokensPerSecond = Double(totalTokenCount) / elapsed
        }
        generationProgress = ""
    }

    private func generationParameters(for model: any STTGenerationModel) -> STTGenerateParameters {
        let defaultParameters = model.defaultGenerationParameters
        return STTGenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: defaultParameters.topP,
            topK: defaultParameters.topK,
            verbose: false,
            language: languageHint,
            chunkDuration: chunkDuration,
            minChunkDuration: defaultParameters.minChunkDuration,
            repetitionPenalty: defaultParameters.repetitionPenalty,
            repetitionContextSize: defaultParameters.repetitionContextSize
        )
    }

    // MARK: - Live Recording & Streaming Transcription

    private var liveTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?
    private var streamingSession: StreamingInferenceSession?
    private var activeRecordingID: UUID?
    private var lastReadPos: Int = 0

    func startRecording() async {
        guard let model else {
            errorMessage = "Model not loaded"
            return
        }

        let recordingID = UUID()
        activeRecordingID = recordingID
        errorMessage = nil
        transcriptionText = ""
        rawEmbeddedLyrics = nil
        embeddedLyricsSource = nil
        usingEmbeddedLyrics = false
        audioPlayer.pause()
        resetGenerationStats()
        lastReadPos = 0

        do {
            try await recorder.startRecording()
        } catch {
            errorMessage = error.localizedDescription
            return
        }

        guard supportsRealtimeRecording else {
            generationProgress = "Recording microphone audio (click Stop Recording to transcribe)..."
            return
        }

        // Create streaming session
        let config = StreamingConfig(
            decodeIntervalSeconds: 1.0,
            maxCachedWindows: 60,
            delayPreset: .custom(ms: streamingDelayMs),
            language: languageHint,
            temperature: temperature,
            maxTokensPerPass: maxTokens
        )
        let session = StreamingInferenceSession(model: model, config: config)
        streamingSession = session

        // Listen to events from the session
        eventTask = Task {
            for await event in session.events {
                guard activeRecordingID == recordingID else { continue }
                switch event {
                case .displayUpdate(let confirmed, let provisional):
                    transcriptionText = confirmed + provisional
                case .confirmed:
                    break  // displayUpdate handles the UI
                case .provisional:
                    break
                case .stats(let stats):
                    tokensPerSecond = stats.tokensPerSecond
                    peakMemory = stats.peakMemoryGB
                case .ended(let fullText):
                    transcriptionText = fullText
                }
            }
            // Stream ended naturally — clean up
            if activeRecordingID == recordingID {
                activeRecordingID = nil
                streamingSession = nil
                eventTask = nil
            }
        }

        // Audio feed loop: read new samples every 100ms and feed to session
        liveTask = Task {
            while !Task.isCancelled && recorder.isRecording {
                if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                    lastReadPos = endPos
                    let samples = audio.asArray(Float.self)
                    session.feedAudio(samples: samples)
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    func stopRecording() {
        liveTask?.cancel()
        liveTask = nil

        // Feed any remaining audio, then stop session.
        if let session = streamingSession {
            if let (audio, endPos) = recorder.getAudio(from: lastReadPos) {
                lastReadPos = endPos
                let samples = audio.asArray(Float.self)
                session.feedAudio(samples: samples)
            }

            if let fullAudio = recorder.stopRecording() {
                saveRecordedAudioForPlayback(fullAudio)
            }

            // Stop promotes all provisional tokens and emits .ended
            // The eventTask will process .ended and clean up naturally
            session.stop()
            return
        }

        let capturedAudio = recorder.stopRecording()
        activeRecordingID = nil
        if let capturedAudio {
            saveRecordedAudioForPlayback(capturedAudio)
            generationTask?.cancel()
            generationTask = Task {
                await transcribeRecording(audioData: capturedAudio, clearExistingText: true)
            }
        } else {
            generationProgress = ""
        }
    }

    private func saveRecordedAudioForPlayback(_ audio: MLXArray) {
        let samples = audio.asArray(Float.self)
        guard !samples.isEmpty else { return }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mic_recording_\(UUID().uuidString).wav")
        if let writer = try? StreamingWAVWriter(url: tempURL, sampleRate: 16_000) {
            try? writer.writeChunk(samples)
            let finalURL = writer.finalize()
            selectedAudioURL = finalURL
            originalAudioURL = finalURL
            rawEmbeddedLyrics = nil
            embeddedLyricsSource = nil
            usingEmbeddedLyrics = false
            audioFileName = "Microphone Recording (\(String(format: "%.1fs", Double(samples.count) / 16_000.0))).wav"
            audioPlayer.loadAudio(from: finalURL)
        }
    }

    func cancelRecording() {
        liveTask?.cancel()
        liveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        recorder.cancelRecording()
        activeRecordingID = nil
        lastReadPos = 0
        generationProgress = ""
    }

    func stop() {
        liveTask?.cancel()
        liveTask = nil
        streamingSession?.cancel()
        streamingSession = nil
        eventTask?.cancel()
        eventTask = nil
        generationTask?.cancel()
        generationTask = nil
        activeRecordingID = nil

        if isRecording {
            recorder.cancelRecording()
            lastReadPos = 0
        }

        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    private func loadSTTModel(_ repo: String) async throws -> any STTGenerationModel {
        try await STT.loadModel(modelRepo: repo, hfToken: ModelCacheHelper.effectiveHFToken)
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

    func seek(to time: TimeInterval) {
        audioPlayer.seek(to: time)
    }

    func copyTranscription() {
        #if os(iOS)
        UIPasteboard.general.string = transcriptionText
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcriptionText, forType: .string)
        #endif
    }

    /// Computes word timings locked to `[start][speaker] ... [end]` ASR/LRC segment boundaries when present,
    /// or falls back to `WordTimingEngine.computeWordTimings` across the audio file duration.
    func computeTranscriptWordTimings(from rawTranscript: String, fallbackDuration: TimeInterval) -> [WordTiming] {
        let trimmed = rawTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let pattern = #"\[(\d+(?:[\.,]\d+)?)\]\[([A-Za-z0-9_ -]{1,24})\]"#
        if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
            let nsRange = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
            let matches = regex.matches(in: trimmed, range: nsRange)
            if !matches.isEmpty {
                let trailingRegex = try? NSRegularExpression(pattern: #"\[(\d+(?:[\.,]\d+)?)\]\s*$"#)
                var wordTimings: [WordTiming] = []
                var nextWordID = 0

                for (segIdx, match) in matches.enumerated() {
                    guard let matchRange = Range(match.range, in: trimmed),
                          let startRange = Range(match.range(at: 1), in: trimmed),
                          let segStart = Double(trimmed[startRange].replacingOccurrences(of: ",", with: ".")) else {
                        continue
                    }
                    let bodyEnd: String.Index
                    var nextSegStart: Double? = nil
                    if segIdx + 1 < matches.count,
                       let nr = Range(matches[segIdx + 1].range, in: trimmed) {
                        bodyEnd = nr.lowerBound
                        if let nsRange1 = Range(matches[segIdx + 1].range(at: 1), in: trimmed) {
                            nextSegStart = Double(trimmed[nsRange1].replacingOccurrences(of: ",", with: "."))
                        }
                    } else {
                        bodyEnd = trimmed.endIndex
                    }

                    var segBody = String(trimmed[matchRange.upperBound..<bodyEnd])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    var explicitEnd: Double? = nil
                    if let trailingRegex {
                        let r = NSRange(segBody.startIndex..<segBody.endIndex, in: segBody)
                        if let tm = trailingRegex.firstMatch(in: segBody, range: r),
                           let tr = Range(tm.range(at: 1), in: segBody),
                           let fullR = Range(tm.range, in: segBody) {
                            explicitEnd = Double(segBody[tr].replacingOccurrences(of: ",", with: "."))
                            segBody.removeSubrange(fullR)
                            segBody = segBody.trimmingCharacters(in: .whitespacesAndNewlines)
                        }
                    }

                    let cleanSeg = Self.cleanTranscriptForVoiceStudio(segBody)
                    let words = cleanSeg.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map(String.init)
                    guard !words.isEmpty else { continue }

                    let segEnd = max(
                        segStart + 0.25,
                        explicitEnd ?? nextSegStart ?? max(fallbackDuration, segStart + Double(words.count) * 0.32)
                    )
                    let span = max(0.2, segEnd - segStart)
                    let totalChars = max(1, words.reduce(0) { $0 + max(1, $1.count) })
                    var cursor = segStart
                    for (wIdx, word) in words.enumerated() {
                        let weight = Double(max(1, word.count)) / Double(totalChars)
                        let wEnd = (wIdx == words.count - 1) ? segEnd : (cursor + span * weight)
                        wordTimings.append(
                            WordTiming(
                                id: nextWordID,
                                word: word,
                                start: cursor,
                                end: max(cursor + 0.04, wEnd),
                                chunkIndex: segIdx,
                                paragraphIndex: segIdx
                            )
                        )
                        nextWordID += 1
                        cursor = wEnd
                    }
                }
                if !wordTimings.isEmpty {
                    return wordTimings
                }
            }
        }

        let cleanText = Self.cleanTranscriptForVoiceStudio(trimmed)
        let effectiveText = cleanText.isEmpty ? trimmed : cleanText
        return WordTimingEngine.computeWordTimings(
            text: effectiveText,
            audioURL: selectedAudioURL,
            duration: fallbackDuration
        )
    }

    func exportTranscription(format: String) {
        let trimmed = transcriptionText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        // Strip speaker/timestamp tags like [0.00][S1] or [12.34] for clean subtitle word timing
        let cleanText = Self.cleanTranscriptForVoiceStudio(trimmed)
        let effectiveText = cleanText.isEmpty ? trimmed : cleanText

        let estDuration = duration > 0 ? duration : max(2.0, Double(effectiveText.count) / 14.0)
        let timings = computeTranscriptWordTimings(from: trimmed, fallbackDuration: estDuration)
        let content: String
        let ext: String

        switch format.lowercased() {
        case "srt":
            content = SubtitleExporter.formatSRT(timings: timings)
            ext = "srt"
        case "vtt", "webvtt":
            content = SubtitleExporter.formatVTT(timings: timings)
            ext = "vtt"
        case "ass", "ssa":
            content = SubtitleExporter.formatASSKaraoke(timings: timings)
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
        case "md", "markdown":
            let title = audioFileName ?? "Studio Transcription"
            content = "# \(title)\n\n\(trimmed)\n"
            ext = "md"
        case "json":
            let payload: [String: Any] = [
                "text": trimmed,
                "model": modelId,
                "language": language,
                "duration_seconds": estDuration,
                "tokens_per_second": tokensPerSecond,
                "words": timings.map { ["word": $0.word, "start": $0.start, "end": $0.end] }
            ]
            if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]),
               let str = String(data: data, encoding: .utf8) {
                content = str
            } else {
                content = trimmed
            }
            ext = "json"
        default:
            content = trimmed
            ext = "txt"
        }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Transcription (.\(ext))"
        savePanel.nameFieldStringValue = "transcription.\(ext)"
        savePanel.begin { [weak self] response in
            guard response == .OK, let dest = savePanel.url else { return }
            do {
                try content.write(to: dest, atomically: true, encoding: .utf8)
            } catch {
                DispatchQueue.main.async {
                    self?.errorMessage = "Export failed: \(error.localizedDescription)"
                }
            }
        }
        #endif
    }

    func exportReadAlongVideo(
        aspect: ReadAlongVideoAspect = .landscape16x9,
        theme: ReadAlongTheme = .midnight
    ) {
        let trimmed = transcriptionText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            errorMessage = "Transcribe an audio file or recording before exporting Read-Along video"
            return
        }
        guard let sourceURL = selectedAudioURL ?? originalAudioURL ?? Voice.bundledStudioSamples().first?.url else {
            errorMessage = "No audio source available for Read-Along MP4 export"
            return
        }

        let cleanText = Self.cleanTranscriptForVoiceStudio(trimmed)
        let effectiveText = cleanText.isEmpty ? trimmed : cleanText
        let estDuration = duration > 0 ? duration : max(2.0, Double(effectiveText.count) / 14.0)
        let timings = computeTranscriptWordTimings(from: trimmed, fallbackDuration: estDuration)
        let aspectTag = aspect.rawValue.replacingOccurrences(of: ":", with: "x")
        let stem = (audioFileName as NSString?)?.deletingPathExtension ?? "transcription"
        let docTitle = audioFileName ?? "Transcribe Studio Read-Along"
        let modelBadge = modelId.split(separator: "/").last.map(String.init) ?? modelId

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Synchronized Read-Along Karaoke Video (.mp4)"
        savePanel.nameFieldStringValue = "\(stem)_readalong_\(aspectTag).mp4"
        savePanel.begin { [weak self] response in
            guard let self, response == .OK, let dest = savePanel.url else { return }
            let ext = sourceURL.pathExtension.isEmpty ? "wav" : sourceURL.pathExtension
            let snapshotAudio = FileManager.default.temporaryDirectory
                .appendingPathComponent("stt_video_snap_\(UUID().uuidString).\(ext)")
            try? FileManager.default.copyItem(at: sourceURL, to: snapshotAudio)
            let audioForVideo = FileManager.default.fileExists(atPath: snapshotAudio.path) ? snapshotAudio : sourceURL

            self.generationProgress = "Rendering 3-Level Delta-Band Read-Along MP4 (\(aspect.rawValue))..."
            Task.detached(priority: .userInitiated) {
                defer {
                    if audioForVideo == snapshotAudio {
                        try? FileManager.default.removeItem(at: snapshotAudio)
                    }
                }
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try ReadAlongVideoExporter.exportKaraokeVideo(
                        sourceWAV: audioForVideo,
                        timings: timings,
                        theme: theme,
                        aspect: aspect,
                        title: docTitle,
                        voiceName: modelBadge,
                        fallbackText: effectiveText,
                        to: dest
                    )
                    let elapsed = max(0.05, CFAbsoluteTimeGetCurrent() - t0)
                    let audioSec = max(1.0, timings.last?.end ?? estDuration)
                    let rtFactor = max(1, Int((audioSec / elapsed).rounded()))
                    let elapsedStr = String(format: "%.1fs", elapsed)
                    await MainActor.run {
                        self.generationProgress = "Exported Read-Along MP4 + SRT/VTT in \(elapsedStr) (\(rtFactor)× real-time): \(dest.lastPathComponent)"
                        Task { @MainActor in
                            try? await Task.sleep(for: .seconds(3.5))
                            if self.generationProgress.hasPrefix("Exported Read-Along MP4") {
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
        #endif
    }

    func clearTranscription() {
        guard !isGenerating && !isRecording else { return }

        transcriptionText = ""
        errorMessage = nil
        generationProgress = ""
        tokensPerSecond = 0
        peakMemory = 0
    }

    private func resetGenerationStats() {
        tokensPerSecond = 0
        peakMemory = 0
        Memory.clearCache()
        Memory.peakMemory = 0
    }

    private func resampleAudio(_ audio: MLXArray, from sourceSR: Int, to targetSR: Int) throws -> MLXArray {
        let samples = audio.asArray(Float.self)

        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceSR), channels: 1, interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(targetSR), channels: 1, interleaved: false
        ) else {
            throw NSError(domain: "STT", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio formats"])
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "STT", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
        }

        let inputFrameCount = AVAudioFrameCount(samples.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw NSError(domain: "STT", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create input buffer"])
        }
        inputBuffer.frameLength = inputFrameCount
        memcpy(inputBuffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        let ratio = Double(targetSR) / Double(sourceSR)
        let outputFrameCount = AVAudioFrameCount(Double(samples.count) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "STT", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create output buffer"])
        }

        var error: NSError?
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let error { throw error }

        let outputSamples = Array(UnsafeBufferPointer(
            start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)
        ))
        return MLXArray(outputSamples)
    }
}

// MARK: - Real Song Lyrics Extractor (Metadata Tags, ID3v2 USLT/SYLT, MP4 ©lyr, FLAC Vorbis, Sidecar .LRC/.TXT)

struct ExtractedSongLyrics: Sendable {
    let lyrics: String
    let source: String
}

enum SongLyricsExtractor {
    nonisolated static func extractLyrics(from primaryURL: URL, fallbackURL: URL? = nil) -> ExtractedSongLyrics? {
        let inspectURL = fallbackURL ?? primaryURL
        let ext = primaryURL.pathExtension.lowercased()

        // 1. Direct lyric, subtitle, e-book, office, or document file (.lrc, .srt, .vtt, .ass, .ssa, .sub, .sbv, .ttml, .txt, .md, .epub, .pdf, .docx, etc.)
        if ["lrc", "srt", "vtt", "webvtt", "ass", "ssa", "sub", "sbv", "ttml", "dfxp", "smi", "sami", "txt", "lyrics"].contains(ext),
           let text = readTextFile(at: inspectURL) ?? readTextFile(at: primaryURL),
           isValidLyrics(text) {
            return ExtractedSongLyrics(lyrics: text, source: ".\(ext.uppercased()) File")
        }
        if UniversalFileFormatCatalog.allReadableDocumentExtensions.contains(ext) && !UniversalFileFormatCatalog.allAudioAndVideoExtensions.contains(ext),
           let docText = try? SpeechTextCleaner.extractDocumentText(from: inspectURL),
           isValidLyrics(docText) {
            return ExtractedSongLyrics(lyrics: docText, source: ".\(ext.uppercased()) Document")
        }

        // 2. Sidecar lyric/subtitle file in the same directory (<song>.lrc, <song>.lyrics.txt, <song>.txt, <song>.srt, <song>.vtt, <song>.ass)
        if let sidecar = extractSidecarLyrics(near: primaryURL) {
            return sidecar
        }

        // 3. Fast SIMD / C-memmem binary tag parser (ID3v2 USLT/SYLT/TXXX, MP4/M4A ©lyr atom, FLAC/Ogg Vorbis LYRICS=)
        if let binaryLyrics = extractBinaryTagLyrics(from: inspectURL) {
            return binaryLyrics
        }

        // 4. AVFoundation metadata inspection (skip for raw uncompressed .wav/.pcm that had no binary tag header)
        if ext != "wav" && ext != "pcm" {
            if let avLyrics = extractAVAssetLyrics(from: inspectURL) {
                return avLyrics
            }
        }

        // 5. ffprobe metadata fallback (skip for .wav if no tag chunk was present)
        if ext != "wav" {
            if let ffprobeLyrics = extractFFProbeLyrics(from: inspectURL) {
                return ffprobeLyrics
            }
        }

        return nil
    }

    static func formatForTranscriptView(_ rawLyrics: String, duration: TimeInterval) -> String {
        let normalized = rawLyrics
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return "" }

        // Check if already formatted with [seconds][S1]
        if normalized.range(of: #"^\[\d+\.\d+\]\[S\d+\]"#, options: .regularExpression) != nil {
            return normalized
        }

        // Check if LRC timestamped format: [mm:ss.xx] or [mm:ss]
        let lrcLines = parseLRCLines(normalized)
        if !lrcLines.isEmpty {
            return lrcLines
                .map { String(format: "[%.2f][S1] %@", $0.seconds, $0.text) }
                .joined(separator: "\n")
        }

        // Check if SRT / VTT subtitle format: 00:00:12,340 --> 00:00:16,000
        let srtLines = parseSRTOrVTTLines(normalized)
        if !srtLines.isEmpty {
            return srtLines
                .map { String(format: "[%.2f][S1] %@", $0.seconds, $0.text) }
                .joined(separator: "\n")
        }

        // Plain unsynchronized song lyrics: preserve every lyric line and space timestamps across the track duration
        let lines = normalized
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else { return normalized }

        let effectiveDuration = duration > 1.0 ? duration : max(15.0, Double(lines.count) * 3.5)
        let step = effectiveDuration / Double(max(1, lines.count))
        return lines.enumerated().map { idx, line in
            let startSec = Double(idx) * step
            return String(format: "[%.2f][S1] %@", startSec, line)
        }.joined(separator: "\n")
    }

    // MARK: - Sidecar File Lookup

    private static func extractSidecarLyrics(near audioURL: URL) -> ExtractedSongLyrics? {
        let dir = audioURL.deletingLastPathComponent()
        let baseName = audioURL.deletingPathExtension().lastPathComponent
        guard !baseName.isEmpty else { return nil }

        let suffixes = [".lrc", ".LRC", ".lyrics.txt", ".lyrics", ".srt", ".vtt", ".ass", ".ssa", ".sub", ".sbv", ".ttml", ".txt"]
        for suffix in suffixes {
            let candidate = dir.appendingPathComponent(baseName + suffix)
            if FileManager.default.fileExists(atPath: candidate.path),
               let content = readTextFile(at: candidate),
               isValidLyrics(content) {
                let label = suffix.uppercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
                return ExtractedSongLyrics(lyrics: content, source: "Sidecar \(label)")
            }
        }
        return nil
    }

    // MARK: - AVFoundation Metadata Extraction

    private static func extractAVAssetLyrics(from url: URL) -> ExtractedSongLyrics? {
        let asset = AVURLAsset(url: url)

        // Direct asset.lyrics property
        if let directLyrics = asset.lyrics?.trimmingCharacters(in: .whitespacesAndNewlines),
           isValidLyrics(directLyrics) {
            return ExtractedSongLyrics(lyrics: directLyrics, source: "Embedded Audio Lyrics")
        }

        var allItems: [AVMetadataItem] = asset.commonMetadata
        for fmt in asset.availableMetadataFormats {
            allItems.append(contentsOf: asset.metadata(forFormat: fmt))
        }

        for item in allItems {
            let commonKey = item.commonKey?.rawValue.lowercased() ?? ""
            let identifier = item.identifier?.rawValue.lowercased() ?? ""
            let keyString = (item.key as? String)?.lowercased() ?? String(describing: item.key).lowercased()

            let isLyricItem =
                commonKey == "lyrics" ||
                identifier.contains("lyrics") ||
                identifier.contains("uslt") ||
                identifier.contains("sylt") ||
                identifier.contains("%a9lyr") ||
                identifier.contains("©lyr") ||
                keyString.contains("uslt") ||
                keyString.contains("sylt") ||
                keyString == "ult" ||
                keyString == "slt" ||
                keyString.contains("lyr") ||
                keyString.contains("lyrics")

            guard isLyricItem else { continue }

            if let str = item.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
               isValidLyrics(str) {
                return ExtractedSongLyrics(lyrics: str, source: "Embedded Metadata Lyrics")
            }
            if let rawVal = item.value as? String,
               isValidLyrics(rawVal) {
                return ExtractedSongLyrics(lyrics: rawVal.trimmingCharacters(in: .whitespacesAndNewlines), source: "Embedded Metadata Lyrics")
            }
            if let data = item.dataValue,
               let decoded = decodeID3USLTPayload(data) ?? decodeTextData(data),
               isValidLyrics(decoded) {
                return ExtractedSongLyrics(lyrics: decoded, source: "Embedded Tag Lyrics")
            }
        }

        return nil
    }

    // MARK: - Native Binary Tag Parser (Zero-Allocation Data.range(of:) SIMD Search)

    nonisolated private static func extractBinaryTagLyrics(from url: URL) -> ExtractedSongLyrics? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        // Read up to first 512 KB (covers ID3v2 headers, FLAC Vorbis blocks, and MP4 moov atoms at start)
        let headLimit = 512 * 1024
        guard let headData = try? handle.read(upToCount: headLimit), !headData.isEmpty else {
            return nil
        }

        if let found = scanDataChunkForLyrics(headData) {
            return found
        }

        // Check trailing 256 KB of file (for MP4 moov at end of file or WAV/AIFF trailing ID3 chunk)
        if let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.uint64Value,
           fileSize > UInt64(headData.count) {
            let tailSize = min(UInt64(256 * 1024), fileSize)
            try? handle.seek(toOffset: fileSize - tailSize)
            if let tailData = try? handle.read(upToCount: Int(tailSize)), !tailData.isEmpty {
                if let found = scanDataChunkForLyrics(tailData) {
                    return found
                }
            }
        }

        return nil
    }

    nonisolated private static func scanDataChunkForLyrics(_ data: Data) -> ExtractedSongLyrics? {
        // 1. Check ID3v2 only if "ID3" signature exists in data (uses fast C memmem)
        let id3Sig = Data([0x49, 0x44, 0x33])
        if data.range(of: id3Sig) != nil, let id3Lyrics = parseID3v2Lyrics(in: data) {
            return id3Lyrics
        }

        // 2. Check MP4 / M4A ©lyr atom only if [0xA9, 'l', 'y', 'r'] exists in data
        let mp4Sig = Data([0xA9, 0x6C, 0x79, 0x72])
        if data.range(of: mp4Sig) != nil, let mp4Lyrics = parseMP4LyricsAtom(in: data) {
            return mp4Lyrics
        }

        // 3. Check FLAC / Ogg Vorbis comments only if "LYRICS=" / "lyrics=" exists in data
        if let vorbisLyrics = parseVorbisCommentLyrics(in: data) {
            return vorbisLyrics
        }

        return nil
    }

    nonisolated private static func parseID3v2Lyrics(in data: Data) -> ExtractedSongLyrics? {
        let id3Sig = Data([0x49, 0x44, 0x33])
        var searchRange = data.startIndex..<data.endIndex
        var id3Offsets: [Int] = []

        while let match = data.range(of: id3Sig, options: [], in: searchRange) {
            let idx = data.distance(from: data.startIndex, to: match.lowerBound)
            if idx + 10 <= data.count {
                let ver = data[match.lowerBound + 3]
                let rev = data[match.lowerBound + 4]
                if (ver == 2 || ver == 3 || ver == 4) && rev == 0 {
                    id3Offsets.append(idx)
                }
            }
            let nextStart = match.upperBound
            guard nextStart < data.endIndex else { break }
            searchRange = nextStart..<data.endIndex
        }

        guard !id3Offsets.isEmpty else { return nil }
        let bytes = [UInt8](data)

        for base in id3Offsets {
            guard base + 10 <= bytes.count else { continue }
            let version = Int(bytes[base + 3])
            let flags = bytes[base + 5]
            let tagSize = synchsafeInt(bytes[base + 6], bytes[base + 7], bytes[base + 8], bytes[base + 9])
            guard tagSize > 0 else { continue }

            var pos = base + 10
            let endPos = min(bytes.count, pos + tagSize)

            // Skip optional extended header in v2.3 / v2.4
            if (flags & 0x40) != 0 && pos + 4 <= endPos {
                let extSize = version == 4
                    ? synchsafeInt(bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3])
                    : Int(UInt32(bytes[pos]) << 24 | UInt32(bytes[pos + 1]) << 16 | UInt32(bytes[pos + 2]) << 8 | UInt32(bytes[pos + 3]))
                pos += (version == 4 ? extSize : extSize + 4)
            }

            let headerLen = version == 2 ? 6 : 10
            while pos + headerLen <= endPos {
                if bytes[pos] == 0 { break } // Padding

                let frameID: String
                let frameSize: Int
                var frameFlags2: UInt8 = 0

                if version == 2 {
                    frameID = String(bytes: bytes[pos..<(pos + 3)], encoding: .ascii) ?? ""
                    frameSize = (Int(bytes[pos + 3]) << 16) | (Int(bytes[pos + 4]) << 8) | Int(bytes[pos + 5])
                } else {
                    frameID = String(bytes: bytes[pos..<(pos + 4)], encoding: .ascii) ?? ""
                    if version == 4 {
                        frameSize = synchsafeInt(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7])
                    } else {
                        frameSize = (Int(bytes[pos + 4]) << 24) | (Int(bytes[pos + 5]) << 16) | (Int(bytes[pos + 6]) << 8) | Int(bytes[pos + 7])
                    }
                    frameFlags2 = bytes[pos + 9]
                }

                pos += headerLen
                guard frameSize > 0, pos + frameSize <= endPos else { break }

                var payloadSlice = Array(bytes[pos..<(pos + frameSize)])
                pos += frameSize

                // Handle ID3v2.4 data length indicator (flag bit 0x01) and unsynchronization (flag bit 0x02)
                if version == 4 {
                    if (frameFlags2 & 0x01) != 0 && payloadSlice.count >= 4 {
                        payloadSlice.removeFirst(4)
                    }
                    if (frameFlags2 & 0x02) != 0 {
                        payloadSlice = removeID3Unsynchronization(payloadSlice)
                    }
                }

                let payloadData = Data(payloadSlice)

                if frameID == "USLT" || frameID == "ULT" {
                    if let decoded = decodeID3USLTPayload(payloadData), isValidLyrics(decoded) {
                        return ExtractedSongLyrics(lyrics: decoded, source: "ID3v2 \(frameID) Lyrics")
                    }
                } else if frameID == "SYLT" || frameID == "SLT" {
                    if let decoded = decodeID3SYLTPayload(payloadData), isValidLyrics(decoded) {
                        return ExtractedSongLyrics(lyrics: decoded, source: "ID3v2 Synced Lyrics")
                    }
                } else if frameID == "TXXX" || frameID == "TXX" {
                    if let decoded = decodeID3TXXXLyricsPayload(payloadData), isValidLyrics(decoded) {
                        return ExtractedSongLyrics(lyrics: decoded, source: "ID3v2 TXXX Lyrics")
                    }
                }
            }
        }

        return nil
    }

    private static func decodeID3USLTPayload(_ data: Data) -> String? {
        let bytes = [UInt8](data)
        // Minimum: 1 byte encoding + 3 bytes language + 1 byte descriptor terminator
        guard bytes.count >= 5 else { return nil }
        let encoding = bytes[0]
        let contentStart = 4 // after encoding (1) + language (3)

        guard contentStart < bytes.count else { return nil }
        let body = Array(bytes[contentStart...])

        let lyricBytes: [UInt8]
        if encoding == 1 || encoding == 2 {
            // UTF-16: descriptor ends with \0\0 on even index
            var descEnd = 0
            var found = false
            var i = 0
            while i + 1 < body.count {
                if body[i] == 0 && body[i + 1] == 0 {
                    descEnd = i + 2
                    found = true
                    break
                }
                i += 2
            }
            lyricBytes = found && descEnd <= body.count ? Array(body[descEnd...]) : body
        } else {
            // ISO-8859-1 (0) or UTF-8 (3): descriptor ends with single \0
            if let nullIdx = body.firstIndex(of: 0) {
                lyricBytes = Array(body[(nullIdx + 1)...])
            } else {
                lyricBytes = body
            }
        }

        return decodeEncodedBytes(lyricBytes, encodingByte: encoding)
    }

    private static func decodeID3SYLTPayload(_ data: Data) -> String? {
        let bytes = [UInt8](data)
        // 1 encoding + 3 lang + 1 timestampFormat + 1 contentType + descriptor
        guard bytes.count >= 7 else { return nil }
        let encoding = bytes[0]
        let timestampFormat = bytes[4] // 2 = milliseconds
        let body = Array(bytes[6...])

        var idx = 0
        if encoding == 1 || encoding == 2 {
            while idx + 1 < body.count {
                if body[idx] == 0 && body[idx + 1] == 0 {
                    idx += 2
                    break
                }
                idx += 2
            }
        } else {
            if let nullIdx = body.firstIndex(of: 0) {
                idx = nullIdx + 1
            }
        }

        var lines: [String] = []
        while idx < body.count {
            let textEnd: Int
            let nextIdx: Int
            if encoding == 1 || encoding == 2 {
                var j = idx
                var found = body.count
                while j + 1 < body.count {
                    if body[j] == 0 && body[j + 1] == 0 {
                        found = j
                        break
                    }
                    j += 2
                }
                textEnd = found
                nextIdx = found + 2
            } else {
                if let nullPos = body[idx...].firstIndex(of: 0) {
                    textEnd = nullPos
                    nextIdx = nullPos + 1
                } else {
                    break
                }
            }

            let syllableBytes = Array(body[idx..<textEnd])
            guard nextIdx + 4 <= body.count else { break }
            let timeVal = (UInt32(body[nextIdx]) << 24) |
                          (UInt32(body[nextIdx + 1]) << 16) |
                          (UInt32(body[nextIdx + 2]) << 8) |
                          UInt32(body[nextIdx + 3])
            idx = nextIdx + 4

            if let text = decodeEncodedBytes(syllableBytes, encodingByte: encoding)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !text.isEmpty {
                let sec = timestampFormat == 2 ? Double(timeVal) / 1000.0 : Double(timeVal)
                let mm = Int(sec) / 60
                let ss = sec - Double(mm * 60)
                lines.append(String(format: "[%02d:%05.2f] %@", mm, ss, text))
            }
        }

        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }

    private static func decodeID3TXXXLyricsPayload(_ data: Data) -> String? {
        let bytes = [UInt8](data)
        guard bytes.count >= 4 else { return nil }
        let encoding = bytes[0]
        let body = Array(bytes[1...])

        let descBytes: [UInt8]
        let valBytes: [UInt8]
        if encoding == 1 || encoding == 2 {
            var splitIdx: Int?
            var i = 0
            while i + 1 < body.count {
                if body[i] == 0 && body[i + 1] == 0 {
                    splitIdx = i
                    break
                }
                i += 2
            }
            guard let s = splitIdx else { return nil }
            descBytes = Array(body[0..<s])
            valBytes = Array(body[(s + 2)...])
        } else {
            guard let s = body.firstIndex(of: 0) else { return nil }
            descBytes = Array(body[0..<s])
            valBytes = Array(body[(s + 1)...])
        }

        guard let desc = decodeEncodedBytes(descBytes, encodingByte: encoding)?.lowercased() else {
            return nil
        }
        guard desc.contains("lyrics") || desc == "lyr" else { return nil }
        return decodeEncodedBytes(valBytes, encodingByte: encoding)
    }

    nonisolated private static func parseMP4LyricsAtom(in data: Data) -> ExtractedSongLyrics? {
        let mp4Sig = Data([0xA9, 0x6C, 0x79, 0x72])
        var searchRange = data.startIndex..<data.endIndex
        while let match = data.range(of: mp4Sig, options: [], in: searchRange) {
            let i = data.distance(from: data.startIndex, to: match.lowerBound)
            let dataAtomPos = i + 4
            if dataAtomPos + 16 <= data.count {
                let d0 = data[data.startIndex + dataAtomPos + 4]
                let d1 = data[data.startIndex + dataAtomPos + 5]
                let d2 = data[data.startIndex + dataAtomPos + 6]
                let d3 = data[data.startIndex + dataAtomPos + 7]
                if d0 == 0x64 && d1 == 0x61 && d2 == 0x74 && d3 == 0x61 {
                    let s0 = Int(data[data.startIndex + dataAtomPos])
                    let s1 = Int(data[data.startIndex + dataAtomPos + 1])
                    let s2 = Int(data[data.startIndex + dataAtomPos + 2])
                    let s3 = Int(data[data.startIndex + dataAtomPos + 3])
                    let dataSize = (s0 << 24) | (s1 << 16) | (s2 << 8) | s3
                    let payloadStart = dataAtomPos + 16
                    let payloadEnd = dataAtomPos + dataSize
                    if dataSize > 16 && payloadEnd <= data.count {
                        let slice = data.subdata(in: (data.startIndex + payloadStart)..<(data.startIndex + payloadEnd))
                        if let text = decodeTextData(slice), isValidLyrics(text) {
                            return ExtractedSongLyrics(lyrics: text, source: "M4A ©lyr Metadata")
                        }
                    }
                }
            }
            let nextStart = match.upperBound
            guard nextStart < data.endIndex else { break }
            searchRange = nextStart..<data.endIndex
        }
        return nil
    }

    nonisolated private static func parseVorbisCommentLyrics(in data: Data) -> ExtractedSongLyrics? {
        let prefixes: [Data] = [
            Data("UNSYNCEDLYRICS=".utf8),
            Data("unsyncedlyrics=".utf8),
            Data("LYRICS=".utf8),
            Data("lyrics=".utf8),
            Data("Lyrics=".utf8)
        ]

        for p in prefixes {
            var searchRange = data.startIndex..<data.endIndex
            while let match = data.range(of: p, options: [], in: searchRange) {
                let i = data.distance(from: data.startIndex, to: match.lowerBound)
                if i >= 4 {
                    let b0 = UInt32(data[data.startIndex + i - 4])
                    let b1 = UInt32(data[data.startIndex + i - 3]) << 8
                    let b2 = UInt32(data[data.startIndex + i - 2]) << 16
                    let b3 = UInt32(data[data.startIndex + i - 1]) << 24
                    let commentLen = Int(b0 | b1 | b2 | b3)
                    if commentLen > p.count && commentLen <= 524_288 && i + commentLen <= data.count {
                        let valStart = data.startIndex + i + p.count
                        let valEnd = data.startIndex + i + commentLen
                        let slice = data.subdata(in: valStart..<valEnd)
                        if let str = String(data: slice, encoding: .utf8), isValidLyrics(str) {
                            return ExtractedSongLyrics(lyrics: str, source: "FLAC/Vorbis Lyrics")
                        }
                    }
                }
                let nextStart = match.upperBound
                guard nextStart < data.endIndex else { break }
                searchRange = nextStart..<data.endIndex
            }
        }
        return nil
    }

    // MARK: - ffprobe Fallback

    private static func extractFFProbeLyrics(from url: URL) -> ExtractedSongLyrics? {
        #if os(macOS)
        let candidates = ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe"]
        guard let ffprobePath = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            return nil
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffprobePath)
        proc.arguments = ["-v", "quiet", "-print_format", "json", "-show_format", url.path]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice

        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let format = root["format"] as? [String: Any],
                  let tags = format["tags"] as? [String: Any] else {
                return nil
            }
            for (k, v) in tags {
                let lower = k.lowercased()
                if lower == "lyrics" || lower.hasPrefix("lyrics-") || lower.contains("unsyncedlyrics") || lower == "uslt" || lower == "sylt" {
                    if let str = v as? String, isValidLyrics(str) {
                        return ExtractedSongLyrics(lyrics: str, source: "Embedded Audio Tag (\(k))")
                    }
                }
            }
        } catch {
            return nil
        }
        #endif
        return nil
    }

    // MARK: - LRC & Subtitle Parsing

    private static func parseLRCLines(_ text: String) -> [(seconds: Double, text: String)] {
        let lines = text.components(separatedBy: .newlines)
        guard let tagRegex = try? NSRegularExpression(pattern: #"\[(\d{1,3}):(\d{2}(?:[\.,]\d{1,3})?)\]"#) else {
            return []
        }

        var results: [(seconds: Double, text: String)] = []
        for rawLine in lines {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            // Ignore LRC metadata headers like [ar:Artist], [ti:Title], [al:Album], [by:Creator], [offset:0]
            if trimmed.range(of: #"^\[(?:ar|ti|al|au|by|length|offset|re|ve|id|tool):"#, options: [.regularExpression, .caseInsensitive]) != nil {
                continue
            }

            let nsLine = trimmed as NSString
            let matches = tagRegex.matches(in: trimmed, range: NSRange(location: 0, length: nsLine.length))
            guard !matches.isEmpty else { continue }

            var lyricPart = trimmed
            if let lastMatch = matches.last {
                let afterIdx = lastMatch.range.location + lastMatch.range.length
                lyricPart = nsLine.substring(from: afterIdx).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            // Strip inline word-level LRC tags like <00:12.40>
            lyricPart = lyricPart.replacingOccurrences(of: #"<\d{1,3}:\d{2}(?:[\.,]\d{1,3})?>"#, with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !lyricPart.isEmpty else { continue }

            for match in matches {
                let minStr = nsLine.substring(with: match.range(at: 1))
                let secStr = nsLine.substring(with: match.range(at: 2)).replacingOccurrences(of: ",", with: ".")
                if let mins = Double(minStr), let secs = Double(secStr) {
                    results.append((seconds: mins * 60.0 + secs, text: lyricPart))
                }
            }
        }

        return results.sorted { $0.seconds < $1.seconds }
    }

    private static func parseSRTOrVTTLines(_ text: String) -> [(seconds: Double, text: String)] {
        guard text.contains("-->") else { return [] }
        let blocks = text.components(separatedBy: "\n\n")
        guard let timeRegex = try? NSRegularExpression(pattern: #"(?:(\d{1,2}):)?(\d{2}):(\d{2})[\.,](\d{1,3})\s*-->"#) else {
            return []
        }

        var results: [(seconds: Double, text: String)] = []
        for block in blocks {
            let lines = block.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard let timingIdx = lines.firstIndex(where: { $0.contains("-->") }),
                  timingIdx + 1 < lines.count else { continue }
            let timingLine = lines[timingIdx]
            let nsTiming = timingLine as NSString
            guard let m = timeRegex.firstMatch(in: timingLine, range: NSRange(location: 0, length: nsTiming.length)) else { continue }

            let hours = m.range(at: 1).location != NSNotFound ? (Double(nsTiming.substring(with: m.range(at: 1))) ?? 0) : 0
            let mins = Double(nsTiming.substring(with: m.range(at: 2))) ?? 0
            let secs = Double(nsTiming.substring(with: m.range(at: 3))) ?? 0
            let msStr = nsTiming.substring(with: m.range(at: 4))
            let msScale = pow(10.0, Double(msStr.count))
            let ms = (Double(msStr) ?? 0) / max(1.0, msScale)

            let startSec = hours * 3600.0 + mins * 60.0 + secs + ms
            let textLines = lines[(timingIdx + 1)...].joined(separator: " ")
                .replacingOccurrences(of: #"<[^>]+>"#, with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !textLines.isEmpty {
                results.append((seconds: startSec, text: textLines))
            }
        }
        return results
    }

    // MARK: - Helpers

    private static func isValidLyrics(_ text: String) -> Bool {
        let trimmed = text
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 8 else { return false }
        let words = trimmed.split { $0.isWhitespace || $0.isNewline }
        return words.count >= 2
    }

    private static func readTextFile(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decodeTextData(data)
    }

    private static func decodeTextData(_ data: Data) -> String? {
        for enc: String.Encoding in [.utf8, .utf16, .utf16LittleEndian, .utf16BigEndian, .windowsCP1252, .isoLatin1] {
            if let str = String(data: data, encoding: enc)?
                .replacingOccurrences(of: "\0", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !str.isEmpty {
                return str
            }
        }
        return nil
    }

    private static func decodeEncodedBytes(_ bytes: [UInt8], encodingByte: UInt8) -> String? {
        guard !bytes.isEmpty else { return nil }
        let data = Data(bytes)
        let raw: String?
        switch encodingByte {
        case 0:
            raw = String(data: data, encoding: .isoLatin1) ?? String(data: data, encoding: .windowsCP1252)
        case 1:
            raw = String(data: data, encoding: .utf16) ?? String(data: data, encoding: .utf16LittleEndian)
        case 2:
            raw = String(data: data, encoding: .utf16BigEndian)
        case 3:
            raw = String(data: data, encoding: .utf8)
        default:
            raw = decodeTextData(data)
        }
        return raw?.replacingOccurrences(of: "\0", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func synchsafeInt(_ b0: UInt8, _ b1: UInt8, _ b2: UInt8, _ b3: UInt8) -> Int {
        return (Int(b0 & 0x7F) << 21) | (Int(b1 & 0x7F) << 14) | (Int(b2 & 0x7F) << 7) | Int(b3 & 0x7F)
    }

    private static func removeID3Unsynchronization(_ bytes: [UInt8]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(bytes.count)
        var i = 0
        while i < bytes.count {
            out.append(bytes[i])
            if bytes[i] == 0xFF && i + 1 < bytes.count && bytes[i + 1] == 0x00 {
                i += 2
            } else {
                i += 1
            }
        }
        return out
    }
}
