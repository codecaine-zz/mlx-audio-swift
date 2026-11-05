//
//  STSViewModel.swift
//  VoicesApp
//
//  Created by Alpay Calalli on 25.02.26.
//

import AVFoundation
import Combine
import Foundation
import MLX
import MLXAudioCore
import MLXAudioLID
import MLXAudioSTS
import MLXAudioVAD
import SwiftUI
#if os(macOS)
import AppKit
#endif

enum AudioStudioMode: String, CaseIterable, Identifiable {
    case separation = "Source Separation (SAM Audio)"
    case enhancement = "Speech Enhancement & Denoise"
    case diarization = "Speaker Diarization & VAD"
    case languageId = "Spoken Language ID (LID)"

    var id: String { rawValue }
}

@MainActor
@Observable
class STSViewModel {
    private static let defaultModelId = "mlx-community/sam-audio-base"
    static let presetModels: [(name: String, id: String)] = [
        ("SAM Audio Base (Separation)", "mlx-community/sam-audio-base"),
        ("SAM Audio Large FP16 (Separation)", "mlx-community/sam-audio-large-fp16"),
        ("SAM Audio Small (Separation)", "mlx-community/sam-audio-small"),
        ("NVIDIA NemotronLabs RE-USE (8–48kHz Enhance)", "nvidia/NVIDIA-NemotronLabs-RE-USE"),
        ("MossFormer2 SE FP16 (Speech Enhance)", "starkdmi/MossFormer2-SE-fp16"),
        ("DeepFilterNet MLX (Realtime Denoise)", "mlx-community/DeepFilterNet-mlx"),
        ("NVIDIA Nemotron 3 Diarization (8-Spk · BF16)", "mlx-community/Nemotron-3-Diarization"),
        ("NVIDIA Nemotron 3 Diarization (8-Spk · 8-bit)", "mlx-community/Nemotron-3-Diarization-8bit"),
        ("NVIDIA Sortformer v2.1 Diarization (4-Spk · FP16)", "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"),
        ("NVIDIA Sortformer v2 Diarization (4-Spk · FP16)", "mlx-community/diar_streaming_sortformer_4spk-v2-fp16"),
        ("Silero VAD v6 (Voice Activity Detection)", "mlx-community/silero-vad-v6"),
        ("FSMN VAD (Voice Activity Detection)", "mlx-community/fsmn-vad"),
        ("VoxLingua107 ECAPA (107-Lang ID)", "beshkenadze/lang-id-voxlingua107-ecapa-mlx"),
        ("MMS-LID 256 (256-Lang ID)", "facebook/mms-lid-256"),
    ]
    static let separationPresets: [String] = [
        "speech", "music", "vocals", "drums", "bass", "piano", "guitar", "noise", "applause"
    ]
    private static let defaultSeparationDescription = "speech"
    private static let defaultUseResidual = false
    private static let modelIdStorageKey = "VoicesApp.STSViewModel.modelId"
    private static let separationDescriptionStorageKey = "VoicesApp.STSViewModel.separationDescription"
    private static let useResidualStorageKey = "VoicesApp.STSViewModel.useResidual"

    static func mode(forModelId id: String) -> AudioStudioMode {
        let lower = id.lowercased()
        if lower.contains("voxlingua") || lower.contains("mms-lid") || lower.contains("ecapa") {
            return .languageId
        } else if lower.contains("diar") || lower.contains("sortformer") || lower.contains("vad") {
            return .diarization
        } else if lower.contains("mossformer") || lower.contains("deepfilter") || lower.contains("re-use") {
            return .enhancement
        }
        return .separation
    }

    static func defaultOrCachedModelId(for mode: AudioStudioMode) -> String {
        let candidates = presetModels.filter { Self.mode(forModelId: $0.id) == mode }
        if let cached = candidates.first(where: { ModelCacheHelper.isModelCached($0.id) }) {
            return cached.id
        }
        switch mode {
        case .separation:
            return "mlx-community/sam-audio-base"
        case .enhancement:
            return "nvidia/NVIDIA-NemotronLabs-RE-USE"
        case .diarization:
            return "mlx-community/Nemotron-3-Diarization"
        case .languageId:
            return "beshkenadze/lang-id-voxlingua107-ecapa-mlx"
        }
    }

    var studioMode: AudioStudioMode = .separation {
        didSet {
            UserDefaults.standard.set(studioMode.rawValue, forKey: "VoicesApp.STSViewModel.studioMode")
            if Self.mode(forModelId: modelId) != studioMode {
                modelId = Self.defaultOrCachedModelId(for: studioMode)
            }
        }
    }

    var isLoading = false
    var isGenerating = false
    var generationProgress: String = ""
    var errorMessage: String?
    var audioURL: URL?
    var originalInputURL: URL?
    private var targetAudioURL: URL?
    private var residualAudioURL: URL?
    var isPlayingOriginal: Bool = false
    var lidPredictions: [LanguagePrediction] = []
    var lidTopLanguage: String?
    var diarizationSegments: [DiarizationSegment] = []
    var diarizationSpeakerCount: Int = 0
    var diarizationRTTM: String = ""

    // What to isolate. "speech", "music", "drums", etc.
    var separationDescription: String = UserDefaults.standard.string(forKey: STSViewModel.separationDescriptionStorageKey) ?? STSViewModel.defaultSeparationDescription {
        didSet {
            UserDefaults.standard.set(separationDescription, forKey: STSViewModel.separationDescriptionStorageKey)
        }
    }

    // Whether to play the residual (everything *except* the target) instead
    var useResidual: Bool = UserDefaults.standard.object(forKey: STSViewModel.useResidualStorageKey).map { _ in
        UserDefaults.standard.bool(forKey: STSViewModel.useResidualStorageKey)
    } ?? STSViewModel.defaultUseResidual {
        didSet {
            UserDefaults.standard.set(useResidual, forKey: STSViewModel.useResidualStorageKey)
            if studioMode == .separation, let nextURL = (useResidual ? residualAudioURL : targetAudioURL) {
                let savedTime = currentTime
                let wasPlaying = isPlaying
                audioURL = nextURL
                if !isPlayingOriginal {
                    audioPlayer.loadAudio(from: nextURL)
                    if savedTime > 0 {
                        audioPlayer.seek(to: savedTime)
                    }
                    if wasPlaying {
                        audioPlayer.play()
                    }
                }
            }
        }
    }

    // Model config
    var modelId: String = UserDefaults.standard.string(forKey: STSViewModel.modelIdStorageKey) ?? STSViewModel.defaultModelId {
        didSet {
            UserDefaults.standard.set(modelId, forKey: STSViewModel.modelIdStorageKey)
        }
    }
    private var loadedModelId: String?

    // Audio player state (manually synced from AudioPlayer via Combine)
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0

    private var model: SAMAudio?
    private var loadedSTS: LoadedSTSModel?
    private var loadedWav2VecLID: Wav2Vec2ForSequenceClassification?
    private var loadedEcapaLID: EcapaTdnn?
    private var loadedNemotronDiar: NemotronDiarizationModel?
    private var loadedSortformerDiar: SortformerModel?
    private var loadedSileroVAD: SileroVAD?
    private var loadedFSMNVAD: FSMNVAD?
    private let audioPlayer = AudioPlayer()
    private var cancellables = Set<AnyCancellable>()
    private var separationTask: Task<Void, Never>?

    var isModelLoaded: Bool {
        guard loadedModelId == modelId else { return false }
        switch studioMode {
        case .languageId:
            return loadedWav2VecLID != nil || loadedEcapaLID != nil
        case .diarization:
            return loadedNemotronDiar != nil || loadedSortformerDiar != nil || loadedSileroVAD != nil || loadedFSMNVAD != nil
        case .separation, .enhancement:
            return model != nil || loadedSTS != nil
        }
    }

    init() {
        if let savedModeRaw = UserDefaults.standard.string(forKey: "VoicesApp.STSViewModel.studioMode"),
           let savedMode = AudioStudioMode(rawValue: savedModeRaw) {
            studioMode = savedMode
        } else {
            studioMode = Self.mode(forModelId: modelId)
        }
        setupAudioPlayerObservers()
    }

    // MARK: - AudioPlayer Observers

    private func setupAudioPlayerObservers() {
        audioPlayer.$isPlaying
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.isPlaying = value }
            .store(in: &cancellables)

        audioPlayer.$currentTime
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.currentTime = value }
            .store(in: &cancellables)

        audioPlayer.$duration
            .receive(on: DispatchQueue.main)
            .sink { [weak self] value in self?.duration = value }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: .modelCacheDidChange)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] note in
                guard let self,
                      let deletedId = note.userInfo?["deletedModelId"] as? String else { return }
                if self.loadedModelId == deletedId {
                    self.model = nil
                    self.loadedSTS = nil
                    self.loadedEcapaLID = nil
                    self.loadedWav2VecLID = nil
                    self.loadedNemotronDiar = nil
                    self.loadedSortformerDiar = nil
                    self.loadedSileroVAD = nil
                    self.loadedFSMNVAD = nil
                    self.loadedModelId = nil
                    Memory.clearCache()
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard !isModelLoaded || loadedModelId != modelId else { return }

        isLoading = true
        errorMessage = nil
        generationProgress = "Downloading & loading model..."

        do {
            let lower = modelId.lowercased()
            if lower.contains("voxlingua") || lower.contains("ecapa") {
                loadedEcapaLID = try await EcapaTdnn.fromPretrained(modelId)
                loadedWav2VecLID = nil
                loadedNemotronDiar = nil
                loadedSortformerDiar = nil
                loadedSileroVAD = nil
                loadedFSMNVAD = nil
                loadedSTS = nil
                model = nil
                studioMode = .languageId
            } else if lower.contains("mms-lid") || lower.contains("wav2vec2") {
                loadedWav2VecLID = try await Wav2Vec2ForSequenceClassification.fromPretrained(modelId)
                loadedEcapaLID = nil
                loadedNemotronDiar = nil
                loadedSortformerDiar = nil
                loadedSileroVAD = nil
                loadedFSMNVAD = nil
                loadedSTS = nil
                model = nil
                studioMode = .languageId
            } else if lower.contains("nemotron-3-diarization") || lower.contains("nemotron_diarization") {
                loadedNemotronDiar = try await NemotronDiarizationModel.fromPretrained(modelId)
                loadedSortformerDiar = nil
                loadedSileroVAD = nil
                loadedFSMNVAD = nil
                loadedEcapaLID = nil
                loadedWav2VecLID = nil
                loadedSTS = nil
                model = nil
                studioMode = .diarization
            } else if lower.contains("sortformer") || lower.contains("diar_streaming") {
                loadedSortformerDiar = try await SortformerModel.fromPretrained(modelId)
                loadedNemotronDiar = nil
                loadedSileroVAD = nil
                loadedFSMNVAD = nil
                loadedEcapaLID = nil
                loadedWav2VecLID = nil
                loadedSTS = nil
                model = nil
                studioMode = .diarization
            } else if lower.contains("silero-vad") {
                loadedSileroVAD = try await SileroVAD.fromPretrained(modelId)
                loadedNemotronDiar = nil
                loadedSortformerDiar = nil
                loadedFSMNVAD = nil
                loadedEcapaLID = nil
                loadedWav2VecLID = nil
                loadedSTS = nil
                model = nil
                studioMode = .diarization
            } else if lower.contains("fsmn-vad") {
                loadedFSMNVAD = try await FSMNVAD.fromPretrained(modelId)
                loadedNemotronDiar = nil
                loadedSortformerDiar = nil
                loadedSileroVAD = nil
                loadedEcapaLID = nil
                loadedWav2VecLID = nil
                loadedSTS = nil
                model = nil
                studioMode = .diarization
            } else {
                let sts = try await STS.loadModel(
                    modelRepo: modelId,
                    hfToken: ModelCacheHelper.effectiveHFToken
                )
                loadedSTS = sts
                if case .samAudio(let sam) = sts {
                    model = sam
                    studioMode = .separation
                } else {
                    model = nil
                    studioMode = .enhancement
                }
                loadedEcapaLID = nil
                loadedWav2VecLID = nil
                loadedNemotronDiar = nil
                loadedSortformerDiar = nil
                loadedSileroVAD = nil
                loadedFSMNVAD = nil
            }
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
        loadedSTS = nil
        loadedEcapaLID = nil
        loadedWav2VecLID = nil
        loadedNemotronDiar = nil
        loadedSortformerDiar = nil
        loadedSileroVAD = nil
        loadedFSMNVAD = nil
        loadedModelId = nil
        Memory.clearCache()
        await loadModel()
    }

    func resetSettingsToDefaults() {
        modelId = STSViewModel.defaultModelId
        separationDescription = STSViewModel.defaultSeparationDescription
        useResidual = STSViewModel.defaultUseResidual

        UserDefaults.standard.removeObject(forKey: STSViewModel.modelIdStorageKey)
        UserDefaults.standard.removeObject(forKey: STSViewModel.separationDescriptionStorageKey)
        UserDefaults.standard.removeObject(forKey: STSViewModel.useResidualStorageKey)
    }

    // MARK: - Separation, Enhancement & Spoken Language ID

    func startSeparation(inputURL: URL) {
        separationTask?.cancel()
        let scoped = inputURL.startAccessingSecurityScopedResource()
        defer { if scoped { inputURL.stopAccessingSecurityScopedResource() } }

        let tempCopy = FileManager.default.temporaryDirectory
            .appendingPathComponent("sts_in_\(UUID().uuidString)_\(inputURL.lastPathComponent)")
        let effectiveURL: URL
        if (try? FileManager.default.copyItem(at: inputURL, to: tempCopy)) != nil {
            effectiveURL = tempCopy
        } else {
            effectiveURL = inputURL
        }

        originalInputURL = effectiveURL
        if studioMode != .languageId {
            lidPredictions = []
            lidTopLanguage = nil
        }
        if studioMode != .diarization {
            diarizationSegments = []
            diarizationSpeakerCount = 0
            diarizationRTTM = ""
        }
        separationTask = Task {
            await processAudio(inputURL: effectiveURL)
        }
    }

    func processAudio(inputURL: URL) async {
        if !isModelLoaded || loadedModelId != modelId {
            await loadModel()
        }
        switch studioMode {
        case .separation:
            await separate(inputURL: inputURL)
        case .enhancement:
            await enhance(inputURL: inputURL)
        case .diarization:
            await runDiarization(inputURL: inputURL)
        case .languageId:
            await identifyLanguage(inputURL: inputURL)
        }
    }

    func runDiarization(inputURL: URL) async {
        isGenerating = true
        errorMessage = nil
        diarizationSegments = []
        diarizationSpeakerCount = 0
        diarizationRTTM = ""
        generationProgress = "Running neural speaker diarization / voice activity detection..."

        do {
            let (inputSR, rawAudio) = try loadAudioArray(from: inputURL)
            let mono = rawAudio.ndim > 1 ? rawAudio.mean(axis: -1) : rawAudio

            if let nemotron = loadedNemotronDiar {
                let output = try nemotron.generate(audio: mono, sampleRate: inputSR, threshold: 0.5)
                diarizationSegments = output.segments
                diarizationSpeakerCount = output.numSpeakers
                diarizationRTTM = output.text
            } else if let sortformer = loadedSortformerDiar {
                let audio16k = inputSR == 16_000
                    ? mono
                    : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: 16_000)
                let output = try await sortformer.generate(audio: audio16k, sampleRate: 16_000, threshold: 0.5)
                diarizationSegments = output.segments
                diarizationSpeakerCount = output.numSpeakers
                diarizationRTTM = output.text
            } else if let silero = loadedSileroVAD {
                let audio16k = inputSR == 16_000
                    ? mono
                    : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: 16_000)
                let segments = try silero.getSpeechTimestamps(audio16k, sampleRate: 16_000)
                diarizationSegments = segments.map {
                    DiarizationSegment(
                        start: Float($0.start) / 16_000.0,
                        end: Float($0.end) / 16_000.0,
                        speaker: 0
                    )
                }
                diarizationSpeakerCount = diarizationSegments.isEmpty ? 0 : 1
                diarizationRTTM = DiarizationOutput(
                    segments: diarizationSegments,
                    numSpeakers: diarizationSpeakerCount
                ).text
            } else if let fsmn = loadedFSMNVAD {
                let segments = try fsmn.detect(mono, sampleRate: inputSR)
                diarizationSegments = segments.compactMap { pair in
                    guard pair.count >= 2 else { return nil }
                    return DiarizationSegment(
                        start: Float(pair[0]) / 1000.0,
                        end: Float(pair[1]) / 1000.0,
                        speaker: 0
                    )
                }
                diarizationSpeakerCount = diarizationSegments.isEmpty ? 0 : 1
                diarizationRTTM = DiarizationOutput(
                    segments: diarizationSegments,
                    numSpeakers: diarizationSpeakerCount
                ).text
            } else {
                throw NSError(
                    domain: "STSViewModel",
                    code: 11,
                    userInfo: [NSLocalizedDescriptionKey: "Diarization/VAD model not loaded"]
                )
            }
            audioURL = inputURL
            isPlayingOriginal = true
            audioPlayer.loadAudio(from: inputURL)
            generationProgress = ""
        } catch {
            errorMessage = "Diarization failed: \(error.localizedDescription)"
            generationProgress = ""
        }
        isGenerating = false
    }

    func diarizationJSONString() -> String {
        let items: [[String: Any]] = diarizationSegments.map { seg in
            [
                "speaker": "SPEAKER_\(String(format: "%02d", seg.speaker))",
                "speaker_id": seg.speaker,
                "start": Double(String(format: "%.3f", seg.start)) ?? Double(seg.start),
                "end": Double(String(format: "%.3f", seg.end)) ?? Double(seg.end),
                "duration": Double(String(format: "%.3f", max(0, seg.end - seg.start))) ?? Double(max(0, seg.end - seg.start))
            ]
        }
        let payload: [String: Any] = [
            "model": modelId,
            "num_speakers": diarizationSpeakerCount,
            "segments": items
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return "[]"
    }

    func prepareModeDemoAudio() -> URL? {
        let samples = Voice.bundledStudioSamples()
        guard let first = samples.first else { return nil }

        switch studioMode {
        case .diarization:
            let outURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("Codecaine_Demo_MultiSpeaker_Diarization.wav")
            if FileManager.default.fileExists(atPath: outURL.path) {
                originalInputURL = outURL
                audioURL = outURL
                isPlayingOriginal = true
                audioPlayer.loadAudio(from: outURL)
                return outURL
            }
            do {
                let targetSR = 16_000
                var combined: [Float] = []
                let pauseSamples = Array(repeating: Float(0), count: Int(Double(targetSR) * 0.28))
                for sample in samples.prefix(3) {
                    let (sr, arr) = try loadAudioArray(from: sample.url)
                    let mono = arr.ndim > 1 ? arr.mean(axis: -1) : arr
                    let resampled = sr == targetSR ? mono : try MLXAudioCore.resampleAudio(mono, from: sr, to: targetSR)
                    let floats = resampled.asArray(Float.self)
                    let maxSlice = min(floats.count, targetSR * 4)
                    combined.append(contentsOf: floats.prefix(maxSlice))
                    combined.append(contentsOf: pauseSamples)
                }
                if !combined.isEmpty {
                    let writer = try StreamingWAVWriter(url: outURL, sampleRate: Double(targetSR))
                    try writer.writeChunk(combined)
                    _ = writer.finalize()
                    originalInputURL = outURL
                    audioURL = outURL
                    isPlayingOriginal = true
                    audioPlayer.loadAudio(from: outURL)
                    return outURL
                }
            } catch {}
            originalInputURL = first.url
            audioURL = first.url
            isPlayingOriginal = true
            audioPlayer.loadAudio(from: first.url)
            return first.url

        case .enhancement:
            let outURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("Codecaine_Demo_Noisy_Speech.wav")
            if FileManager.default.fileExists(atPath: outURL.path) {
                originalInputURL = outURL
                audioURL = outURL
                isPlayingOriginal = true
                audioPlayer.loadAudio(from: outURL)
                return outURL
            }
            do {
                let sourceURL = samples.first(where: { $0.name.lowercased().contains("sophie") })?.url ?? first.url
                let (sr, arr) = try loadAudioArray(from: sourceURL)
                let mono = arr.ndim > 1 ? arr.mean(axis: -1) : arr
                var floats = mono.asArray(Float.self)
                for i in 0..<floats.count {
                    let t = Float(i) / Float(max(1, sr))
                    let hum = 0.018 * sin(2.0 * .pi * 120.0 * t) + 0.010 * sin(2.0 * .pi * 240.0 * t)
                    let hiss = Float(((i * 1103515245 + 12345) & 0x7fffffff)) / Float(0x7fffffff) * 0.024 - 0.012
                    floats[i] = max(-0.98, min(0.98, floats[i] + hum + hiss))
                }
                let writer = try StreamingWAVWriter(url: outURL, sampleRate: Double(sr))
                try writer.writeChunk(floats)
                _ = writer.finalize()
                originalInputURL = outURL
                audioURL = outURL
                isPlayingOriginal = true
                audioPlayer.loadAudio(from: outURL)
                return outURL
            } catch {}
            originalInputURL = first.url
            audioURL = first.url
            isPlayingOriginal = true
            audioPlayer.loadAudio(from: first.url)
            return first.url

        case .separation, .languageId:
            originalInputURL = first.url
            audioURL = first.url
            isPlayingOriginal = true
            audioPlayer.loadAudio(from: first.url)
            return first.url
        }
    }

    func enhance(inputURL: URL) async {
        guard let loadedSTS else {
            errorMessage = "Enhancement model not loaded"
            return
        }
        isGenerating = true
        errorMessage = nil
        generationProgress = "Loading & resampling audio..."

        do {
            let (inputSR, rawAudio) = try loadAudioArray(from: inputURL)
            let mono = rawAudio.ndim > 1 ? rawAudio.mean(axis: -1) : rawAudio
            var targetSR = loadedSTS.sampleRate

            generationProgress = "Enhancing speech clarity & removing background noise..."
            let enhanced: MLXArray
            switch loadedSTS {
            case .nemotronReuse(let reuse):
                let clampedRate = min(max(inputSR, 8_000), 48_000)
                let inputForReuse = inputSR == clampedRate
                    ? mono
                    : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: clampedRate)
                let out = try reuse.generate(audio: inputForReuse, sampleRate: clampedRate)
                enhanced = out.audio
                targetSR = out.sampleRate
            case .mossFormer2SE(let mf):
                let audioData = inputSR == targetSR
                    ? mono
                    : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: targetSR)
                enhanced = try mf.enhance(audioData)
            case .deepFilterNet(let dfn):
                let audioData = inputSR == targetSR
                    ? mono
                    : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: targetSR)
                enhanced = try dfn.enhance(audioData)
            case .samAudio(let sam):
                let res = try await sam.separateLong(audioPaths: [inputURL.path], descriptions: ["speech"])
                enhanced = res.target.first ?? mono
            case .lfmAudio:
                enhanced = mono
                targetSR = inputSR
            }
            eval(enhanced)
            let samples = enhanced.squeezed().asArray(Float.self)

            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("wav")
            let wavWriter = try StreamingWAVWriter(url: tempURL, sampleRate: Double(targetSR))
            try wavWriter.writeChunk(samples)
            let finalURL = wavWriter.finalize()

            Memory.clearCache()
            audioURL = finalURL
            isPlayingOriginal = false
            generationProgress = ""
            audioPlayer.loadAudio(from: finalURL)
            audioPlayer.play()
        } catch {
            errorMessage = "Speech enhancement failed: \(error.localizedDescription)"
            generationProgress = ""
            Memory.clearCache()
        }
        isGenerating = false
    }

    func identifyLanguage(inputURL: URL) async {
        isGenerating = true
        errorMessage = nil
        lidPredictions = []
        lidTopLanguage = nil
        generationProgress = "Analyzing spoken language..."

        do {
            let (_, waveform) = try loadAudioArray(from: inputURL, sampleRate: 16_000)
            let output: LIDOutput
            if let ecapa = loadedEcapaLID {
                output = ecapa.predict(waveform: waveform, topK: 5)
            } else if let w2v = loadedWav2VecLID {
                output = w2v.predict(waveform: waveform, topK: 5)
            } else {
                throw NSError(domain: "STSViewModel", code: 10, userInfo: [NSLocalizedDescriptionKey: "LID model not loaded"])
            }
            lidTopLanguage = "\(output.language) (\(String(format: "%.1f%%", output.confidence * 100)))"
            lidPredictions = output.topLanguages
            audioURL = inputURL
            isPlayingOriginal = true
            audioPlayer.loadAudio(from: inputURL)
            generationProgress = ""
        } catch {
            errorMessage = "Language identification failed: \(error.localizedDescription)"
            generationProgress = ""
        }
        isGenerating = false
    }

    func switchPlaybackSource(original: Bool) {
        let savedTime = currentTime
        let wasPlaying = isPlaying
        isPlayingOriginal = original
        if original, let orig = originalInputURL {
            audioPlayer.loadAudio(from: orig)
            if savedTime > 0 {
                audioPlayer.seek(to: min(savedTime, max(0, audioPlayer.duration - 0.05)))
            }
            if wasPlaying || savedTime == 0 {
                audioPlayer.play()
            }
        } else if !original, let proc = audioURL {
            audioPlayer.loadAudio(from: proc)
            if savedTime > 0 {
                audioPlayer.seek(to: min(savedTime, max(0, audioPlayer.duration - 0.05)))
            }
            if wasPlaying || savedTime == 0 {
                audioPlayer.play()
            }
        }
    }

    func saveAudioInFormat(_ format: AudioExportFormat) {
        guard let sourceURL = (isPlayingOriginal ? originalInputURL : audioURL) ?? audioURL else { return }
        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Export Processed Audio as \(format.label)"
        savePanel.nameFieldStringValue = "processed_audio.\(format.rawValue)"
        savePanel.begin { [weak self] response in
            guard response == .OK, let dest = savePanel.url else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.generationProgress = "Exporting \(format.label)..."
                do {
                    try await Task.detached(priority: .userInitiated) {
                        try AudioFormatExporter.convert(sourceWAV: sourceURL, to: dest, format: format)
                    }.value
                    self.generationProgress = ""
                } catch {
                    self.generationProgress = ""
                    self.errorMessage = "Export failed: \(error.localizedDescription)"
                }
            }
        }
        #endif
    }

    func separate(inputURL: URL) async {
        guard let model else {
            errorMessage = "Model not loaded"
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = "Starting separation..."

        do {
            try Task.checkCancellation()

            generationProgress = "Separating \"\(separationDescription)\"..."

            let result = try await model.separateLong(
                audioPaths: [inputURL.path],
                descriptions: [separationDescription]
            )

            try Task.checkCancellation()

            guard let targetArr = result.target.first else {
                throw NSError(
                    domain: "STSViewModel", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No target audio in result"]
                )
            }
            let targetSamples = targetArr.asArray(Float.self)
            guard !targetSamples.isEmpty else {
                throw NSError(
                    domain: "STSViewModel", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Separated audio was empty"]
                )
            }

            let sampleRate = Double(model.sampleRate)

            // Write target WAV
            let targetTemp = FileManager.default.temporaryDirectory
                .appendingPathComponent("target_\(UUID().uuidString).wav")
            let targetWriter = try StreamingWAVWriter(url: targetTemp, sampleRate: sampleRate)
            try targetWriter.writeChunk(targetSamples)
            let writtenTargetURL = targetWriter.finalize()
            targetAudioURL = writtenTargetURL

            // Write residual WAV so user can flip Isolate Target vs Keep Background instantaneously
            if let residualArr = result.residual.first {
                let resSamples = residualArr.asArray(Float.self)
                if !resSamples.isEmpty {
                    let resTemp = FileManager.default.temporaryDirectory
                        .appendingPathComponent("residual_\(UUID().uuidString).wav")
                    if let resWriter = try? StreamingWAVWriter(url: resTemp, sampleRate: sampleRate) {
                        try? resWriter.writeChunk(resSamples)
                        residualAudioURL = resWriter.finalize()
                    }
                }
            }

            if let peakMem = result.peakMemoryGB {
                print("[STS] Peak memory: \(String(format: "%.2f", peakMem)) GB")
            }

            let finalURL = (useResidual ? residualAudioURL : targetAudioURL) ?? writtenTargetURL

            Memory.clearCache()

            audioURL = finalURL
            isPlayingOriginal = false
            generationProgress = ""

            // Load and play — no streaming since separate() isn't an AsyncSequence
            audioPlayer.loadAudio(from: finalURL)
            audioPlayer.play()

        } catch is CancellationError {
            audioPlayer.stop()
            Memory.clearCache()
            generationProgress = ""
        } catch {
            errorMessage = "Separation failed: \(error.localizedDescription)"
            generationProgress = ""
            Memory.clearCache()
        }

        isGenerating = false
    }

    // MARK: - Playback Controls

    func play()           { audioPlayer.play() }
    func pause()          { audioPlayer.pause() }
    func togglePlayPause(){ audioPlayer.togglePlayPause() }
    func seek(to time: TimeInterval) { audioPlayer.seek(to: time) }

    func stop() {
        separationTask?.cancel()
        separationTask = nil
        audioPlayer.stop()
        if isGenerating {
            isGenerating = false
            generationProgress = ""
        }
    }

    // MARK: - Save

    func saveAudioFile() {
        guard let audioURL else { return }

        #if os(macOS)
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.wav]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false
        savePanel.title = "Save Separated Audio"
        savePanel.nameFieldStringValue = "separated_audio.wav"

        savePanel.begin { response in
            guard response == .OK, let dest = savePanel.url else { return }
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: audioURL, to: dest)
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to save file: \(error.localizedDescription)"
                }
            }
        }
        #else
        errorMessage = "Save functionality is available on macOS"
        #endif
    }
}
