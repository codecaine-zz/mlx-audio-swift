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
import SwiftUI
#if os(macOS)
import AppKit
#endif

enum AudioStudioMode: String, CaseIterable, Identifiable {
    case separation = "Source Separation (SAM Audio)"
    case enhancement = "Speech Enhancement & Denoise"
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
        ("MossFormer2 SE FP16 (Speech Enhance)", "starkdmi/MossFormer2-SE-fp16"),
        ("DeepFilterNet MLX (Realtime Denoise)", "mlx-community/DeepFilterNet-mlx"),
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

    var studioMode: AudioStudioMode = .separation {
        didSet {
            switch studioMode {
            case .separation:
                if !modelId.lowercased().contains("sam-audio") {
                    modelId = "mlx-community/sam-audio-base"
                }
            case .enhancement:
                if !modelId.lowercased().contains("mossformer") && !modelId.lowercased().contains("deepfilter") {
                    modelId = "starkdmi/MossFormer2-SE-fp16"
                }
            case .languageId:
                if !modelId.lowercased().contains("voxlingua") && !modelId.lowercased().contains("mms-lid") {
                    modelId = "beshkenadze/lang-id-voxlingua107-ecapa-mlx"
                }
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
    private let audioPlayer = AudioPlayer()
    private var cancellables = Set<AnyCancellable>()
    private var separationTask: Task<Void, Never>?

    var isModelLoaded: Bool {
        guard loadedModelId == modelId else { return false }
        if studioMode == .languageId {
            return loadedWav2VecLID != nil || loadedEcapaLID != nil
        }
        return model != nil || loadedSTS != nil
    }

    init() {
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
                loadedSTS = nil
                model = nil
                studioMode = .languageId
            } else if lower.contains("mms-lid") || lower.contains("wav2vec2") {
                loadedWav2VecLID = try await Wav2Vec2ForSequenceClassification.fromPretrained(modelId)
                loadedEcapaLID = nil
                loadedSTS = nil
                model = nil
                studioMode = .languageId
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
        case .languageId:
            await identifyLanguage(inputURL: inputURL)
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
            let targetSR = loadedSTS.sampleRate
            let audioData = inputSR == targetSR
                ? mono
                : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: targetSR)

            generationProgress = "Enhancing speech clarity & removing background noise..."
            let enhanced: MLXArray
            switch loadedSTS {
            case .mossFormer2SE(let mf):
                enhanced = try mf.enhance(audioData)
            case .deepFilterNet(let dfn):
                enhanced = try dfn.enhance(audioData)
            case .samAudio(let sam):
                let res = try await sam.separateLong(audioPaths: [inputURL.path], descriptions: ["speech"])
                enhanced = res.target.first ?? audioData
            case .lfmAudio:
                enhanced = audioData
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
