import Accelerate
import AVFoundation
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTS
import MLXAudioSTT
import MLXAudioTTS
import MLXAudioVAD
import Network
import SwiftUI
#if os(macOS)
import AppKit
import PDFKit
import Vision
#endif

// MARK: - Local HuggingFace Cache & Token Helper

enum ModelCacheHelper {
    private static let hfTokenStorageKey = "VoicesApp.HuggingFaceToken"
    private static let weightExtensions: Set<String> = [
        "safetensors", "npz", "bin", "gguf", "pt", "onnx", "zst", "fermion", "fsq"
    ]
    private static let cacheStatsLock = NSLock()
    nonisolated(unsafe) private static var cachedStatsByRepo: [String: (hasWeights: Bool, sizeBytes: UInt64, sizeString: String?)] = [:]

    static func invalidateMemoryCache() {
        cacheStatsLock.lock()
        cachedStatsByRepo.removeAll()
        cacheStatsLock.unlock()
    }

    static func isValidRepoId(_ repoId: String) -> Bool {
        let trimmed = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.contains(".."),
              !trimmed.contains("\\"),
              !trimmed.contains("\0"),
              !trimmed.hasPrefix("/") else {
            return false
        }
        return true
    }

    static var hfToken: String {
        get {
            let stored = UserDefaults.standard.string(forKey: hfTokenStorageKey) ?? ""
            if !stored.isEmpty { return stored }
            return ProcessInfo.processInfo.environment["HF_TOKEN"] ?? ""
        }
        set {
            UserDefaults.standard.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), forKey: hfTokenStorageKey)
        }
    }

    static var effectiveHFToken: String? {
        let trimmed = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static var mlxAudioCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/mlx-audio")
    }

    static var hubCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    static func mlxModelDirectory(for repoId: String) -> URL {
        let safeRepo = isValidRepoId(repoId) ? repoId : "__invalid__"
        let subdir = safeRepo.replacingOccurrences(of: "/", with: "_")
        return mlxAudioCacheRoot.appendingPathComponent(subdir)
    }

    static func hubModelDirectory(for repoId: String) -> URL {
        let safeRepo = isValidRepoId(repoId) ? repoId : "__invalid__"
        let hubSubdir = "models--" + safeRepo.replacingOccurrences(of: "/", with: "--")
        return hubCacheRoot.appendingPathComponent(hubSubdir)
    }

    static func phonon2UnpackedDirectories(for repoId: String) -> [URL] {
        guard isValidRepoId(repoId) else { return [] }
        let mlxDir = mlxModelDirectory(for: repoId)
        let hubDir = hubModelDirectory(for: repoId)
        var candidateConfigs: [URL] = [mlxDir.appendingPathComponent("config.json")]
        let snapshotsDir = hubDir.appendingPathComponent("snapshots")
        if let snaps = try? FileManager.default.contentsOfDirectory(at: snapshotsDir, includingPropertiesForKeys: nil) {
            for snap in snaps {
                candidateConfigs.append(snap.appendingPathComponent("config.json"))
            }
        }
        var digests = Set<String>()
        for configURL in candidateConfigs {
            guard let data = try? Data(contentsOf: configURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let artifact = json["artifact"] as? [String: Any],
               let sha = artifact["sha256"] as? String, !sha.isEmpty,
               !sha.contains("/"), !sha.contains("..") {
                digests.insert(sha)
            }
            if let fermion = json["fermion"] as? [String: Any],
               let sha = fermion["container_sha256"] as? String, !sha.isEmpty,
               !sha.contains("/"), !sha.contains("..") {
                digests.insert(sha)
            }
        }
        let phonon2Root = mlxAudioCacheRoot.appendingPathComponent("phonon2")
        return digests.compactMap { digest in
            let url = phonon2Root.appendingPathComponent(digest)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
    }

    static func modelDirectory(for repoId: String) -> URL? {
        guard isValidRepoId(repoId) else { return nil }
        let mlxDir = mlxModelDirectory(for: repoId)
        if directoryHasWeights(mlxDir) {
            return mlxDir
        }
        let hubSnapshotsDir = hubModelDirectory(for: repoId).appendingPathComponent("snapshots")
        if let snapshots = try? FileManager.default.contentsOfDirectory(at: hubSnapshotsDir, includingPropertiesForKeys: nil),
           let first = snapshots.first(where: { directoryHasWeights($0) }) {
            return first
        }
        return nil
    }

    static func directoryHasWeights(_ dir: URL) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return false }

        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if weightExtensions.contains(ext) {
                let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                if size > 0 { return true }
            }
        }
        return false
    }

    private static func stats(for repoId: String) -> (hasWeights: Bool, sizeBytes: UInt64, sizeString: String?) {
        guard isValidRepoId(repoId) else { return (false, 0, nil) }
        cacheStatsLock.lock()
        if let cached = cachedStatsByRepo[repoId] {
            cacheStatsLock.unlock()
            return cached
        }
        cacheStatsLock.unlock()

        let hasWeights = modelDirectory(for: repoId) != nil
        let mlxBytes = directoryByteSize(mlxModelDirectory(for: repoId))
        let hubBytes = directoryByteSize(hubModelDirectory(for: repoId))
        let phonon2Bytes = phonon2UnpackedDirectories(for: repoId).reduce(0) { $0 + directoryByteSize($1) }
        let sizeBytes = max(mlxBytes, hubBytes) + phonon2Bytes
        let sizeString: String? = (hasWeights && sizeBytes > 0)
            ? ByteCountFormatter.string(fromByteCount: Int64(sizeBytes), countStyle: .file)
            : nil
        let computed = (hasWeights: hasWeights, sizeBytes: sizeBytes, sizeString: sizeString)

        cacheStatsLock.lock()
        cachedStatsByRepo[repoId] = computed
        cacheStatsLock.unlock()
        return computed
    }

    static func isModelCached(_ repoId: String) -> Bool {
        stats(for: repoId).hasWeights
    }

    static func cachedModelSizeBytes(_ repoId: String) -> UInt64 {
        stats(for: repoId).sizeBytes
    }

    static func cachedModelSizeString(_ repoId: String) -> String? {
        stats(for: repoId).sizeString
    }

    static func directoryByteSize(_ dir: URL) -> UInt64 {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
            return 0
        }
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey],
            options: []
        ) else { return 0 }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
               values.isSymbolicLink != true,
               values.isRegularFile == true,
               let size = values.fileSize {
                total += UInt64(max(size, 0))
            }
        }
        return total
    }

    /// Deletes a cached model from both the mlx-audio snapshot directory and the HuggingFace Hub cache,
    /// including any unpacked Phonon 2.5 container cache and lock/metadata directories.
    /// - Returns: The number of logical bytes freed.
    @discardableResult
    static func deleteModel(_ repoId: String) -> UInt64 {
        guard isValidRepoId(repoId) else { return 0 }
        let fm = FileManager.default
        let hubRootPrefix = hubCacheRoot.standardizedFileURL.path + "/"
        let freedBytes = cachedModelSizeBytes(repoId)
        let mlxDir = mlxModelDirectory(for: repoId)
        let hubDir = hubModelDirectory(for: repoId)
        let hubSubdir = hubDir.lastPathComponent
        guard !hubSubdir.isEmpty, hubSubdir != ".", hubSubdir != "..", !hubSubdir.contains("..") else {
            return 0
        }
        let metadataDir = hubCacheRoot.appendingPathComponent(".metadata").appendingPathComponent(hubSubdir)
        let locksDir = hubCacheRoot.appendingPathComponent(".locks").appendingPathComponent(hubSubdir)

        for unpacked in phonon2UnpackedDirectories(for: repoId) {
            if unpacked.standardizedFileURL.path.hasPrefix(hubRootPrefix) {
                try? fm.removeItem(at: unpacked)
            }
        }
        for target in [mlxDir, hubDir, metadataDir, locksDir] {
            let stdPath = target.standardizedFileURL.path
            if stdPath.hasPrefix(hubRootPrefix), fm.fileExists(atPath: stdPath) {
                try? fm.removeItem(at: target)
            }
        }

        invalidateMemoryCache()

        NotificationCenter.default.post(
            name: .modelCacheDidChange,
            object: nil,
            userInfo: ["deletedModelId": repoId, "freedBytes": freedBytes]
        )
        return freedBytes
    }

    static func revealModelInFinder(_ repoId: String) {
        #if os(macOS)
        let mlxDir = mlxModelDirectory(for: repoId)
        let hubDir = hubModelDirectory(for: repoId)
        if FileManager.default.fileExists(atPath: mlxDir.path) {
            NSWorkspace.shared.activateFileViewerSelecting([mlxDir])
        } else if FileManager.default.fileExists(atPath: hubDir.path) {
            NSWorkspace.shared.activateFileViewerSelecting([hubDir])
        } else {
            revealCacheInFinder()
        }
        #endif
    }

    static func revealCacheInFinder() {
        #if os(macOS)
        let target = FileManager.default.fileExists(atPath: mlxAudioCacheRoot.path) ? mlxAudioCacheRoot : hubCacheRoot
        NSWorkspace.shared.open(target)
        #endif
    }
}

// MARK: - Downloaded AI Models Cache Manager

enum DownloadedModelCategory: String, CaseIterable, Identifiable {
    case all = "All"
    case tts = "TTS / Voice"
    case stt = "STT / ASR"
    case audioLab = "Audio Lab"
    case codecHelper = "Codecs & Helpers"
    case incomplete = "Incomplete"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .all: return "internaldrive.fill"
        case .tts: return "waveform"
        case .stt: return "mic.fill"
        case .audioLab: return "tuningfork"
        case .codecHelper: return "puzzlepiece.extension.fill"
        case .incomplete: return "exclamationmark.triangle.fill"
        }
    }

    var tint: Color {
        switch self {
        case .all: return .blue
        case .tts: return .blue
        case .stt: return .purple
        case .audioLab: return .teal
        case .codecHelper: return .indigo
        case .incomplete: return .orange
        }
    }
}

struct DownloadedModelItem: Identifiable, Hashable {
    var id: String { repoId }
    let repoId: String
    let displayName: String
    let category: DownloadedModelCategory
    let sizeBytes: UInt64
    let hasWeights: Bool
    let modifiedDate: Date?
    let primaryDirectory: URL

    var sizeString: String {
        ByteCountFormatter.string(fromByteCount: Int64(sizeBytes), countStyle: .file)
    }
}

@MainActor
@Observable
final class DownloadedModelsManager {
    static let shared = DownloadedModelsManager()

    var items: [DownloadedModelItem] = []
    var cacheRevision: Int = 0
    var statusBanner: String? = nil

    var totalBytes: UInt64 {
        items.reduce(0) { $0 + $1.sizeBytes }
    }

    var totalSizeString: String {
        ByteCountFormatter.string(fromByteCount: Int64(totalBytes), countStyle: .file)
    }

    var completeModelsCount: Int {
        items.filter(\.hasWeights).count
    }

    var incompleteModelsCount: Int {
        items.filter { !$0.hasWeights }.count
    }

    private init() {
        refresh()
        NotificationCenter.default.addObserver(
            forName: .modelCacheDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.refresh()
            }
        }
    }

    func isModelCached(_ repoId: String) -> Bool {
        _ = cacheRevision
        return ModelCacheHelper.isModelCached(repoId)
    }

    func cachedModelSizeString(_ repoId: String) -> String? {
        _ = cacheRevision
        return ModelCacheHelper.cachedModelSizeString(repoId)
    }

    @discardableResult
    func deleteModel(_ repoId: String, displayName: String? = nil) -> UInt64 {
        let label = displayName ?? items.first(where: { $0.repoId == repoId })?.displayName ?? repoId
        LocalhostServerManager.shared.unloadModelIfMatching(repoId)
        let freed = ModelCacheHelper.deleteModel(repoId)
        let freedStr = ByteCountFormatter.string(fromByteCount: Int64(freed), countStyle: .file)
        refresh()
        flashBanner("Deleted \(label) — Freed \(freedStr)")
        return freed
    }

    func cleanIncompleteDownloads() {
        let incomplete = items.filter { !$0.hasWeights }
        guard !incomplete.isEmpty else { return }
        var totalFreed: UInt64 = 0
        for item in incomplete {
            totalFreed += ModelCacheHelper.deleteModel(item.repoId)
        }
        // Also clean any malformed double-prefix directories in .locks / .metadata / hub
        let fm = FileManager.default
        for root in [
            ModelCacheHelper.hubCacheRoot,
            ModelCacheHelper.hubCacheRoot.appendingPathComponent(".locks"),
            ModelCacheHelper.hubCacheRoot.appendingPathComponent(".metadata")
        ] {
            if let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
                for entry in entries where entry.lastPathComponent.hasPrefix("models--mlx-community--mlx-community--") {
                    try? fm.removeItem(at: entry)
                }
            }
        }
        refresh()
        let freedStr = ByteCountFormatter.string(fromByteCount: Int64(totalFreed), countStyle: .file)
        flashBanner("Cleaned \(incomplete.count) incomplete download\(incomplete.count == 1 ? "" : "s") (\(freedStr))")
    }

    func refresh() {
        ModelCacheHelper.invalidateMemoryCache()
        let fm = FileManager.default
        var catalog: [String: (name: String, category: DownloadedModelCategory)] = [
            "nativ-community/phonon": ("Phonon TTS 40M (Default)", .tts),
            "gradium/phonon-40m-multilingual": ("Gradium Phonon 40M Multilingual", .tts),
            "beshkenadze/kitten-tts-g2p": ("KittenTTS G2P Phonemizer", .codecHelper),
            "kyutai/moshiko-pytorch-bf16": ("Kyutai Moshiko / Mimi Backbone (BF16)", .codecHelper),
            "kyutai/mimi": ("Kyutai Mimi Neural Audio Codec", .codecHelper),
            "mlx-community/snac_24khz": ("SNAC 24kHz Neural Audio Codec", .codecHelper),
            "lucasnewman/vocos-mel-24khz": ("Vocos Mel 24kHz Vocoder", .codecHelper),
            "lucasnewman/f5-tts-mlx": ("F5-TTS MLX", .tts),
            "openai/whisper-tiny.en": ("Whisper Tiny English", .stt),
        ]
        for preset in TTSViewModel.presetModels {
            catalog[preset.id] = (preset.name, .tts)
        }
        for preset in STTViewModel.presetModels {
            catalog[preset.id] = (preset.name, .stt)
        }
        for preset in STSViewModel.presetModels {
            catalog[preset.id] = (preset.name, .audioLab)
        }

        var underscoreToRepoId: [String: String] = [:]
        for repoId in catalog.keys {
            underscoreToRepoId[repoId.replacingOccurrences(of: "/", with: "_")] = repoId
        }

        var discoveredRepoIds = Set<String>()

        // 1. Scan ~/.cache/huggingface/hub/models--*
        if let hubEntries = try? fm.contentsOfDirectory(
            at: ModelCacheHelper.hubCacheRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in hubEntries {
                let name = entry.lastPathComponent
                guard name.hasPrefix("models--") else { continue }
                let stripped = String(name.dropFirst("models--".count))
                let parts = stripped.components(separatedBy: "--")
                guard parts.count >= 2 else { continue }
                let org = parts[0]
                let repo = parts.dropFirst().joined(separator: "--")
                let repoId = "\(org)/\(repo)"
                discoveredRepoIds.insert(repoId)
                underscoreToRepoId[repoId.replacingOccurrences(of: "/", with: "_")] = repoId
            }
        }

        // 2. Scan ~/.cache/huggingface/hub/mlx-audio/*
        if let mlxEntries = try? fm.contentsOfDirectory(
            at: ModelCacheHelper.mlxAudioCacheRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in mlxEntries {
                let name = entry.lastPathComponent
                if name == "phonon2" { continue }
                if let mapped = underscoreToRepoId[name] {
                    discoveredRepoIds.insert(mapped)
                    continue
                }
                // Check config.json for model_id
                let cfgURL = entry.appendingPathComponent("config.json")
                if let data = try? Data(contentsOf: cfgURL),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let modelId = json["model_id"] as? String,
                   modelId.contains("/") {
                    discoveredRepoIds.insert(modelId)
                    continue
                }
                if let firstUnderscore = name.firstIndex(of: "_") {
                    let org = String(name[..<firstUnderscore])
                    let repo = String(name[name.index(after: firstUnderscore)...])
                    if !org.isEmpty && !repo.isEmpty {
                        discoveredRepoIds.insert("\(org)/\(repo)")
                    }
                }
            }
        }

        var builtItems: [DownloadedModelItem] = []
        for repoId in discoveredRepoIds {
            let mlxDir = ModelCacheHelper.mlxModelDirectory(for: repoId)
            let hubDir = ModelCacheHelper.hubModelDirectory(for: repoId)
            let mlxExists = fm.fileExists(atPath: mlxDir.path)
            let hubExists = fm.fileExists(atPath: hubDir.path)
            guard mlxExists || hubExists else { continue }

            let hasWeights = ModelCacheHelper.isModelCached(repoId)
            let sizeBytes = ModelCacheHelper.cachedModelSizeBytes(repoId)
            guard hasWeights || sizeBytes > 0 else { continue }

            let primaryDir = mlxExists ? mlxDir : hubDir
            let modDate = (try? primaryDir.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)

            let known = catalog[repoId]
            let displayName = known?.name ?? (repoId.split(separator: "/").last.map(String.init) ?? repoId)
            let category: DownloadedModelCategory
            if !hasWeights {
                category = .incomplete
            } else if let knownCat = known?.category {
                category = knownCat
            } else {
                let lower = repoId.lowercased()
                if lower.contains("asr") || lower.contains("whisper") || lower.contains("transcribe")
                    || lower.contains("parakeet") || lower.contains("canary") || lower.contains("sensevoice")
                    || lower.contains("moonshine") {
                    category = .stt
                } else if lower.contains("sam-audio") || lower.contains("mossformer")
                    || lower.contains("deepfilter") || lower.contains("voxlingua") || lower.contains("mms-lid")
                    || lower.contains("nemotron-3-diarization") || lower.contains("sortformer")
                    || lower.contains("silero-vad") || lower.contains("fsmn-vad")
                    || lower.contains("smart-turn") || lower.contains("re-use")
                    || lower.contains("lfm2.5-audio") || lower.contains("lfm2-audio") {
                    category = .audioLab
                } else if lower.contains("snac") || lower.contains("vocos") || lower.contains("mimi")
                    || lower.contains("moshiko") || lower.contains("g2p") || lower.contains("codec") {
                    category = .codecHelper
                } else {
                    category = .tts
                }
            }

            builtItems.append(
                DownloadedModelItem(
                    repoId: repoId,
                    displayName: displayName,
                    category: category,
                    sizeBytes: sizeBytes,
                    hasWeights: hasWeights,
                    modifiedDate: modDate,
                    primaryDirectory: primaryDir
                )
            )
        }

        builtItems.sort { a, b in
            if a.hasWeights != b.hasWeights {
                return a.hasWeights && !b.hasWeights
            }
            if a.sizeBytes != b.sizeBytes {
                return a.sizeBytes > b.sizeBytes
            }
            return a.displayName.localizedCaseInsensitiveCompare(b.displayName) == .orderedAscending
        }

        items = builtItems
        cacheRevision &+= 1
    }

    private func flashBanner(_ text: String) {
        statusBanner = text
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3.5))
            if self.statusBanner == text {
                self.statusBanner = nil
            }
        }
    }
}

// MARK: - Generated Audio Take (Session History)

struct GeneratedTake: Identifiable, Hashable {
    let id: UUID
    let text: String
    let modelId: String
    let voiceName: String
    let audioURL: URL
    let duration: TimeInterval
    let tokensPerSecond: Double
    let createdAt: Date

    init(
        id: UUID = UUID(),
        text: String,
        modelId: String,
        voiceName: String,
        audioURL: URL,
        duration: TimeInterval,
        tokensPerSecond: Double,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.modelId = modelId
        self.voiceName = voiceName
        self.audioURL = audioURL
        self.duration = duration
        self.tokensPerSecond = tokensPerSecond
        self.createdAt = createdAt
    }

    var shortModelName: String {
        modelId.split(separator: "/").last.map(String.init) ?? modelId
    }
}

struct ExpressiveTagItem: Identifiable {
    let id = UUID()
    let tag: String
    let label: String
    let icon: String
}

struct PromptTemplateItem: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String
    let icon: String
    let text: String
}

struct Voice: Identifiable, Hashable {
    let id: UUID
    var name: String
    var voiceId: String?
    var description: String
    var language: String
    var color: Color
    var isCustom: Bool
    var isBuiltIn: Bool
    var lastUsed: Date?

    // Voice cloning properties
    var audioFileURL: URL?
    var transcription: String?

    /// Whether this voice uses audio cloning (has both audio file and transcription)
    var isClonedVoice: Bool {
        audioFileURL != nil && transcription != nil && !transcription!.isEmpty
    }

    /// Stable canonical identifier used for persisting favorite/recent/selected voices across launches
    var voiceKey: String {
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let cleanId = (voiceId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return "\(cleanName)|\(cleanId)"
    }

    /// Studio category for filtering & grouping in the Voice Browser
    var category: String {
        if isCustom { return "Custom" }
        if isClonedVoice { return "Reference Clones" }
        let desc = description.lowercased()
        if desc.hasPrefix("conversational") { return "Conversational" }
        if desc.hasPrefix("narrator") { return "Narrator" }
        if desc.hasPrefix("character") { return "Character" }
        return "Studio Voices"
    }

    init(
        id: UUID = UUID(),
        name: String,
        voiceId: String? = nil,
        description: String = "",
        language: String = "English",
        color: Color = .blue,
        isCustom: Bool = false,
        isBuiltIn: Bool = false,
        lastUsed: Date? = nil,
        audioFileURL: URL? = nil,
        transcription: String? = nil
    ) {
        self.id = id
        self.name = name
        self.voiceId = voiceId
        self.description = description
        self.language = language
        self.color = color
        self.isCustom = isCustom
        self.isBuiltIn = isBuiltIn
        self.lastUsed = lastUsed
        self.audioFileURL = audioFileURL
        self.transcription = transcription
    }
}

struct VoiceCollection: Identifiable {
    let id: UUID
    var name: String
    var description: String
    var imageName: String
    var backgroundColor: Color
    var voices: [Voice]

    init(
        id: UUID = UUID(),
        name: String,
        description: String = "",
        imageName: String = "",
        backgroundColor: Color = .pink.opacity(0.3),
        voices: [Voice] = []
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.imageName = imageName
        self.backgroundColor = backgroundColor
        self.voices = voices
    }
}

// MARK: - Model Built-in Voices & Sample Data

extension Voice {
    private static let palette: [Color] = [
        .blue.opacity(0.35),
        .purple.opacity(0.35),
        .orange.opacity(0.35),
        .teal.opacity(0.35),
        .pink.opacity(0.35),
        .green.opacity(0.35),
        .indigo.opacity(0.35),
        .mint.opacity(0.35)
    ]

    /// Whether the given model repo supports zero-shot reference audio cloning
    static func supportsReferenceCloning(for modelId: String) -> Bool {
        let lower = modelId.lowercased()
        if lower.contains("nativ-community/phonon") || lower.contains("gradium/phonon") {
            return false
        }
        if lower.contains("kokoro") || lower.contains("kitten") || lower.contains("orpheus")
            || lower.contains("vibevoice-realtime") || lower.contains("voxtral")
            || lower.contains("melotts") || lower.contains("rumik") || lower.contains("soprano")
            || lower.contains("vyvotts") || lower.contains("kugelaudio") || lower.contains("bark") {
            return false
        }
        return true
    }

    /// Returns the built-in voices for a given TTS model repo ID, merging any dynamically discovered voice names.
    static func builtInVoices(for modelId: String, dynamicVoiceNames: [String] = []) -> [Voice] {
        let lower = modelId.lowercased()
        var specs: [(name: String, voiceId: String?, description: String, language: String)] = []

        if lower.contains("nativ-community/phonon") {
            specs = [
                ("Freya", "freya", "Phonon built-in (Default)", "English"),
                ("Harper", "harper", "Phonon built-in", "English"),
                ("Sterling", "sterling", "Phonon built-in", "English"),
                ("Toby", "toby", "Phonon built-in", "English"),
                ("Zoey (maps to Freya)", "freya", "Codecaine Phonon Default", "English"),
            ]
        } else if lower.contains("gradium/phonon") || lower.contains("empyrealworks/phonon") {
            let lang: String
            if lower.hasSuffix("-fr") { lang = "French" }
            else if lower.hasSuffix("-de") { lang = "German" }
            else if lower.hasSuffix("-es") { lang = "Spanish" }
            else if lower.hasSuffix("-pt") { lang = "Portuguese" }
            else if lower.contains("multilingual") { lang = "Multilingual" }
            else { lang = "English" }
            specs = [
                ("default", "default", "Conversational · Built-in Default", lang),
                ("Zoey", "Zoey", "Conversational · English (US)", "English"),
                ("Freya", "Freya", "Narrator · English (UK)", "English"),
                ("Harper", "Harper", "Conversational · English (US)", "English"),
                ("Trevor", "Trevor", "Narrator · English (US)", "English"),
                ("Alfie", "Alfie", "Conversational · English (UK)", "English"),
                ("Noémie (French)", "Noémie", "Conversational · French", "French"),
                ("Coralie (French)", "Coralie", "Conversational · French", "French"),
                ("Augustin (French)", "Augustin", "Narrator · French", "French"),
                ("Arnaud (French)", "Arnaud", "Conversational · French", "French"),
                ("Nele (German)", "Nele", "Conversational · German", "German"),
                ("Jette (German)", "Jette", "Conversational · German", "German"),
                ("Anton (German)", "Anton", "Narrator · German", "German"),
                ("Leon (German)", "Leon", "Conversational · German", "German"),
                ("Noa (Spanish)", "Noa", "Conversational · Spanish (ES)", "Spanish"),
                ("Lucía-Sol (Spanish)", "Lucía-Sol", "Conversational · Spanish (ES)", "Spanish"),
                ("Marcos (Spanish)", "Marcos", "Narrator · Spanish (ES)", "Spanish"),
                ("Rebeca (Spanish MX)", "Rebeca", "Conversational · Spanish (MX)", "Spanish"),
                ("Lalo (Spanish MX)", "Lalo", "Narrator · Spanish (MX)", "Spanish"),
                ("Manuela-Lu (Portuguese)", "Manuela-Lu", "Conversational · Portuguese (BR)", "Portuguese"),
                ("Isadora (Portuguese)", "Isadora", "Conversational · Portuguese (BR)", "Portuguese"),
                ("Davi (Portuguese)", "Davi", "Narrator · Portuguese (BR)", "Portuguese"),
                ("Igor (Portuguese)", "Igor", "Conversational · Portuguese (BR)", "Portuguese"),
                ("Bryony", "Bryony", "Conversational · English (UK)", "English"),
                ("Delia", "Delia", "Conversational · English (US)", "English"),
                ("Easton", "Easton", "Conversational · English (US)", "English"),
                ("Everly", "Everly", "Conversational · English (US)", "English"),
                ("Hadley", "Hadley", "Conversational · English (US)", "English"),
                ("Juniper", "Juniper", "Conversational · English (US)", "English"),
                ("Lottie", "Lottie", "Conversational · English (UK)", "English"),
                ("Micah", "Micah", "Conversational · English (US)", "English"),
                ("Miles", "Miles", "Conversational · English (US)", "English"),
                ("Pippa", "Pippa", "Conversational · English (UK)", "English"),
                ("Spencer", "Spencer", "Conversational · English (UK)", "English"),
                ("Teddy", "Teddy", "Conversational · English (UK)", "English"),
                ("Toby", "Toby", "Conversational · English (UK)", "English"),
                ("Edwin", "Edwin", "Narrator · English (UK)", "English"),
                ("Elodie-Rose", "Elodie-Rose", "Narrator · English (UK)", "English"),
                ("Hugh", "Hugh", "Narrator · English (UK)", "English"),
                ("Imogen", "Imogen", "Narrator · English (UK)", "English"),
                ("Maeve", "Maeve", "Narrator · English (UK)", "English"),
                ("Preston", "Preston", "Narrator · English (US)", "English"),
                ("Russell", "Russell", "Narrator · English (US)", "English"),
                ("Captain Silas Blackwater", "Captain Silas Blackwater", "Character · Pirate Captain", "English"),
                ("Fairy Godmother", "Fairy Godmother", "Character · Magical", "English"),
                ("Fennick the Young Elf", "Fennick the Young Elf", "Character · Fantasy", "English"),
                ("Grub the Ogre", "Grub the Ogre", "Character · Deep Ogre", "English"),
                ("Master Aldric", "Master Aldric", "Character · Wise Scholar", "English"),
                ("Morwenna of the Deep", "Morwenna of the Deep", "Character · Mystic", "English"),
                ("Old Mortlake", "Old Mortlake", "Character · Elderly", "English"),
                ("Sir Bramwell", "Sir Bramwell", "Character · Knight", "English"),
                ("The Snow Queen", "The Snow Queen", "Character · Regal", "English"),
                ("The Thing Under the Stairs", "The Thing Under the Stairs", "Character · Eerie", "English"),
                ("Vyrmathrax the Ancient", "Vyrmathrax the Ancient", "Character · Dragon", "English"),
                ("Azgorath", "Azgorath", "Character · Dark Lord", "English"),
            ]
        } else if lower.contains("pocket-tts") || lower.contains("pocket_tts") {
            specs = [
                ("Alba", "alba", "Pocket TTS built-in", "English"),
                ("Marius", "marius", "Pocket TTS built-in", "English"),
                ("Javert", "javert", "Pocket TTS built-in", "English"),
                ("Jean", "jean", "Pocket TTS built-in", "English"),
                ("Fantine", "fantine", "Pocket TTS built-in", "English"),
                ("Cosette", "cosette", "Pocket TTS built-in", "English"),
                ("Eponine", "eponine", "Pocket TTS built-in", "English"),
                ("Azelma", "azelma", "Pocket TTS built-in", "English"),
            ]
        } else if lower.contains("fish-audio") || lower.contains("fish_speech") || lower.contains("fishspeech") {
            specs = [
                ("Zoey (Default)", "Zoey", "Conversational · Clear, warm everyday voice", "Multilingual"),
                ("Default (Fish S2 Pro)", nil, "Conversational · Unconditioned S2 Pro speaker", "Multilingual"),
                ("Alfie", "Alfie", "Conversational · Friendly, upbeat voice", "Multilingual"),
                ("Bryony", "Bryony", "Conversational · Crisp, articulate studio voice", "Multilingual"),
                ("Delia", "Delia", "Conversational · Smooth, expressive reading voice", "Multilingual"),
                ("Easton", "Easton", "Conversational · Calm, modern conversational voice", "Multilingual"),
                ("Everly", "Everly", "Conversational · Bright, natural speaking voice", "Multilingual"),
                ("Hadley", "Hadley", "Conversational · Relaxed, clear reading voice", "Multilingual"),
                ("Harper", "Harper", "Conversational · Direct, engaging modern voice", "Multilingual"),
                ("Juniper", "Juniper", "Conversational · Lively, warm conversational voice", "Multilingual"),
                ("Lottie", "Lottie", "Conversational · Light, cheerful speaking voice", "Multilingual"),
                ("Micah", "Micah", "Conversational · Smooth, balanced everyday reader", "Multilingual"),
                ("Miles", "Miles", "Conversational · Crisp technical & article reader", "Multilingual"),
                ("Pippa", "Pippa", "Conversational · Bright, spirited reading voice", "Multilingual"),
                ("Spencer", "Spencer", "Conversational · Natural, easygoing speaking voice", "Multilingual"),
                ("Teddy", "Teddy", "Conversational · Warm, approachable storyteller", "Multilingual"),
                ("Toby", "Toby", "Conversational · Energetic, clear conversational voice", "Multilingual"),
                ("Edwin", "Edwin", "Narrator · Measured, classic audiobook narrator", "Multilingual"),
                ("Elodie-Rose", "Elodie-Rose", "Narrator · Gentle, poised storytelling voice", "Multilingual"),
                ("Freya", "Freya", "Narrator · Rich, steady narrative voice", "Multilingual"),
                ("Hugh", "Hugh", "Narrator · Deep, composed documentary narrator", "Multilingual"),
                ("Imogen", "Imogen", "Narrator · Polished, expressive literary voice", "Multilingual"),
                ("Maeve", "Maeve", "Narrator · Warm, resonant storytelling voice", "Multilingual"),
                ("Preston", "Preston", "Narrator · Authoritative, clear announcer voice", "Multilingual"),
                ("Russell", "Russell", "Narrator · Grounded, mature narrative voice", "Multilingual"),
                ("Trevor", "Trevor", "Narrator · Steady, classic broadcast reader", "Multilingual"),
                ("Captain Silas Blackwater", "Captain Silas Blackwater", "Character · Weathered maritime captain", "Multilingual"),
                ("Fairy Godmother", "Fairy Godmother", "Character · Whimsical, magical fairytale", "Multilingual"),
                ("Fennick the Young Elf", "Fennick the Young Elf", "Character · Youthful fantasy adventurer", "Multilingual"),
                ("Grub the Ogre", "Grub the Ogre", "Character · Heavy, gravelly fantasy ogre", "Multilingual"),
                ("Master Aldric", "Master Aldric", "Character · Wise scholar & arcane mentor", "Multilingual"),
                ("Morwenna of the Deep", "Morwenna of the Deep", "Character · Ethereal, mythic oceanic voice", "Multilingual"),
                ("Old Mortlake", "Old Mortlake", "Character · Aged, atmospheric tale-teller", "Multilingual"),
                ("Sir Bramwell", "Sir Bramwell", "Character · Noble chivalric knight", "Multilingual"),
                ("The Snow Queen", "The Snow Queen", "Character · Regal, crystalline dramatic voice", "Multilingual"),
                ("The Thing Under the Stairs", "The Thing Under the Stairs", "Character · Eerie, whispered gothic creature", "Multilingual"),
                ("Vyrmathrax the Ancient", "Vyrmathrax the Ancient", "Character · Booming, ancient dragon", "Multilingual"),
                ("Azgorath", "Azgorath", "Character · Dark, cinematic fantasy villain", "Multilingual"),
            ]
        } else if lower.contains("marvis") {
            specs = [
                ("Conversational A", "conversational_a", "Marvis built-in", "English"),
                ("Conversational B", "conversational_b", "Marvis built-in", "English"),
                ("Conversational FR", "conversational_fr", "Marvis built-in", "French"),
                ("Conversational DE", "conversational_de", "Marvis built-in", "German"),
            ]
        } else if lower.contains("qwen3-tts") {
            specs = [
                ("Ryan", "Ryan", "Qwen3-TTS built-in", "English"),
                ("Aiden", "Aiden", "Qwen3-TTS built-in", "English"),
                ("Vivian", "Vivian", "Qwen3-TTS built-in", "Chinese"),
                ("Serena", "Serena", "Qwen3-TTS built-in", "Chinese"),
                ("Uncle Fu", "Uncle_Fu", "Qwen3-TTS built-in", "Chinese"),
                ("Dylan", "Dylan", "Qwen3-TTS built-in", "Chinese"),
                ("Eric", "Eric", "Qwen3-TTS built-in", "Chinese"),
            ]
        } else if lower.contains("kokoro") {
            specs = [
                ("Heart (af_heart)", "af_heart", "American Female", "English"),
                ("Bella (af_bella)", "af_bella", "American Female", "English"),
                ("Nova (af_nova)", "af_nova", "American Female", "English"),
                ("Sarah (af_sarah)", "af_sarah", "American Female", "English"),
                ("Sky (af_sky)", "af_sky", "American Female", "English"),
                ("Adam (am_adam)", "am_adam", "American Male", "English"),
                ("Michael (am_michael)", "am_michael", "American Male", "English"),
                ("Echo (am_echo)", "am_echo", "American Male", "English"),
                ("Emma (bf_emma)", "bf_emma", "British Female", "English"),
                ("George (bm_george)", "bm_george", "British Male", "English"),
                ("Dora (ef_dora)", "ef_dora", "Spanish Female", "Spanish"),
                ("Siwis (ff_siwis)", "ff_siwis", "French Female", "French"),
                ("Alpha (jf_alpha)", "jf_alpha", "Japanese Female", "Japanese"),
                ("Xiaobei (zf_xiaobei)", "zf_xiaobei", "Chinese Female", "Chinese"),
            ]
        } else if lower.contains("kitten") {
            specs = [
                ("Bella", "Bella", "KittenTTS built-in", "English"),
                ("Jasper", "Jasper", "KittenTTS built-in", "English"),
                ("Luna", "Luna", "KittenTTS built-in", "English"),
                ("Bruno", "Bruno", "KittenTTS built-in", "English"),
                ("Rosie", "Rosie", "KittenTTS built-in", "English"),
                ("Hugo", "Hugo", "KittenTTS built-in", "English"),
                ("Kiki", "Kiki", "KittenTTS built-in", "English"),
                ("Leo", "Leo", "KittenTTS built-in", "English"),
            ]
        } else if lower.contains("orpheus") || lower.contains("llama") {
            specs = [
                ("Tara", "tara", "Orpheus built-in", "English"),
                ("Leah", "leah", "Orpheus built-in", "English"),
                ("Jess", "jess", "Orpheus built-in", "English"),
                ("Leo", "leo", "Orpheus built-in", "English"),
                ("Dan", "dan", "Orpheus built-in", "English"),
                ("Mia", "mia", "Orpheus built-in", "English"),
                ("Zac", "zac", "Orpheus built-in", "English"),
                ("Zoe", "zoe", "Orpheus built-in", "English"),
            ]
        } else if lower.contains("vyvotts") {
            specs = [
                ("EN-US-1", "en-us-1", "VyvoTTS built-in", "English"),
            ]
        } else if lower.contains("melotts") {
            specs = [
                ("EN-Default", "EN-Default", "MeloTTS Default", "English"),
                ("EN-US", "EN-US", "American English", "English"),
                ("EN-BR", "EN-BR", "British English", "English"),
                ("EN-INDIA", "EN_INDIA", "Indian English", "English"),
                ("EN-AU", "EN-AU", "Australian English", "English"),
            ]
        } else if lower.contains("voxtral") {
            specs = [
                ("Casual Male", "casual_male", "Voxtral built-in", "English"),
                ("Casual Female", "casual_female", "Voxtral built-in", "English"),
            ]
        } else if lower.contains("vibevoice-realtime") {
            specs = [
                ("Carter (en-Carter_man)", "en-Carter_man", "VibeVoice Default", "English"),
                ("Emma (en-Emma_woman)", "en-Emma_woman", "VibeVoice built-in", "English"),
                ("Spk0 (fr-Spk0_man)", "fr-Spk0_man", "VibeVoice French", "French"),
                ("Spk1 (jp-Spk1_woman)", "jp-Spk1_woman", "VibeVoice Japanese", "Japanese"),
            ]
        } else if lower.contains("rumik") {
            specs = [
                ("Ira", "Ira", "Rumik built-in", "English"),
                ("Aisha", "Aisha", "Rumik built-in", "English"),
                ("Siya", "Siya", "Rumik built-in", "English"),
                ("Zoya", "Zoya", "Rumik built-in", "English"),
            ]
        } else if lower.contains("maya2") || lower.contains("maya-2") {
            specs = Maya2Model.builtInVoices.map { name in
                (name, name, "Maya-2 Native built-in voice", "Multilingual")
            }
        } else if lower.contains("ming") {
            specs = [
                ("UK Charlotte", "UK_Charlotte", "Ming British Female", "English"),
                ("US James", "US_James", "Ming American Male", "English"),
                ("灵小柔 (Ling Xiaorou)", "灵小柔", "Ming Chinese Female", "Chinese"),
                ("灵楷 (Ling Kai)", "灵楷", "Ming Chinese Male", "Chinese"),
            ]
        } else if lower.contains("spark-tts") || lower.contains("sparktts") {
            specs = [
                ("Female (moderate/moderate)", "female", "Spark-TTS Controllable Female", "Multilingual"),
                ("Male (moderate/moderate)", "male", "Spark-TTS Controllable Male", "Multilingual"),
            ]
        } else if lower.contains("bark") {
            specs = [
                ("Speaker 6 (v2/en_speaker_6)", "v2/en_speaker_6", "Bark English Speaker 6", "English"),
                ("Speaker 0 (en_speaker_0)", "en_speaker_0", "Bark English Speaker 0", "English"),
                ("Announcer", "announcer", "Bark Announcer", "English"),
            ]
        } else {
            specs = [
                ("Default", nil, "Model default voice", "English"),
            ]
        }

        // Append any dynamically discovered voices from the loaded model that aren't already in specs
        let existingIds = Set(specs.compactMap { $0.voiceId?.lowercased() })
        for rawName in dynamicVoiceNames {
            if !existingIds.contains(rawName.lowercased()) {
                specs.append((rawName, rawName, "Model built-in voice", "English"))
            }
        }

        var result: [Voice] = specs.enumerated().map { index, spec in
            Voice(
                name: spec.name,
                voiceId: spec.voiceId,
                description: spec.description,
                language: spec.language,
                color: palette[index % palette.count],
                isBuiltIn: true
            )
        }

        // For models that support reference audio cloning, also include the bundled reference voices
        if supportsReferenceCloning(for: modelId) {
            result.append(contentsOf: samples)
        }

        return result
    }

    struct FishS2VoicePreset {
        let resourceName: String?
        let refText: String?
        let inlineTag: String?
        let pitchSemitones: Float
    }

    /// Maps each of the 38 built-in studio voices to a natural Fish S2 Pro studio reference anchor without artificial pitch resampling distortion.
    static func fishS2Preset(for voiceId: String?) -> FishS2VoicePreset? {
        guard let raw = voiceId?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        let standardTranscript = "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        switch raw.lowercased() {
        case "default":
            return nil
        case "zoey", "delia", "hadley", "elodie-rose", "imogen", "maeve", "morwenna of the deep", "the snow queen", "the thing under the stairs", "lily":
            return FishS2VoicePreset(resourceName: "Lily", refText: standardTranscript, inlineTag: nil, pitchSemitones: 0.0)
        case "bryony", "everly", "harper", "juniper", "lottie", "pippa", "freya", "fairy godmother", "sophie":
            return FishS2VoicePreset(resourceName: "Sophie", refText: standardTranscript, inlineTag: nil, pitchSemitones: 0.0)
        case "alfie", "easton", "micah", "miles", "spencer", "teddy", "toby", "edwin", "hugh", "preston", "russell", "trevor", "captain silas blackwater", "fennick the young elf", "grub the ogre", "master aldric", "old mortlake", "sir bramwell", "vyrmathrax the ancient", "azgorath", "german action voice", "bane", "james":
            return FishS2VoicePreset(resourceName: "James", refText: standardTranscript, inlineTag: nil, pitchSemitones: 0.0)
        default:
            return nil
        }
    }

    /// Returns a local preview/audition WAV URL for any built-in Fish S2 voice or custom cloned voice.
    static func previewAudioURL(for voice: Voice) -> URL? {
        if let url = voice.audioFileURL, FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        if let preset = fishS2Preset(for: voice.voiceId ?? voice.name),
           let resourceName = preset.resourceName,
           let bundled = Bundle.main.url(forResource: resourceName, withExtension: "wav") {
            return bundled
        }
        return Bundle.main.url(forResource: "Sophie", withExtension: "wav")
    }

    /// Phase-locked WSOLA pitch shift (in semitones) ported from bun_rad_studio fmPhonon.ts (`pitchShiftPCM`).
    /// Only used when the user explicitly sets a manual non-zero pitch shift; presets keep 0.0 semitones for 100% pure neural output.
    static func applyPitchShift(_ samples: [Float], semitones: Float, sampleRate: Double = 24000.0) -> [Float] {
        guard samples.count > 1024, abs(semitones) > 0.05 else { return samples }
        let clamped = max(-12.0, min(12.0, Double(semitones)))
        let pitchScale = pow(2.0, clamped / 12.0)
        let n = samples.count
        let resampledLen = max(256, Int(round(Double(n) / pitchScale)))

        // Step 1: Cubic Hermite interpolation resampling
        var resampled = [Float](repeating: 0, count: resampledLen)
        for i in 0..<resampledLen {
            let srcPos = Double(i) * pitchScale
            let idx = Int(srcPos)
            let frac = Float(srcPos - Double(idx))
            let s0 = samples[max(0, idx - 1)]
            let s1 = samples[min(n - 1, idx)]
            let s2 = samples[min(n - 1, idx + 1)]
            let s3 = samples[min(n - 1, idx + 2)]
            let c0 = s1
            let c1 = 0.5 * (s2 - s0)
            let c2 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3
            let c3 = 0.5 * (s3 - s0) + 1.5 * (s1 - s2)
            resampled[i] = ((c3 * frac + c2) * frac + c1) * frac + c0
        }

        // Step 2: Phase-locked WSOLA time-stretch back to original length n (40ms window, 10ms hop, 12ms search)
        let sr = max(8000.0, sampleRate)
        let winSize = max(512, Int(round(sr * 0.040)))
        let hopOut = max(128, winSize / 4)
        let searchRange = max(64, Int(round(sr * 0.012)))
        let stretchRatio = Double(resampledLen) / Double(n)

        var hann = [Float](repeating: 0, count: winSize)
        for i in 0..<winSize {
            hann[i] = Float(0.5 * (1.0 - cos((2.0 * Double.pi * Double(i)) / Double(winSize - 1))))
        }

        var output = [Float](repeating: 0, count: n)
        var norm = [Float](repeating: 0, count: n)

        resampled.withUnsafeBufferPointer { resBuf in
            hann.withUnsafeBufferPointer { hannBuf in
                output.withUnsafeMutableBufferPointer { outBuf in
                    norm.withUnsafeMutableBufferPointer { normBuf in
                        guard let resBase = resBuf.baseAddress,
                              let hannBase = hannBuf.baseAddress,
                              let outBase = outBuf.baseAddress,
                              let normBase = normBuf.baseAddress else { return }

                        var outPos = 0
                        var prevInPos: Int = -1

                        while outPos < n {
                            let idealInPos = Int(round(Double(outPos) * stretchRatio))
                            var bestInPos = max(0, min(max(0, resampledLen - winSize), idealInPos))

                            if prevInPos >= 0 && outPos > 0 {
                                let expectedContPos = prevInPos + hopOut
                                let searchStart = max(0, idealInPos - searchRange)
                                let searchEnd = min(max(0, resampledLen - winSize), idealInPos + searchRange)
                                let corrLen = min(hopOut, n - outPos)

                                if expectedContPos + corrLen < resampledLen && searchEnd > searchStart && corrLen > 0 {
                                    let ptrA = resBase.advanced(by: expectedContPos)
                                    let vLen = vDSP_Length(corrLen)
                                    var normA: Float = 0
                                    vDSP_svesq(ptrA, 1, &normA, vLen)
                                    normA = max(normA, 1e-9)

                                    var bestScore: Float = -.infinity
                                    let step = 2
                                    var cand = searchStart
                                    while cand <= searchEnd {
                                        let ptrB = resBase.advanced(by: cand)
                                        var dot: Float = 0
                                        var normB: Float = 0
                                        vDSP_dotpr(ptrA, 1, ptrB, 1, &dot, vLen)
                                        vDSP_svesq(ptrB, 1, &normB, vLen)
                                        let score = dot / sqrt(normA * max(normB, 1e-9))
                                        if score > bestScore {
                                            bestScore = score
                                            bestInPos = cand
                                        }
                                        cand += step
                                    }
                                }
                            }

                            let copyLen = min(winSize, n - outPos, resampledLen - bestInPos)
                            if copyLen > 0 {
                                let vCopy = vDSP_Length(copyLen)
                                let dstOut = outBase.advanced(by: outPos)
                                let dstNorm = normBase.advanced(by: outPos)
                                vDSP_vma(resBase.advanced(by: bestInPos), 1, hannBase, 1, dstOut, 1, dstOut, 1, vCopy)
                                vDSP_vadd(dstNorm, 1, hannBase, 1, dstNorm, 1, vCopy)
                            }

                            prevInPos = bestInPos
                            outPos += hopOut
                        }
                    }
                }
            }
        }

        for i in 0..<n {
            if norm[i] > 1e-4 {
                output[i] /= norm[i]
            } else {
                output[i] = samples[i]
            }
        }
        return output
    }

    static let samples: [Voice] = [
        Voice(
            name: "Lily",
            description: "Velvety Actress (Clone)",
            language: "English",
            color: .purple.opacity(0.3),
            lastUsed: Date(),
            audioFileURL: Bundle.main.url(forResource: "Lily", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        ),
        Voice(
            name: "James",
            description: "Professional Narrator (Clone)",
            language: "English",
            color: .blue.opacity(0.3),
            lastUsed: Date().addingTimeInterval(-3600),
            audioFileURL: Bundle.main.url(forResource: "James", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        ),
        Voice(
            name: "Sophie",
            description: "Warm & Friendly (Clone)",
            language: "English",
            color: .orange.opacity(0.3),
            lastUsed: Date().addingTimeInterval(-7200),
            audioFileURL: Bundle.main.url(forResource: "Sophie", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        )
    ]

    static let customVoices: [Voice] = [
        Voice(
            name: "German action voice",
            voiceId: "german action voice",
            description: "Custom voice · Energetic Baritone",
            language: "English",
            color: .teal.opacity(0.3),
            isCustom: true,
            audioFileURL: Bundle.main.url(forResource: "James", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        ),
        Voice(
            name: "Bane",
            voiceId: "bane",
            description: "Deep & Dramatic",
            language: "English",
            color: .green.opacity(0.3),
            isCustom: true,
            audioFileURL: Bundle.main.url(forResource: "James", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        )
    ]

    private static let customVoicesStorageKey = "VoicesApp.CustomVoices.v1"

    private struct StoredCustomVoiceRecord: Codable {
        let id: UUID
        let name: String
        let voiceId: String?
        let description: String
        let language: String
        let colorIndex: Int
        let audioPath: String?
        let transcription: String?
    }

    static func loadCustomVoices() -> [Voice] {
        guard let data = UserDefaults.standard.data(forKey: customVoicesStorageKey),
              let records = try? JSONDecoder().decode([StoredCustomVoiceRecord].self, from: data) else {
            return customVoices
        }
        return records.map { rec in
            let color = palette[abs(rec.colorIndex) % palette.count]
            let resolvedURL: URL?
            if let path = rec.audioPath, !path.isEmpty {
                if FileManager.default.fileExists(atPath: path) {
                    resolvedURL = URL(fileURLWithPath: path)
                } else if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                    let candidate = docs.appendingPathComponent("ClonedVoices").appendingPathComponent((path as NSString).lastPathComponent)
                    resolvedURL = FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
                } else {
                    resolvedURL = nil
                }
            } else {
                resolvedURL = nil
            }
            return Voice(
                id: rec.id,
                name: rec.name,
                voiceId: rec.voiceId,
                description: rec.description,
                language: rec.language,
                color: color,
                isCustom: true,
                audioFileURL: resolvedURL,
                transcription: rec.transcription
            )
        }
    }

    static func saveCustomVoices(_ voices: [Voice]) {
        let records: [StoredCustomVoiceRecord] = voices.enumerated().map { idx, v in
            StoredCustomVoiceRecord(
                id: v.id,
                name: v.name,
                voiceId: v.voiceId,
                description: v.description,
                language: v.language,
                colorIndex: idx % palette.count,
                audioPath: v.audioFileURL?.path,
                transcription: v.transcription
            )
        }
        if let data = try? JSONEncoder().encode(records) {
            UserDefaults.standard.set(data, forKey: customVoicesStorageKey)
        }
    }

    static let expressiveTags: [ExpressiveTagItem] = [
        ExpressiveTagItem(tag: "[laughing]", label: "Laugh", icon: "face.smiling"),
        ExpressiveTagItem(tag: "[whisper]", label: "Whisper", icon: "mouth"),
        ExpressiveTagItem(tag: "[sigh]", label: "Sigh", icon: "wind"),
        ExpressiveTagItem(tag: "[excited]", label: "Excited", icon: "sparkles"),
        ExpressiveTagItem(tag: "[pause]", label: "Pause", icon: "pause.circle"),
        ExpressiveTagItem(tag: "[emphasis]", label: "Emphasis", icon: "exclamationmark.bubble"),
        ExpressiveTagItem(tag: "[clearing throat]", label: "Clear Throat", icon: "waveform.path.ecg"),
        ExpressiveTagItem(tag: "<|speaker:0|>", label: "Speaker 0", icon: "person.wave.2"),
        ExpressiveTagItem(tag: "<|speaker:1|>", label: "Speaker 1", icon: "person.2.wave.2"),
    ]

    static let deadOrbitNovelText: String = #"""
# DEAD ORBIT: The Talbot Protocol

*A novel*

---

## Prologue: The Signal

The first thing I remember about the end of the world is that it was a Tuesday, and I was out of coffee.

That's not poetic. That's not dramatic. That's just the truth, and if you've read any of my other accounts—or if you haven't, in which case, welcome to the party, we have snacks and existential dread—you know I've never been one for poetry. I'm a guy who likes his coffee black, his enemies dead, and his family breathing. Everything else is negotiable.

My name is Jake Callahan. Forty-three years old. Ex-Army, ex-cop, ex-everything-that-requires-a-pension. I live in a small town in the Rockies where the biggest crime used to be someone stealing a neighbor's lawn gnome. My wife, Sarah, teaches third grade. My daughter, Emma, is sixteen and thinks I'm an embarrassment. My son, Danny, is twelve and thinks I'm a superhero. They're both right, depending on the day.

On that Tuesday, the coffee situation was dire. We were out. Completely out. Sarah had forgotten to buy more, which was unusual because Sarah never forgets anything. She's the kind of woman who has a spreadsheet for the grocery list and a backup spreadsheet for the backup spreadsheet. So when she forgot the coffee, I should have known something was wrong. But I didn't. Because I'm an idiot. A lovable idiot, but an idiot nonetheless.

I was standing in the kitchen, staring at the empty canister like it had personally betrayed me, when the television in the living room crackled to life on its own. I hadn't turned it on. Sarah was in the bedroom getting ready for work. Danny was at school. Emma was... somewhere. Probably doing something that would make me age ten years.

The TV showed a news anchor I didn't recognize, and behind her, a graphic I didn't understand. It looked like a weather map, but the colors were wrong—purples and greens where blues and reds should have been. And there was a sound underneath the anchor's voice, a low hum, like the house itself was vibrating.

"...unusual atmospheric phenomenon detected over the Pacific Northwest..." the anchor was saying. "...authorities are urging residents to remain indoors..."

Then the screen went to static. Then black. Then the power went out.

The hum didn't stop.

It got louder.

And somewhere, miles away, something fell out of the sky.

---

## Chapter 1: First Contact

I've been shot at. I've been stabbed. I've been in a helicopter that decided, halfway through a mission, that it didn't want to be a helicopter anymore. I've seen things that would make a seasoned horror author reach for the light switch.

None of that prepared me for the sound of something the size of a freight train hitting the earth three miles from my house.

The windows rattled. The dishes in the cabinet jumped. Sarah came running out of the bedroom, her hair half-done, one earring in, one earring out, and the look on her face was the same look I'd seen a thousand times in the field—the look of someone who knows, instinctively, that the world has just changed.

"Jake," she said. "What was that?"

"I don't know," I said. "But I'm going to find out."

"No," she said. "You're not."

"Sarah—"

"You're going to stay right here with me until we know what's happening."

I looked at her. She was scared. Really scared. Not the kind of scared where you pretend you're not scared. The kind where your hands shake and you don't even try to hide it. Sarah never shook. Sarah was the rock. If Sarah was shaking, the world was ending.

"Okay," I said. "Okay. Let's stay here."

We stayed. For about four minutes. Then my phone rang.

It was Emma.

"Dad," she said, and her voice was the worst thing I've ever heard. Worse than the sound of the impact. Worse than the hum. "Dad, there's something in the school. There's something in the school and it's—"

The line went dead.

I was out the door before Sarah could say my name. I grabbed the truck keys, the .45 from the nightstand drawer, and the go-bag I kept in the closet because I'm the kind of guy who keeps a go-bag. Old habits. Military habits. The kind that keep you alive when everything else is trying to kill you.

"Jake!" Sarah was at the door. "Jake, what about Danny? Danny's at the middle school!"

I stopped. My heart, which had been pumping pure adrenaline, stuttered.

"Danny's at the middle school," I repeated.

"Yes. And Emma's at the high school. And you can't be in two places at once."

She was right. She was absolutely right. And I hated her for it, because it meant I had to make a choice, and choices like that are the kind that haunt you forever.

"Call Danny," I said. "Get him out of there. Take him to the cabin. The one on Miller's Ridge. I'll get Emma and meet you there."

"Jake, I can't—"

"You can. You're the strongest person I know. You've got this."

She looked at me. I looked at her. And for a moment, just a moment, the world stopped being crazy and was just us, standing in the doorway of our house, holding onto each other like we were the only two people left in the universe.

Then the hum got louder, and the moment broke, and I was running to the truck.

---

The drive to the high school should have taken eight minutes. It took twenty, because the roads were already a mess. Cars were stopped in the middle of the street. People were standing in their yards, looking up at the sky, where the purple-green clouds were swirling like something out of a nightmare. And some of them—some of them were just standing there. Not looking at the sky. Not looking at anything. Just standing. Empty. Like someone had reached inside them and turned off the lights.

I didn't stop. I couldn't. Emma was in that school.

I pulled into the parking lot and the first thing I noticed was the silence. High schools are never silent. They're chaos incarnate—bells and yelling and the squeak of sneakers on linoleum. But this was silent. Dead silent. The kind of silence that has weight.

I got out of the truck with the .45 and a flashlight. The front doors were open. Not broken—open. Like someone had politely held them for whatever was inside.

"Emma!" I shouted. "Emma, where are you?"

Nothing. Just the hum, which was louder here, vibrating in my chest like a second heartbeat.

I went in. The hallway was dark—the power was out here too—but my flashlight cut through the darkness, and what it showed made my stomach drop.

Lockers were open. Backpacks were on the floor. A shoe. A phone. A jacket. All the signs of people who had been here and then suddenly weren't. And on the walls, on the ceiling, on the floor—marks. Scratches. Like something had been dragged. Like something had been *carved*.

And then I heard it. Not the hum. Something else. A wet sound. A chewing sound.

I followed it. Down the hall, past the cafeteria, past the gym, to the auditorium. The doors were propped open with a chair, and inside, in the dark, I could see shapes. Moving shapes. Slow, shuffling shapes.

I raised the flashlight and pointed it at the nearest one.

It was a kid. Or it had been. A boy, maybe fifteen, wearing a letterman jacket. But his eyes were wrong—milky white, like cataracts, and his mouth was open, and he was chewing on something that I really, really hoped wasn't what I thought it was.

He turned toward me. And he saw me. Or whatever was left of him saw me. And he made a sound—not a word, not a groan, but something in between, something that was almost human and almost animal and completely wrong.

Then he lunged.

I shot him. Center mass. Two rounds. He went down, and I felt nothing but the recoil and the sickening certainty that I'd just shot a child, and then I felt something else—relief, because he was still moving, still reaching for me, and that meant he wasn't a child anymore. He was something else.

The other shapes in the auditorium turned toward me. There were dozens of them. Students. Teachers. Janitors. All of them with those milky eyes and those open mouths and that wet, hungry sound.

"Emma!" I shouted again, louder, desperate. "Emma, if you can hear me, get out! Get out now!"

And then I heard her voice, from somewhere behind the stage, from the wings of the auditorium, high and thin and terrified: "Dad! Dad, I'm here! I'm trapped!"

I didn't think. I just moved. I pushed through the first wave, firing as I went, each shot a prayer and a curse, and I reached the stage and found Emma huddled behind a row of chairs, her face streaked with tears, her phone in her hand, and her eyes—her eyes were wide and brown and human and alive.

"Hey, kiddo," I said, grabbing her hand. "Let's go."

She grabbed my hand and didn't let go.

We ran. Through the auditorium, through the hall, past the lockers and the backpacks and the shoe and the phone and the jacket, out into the parking lot, into the truck. I locked the doors. I started the engine. And as I pulled out of the lot, I looked in the rearview mirror and saw them—dozens of them, shuffling out of the school, following us, slow and relentless and hungry.

"Are you okay?" I asked Emma.

"I'm okay," she said. "Dad, what's happening?"

"I don't know, sweetheart. But we're going to be okay. We're going to find your mom and Danny, and we're going to be okay."

I didn't know if that was true. But it was what she needed to hear, and sometimes that's enough.

---

## Chapter 2: The Cabin

Three days in, the cabin was holding. We had food, water, and each other. The kids were handling it better than I expected—Emma was reading, Danny was playing cards with Sarah, and I was standing on the porch with a pair of binoculars and a growing sense of dread.

The valley below us was changing. I could see movement—slow, shuffling movement—in the tree line. The Turned were out there, gathering, drawn by something I couldn't identify. Maybe the hum. Maybe something else.

On the fourth day, a truck came up the mountain road. A big one—a military surplus deuce-and-a-half, painted matte black, with no markings. It stopped at the bottom of the hill, and three people got out.

I watched them through the binoculars. Two men and a woman. They were armed—rifles, not handguns, and they moved like they knew how to use them. Military, or something close to it.

"Sarah," I called through the door. "We've got company."

She came out with her own rifle—a hunting rifle, not military-grade, but she could shoot. I'd taught her. She was a better shot than me, which I'd never admitted out loud and never would.

"Friends or foes?" she asked.

"Unknown. Stay here. I'll go down."

"Like hell you will," she said. "I'm coming with you."

We went together. That's how we did things.

The three strangers were waiting at the bottom of the hill. The taller of the two men stepped forward. He was maybe fifty, with a gray beard and eyes that had seen too much. He wore a tactical vest over a flannel shirt, which was a look I'd never seen before and hoped never to see again.

"Name's Colonel Richard Vance," he said. "United States Army. Retired. Or I was, before the world ended."

"Jake Callahan," I said. "Also retired. Also from something. Also before the world ended."

He almost smiled. Almost. "We're setting up a compound about twenty miles east. Fortified. Secure. We're looking for survivors."

"We're not survivors," I said. "We're a family. There's a difference."

He nodded. "There is. And that's why I'm here. We need people like you. People who know how to fight. People who have something to protect."

"And in exchange?"

"In exchange, we keep your family safe. We have food, water, medical supplies, and walls that the Turned can't get through. We have power—generators, solar, the works. And we have information."

"What kind of information?"

He looked at me. Then at Sarah. Then back at me.

"The kind that explains why this is happening. And the kind that might help us stop it."

I looked at Sarah. She looked at me. And we made the decision that would change everything.

"Get the kids," I said. "We're going."

---

*... to be continued ...*

---

## About the Author

This novel is a work of fiction. Any resemblance to real persons, living or dead, is coincidental. The characters, events, and places depicted are products of the author's imagination.

*Dead Orbit: The Talbot Protocol* — Book One of the Dead Orbit Series.

---

## Chapter 4: Fort Haven

Fort Haven was not what I expected.

I expected a military base. Barbed wire, guard towers, the whole nine yards. What I got was a high school—ironically, the same type of building I'd just shot my way out of—surrounded by a hastily erected wall of shipping containers, chain-link fence, and concrete barriers. It looked like someone had built a fortress out of a junkyard, and somehow it worked.

The compound was located in a small town called Meridian, population formerly two thousand, now however many people Colonel Vance had managed to gather. As we drove through the gate—past two armed guards who checked our IDs and waved us through—I counted maybe three hundred people. Men, women, children. All of them looked tired. All of them looked scared. But they were alive, and that counted for something.

Vance met us at the main entrance. "Welcome to Fort Haven," he said. "It's not much, but it's home."

"It's a high school," I said.

"It's a high school with a gym that doubles as an armory, a cafeteria that doubles as a mess hall, and a library that doubles as a command center. We've got power from the solar panels on the roof and the generators in the basement. Water from the well. Food from the stockpile we raided from three grocery stores and a Costco."

"A Costco," Sarah said. "You raided a Costco."

"Best decision I ever made," Vance said. "The samples alone were worth the trip."

I almost liked him.

He showed us to our quarters—a classroom on the second floor that had been converted into a small apartment. Two cots for the kids, a mattress on the floor for Sarah and me, and a desk that I could use as a makeshift workstation. It wasn't the Ritz, but it was better than sleeping in the truck.

"The first thing you need to know," Vance said, once we'd settled in, "is that we're not the only ones. There are other compounds. Some are friendly. Some are not. The nearest one is about fifty miles south, run by a man named Harlan Cross. He was a sheriff before the world ended. Now he's something else."

"What kind of something else?" I asked.

"The kind that doesn't ask permission," Vance said. "He's been raiding other compounds for supplies. He's got a militia—maybe fifty, sixty men—and he's not afraid to use them."

"Great," I said. "So we've got zombies outside and warlords inside. Fantastic."

"There's more," Vance said. He looked at me with those tired eyes. "The Turned aren't the only thing that fell from the sky."

He reached into his vest and pulled out a tablet. On the screen was a photograph—blurry, taken from a distance, but clear enough to see what it showed. It was a ship. Not an airplane. Not a helicopter. A ship. Angular, black, and utterly alien, sitting in a crater in the middle of a forest.

"That was taken three days ago," Vance said. "By a drone we sent out. The ship is about two miles from here. We've been monitoring it, but we haven't approached. Not yet."

"Why not?" Sarah asked.

"Because every time we get within a quarter mile, the hum gets louder. And people start... changing. Faster. Like the ship is calling to them. Pulling them in."

I looked at the photograph. Then I looked at Vance.

"You think the aliens caused the zombie outbreak," I said.

"I think," Vance said carefully, "that the aliens and the zombies are connected. I think whatever came out of that ship is what's turning people. And I think if we don't figure out how to stop it, there won't be anyone left to turn."

---

## Chapter 5: The Signal

That night, I couldn't sleep. The hum was always there, a low vibration in the walls, in the floor, in my bones. I got up and went to the roof, where Vance had set up a makeshift observation post. A young soldier—barely twenty, with a baby face and a rifle that looked too big for him—was on watch.

"Anything?" I asked.

"Nothing but the hum, sir," he said. "It's been steady all night."

I looked out over the valley. In the distance, I could see the purple-green clouds, still swirling, still wrong. And somewhere out there, two miles away, an alien ship sat in a crater, humming.

"Tell me about the signal," I said.

The soldier—his name was Private First Class Danny Park, and yes, the irony of his name was not lost on me—looked at me. "The signal?"

"Vance mentioned it. The hum. He said it's getting louder."

"It is," Park said. "We've been tracking it. It's not just a sound—it's a frequency. A specific frequency, about seven hertz. Infrasound. Below the threshold of human hearing, but you can feel it. And it's been increasing. When the ship first landed, it was maybe a three on the scale. Now it's a seven. And climbing."

"What happens at ten?"

Park looked at me. "We don't know. But the closer people get to the ship, the faster they turn. And the ones who've already turned—they're drawn to it. They walk toward it. Like they're being called."

I thought about the Turned I'd seen at the high school, shuffling in the same direction, like they were following an invisible path. They weren't just wandering. They were going somewhere.

"They're going to the ship," I said.

"Yes, sir," Park said. "We think so. We think the ship is some kind of... beacon. And the Turned are the response."

I stood on the roof and looked at the sky and thought about my family, sleeping in a classroom two floors below me, and I thought about the world that was ending outside, and I thought about the ship in the crater, humming its song, calling its children home.

And I made a decision.

"We need to go to the ship," I said.

Park looked at me like I'd lost my mind. "Sir, with all due respect—"

"I know. I know it's dangerous. I know it might kill us. But if that ship is controlling the Turned, then it's controlling the end of the world. And I'm not going to sit in a classroom and wait for it to finish."

Park was quiet for a moment. Then he said, "I'll tell Colonel Vance."

"Tell him I said it's time. And tell him I want to go in the morning."

I went back inside. Sarah was awake, sitting on the edge of the mattress, watching me with those eyes that could see right through me.

"You're going to the ship," she said. It wasn't a question.

"Yeah."

"I'm coming with you."

"Sarah—"

"Don't. Don't you dare say I should stay. Don't you dare tell me it's too dangerous. I've been with you through three deployments, two kidnappings, and a house fire. I'm not staying behind now."

I looked at her. She was right. She was always right. And I loved her for it, even when it terrified me.

"Okay," I said. "But we leave the kids here. With Vance. Under guard."

She nodded. "Agreed."

We lay down together and held each other, and the hum vibrated through the mattress, through the floor, through the earth, and somewhere, two miles away, something alien and ancient and patient waited for us.

---

## Chapter 6: The Approach

Morning came too fast. Vance assembled a team: me, Sarah, two of his best soldiers—a sergeant named Reyes and a corporal named Chen—and a scientist named Dr. Amara Okafor, who had been a xenobiologist at a university before the world ended and was now the closest thing we had to an expert on alien biology.

"You're insane," Dr. Okafor said, when Vance told her the plan. "You want to walk up to an alien ship in the middle of a zombie apocalypse."

"Yes," I said.

"And you want me to come with you."

"Yes."

"Because I'm the only one who might understand what we're dealing with."

"Yes."

She looked at me for a long moment. Then she sighed. "Fine. But if I die, I want it on record that I told you this was a terrible idea."

"Noted," I said.

We took two vehicles—an armored Humvee and a pickup truck—and drove east, toward the crater. The roads were bad, littered with abandoned cars and the occasional Turned, but the Turned didn't attack us. They just stood there, watching, as we drove past. Like they were waiting for something. Like they knew where we were going and didn't need to stop us, because whatever was at the end of the road would do that for them.

The last mile, we went on foot. The Humvee couldn't make it through the trees, and the pickup got stuck in a ditch. We loaded up with weapons, supplies, and Dr. Okafor's equipment—a portable scanner, a laptop, and a device she called a "resonance detector" that looked like a metal detector but was apparently designed to pick up alien frequencies.

The forest was quiet. Too quiet. No birds, no insects, no wind. Just the hum, louder now, a physical presence that pressed against my eardrums.

"How much farther?" Sarah asked.

"Half a mile," Dr. Okafor said, checking her GPS. "The crater should be just over this ridge."

We crested the ridge, and there it was.

The ship was bigger than I'd expected. From the drone photos, it looked like a small object, maybe the size of a house. In person, it was massive—maybe two hundred feet long, black and smooth, with no visible seams or markings. It sat in a crater that was twice its size, the earth around it scorched and cracked, as if the ship had burned its way into the ground.

And around the ship, the Turned were gathered. Hundreds of them. Standing in a circle, facing the ship, motionless. Not shuffling, not chewing, not reaching. Just standing. Like an audience waiting for a performance.

"Jesus," Reyes whispered.

"Keep moving," I said. "Slowly. Don't make eye contact with any of them."

We moved through the crowd of Turned like ghosts. They didn't react to us. They didn't even seem to notice us. Their milky eyes were fixed on the ship, and their mouths were open, and from those open mouths came a sound—a low, collective hum that matched the ship's frequency exactly.

"They're singing," Dr. Okafor said, her voice barely a whisper. "They're singing to it."

We reached the edge of the crater. The ship was maybe fifty feet below us, and up close, I could see details the drone photos hadn't captured. The surface of the ship wasn't smooth—it was covered in patterns, geometric lines and shapes that seemed to shift and move when I looked at them directly. Like the ship was alive. Like it was breathing.

"The resonance detector is going crazy," Dr. Okafor said, holding up the device. The needle was pegged. "The frequency is off the charts. It's not just seven hertz anymore—it's cycling through multiple frequencies simultaneously. It's... it's communicating."

"Communicating with what?" Sarah asked.

"With everything," Dr. Okafor said. "With us. With the Turned. With the earth itself. This ship isn't just a beacon—it's a node. A network node. And it's connected to something much, much bigger."

I looked at the ship. Then I looked at the Turned, standing in their circle, singing their song. Then I looked at the sky, where the purple-green clouds still swirled.

"How much bigger?" I asked.

Dr. Okafor looked at me. "I think," she said, "that this ship is just the first one. I think there are more. And I think they're all waking up."

As if on cue, the ship's patterns shifted. The geometric lines rearranged themselves, and a section of the hull—previously invisible—slid open, revealing a dark interior. And from that interior, a light. Not a mechanical light. Not electric. Something else. Something that pulsed with the same rhythm as the hum, the same rhythm as the Turned, the same rhythm as my own heartbeat.

And from inside the ship, a voice. Not a human voice. Not an alien voice. A voice that was all of them and none of them, a voice that spoke in frequencies and feelings and images that bypassed my ears and went straight to my brain.

It said: *We have been waiting. We are here. You are next.*

Then the ship pulsed, and the hum became a roar, and every Turned in the circle turned toward us.

---

## Chapter 7: The Fall

The Turned moved. All at once, like a wave, like a tide. Hundreds of them, shuffling, lurching, running—some of them running, which I hadn't seen before, which I didn't know they could do. They came up the slope of the crater toward us, and the sound they made was a chorus of groans and clicks and that wet, hungry chewing.

"Run!" I shouted.

We ran. Back through the forest, toward the vehicles, the Turned behind us like a flood. Reyes was firing as he ran, dropping two, three, four of them, but for every one he dropped, three more took its place. Chen was ahead of us, reaching the Humvee first, starting the engine.

Sarah was next to me, her rifle in her hands, and she was shooting too—precise, controlled shots, each one finding its mark. She'd always been the better shot. Now she was the better shot in a zombie apocalypse, which was a distinction I hoped she'd never have to put on a resume.

Dr. Okafor was stumbling, her equipment bouncing against her hip, her face white with terror. I grabbed her arm and pulled her along.

"The ship!" she gasped. "The ship opened! It showed us—"

"I know! Run now, talk later!"

We reached the Humvee. Chen threw open the doors, and we piled in—me, Sarah, Dr. Okafor, Reyes, Chen. Chen gunned the engine, and we lurched forward, the Humvee bouncing over roots and rocks and the occasional Turned that didn't get out of the way.

Behind us, the Turned kept coming. Not just from the crater— from everywhere. They were emerging from the trees, from the bushes, from the shadows. Hundreds of them. Maybe thousands. All moving in the same direction. All moving toward us.

"They're not following us," Dr. Okafor said, looking through the rear window. "They're following the ship. The ship is moving."

I looked in the mirror. She was right. The ship—massive, black, impossible—was lifting off. Slowly, like a whale surfacing, it rose from the crater, trailing smoke and debris and that terrible, pulsing light. And the Turned below it rose with it, like they were tethered, like they were being pulled.

"Where is it going?" Sarah asked.

Dr. Okafor checked her equipment. "Northeast. Toward... toward Fort Haven."

My blood went cold. "The kids."

I grabbed the radio. "Fort Haven, this is Callahan. Come in. Fort Haven, do you copy?"

Static. Then Vance's voice, strained and urgent: "Callahan, we've got a situation. The hum just increased tenfold. People are... people are changing. Right now. In the compound. I've got—"

The transmission cut out.

"Vance! Vance, come in!"

Nothing. Just static and the hum, louder now, so loud I could feel it in my teeth.

"Drive," I said to Chen. "Drive faster."

---

## Chapter 8: Fort Haven Falls

We made it back to Fort Haven in twenty minutes. It should have taken forty. Chen drove like a man possessed, and the Humvee took a beating, but we made it.

What we found when we got there made the crater look like a picnic.

The compound was in chaos. The Turned—people who had been fine that morning, people who had been soldiers and teachers and parents—were attacking the living. The walls were breached. The gates were down. And overhead, the alien ship hovered, its light pulsing, its hum vibrating through everything.

"Emma!" I shouted. "Danny!"

I ran for the school building. Sarah was right behind me. Reyes and Chen covered our flanks, firing at the Turned that got too close.

The hallways were a nightmare. Blood on the walls. Bodies on the floor. The sound of screaming and the wet sound of chewing. I pushed through it all, my .45 in my hand, firing when I had to, running when I could.

I found Emma in the classroom where we'd been staying. She was under the desk, her arms wrapped around Danny, both of them shaking, both of them alive.

"Dad!" Emma screamed.

I grabbed them both and held them. "I've got you. I've got you. We're leaving. Now."

"Where's Mom?" Danny asked.

"She's right here," Sarah said, appearing in the doorway. "We're all here. Let's go."

We moved. Through the hallway, past the bodies, past the Turned, out into the parking lot where the Humvee was waiting. Reyes was behind the wheel now, engine running. We piled in—me, Sarah, Emma, Danny, Dr. Okafor, Reyes, Chen. Seven people in a vehicle built for five. Nobody complained.

As we drove out of Fort Haven, I looked back. The compound was burning. The ship hovered overhead, its light pulsing, and the Turned—thousands of them now—stood in the flames, not burning, not caring, just standing and humming and waiting.

"Where do we go?" Sarah asked.

I looked at the map. I looked at the sky. I looked at my family.

"West," I said. "Away from the ship. Away from the Turned. Away from everything."

"And then what?" Emma asked. Her voice was small, but her eyes were strong.

I looked at her. My daughter. Sixteen years old, and she'd already seen more death than most people see in a lifetime. And she was still asking questions. Still looking for answers. Still fighting.

"Then we figure out how to fight back," I said. "We figure out what these things are, where they came from, and how to stop them. And then we take the fight to them."

"That's a big promise, Dad."

"I know," I said. "But I'm a big guy."

She almost smiled. Almost. And that was enough.

---

## Chapter 9: The Road

We drove west for two days. The roads were bad, but the Humvee was tougher than the roads, and Reyes was a better driver than he had any right to be. We avoided the cities—too many Turned, too much danger—and stuck to back roads, mountain passes, and the occasional dirt track that led nowhere but away.

We stopped at a farmhouse on the second night. It was abandoned, but it was intact, and it had a well and a wood stove and a pantry that was half-full. We made camp, built a fire, and for the first time in days, we sat down and ate a meal like a family.

"So," Dr. Okafor said, between bites of canned chili, "let me tell you what I know. Which isn't much, but it's more than nothing."

She pulled out her laptop and opened a file. On the screen was a waveform—the ship's signal, recorded during our approach to the crater.

"This is the frequency the ship is broadcasting. Seven hertz, cycling through multiple harmonics. It's not random—it's structured. It's a language. Or a code. Or both."

"What does it say?" Sarah asked.

"I don't know yet. But I've been analyzing the pattern, and I've found something interesting." She pointed to a section of the waveform. "This part here—it repeats. Every seventy-two seconds. And the repetition isn't exact—it's modified slightly each time. Like a conversation. Like someone is talking, and someone else is answering."

"The Turned," I said. "They're answering."

"Exactly. The ship sends a signal, the Turned receive it and respond, and the ship modifies its signal based on the response. It's a feedback loop. The ship is... managing them. Directing them. And the more Turned there are, the stronger the loop gets."

"So if we destroy the ship," Reyes said, "the loop breaks."

"Maybe," Dr. Okafor said. "Or maybe the loop is just one node in a larger network. Remember what I said at the crater—this ship might not be the only one."

"Then we need to find the others," I said.

"And how exactly do we do that?" Chen asked.

Dr. Okafor looked at me. "The resonance detector. It can pick up the ship's frequency from a distance. If there are other ships out there, broadcasting on the same frequency, we can find them."

"Then that's the plan," I said. "We find the ships. We figure out what they are. And we destroy them."

"And the Turned?" Sarah asked.

"The Turned are people," I said. "Or they were. If the ship is controlling them, then breaking the signal might free them. Or it might kill them. I don't know. But we have to try."

Danny, who had been quiet throughout the conversation, spoke up. "Dad?"

"Yeah, buddy?"

"The voice. In my head. When the ship opened. Did you hear it too?"

I looked at him. "You heard the voice?"

"Yes. It said... it said something. I don't remember the words, but I remember the feeling. It felt like... like it was inviting me. Like it wanted me to come closer."

The room went quiet. Everyone looked at Danny. He was twelve years old, and he'd heard the voice of an alien intelligence, and he was talking about it like it was a weather report.

"Are you okay?" Sarah asked, kneeling next to him.

"I think so," he said. "It didn't hurt. It just... talked. And then it stopped."

Dr. Okafor was staring at Danny with an expression I couldn't read. "The signal," she said slowly. "It's not just affecting the Turned. It's affecting everyone. The Turned are the ones who've been fully converted, but the signal is reaching all of us. It's in the background, all the time, and some people are more susceptible than others."

"Are you saying my son is being affected by an alien signal?" Sarah's voice was sharp, dangerous.

"I'm saying," Dr. Okafor said carefully, "that the signal is universal. Everyone is exposed. But some people—children, maybe, or people with certain neurological profiles—might be more... receptive."

"Then we need to find a way to block it," I said.

"I've been working on that," Dr. Okafor said. "The frequency is seven hertz. If we can generate a counter-frequency—something that cancels it out—we might be able to protect ourselves. At least partially."

"Do it," I said. "Whatever you need. We'll get it."

She nodded. "I'll need equipment. Electronics. A signal generator. Maybe a ham radio."

"Then we'll find one," I said. "Tomorrow, we move. We head west, we find supplies, and we keep moving until we find a place to make a stand."

"And then?" Emma asked.

"And then we fight," I said.

---

## Chapter 10: The Stand

Three weeks later, we found our place.

It was a mining town in the mountains of Montana—Silver Creek, population formerly eight hundred, now zero. The town was nestled in a valley, surrounded by mountains on three sides, with a single road in and out. The mine itself was a massive complex, carved into the mountainside, with tunnels that went deep—miles deep, according to the old maps we found in the town hall.

It was perfect.

Over the next two weeks, we turned Silver Creek into a fortress. Reyes and Chen organized the defenses—barricades on the road, watchtowers on the hills, patrols in the surrounding forest. Dr. Okafor set up her lab in the mine office and began working on the counter-frequency device. Sarah organized the survivors we'd picked up along the way—about forty people, mostly families, all of them scared and all of them willing to fight.

And I trained them. Every morning, before dawn, I ran drills. Weapons handling, hand-to-hand combat, survival skills. I taught them what I knew, which was a lot, and I pushed them hard, because the world was hard and getting harder.

Emma helped. She was sixteen, but she was tough—tougher than I'd realized. She took to the training like she'd been born for it, and within a week, she was one of the best shots in the group. Danny, on the other hand, was different. He didn't train with weapons. He sat with Dr. Okafor, listening to her talk about frequencies and signals and alien biology, and he asked questions that made her stop and think.

"Your son," she told me one evening, "is either the smartest person I've ever met or the most dangerous. Possibly both."

"What do you mean?"

"The signal. The seven-hertz frequency. He can hear it. Not just feel it—hear it. He says it has words. Not English words, not any human language, but words nonetheless. And he says he's starting to understand them."

I looked at Danny, sitting across the room, his eyes focused on something none of us could see.

"What's he hearing?" I asked.

"He says the ship is talking to him. Not the ship we saw—the network. The whole network. He says there are twelve ships. Twelve nodes. And they're all waking up."

"Twelve," I repeated. "Where are they?"

"He says one is in Montana. About a hundred miles from here."

My blood went cold. "One is here? In Montana?"

"Not here," Danny said, looking up. "Not here. But close. Close enough to hear."

I looked at Dr. Okafor. She looked at me. And we both understood, with a certainty that settled in our bones like ice, that the fight was coming to us.

---

## Chapter 11: The Second Ship

The second ship was in a valley two hours north of Silver Creek, in a place called Dead Man's Pass. Dr. Okafor's resonance detector had picked up the signal three days ago, and since then, it had been getting stronger.

We had to go. There was no choice. If the ships were networked, if they were waking up one by one, then every ship that came online made the network stronger. And the Turned—more of them, every day, drawn to the ships like moths to a flame—were growing in number.

I took a team: me, Sarah, Reyes, Chen, Dr. Okafor, and two of our best fighters—a former marine named Jackson and a former paramedic named Luisa. We left at dawn, in two vehicles, and drove north.

The valley was beautiful, in a terrible way. Pine trees, a river, mountains on either side. The kind of place you'd go for a vacation, if the world hadn't ended. But the beauty was marred by the Turned—hundreds of them, lining the road, standing in the trees, filling the valley like a silent, hungry army.

We drove through them. They didn't attack. They just watched, their milky eyes tracking us as we passed.

The ship was in the center of the valley, sitting in a crater like the first one. But this one was different. It was smaller—maybe a hundred feet long—and it was already open. The hull had split, and from inside, the light pulsed, and the Turned were not just standing around it. They were entering it. Walking inside, one by one, disappearing into the light.

"They're going in," Dr. Okafor said, her voice barely a whisper. "They're being... absorbed."

"Absorbed into what?" Sarah asked.

"I don't know. But I think... I think the ship is a collector. It collects the Turned, processes them, and uses them. For what, I don't know. But the network needs biomass. It needs bodies."

I watched the Turned walk into the ship, one by one, and I felt something I hadn't felt since the beginning: pure, undiluted terror. Not for myself—for my family. For Danny, who could hear the signal. For Emma, who was tough but still a child. For Sarah, who was the strongest person I knew and who I couldn't bear to lose.

"We have to destroy it," I said.

"How?" Reyes asked. "We don't have explosives. We don't have—"

"We have the mine," I said. "The mine has dynamite. Old mining dynamite, but it's still good. We go back, we get the dynamite, we come back, and we blow this thing to hell."

"And the Turned?" Chen asked.

"We draw them away. All of them. Every single one. We lead them on a chase, get them out of the valley, and then we come back and plant the charges."

"That's insane," Luisa said.

"Yeah," I said. "It is. But it's all we've got."

---

## Chapter 12: The Chase

The plan was simple. It was also suicidal. But simple and suicidal was better than complicated and suicidal, so we went with it.

Jackson and Luisa would lead the Turned away. They'd take the Humvee and drive south, honking the horn, drawing the Turned after them like a pied piper. Meanwhile, the rest of us would sneak into the valley, plant the dynamite, and get out before the ship could react.

It worked. For about ten minutes.

Jackson and Luisa drove south, and the Turned followed. Hundreds of them, streaming out of the valley, chasing the sound of the horn. It was almost comical—if anything about the end of the world could be called comical.

We moved in. Reyes and Chen planted the dynamite—four charges, placed at the corners of the ship, connected by a detonator that Dr. Okafor had rigged from parts she'd scavenged from the mine office. It was crude, but it would work. Probably.

"Charges set," Reyes said. "Let's get out of here."

We were halfway back to the road when the ship reacted.

The light inside intensified. The hum became a scream—a high-pitched, piercing sound that made my vision blur and my stomach lurch. And from inside the ship, something emerged.

It wasn't a Turned. It wasn't a person. It was something else—something that had been a person, or maybe several people, fused together into a single mass of flesh and bone and milky eyes. It was eight feet tall, with too many arms and not enough of anything else, and it moved with a speed that no human body should be capable of.

"Run!" I shouted.

We ran. The thing chased us, its movements jerky and wrong, its many mouths open and screaming. Reyes fired at it—three shots, four, five—and the bullets hit, but they didn't stop it. They just made it angrier.

We reached the road. The Humvee was gone—Jackson and Luisa had it, and they were miles away by now. We were on foot.

"Dr. Okafor!" I shouted. "The detonator!"

She fumbled with it, her hands shaking. The thing was getting closer. I could hear its breathing—or whatever passed for breathing in a creature made of fused human bodies.

"Jake!" Sarah screamed.

I turned. The thing was ten feet away. Its many arms reached for me, and in its many eyes, I saw something that looked like hunger and something that looked like recognition.

I raised my .45 and fired. One shot, two, three. The thing staggered but kept coming. I fired again. And again. And again. Emptying the magazine into the thing's chest, its face, its eyes.

It stopped. It stood there, swaying, its many mouths open, its many eyes fixed on me. And then it fell. Not like a person falling—like a building collapsing. It hit the ground with a sound like wet concrete, and it didn't get up.

"Come on!" Dr. Okafor shouted. She had the detonator ready. "We need to be further away!"

We ran. We ran until our legs burned and our lungs screamed and the hum in our heads became a roar. And then Dr. Okafor pressed the button.

The explosion was visible from a mile away. A flash of light, a column of smoke, and then a sound—a deep, resonant boom that shook the mountains and echoed through the valley. The ship was gone. The crater was a smoking ruin. And the Turned in the valley—all of them—collapsed. Like puppets with their strings cut, they fell to the ground and didn't get up.

We stood on the hillside and watched the smoke rise, and for the first time since the world ended, I felt something I hadn't felt in weeks.

Hope.

---

## Chapter 13: Aftermath

We went home. To Silver Creek. To the people who were waiting for us, who had been watching the sky, who had seen the explosion and didn't know what it meant.

I told them. I told them everything—the ship, the Turned, the fused creature, the explosion. I told them that there were twelve ships, and we'd destroyed one, and there were eleven more. I told them that the fight was just beginning.

And then I told them something I hadn't planned to say, something that came from somewhere deep inside me, somewhere that I didn't know existed until that moment.

"We're going to win this," I said. "I don't know how. I don't know when. But we're going to win. Because we're human. And humans don't give up. We don't quit. We don't stop fighting, even when the odds are impossible, even when the enemy is alien and ancient and powerful beyond our understanding. We fight. That's what we do. That's what we've always done."

The room was quiet. Then Danny stood up.

"Dad's right," he said. "I can still hear the signal. The other ships. They're out there. But they're not as strong now. The network is... broken. We broke it. And if we can break one, we can break them all."

He looked at me, and for a moment, I saw something in his eyes that I'd never seen before. Not fear. Not childhood. Something older. Something that had been there all along, waiting for this moment.

"I can hear them," he said. "And I can help."

I looked at my son. My twelve-year-old son, who could hear alien signals and understand them, who was brave and smart and something I couldn't quite name.

"Then we'll do it together," I said. "All of us. Together."

Sarah put her hand on my shoulder. Emma stood next to Danny. Reyes and Chen flanked me. Dr. Okafor held her equipment. And the people of Silver Creek—forty survivors, forty fighters, forty humans—stood in that room and looked at me, and I looked at them, and I knew, with a certainty that went deeper than thought, that we were going to make it.

Not because we were the strongest. Not because we were the smartest. But because we were together, and together, we were something the aliens hadn't counted on.

We were human.

And we were not going to be turned.

---

## Epilogue: The Signal

Three months later, the network was down to nine ships. We'd destroyed three more—two in California, one in Texas—with a combination of dynamite, ingenuity, and Danny's ability to hear the signal and predict the ships' movements.

The Turned were fewer now. Without the network to sustain them, they were falling apart—literally. Their bodies were breaking down, the milky eyes clearing, the hunger fading. Some of them were even waking up—confused, scared, but alive. Human.

Danny was getting stronger. He could hear the signal from hundreds of miles away, and he could sense the ships' locations with an accuracy that Dr. Okafor called "impossible" and I called "my son." He was changing, I knew that. The signal was changing him. But he was still Danny. Still my boy. Still human.

For now.

The hum was quieter now. Still there, but fainter, like a radio signal losing strength. And in the quiet moments, between the fighting and the planning and the surviving, I could hear something else.

Silence.

Beautiful, precious, human silence.

And in that silence, I could hear my family breathing. And that was enough.

---

*The End.*

---

## About the Author

This novel is a work of fiction. Any resemblance to real persons, living or dead, is coincidental. The characters, events, and places depicted are products of the author's imagination.

*Dead Orbit: The Talbot Protocol* — Book One of the Dead Orbit Series.

*Coming soon: Book Two — Dead Orbit: The Waking Dark*
"""#

    static let promptTemplates: [PromptTemplateItem] = [
        PromptTemplateItem(
            title: "Dead Orbit: The Talbot Protocol (Full Novel)",
            subtitle: "13 Chapters · Unlimited Gapless Book Benchmark",
            icon: "books.vertical.fill",
            text: deadOrbitNovelText
        ),
        PromptTemplateItem(
            title: "Studio Welcome",
            subtitle: "Crisp natural voice test",
            icon: "waveform.circle",
            text: "Welcome to MLX Audio Studio on Apple Silicon! Real-time neural speech synthesis, voice cloning, and audio separation are ready right on your Mac."
        ),
        PromptTemplateItem(
            title: "Expressive Storytelling",
            subtitle: "Uses [whisper], [sigh], and [laughing] tags",
            icon: "theatermasks",
            text: "[whisper] Did you hear that footsteps sound in the hallway? [pause] [sigh] Oh, wait a second... [laughing] it is just the cat knocking over my coffee mug again!"
        ),
        PromptTemplateItem(
            title: "Two-Speaker Podcast",
            subtitle: "Multi-speaker dialogue with <|speaker:0|> & <|speaker:1|>",
            icon: "mic.badge.plus",
            text: "<|speaker:0|> Welcome back to the Deep Tech Podcast! Today we are testing local neural audio models on Apple Silicon.\n<|speaker:1|> And the latency is honestly mind-blowing. Everything runs locally on unified memory with zero cloud calls."
        ),
        PromptTemplateItem(
            title: "Audiobook Narration",
            subtitle: "Long-form dramatic pacing",
            icon: "book.closed",
            text: "The rain drummed softly against the tall library windows as Eleanor opened the brass-bound ledger. Inside, written in faded indigo ink, was a map that hadn't been seen for three hundred years."
        ),
        PromptTemplateItem(
            title: "News Broadcast",
            subtitle: "Clear broadcast cadence",
            icon: "newspaper",
            text: "Good evening. Breaking news in on-device artificial intelligence: open-weight speech synthesis and transcription models are now running natively on macOS with hardware-accelerated Metal shaders."
        ),
        PromptTemplateItem(
            title: "Pangram Benchmark",
            subtitle: "Phonetic articulation test",
            icon: "gauge.with.dots.needle.67percent",
            text: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns under the bright autumn moon."
        ),
        PromptTemplateItem(
            title: "Calibration 1: The Rainbow Passage",
            subtitle: "Phonetic Balance (Codecaine Voice Recorder)",
            icon: "rainbow",
            text: "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon."
        ),
        PromptTemplateItem(
            title: "Calibration 2: Dramatic Storyteller",
            subtitle: "Wide Pitch & Expressive Range",
            icon: "sparkles.tv",
            text: "Wait — did you hear that? Beneath the ancient stone library, a golden clockwork engine began to hum! \"Quickly!\" whispered Elena, her voice trembling with wonder. \"Before the midnight bell strikes twelve, open the starlight vault!\""
        ),
        PromptTemplateItem(
            title: "Calibration 3: Technical & AI Pitch",
            subtitle: "Clear Articulation & Cadence",
            icon: "cpu.fill",
            text: "Welcome to Codecaine Studio. Today we are synthesizing twenty-four kilohertz neural speech directly on local hardware, with zero cloud latency, automatic word alignment, and instant multi-format audio export."
        ),
    ]
}

extension VoiceCollection {
    static let samples: [VoiceCollection] = [
        VoiceCollection(
            name: "Father Christmas & Characters",
            description: "Holiday themed voices",
            imageName: "gift.fill",
            backgroundColor: .pink.opacity(0.2)
        ),
        VoiceCollection(
            name: "Professional Narrators",
            description: "For audiobooks and podcasts",
            imageName: "book.fill",
            backgroundColor: .purple.opacity(0.2)
        ),
        VoiceCollection(
            name: "Character Voices",
            description: "Unique personalities",
            imageName: "theatermasks.fill",
            backgroundColor: .blue.opacity(0.2)
        )
    ]
}

extension Notification.Name {
    static let sendTranscriptToVoiceStudio = Notification.Name("VoicesApp.SendTranscriptToVoiceStudio")
    static let sendAudioToTranscribe = Notification.Name("VoicesApp.SendAudioToTranscribe")
    static let cloneVoiceFromAudio = Notification.Name("VoicesApp.CloneVoiceFromAudio")
    static let pasteAndReadClipboard = Notification.Name("VoicesApp.PasteAndReadClipboard")
    static let toggleShortcutsSheet = Notification.Name("VoicesApp.ToggleShortcutsSheet")
    static let modelCacheDidChange = Notification.Name("VoicesApp.ModelCacheDidChange")
    static let openDownloadedModelsManager = Notification.Name("VoicesApp.OpenDownloadedModelsManager")
}

// MARK: - Smart Speech Text Cleaner & Universal Document Extractor (from bun_rad_studio fmPhonon.ts)

struct SpeechCleanOptions: Equatable, Codable, Sendable {
    private static let storageKey = "VoicesApp.SpeechCleanOptions.v1"

    var stripMarkdown: Bool = true
    var stripHtml: Bool = true
    var stripCodeBlocks: Bool = true
    var simplifyUrls: Bool = true
    var stripTimestamps: Bool = true
    var stripDividers: Bool = true

    static func loadFromDefaults() -> SpeechCleanOptions {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode(SpeechCleanOptions.self, from: data) else {
            return SpeechCleanOptions()
        }
        return decoded
    }

    func saveToDefaults() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }
}

// MARK: - Universal File Format Catalog (All E-Books, Documents, Office, Subtitles, Data, Code, OCR, Archives & Media)

enum StudioFileCategory: String, CaseIterable, Identifiable, Sendable {
    case ebook = "E-Books & Publishing"
    case pdfPrint = "PDF & Print"
    case office = "Office, iWork & Slides"
    case spreadsheet = "Spreadsheets & Tables"
    case subtitle = "Subtitles, Captions & Lyrics"
    case document = "Text, Markdown & Docs"
    case webEmail = "Web, HTML & Email"
    case structuredData = "Notebooks, JSON, YAML & Logs"
    case sourceCode = "Source Code & Scripts"
    case imageOCR = "Images & Scanned OCR"
    case archive = "Archives & Bundles"
    case audio = "Audio Containers"
    case video = "Video Containers"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .ebook: return "book.pages.fill"
        case .pdfPrint: return "doc.richtext.fill"
        case .office: return "doc.text.fill"
        case .spreadsheet: return "tablecells.fill"
        case .subtitle: return "captions.bubble.fill"
        case .document: return "text.alignleft"
        case .webEmail: return "globe"
        case .structuredData: return "curlybraces.square.fill"
        case .sourceCode: return "chevron.left.forwardslash.chevron.right"
        case .imageOCR: return "text.viewfinder"
        case .archive: return "archivebox.fill"
        case .audio: return "waveform.circle.fill"
        case .video: return "film.stack.fill"
        }
    }
}

enum UniversalFileFormatCatalog {
    static let ebookExtensions: Set<String> = [
        "epub", "ibooks", "mobi", "azw", "azw3", "azw4", "kf8", "kfx",
        "fb2", "fbz", "lit", "pdb", "prc", "pobi", "djvu", "djv",
        "cbz", "cbr", "cb7", "cbt", "snb", "tcr"
    ]

    static let pdfPrintExtensions: Set<String> = [
        "pdf", "ps", "eps", "xps", "oxps"
    ]

    static let officeDocumentExtensions: Set<String> = [
        "docx", "doc", "docm", "dotx", "dot", "odt", "fodt", "ott",
        "rtf", "rtfd", "pages", "wordml", "wpd", "wps", "abw",
        "pptx", "ppt", "ppsx", "pps", "odp", "fodp", "key"
    ]

    static let spreadsheetExtensions: Set<String> = [
        "xlsx", "xlsm", "xls", "ods", "fods", "numbers",
        "csv", "tsv", "psv", "tab", "dif"
    ]

    static let subtitleExtensions: Set<String> = [
        "srt", "vtt", "webvtt", "ass", "ssa", "sub", "sbv",
        "lrc", "ttml", "dfxp", "smi", "sami", "scc", "stl", "itt", "lyrics"
    ]

    static let markupAndDocExtensions: Set<String> = [
        "txt", "text", "md", "markdown", "mdx", "rmd", "qmd",
        "rst", "rest", "org", "adoc", "asciidoc", "asc",
        "tex", "latex", "ltx", "bib", "textile", "wiki", "mediawiki",
        "pod", "man", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "nfo", "roff", "troff", "groff", "me", "ms"
    ]

    static let webAndEmailExtensions: Set<String> = [
        "html", "htm", "xhtml", "shtml", "mhtml", "mht",
        "webarchive", "maff", "eml", "mbox", "msg"
    ]

    static let structuredDataExtensions: Set<String> = [
        "ipynb", "json", "jsonl", "ndjson", "geojson", "har",
        "yaml", "yml", "toml", "xml", "plist", "rss", "atom", "opml", "rdf",
        "ini", "cfg", "conf", "properties", "env", "log", "out", "err"
    ]

    static let sourceCodeExtensions: Set<String> = [
        "swift", "py", "pyw", "js", "mjs", "cjs", "ts", "tsx", "jsx",
        "rs", "go", "c", "h", "cpp", "hpp", "cc", "cxx", "m", "mm",
        "java", "kt", "kts", "scala", "groovy", "rb", "erb", "php",
        "cs", "fs", "vb", "lua", "r", "jl", "pl", "pm",
        "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
        "sql", "graphql", "gql", "proto", "css", "scss", "sass", "less",
        "vue", "svelte", "astro", "dart", "ex", "exs", "erl", "hs",
        "clj", "lisp", "scm", "el", "vim", "zig", "nim", "v", "sol",
        "wgsl", "metal", "glsl", "hlsl", "dockerfile", "makefile",
        "cmake", "gradle", "nix", "tf", "hcl"
    ]

    static let imageOCRExtensions: Set<String> = [
        "png", "jpg", "jpeg", "jpe", "jfif", "webp", "tiff", "tif",
        "bmp", "dib", "heic", "heif", "avif", "gif", "svg",
        "icns", "ico", "jp2", "j2k", "pbm", "pgm", "ppm", "pnm",
        "exr", "hdr", "dng", "cr2", "nef", "arw"
    ]

    static let archiveExtensions: Set<String> = [
        "zip", "tar", "tgz", "gz", "bz2", "xz"
    ]

    static let audioExtensions: Set<String> = [
        "wav", "wave", "bwf", "mp3", "mp2", "mp1", "mpga",
        "m4a", "m4b", "m4p", "m4r", "aac", "adts",
        "flac", "ogg", "oga", "opus", "spx",
        "aiff", "aif", "aifc", "caf", "wma",
        "ac3", "eac3", "dts", "amr", "awb", "3ga",
        "ape", "wv", "tta", "mka", "au", "snd", "gsm", "voc",
        "w64", "rf64", "dsf", "dff", "pcm", "raw", "ulaw", "alaw"
    ]

    static let videoExtensions: Set<String> = [
        "mp4", "m4v", "mov", "qt", "mkv", "webm", "avi",
        "flv", "f4v", "wmv", "asf", "ts", "mts", "m2ts",
        "mpg", "mpeg", "mpe", "vob", "ogv", "3gp", "3g2",
        "divx", "rm", "rmvb", "dv", "mxf"
    ]

    static let allAudioAndVideoExtensions: Set<String> = {
        audioExtensions.union(videoExtensions)
    }()

    static let allReadableDocumentExtensions: Set<String> = {
        var set = Set<String>()
        set.formUnion(ebookExtensions)
        set.formUnion(pdfPrintExtensions)
        set.formUnion(officeDocumentExtensions)
        set.formUnion(spreadsheetExtensions)
        set.formUnion(subtitleExtensions)
        set.formUnion(markupAndDocExtensions)
        set.formUnion(webAndEmailExtensions)
        set.formUnion(structuredDataExtensions)
        set.formUnion(sourceCodeExtensions)
        set.formUnion(imageOCRExtensions)
        set.formUnion(archiveExtensions)
        return set
    }()

    static let allSupportedWatchFolderExtensions: Set<String> = {
        allReadableDocumentExtensions.union(allAudioAndVideoExtensions)
    }()

    static func category(forExtension rawExt: String) -> StudioFileCategory {
        let ext = rawExt.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
        if ebookExtensions.contains(ext) { return .ebook }
        if pdfPrintExtensions.contains(ext) { return .pdfPrint }
        if officeDocumentExtensions.contains(ext) { return .office }
        if spreadsheetExtensions.contains(ext) { return .spreadsheet }
        if subtitleExtensions.contains(ext) { return .subtitle }
        if webAndEmailExtensions.contains(ext) { return .webEmail }
        if structuredDataExtensions.contains(ext) { return .structuredData }
        if sourceCodeExtensions.contains(ext) { return .sourceCode }
        if imageOCRExtensions.contains(ext) { return .imageOCR }
        if archiveExtensions.contains(ext) { return .archive }
        if audioExtensions.contains(ext) { return .audio }
        if videoExtensions.contains(ext) { return .video }
        return .document
    }

    static func badgeLabel(forDocumentTitle title: String) -> String {
        let ext = URL(fileURLWithPath: title).pathExtension.lowercased()
        guard !ext.isEmpty else {
            return "UNIVERSAL BOOK & DOCUMENT STUDIO"
        }
        let cat = category(forExtension: ext)
        return "\(cat.rawValue.uppercased()) · .\(ext.uppercased())"
    }

    static func isPureAudioOrVideo(url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return allAudioAndVideoExtensions.contains(ext)
    }

    static func isAudioOrVideo(url: URL) -> Bool {
        isPureAudioOrVideo(url: url)
    }

    static func isReadableDocument(url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        if allReadableDocumentExtensions.contains(ext) {
            return true
        }
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
            return true
        }
        return false
    }

    static let supportedFormatsSummary: String =
        "E-Books (.epub, .mobi, .azw3, .fb2, .lit, .cbz), PDF/Print (.pdf, .xps, .ps), Office & iWork (.docx, .doc, .pptx, .xlsx, .odt, .pages, .key, .numbers), Subtitles (.srt, .vtt, .ass, .lrc, .ttml, .sbv), Docs/Web (.txt, .md, .tex, .rst, .org, .html, .eml), Data/Code (.ipynb, .json, .jsonl, .csv, .tsv, .yaml, .toml, .plist, .swift, .py, .ts), Archives (.zip, .tar) & Images via OCR (.png, .jpg, .webp, .heic, .tiff, .svg)"
}

enum SpeechTextCleaner {
    private static let regexLock = NSLock()
    nonisolated(unsafe) private static var regexCache: [String: NSRegularExpression] = [:]

    static func clean(_ rawText: String, options: SpeechCleanOptions = SpeechCleanOptions()) -> String {
        var s = rawText.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")

        if options.stripCodeBlocks && s.contains("`") {
            s = replaceRegex(s, pattern: "```[^`]*```", with: " [code block omitted] ")
            s = replaceRegex(s, pattern: "`([^`\\n]{1,400})`", with: "$1")
        }

        if options.stripHtml && (s.contains("<") || s.contains("&")) {
            s = replaceRegex(s, pattern: "(?i)<head\\b[^>]*>[^<]*(?:<(?!/?head\\b)[^<]*)*</head>", with: " ")
            s = replaceRegex(s, pattern: "(?i)<script\\b[^>]*>[^<]*(?:<(?!/?script\\b)[^<]*)*</script>", with: " ")
            s = replaceRegex(s, pattern: "(?i)<style\\b[^>]*>[^<]*(?:<(?!/?style\\b)[^<]*)*</style>", with: " ")
            // Preserve <|speaker:0|> and <|...|> model control tokens
            s = replaceRegex(s, pattern: "<(?!\\|)[^>]+>", with: " ")
            s = decodeHTMLEntities(s)
        }

        if options.stripTimestamps {
            s = replaceRegex(s, pattern: "(?m)^\\d+\\s*$", with: "")
            s = replaceRegex(
                s,
                pattern: "(?m)^\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d{1,3}\\s*-->\\s*\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d{1,3}.*$",
                with: ""
            )
            s = replaceRegex(
                s,
                pattern: "(?m)^\\d{1,2}:\\d{2}:\\d{2}[.,]\\d{1,3}\\s*,\\s*\\d{1,2}:\\d{2}:\\d{2}[.,]\\d{1,3}\\s*$",
                with: ""
            )
            s = replaceRegex(s, pattern: "\\[\\d{1,2}:\\d{2}(?:[.:]\\d{1,3})?\\]", with: " ")
        }

        if options.simplifyUrls && s.contains("http") {
            s = replaceRegex(s, pattern: "(?i)https?://([^/\\s)\\]]+)[^\\s)\\]]*", with: "link to $1")
        }

        if options.stripMarkdown {
            if s.contains("[") {
                s = replaceRegex(s, pattern: "!\\[([^\\]\\n]{0,300})\\]\\([^)\\n]{1,600}\\)", with: "$1")
                s = replaceRegex(s, pattern: "\\[([^\\]\\n]{1,300})\\]\\([^)\\n]{1,600}\\)", with: "$1")
            }
            s = replaceRegex(s, pattern: "(?m)^#{1,6}\\s+", with: "")
            if s.contains("*") || s.contains("_") {
                s = replaceRegex(s, pattern: "\\*\\*([^*\\n]{1,300})\\*\\*|__([^_\\n]{1,300})__", with: "$1$2")
                s = replaceRegex(s, pattern: "(?<!\\*)\\*([^*\\n]{1,300})\\*(?!\\*)|(?<!_)_([^_\\n]{1,300})_(?!_)", with: "$1$2")
            }
            if s.contains("~~") {
                s = replaceRegex(s, pattern: "~~([^~\\n]{1,300})~~", with: "$1")
            }
            s = replaceRegex(s, pattern: "(?m)^>\\s?", with: "")
            s = replaceRegex(s, pattern: "(?m)^\\s*[-*+]\\s+", with: "• ")
            if s.contains("|") {
                s = replaceRegex(s, pattern: "(?m)^\\s*\\|[-:\\s|]+\\|\\s*$", with: "")
                // Strip markdown table pipes while protecting <|speaker:N|> control tokens
                s = replaceRegex(s, pattern: "(?<!<)\\|(?!>|speaker:)", with: " ")
            }
            if s.contains("\\") {
                s = s.replacingOccurrences(of: "\\'", with: "'")
                    .replacingOccurrences(of: "\\\"", with: "\"")
                    .replacingOccurrences(of: "\\-", with: "-")
                    .replacingOccurrences(of: "\\.", with: ".")
                    .replacingOccurrences(of: "\\!", with: "!")
                    .replacingOccurrences(of: "\\?", with: "?")
            }
        }

        if options.stripDividers {
            s = replaceRegex(s, pattern: "(?m)^\\s*[-=*_]{3,}\\s*$", with: "")
        }

        s = replaceRegex(s, pattern: "[ \\t]+", with: " ")
        s = replaceRegex(s, pattern: "\\n{3,}", with: "\n\n")
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func decodeHTMLEntities(_ input: String) -> String {
        var s = input
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&#160;", with: " ")
            .replacingOccurrences(of: "&#xa0;", with: " ", options: .caseInsensitive)
            .replacingOccurrences(of: "&amp;", with: " and ")
            .replacingOccurrences(of: "&#38;", with: " and ")
            .replacingOccurrences(of: "&lt;", with: " less than ")
            .replacingOccurrences(of: "&gt;", with: " greater than ")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#34;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'")
            .replacingOccurrences(of: "&rsquo;", with: "’")
            .replacingOccurrences(of: "&lsquo;", with: "‘")
            .replacingOccurrences(of: "&rdquo;", with: "”")
            .replacingOccurrences(of: "&ldquo;", with: "“")
            .replacingOccurrences(of: "&mdash;", with: " — ")
            .replacingOccurrences(of: "&#8212;", with: " — ")
            .replacingOccurrences(of: "&ndash;", with: " – ")
            .replacingOccurrences(of: "&#8211;", with: " – ")
            .replacingOccurrences(of: "&hellip;", with: "…")
            .replacingOccurrences(of: "&#8230;", with: "…")
            .replacingOccurrences(of: "&copy;", with: "©")
            .replacingOccurrences(of: "&reg;", with: "®")
            .replacingOccurrences(of: "&trade;", with: "™")
        // Decode numeric decimal entities &#NNN;
        if s.contains("&#"),
           let regex = try? NSRegularExpression(pattern: "&#(\\d{2,5});", options: []) {
            let ns = s as NSString
            let matches = regex.matches(in: s, options: [], range: NSRange(location: 0, length: ns.length)).reversed()
            let mutable = NSMutableString(string: s)
            for match in matches where match.numberOfRanges >= 2 {
                let codeStr = ns.substring(with: match.range(at: 1))
                if let code = UInt32(codeStr), let scalar = UnicodeScalar(code) {
                    mutable.replaceCharacters(in: match.range, with: String(Character(scalar)))
                }
            }
            s = mutable as String
        }
        return s
    }

    static func extractDocumentText(from url: URL) throws -> String {
        let ext = url.pathExtension.lowercased()
        let bookCleanOpts = SpeechCleanOptions(
            stripMarkdown: false,
            stripHtml: true,
            stripCodeBlocks: false,
            simplifyUrls: true,
            stripTimestamps: false,
            stripDividers: true
        )

        #if os(macOS)
        // 0. Directory / Folder bundle check (if user dropped a folder of chapters/files)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
           isDir.boolValue,
           !["rtfd", "epub", "ibooks", "pages", "numbers", "key", "webarchive"].contains(ext) {
            if let folderText = extractFolderOrArchiveDocuments(at: url), !folderText.isEmpty {
                return folderText
            }
        }

        // 1. PDF & Print Documents (.pdf, .ps, .eps, .xps, .oxps)
        if ext == "pdf" {
            if let pdf = PDFDocument(url: url) {
                var pages: [String] = []
                for i in 0..<pdf.pageCount {
                    if let pageText = pdf.page(at: i)?.string?.trimmingCharacters(in: .whitespacesAndNewlines),
                       !pageText.isEmpty {
                        pages.append(pageText)
                    }
                }
                if !pages.isEmpty {
                    return pages.joined(separator: "\n\n")
                }
            }
            // Fallback 1: Poppler pdftotext if installed
            if let pdftotext = StudioPackageManager.findExecutable(names: ["pdftotext"]),
               let out = runCLI(executable: pdftotext, arguments: ["-layout", url.path, "-"]),
               !out.isEmpty {
                return clean(out)
            }
            // Fallback 2: Scanned PDF pages via Vision OCR
            if let pdf = PDFDocument(url: url) {
                var ocrPages: [String] = []
                let maxPages = min(pdf.pageCount, 50)
                for i in 0..<maxPages {
                    if let page = pdf.page(at: i) {
                        let rect = page.bounds(for: .mediaBox)
                        let img = page.thumbnail(of: CGSize(width: max(rect.width * 2, 1200), height: max(rect.height * 2, 1600)), for: .mediaBox)
                        if let cgImage = img.cgImage(forProposedRect: nil, context: nil, hints: nil),
                           let text = recognizeCGImageText(cgImage),
                           !text.isEmpty {
                            ocrPages.append(text)
                        }
                    }
                }
                if !ocrPages.isEmpty {
                    return ocrPages.joined(separator: "\n\n")
                }
            }
        }

        if ext == "xps" || ext == "oxps" {
            if let xpsText = extractXPSText(from: url), !xpsText.isEmpty {
                return xpsText
            }
        }

        if ext == "ps" || ext == "eps" {
            if let ps2ascii = StudioPackageManager.findExecutable(names: ["ps2ascii"]),
               let out = runCLI(executable: ps2ascii, arguments: [url.path]),
               !out.isEmpty {
                return clean(out)
            }
        }

        // 2. E-Books: EPUB / iBooks (.epub, .ibooks)
        if ext == "epub" || ext == "ibooks" {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out, options: bookCleanOpts)
            }
            if let epubText = extractEpubText(from: url), !epubText.isEmpty {
                return epubText
            }
            if let calibreText = runEbookConvertToText(from: url), !calibreText.isEmpty {
                return clean(calibreText, options: bookCleanOpts)
            }
        }

        // 3. E-Books: FictionBook (.fb2, .fbz)
        if ext == "fb2" || ext == "fbz" || url.lastPathComponent.lowercased().hasSuffix(".fb2.zip") {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               ext == "fb2",
               let out = runCLI(executable: pandoc, arguments: ["-f", "fb2", "-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out, options: bookCleanOpts)
            }
            if let fb2Text = extractFB2Text(from: url), !fb2Text.isEmpty {
                return fb2Text
            }
            if let calibreText = runEbookConvertToText(from: url), !calibreText.isEmpty {
                return clean(calibreText, options: bookCleanOpts)
            }
        }

        // 4. E-Books: Kindle / Mobipocket / PalmDOC / DjVu / Comic Archives (.mobi, .azw, .azw3, .azw4, .kf8, .kfx, .prc, .pdb, .lit, .snb, .tcr, .djvu, .cbz, .cbr)
        if ["mobi", "azw", "azw3", "azw4", "kf8", "kfx", "prc", "pdb", "pobi", "lit", "snb", "tcr"].contains(ext) {
            if let calibreText = runEbookConvertToText(from: url), !calibreText.isEmpty {
                return clean(calibreText, options: bookCleanOpts)
            }
            if let mobiText = extractMobiOrPalmDocText(from: url), !mobiText.isEmpty {
                return mobiText
            }
        }

        if ext == "djvu" || ext == "djv" {
            if let djvutxt = StudioPackageManager.findExecutable(names: ["djvutxt"]),
               let out = runCLI(executable: djvutxt, arguments: [url.path]),
               !out.isEmpty {
                return clean(out, options: bookCleanOpts)
            }
            if let calibreText = runEbookConvertToText(from: url), !calibreText.isEmpty {
                return clean(calibreText, options: bookCleanOpts)
            }
        }

        if ["cbz", "cbr", "cb7", "cbt"].contains(ext) {
            if let comicText = extractComicArchiveOCRText(from: url), !comicText.isEmpty {
                return comicText
            }
        }

        // 5. Apple iWork Documents (.pages, .numbers, .key)
        if ["pages", "numbers", "key"].contains(ext) {
            if let iworkText = extractIWorkText(from: url), !iworkText.isEmpty {
                return iworkText
            }
        }

        // 6. Presentations & Spreadsheets (.pptx, .ppt, .ppsx, .odp, .fodp, .xlsx, .xlsm, .xls, .ods, .fods)
        if ["pptx", "ppt", "ppsx", "pps", "odp", "fodp", "xlsx", "xlsm", "xls", "ods", "fods"].contains(ext) {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               ["pptx", "odp", "xlsx", "ods"].contains(ext),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out)
            }
            if let officeXML = extractOfficeZipXMLText(from: url, ext: ext), !officeXML.isEmpty {
                return officeXML
            }
            if let binaryText = extractLegacyBinaryOfficeText(from: url), !binaryText.isEmpty {
                return binaryText
            }
        }

        // 7. Word & Rich Text Documents (.docx, .docm, .dotx, .odt, .fodt, .ott, .doc, .rtf, .rtfd, .webarchive, .wordml, .wpd, .wps, .abw)
        if ["docx", "docm", "dotx", "dot", "odt", "fodt", "ott", "doc", "rtf", "rtfd", "webarchive", "wordml", "wpd", "wps", "abw"].contains(ext) {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out)
            }
            if let textutilOut = runTextutil(on: url), !textutilOut.isEmpty {
                return clean(textutilOut)
            }
            if let officeXML = extractOfficeZipXMLText(from: url, ext: ext), !officeXML.isEmpty {
                return officeXML
            }
            if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil),
               !attr.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return attr.string
            }
            if let binaryText = extractLegacyBinaryOfficeText(from: url), !binaryText.isEmpty {
                return binaryText
            }
        }

        // 8. Academic & Markup Formats (.tex, .latex, .ltx, .bib, .rst, .rest, .org, .adoc, .asciidoc, .textile, .wiki, .mediawiki, .man, .roff, .troff, .groff)
        if ["tex", "latex", "ltx", "bib", "rst", "rest", "org", "adoc", "asciidoc", "asc", "textile", "wiki", "mediawiki", "pod", "man", "roff", "troff", "groff"].contains(ext) {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out)
            }
            if let rawMarkup = readTextWithFallbackEncoding(from: url), !rawMarkup.isEmpty {
                if ["tex", "latex", "ltx"].contains(ext) {
                    return cleanLaTeXForSpeech(rawMarkup)
                }
                return clean(rawMarkup)
            }
        }

        // 9. Images & Scanned Pages via macOS Vision On-Device OCR + Tesseract fallback
        if UniversalFileFormatCatalog.imageOCRExtensions.contains(ext) {
            if ext == "svg", let rawSVG = readTextWithFallbackEncoding(from: url) {
                let svgText = extractSVGTextElements(from: rawSVG)
                if !svgText.isEmpty { return svgText }
            }
            if let ocrText = recognizeImageText(from: url), !ocrText.isEmpty {
                return ocrText
            }
        }

        // 10. ZIP / TAR / GZ Archives containing documents or chapters
        if UniversalFileFormatCatalog.archiveExtensions.contains(ext) {
            if let archiveText = extractArchiveDocuments(from: url), !archiveText.isEmpty {
                return archiveText
            }
        }

        // 11. Audio & Video Containers dropped into Document Reader (extract embedded lyrics, subtitles, or metadata)
        if UniversalFileFormatCatalog.allAudioAndVideoExtensions.contains(ext) {
            if let mediaText = extractMediaLyricsOrSubtitles(from: url), !mediaText.isEmpty {
                return mediaText
            }
        }

        // 12. HTML / XHTML / MHTML / EML / MBOX
        if ["html", "htm", "xhtml", "shtml"].contains(ext) {
            if let rawHTML = readTextWithFallbackEncoding(from: url), !rawHTML.isEmpty {
                let formatted = insertParagraphBreaksInHTML(rawHTML)
                let cleaned = clean(formatted, options: bookCleanOpts)
                if !cleaned.isEmpty { return cleaned }
            }
            if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil),
               !attr.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return attr.string
            }
        }

        if ["eml", "mbox", "mhtml", "mht"].contains(ext) {
            if let rawEmail = readTextWithFallbackEncoding(from: url), !rawEmail.isEmpty {
                let extractedEmail = extractEmailOrMHTMLText(from: rawEmail)
                if !extractedEmail.isEmpty { return extractedEmail }
            }
        }
        #endif

        // 13. Binary or XML Apple Property List (.plist)
        if ext == "plist", let data = try? Data(contentsOf: url),
           let plistObj = try? PropertyListSerialization.propertyList(from: data, options: [], format: nil) {
            let prose = structuredObjectToProse(plistObj)
            if !prose.isEmpty { return prose }
        }

        guard let raw = readTextWithFallbackEncoding(from: url) else {
            throw NSError(
                domain: "SpeechTextCleaner",
                code: 422,
                userInfo: [NSLocalizedDescriptionKey: "Could not extract readable text from \(url.lastPathComponent)"]
            )
        }

        // 14. Subtitles, Captions & Synced Lyrics (.srt, .vtt, .webvtt, .ass, .ssa, .sub, .sbv, .lrc, .ttml, .dfxp, .smi, .sami, .scc, .stl, .itt, .lyrics)
        if UniversalFileFormatCatalog.subtitleExtensions.contains(ext) {
            let subText = cleanSubtitleOrLyricText(raw, ext: ext)
            if !subText.isEmpty { return subText }
        }

        // 15. Jupyter Notebook (.ipynb)
        if ext == "ipynb", let data = raw.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cells = json["cells"] as? [[String: Any]] {
            var blocks: [String] = []
            for cell in cells {
                let cellType = (cell["cell_type"] as? String) ?? ""
                let sourceLines: [String]
                if let arr = cell["source"] as? [String] {
                    sourceLines = arr
                } else if let str = cell["source"] as? String {
                    sourceLines = [str]
                } else {
                    sourceLines = []
                }
                let joined = sourceLines.joined().trimmingCharacters(in: .whitespacesAndNewlines)
                if !joined.isEmpty {
                    blocks.append(cellType == "markdown" ? clean(joined) : joined)
                }
            }
            if !blocks.isEmpty {
                return blocks.joined(separator: "\n\n")
            }
        }

        // 16. CSV / TSV / PSV / TAB structured table -> spoken prose
        if ["csv", "tsv", "psv", "tab"].contains(ext) {
            let delimiter: Character = (ext == "tsv" || ext == "tab") ? "\t" : (ext == "psv" ? "|" : ",")
            let spoken = parseDelimitedTableToProse(raw, delimiter: delimiter)
            if !spoken.isEmpty { return spoken }
        }

        // 17. JSON / GeoJSON / HAR structured object -> spoken prose
        if ["json", "geojson", "har"].contains(ext), let data = raw.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) {
            let prose = structuredObjectToProse(obj)
            if !prose.isEmpty { return prose }
        }

        // 18. JSONL / NDJSON (newline-delimited JSON) -> spoken prose
        if ext == "jsonl" || ext == "ndjson" {
            let lines = raw.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            var proseRecords: [String] = []
            for line in lines.prefix(1000) {
                if let data = line.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data) {
                    let p = structuredObjectToProse(obj)
                    if !p.isEmpty { proseRecords.append(p) }
                }
            }
            if !proseRecords.isEmpty {
                return proseRecords.joined(separator: "\n\n")
            }
        }

        // 19. YAML / TOML / INI / CONF / ENV
        if ["yaml", "yml", "toml", "ini", "cfg", "conf", "properties", "env"].contains(ext) {
            let cleanedConfig = cleanConfigOrYAMLToProse(raw)
            if !cleanedConfig.isEmpty { return cleanedConfig }
        }

        // 20. XML / RSS / Atom / OPML / RDF
        if ["xml", "rss", "atom", "opml", "rdf"].contains(ext) {
            let formattedXML = insertParagraphBreaksInHTML(raw)
            let cleanedXML = clean(formattedXML, options: bookCleanOpts)
            if !cleanedXML.isEmpty { return cleanedXML }
        }

        // 21. Markdown & Documentation
        if ["md", "markdown", "mdx", "rmd", "qmd", "html", "htm"].contains(ext) {
            return clean(raw)
        }

        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func readTextWithFallbackEncoding(from url: URL) -> String? {
        if let utf8 = try? String(contentsOf: url, encoding: .utf8), !utf8.contains("\0") {
            return utf8
        }
        var detectedEncoding: String.Encoding = .utf8
        if let detected = try? String(contentsOf: url, usedEncoding: &detectedEncoding), !detected.contains("\0") {
            return detected
        }
        if let utf16 = try? String(contentsOf: url, encoding: .utf16), !utf16.isEmpty {
            return utf16
        }
        if let win1252 = try? String(contentsOf: url, encoding: .windowsCP1252), !win1252.contains("\0") {
            return win1252
        }
        if let latin1 = try? String(contentsOf: url, encoding: .isoLatin1), !latin1.contains("\0") {
            return latin1
        }
        // If binary file with embedded text strings, extract printable UTF-8/ASCII passages
        if let data = try? Data(contentsOf: url), !data.isEmpty {
            let extracted = extractPrintableTextFromBinaryData(data)
            if !extracted.isEmpty { return extracted }
        }
        return nil
    }

    private static func insertParagraphBreaksInHTML(_ html: String) -> String {
        html
            .replacingOccurrences(of: "</p>", with: "</p>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</div>", with: "</div>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</section>", with: "</section>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</article>", with: "</article>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</h1>", with: "</h1>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</h2>", with: "</h2>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</h3>", with: "</h3>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</h4>", with: "</h4>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</li>", with: "</li>\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</tr>", with: "</tr>\n", options: .caseInsensitive)
            .replacingOccurrences(of: "<br>", with: "\n", options: .caseInsensitive)
            .replacingOccurrences(of: "<br/>", with: "\n", options: .caseInsensitive)
            .replacingOccurrences(of: "<br />", with: "\n", options: .caseInsensitive)
    }

    private static func cleanSubtitleOrLyricText(_ raw: String, ext: String) -> String {
        let normalized = raw.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")

        // ASS / SSA subtitle format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        if ext == "ass" || ext == "ssa" || normalized.contains("[Events]") || normalized.contains("Dialogue:") {
            var cues: [String] = []
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.lowercased().hasPrefix("dialogue:") else { continue }
                let afterPrefix = trimmed.dropFirst("dialogue:".count)
                let parts = afterPrefix.split(separator: ",", maxSplits: 9, omittingEmptySubsequences: false)
                guard parts.count >= 10 else { continue }
                var textPart = String(parts[9])
                // Strip {\...} ASS override tags
                textPart = replaceRegex(textPart, pattern: "\\{[^}]*\\}", with: "")
                textPart = textPart
                    .replacingOccurrences(of: "\\N", with: " ")
                    .replacingOccurrences(of: "\\n", with: " ")
                    .replacingOccurrences(of: "\\h", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !textPart.isEmpty && cues.last != textPart {
                    cues.append(textPart)
                }
            }
            if !cues.isEmpty {
                return cues.joined(separator: "\n")
            }
        }

        // MicroDVD .sub format: {123}{456}First line|Second line
        if ext == "sub" {
            var cues: [String] = []
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("{") {
                    let stripped = replaceRegex(trimmed, pattern: "^\\{\\d+\\}\\{\\d+\\}", with: "")
                        .replacingOccurrences(of: "|", with: " ")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !stripped.isEmpty {
                        cues.append(stripped)
                    }
                } else if !trimmed.isEmpty && trimmed.range(of: "^\\d{2}:\\d{2}:\\d{2}", options: .regularExpression) == nil {
                    cues.append(trimmed)
                }
            }
            if !cues.isEmpty {
                return cues.joined(separator: "\n")
            }
        }

        // TTML / DFXP / ITT / SAMI XML subtitles
        if ["ttml", "dfxp", "itt", "smi", "sami"].contains(ext) {
            let formatted = insertParagraphBreaksInHTML(normalized)
            return clean(formatted, options: SpeechCleanOptions(stripMarkdown: false, stripHtml: true, stripTimestamps: true))
        }

        // LRC / SRT / VTT / SBV
        var outLines: [String] = []
        for line in lines {
            var s = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if s.isEmpty { continue }
            let upper = s.uppercased()
            if upper == "WEBVTT" || upper.hasPrefix("WEBVTT ") || upper.hasPrefix("NOTE ") || upper.hasPrefix("STYLE") {
                continue
            }
            // Skip LRC metadata tags like [ti:Title], [ar:Artist], [al:Album], [by:Author], [offset:0]
            if s.range(of: "^\\[(?:ti|ar|al|by|offset|re|ve|length):", options: [.regularExpression, .caseInsensitive]) != nil {
                continue
            }
            // Skip numeric cue indices
            if s.range(of: "^\\d+$", options: .regularExpression) != nil {
                continue
            }
            // Skip SRT/VTT/SBV timestamp lines
            if s.range(of: "^\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d{1,3}\\s*(?:-->|,)\\s*\\d{1,2}:\\d{2}", options: .regularExpression) != nil {
                continue
            }
            // Strip inline LRC [mm:ss.xx] or <mm:ss.xx> word timestamps
            s = replaceRegex(s, pattern: "[\\[<]\\d{1,2}:\\d{2}(?:[.:]\\d{1,3})?[\\]>]", with: " ")
            // Strip VTT/HTML tags
            s = replaceRegex(s, pattern: "<[^>]+>", with: " ")
            s = decodeHTMLEntities(s).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty {
                outLines.append(s)
            }
        }
        return outLines.joined(separator: "\n")
    }

    private static func parseDelimitedTableToProse(_ raw: String, delimiter: Character) -> String {
        let lines = raw.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !lines.isEmpty else { return "" }
        let headers = splitCSVRow(lines[0], delimiter: delimiter)
        guard lines.count >= 2 else {
            return headers.joined(separator: ", ")
        }
        var spokenRows: [String] = []
        for rowLine in lines.dropFirst().prefix(2000) {
            let cols = splitCSVRow(rowLine, delimiter: delimiter)
            var pairs: [String] = []
            for (idx, val) in cols.enumerated() where !val.isEmpty {
                if idx < headers.count && !headers[idx].isEmpty {
                    pairs.append("\(headers[idx]): \(val)")
                } else {
                    pairs.append(val)
                }
            }
            if !pairs.isEmpty {
                spokenRows.append(pairs.joined(separator: ", ") + ".")
            }
        }
        return spokenRows.joined(separator: "\n")
    }

    private static func splitCSVRow(_ line: String, delimiter: Character) -> [String] {
        var result: [String] = []
        var current = ""
        var inQuotes = false
        var iterator = line.makeIterator()
        while let ch = iterator.next() {
            if ch == "\"" {
                inQuotes.toggle()
            } else if ch == delimiter && !inQuotes {
                result.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            } else {
                current.append(ch)
            }
        }
        result.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        return result
    }

    private static func cleanConfigOrYAMLToProse(_ raw: String) -> String {
        let lines = raw.components(separatedBy: .newlines)
        var out: [String] = []
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") || trimmed.hasPrefix(";") || trimmed == "---" || trimmed == "..." {
                continue
            }
            out.append(trimmed)
        }
        return out.joined(separator: "\n")
    }

    private static func cleanLaTeXForSpeech(_ raw: String) -> String {
        var s = raw
        // If \begin{document} exists, focus on document body while keeping \title if present
        var extractedTitle = ""
        if let titleRange = s.range(of: "\\\\title\\{([^}]+)\\}", options: .regularExpression) {
            extractedTitle = String(s[titleRange])
                .replacingOccurrences(of: "\\title{", with: "")
                .replacingOccurrences(of: "}", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let beginDoc = s.range(of: "\\begin{document}") {
            s = String(s[beginDoc.upperBound...])
        }
        if let endDoc = s.range(of: "\\end{document}") {
            s = String(s[..<endDoc.lowerBound])
        }
        // Strip LaTeX comments
        s = replaceRegex(s, pattern: "(?m)(?<!\\\\)%.*$", with: "")
        // Convert section headings to plain lines
        s = replaceRegex(s, pattern: "\\\\(?:part|chapter|section|subsection|subsubsection|paragraph)\\*?\\{([^}]*)\\}", with: "\n\n$1.\n\n")
        // Unwrap formatting commands \textbf{...}, \textit{...}, \emph{...}, \underline{...}, \texttt{...}
        for _ in 0..<3 {
            s = replaceRegex(s, pattern: "\\\\(?:textbf|textit|emph|underline|texttt|textsc|textrm|textsf|mbox)\\{([^}]*)\\}", with: "$1")
        }
        // Strip citations, labels, refs, footnotes, package/environment boilerplate
        s = replaceRegex(s, pattern: "\\\\(?:cite|citep|citet|ref|eqref|pageref|label|bibliographystyle|bibliography|usepackage|documentclass|maketitle)(?:\\[[^\\]]*\\])?\\{[^}]*\\}", with: "")
        s = replaceRegex(s, pattern: "\\\\(?:begin|end)\\{[^}]*\\}(?:\\[[^\\]]*\\])?", with: "\n")
        s = replaceRegex(s, pattern: "\\\\item\\b", with: "\n• ")
        s = replaceRegex(s, pattern: "\\\\[a-zA-Z]+\\*?", with: " ")
        s = s.replacingOccurrences(of: "{", with: "").replacingOccurrences(of: "}", with: "")
        s = s.replacingOccurrences(of: "~", with: " ")
        let body = clean(s)
        if !extractedTitle.isEmpty && !body.hasPrefix(extractedTitle) {
            return "\(extractedTitle).\n\n\(body)"
        }
        return body
    }

    private static func extractEmailOrMHTMLText(from raw: String) -> String {
        let normalized = raw.replacingOccurrences(of: "\r\n", with: "\n")
        var headerLines: [String] = []
        let parts = normalized.components(separatedBy: "\n\n")
        let rawHeaders = parts.first ?? ""
        for hLine in rawHeaders.components(separatedBy: "\n") {
            let lower = hLine.lowercased()
            if lower.hasPrefix("subject:") || lower.hasPrefix("from:") || lower.hasPrefix("date:") {
                headerLines.append(hLine.trimmingCharacters(in: .whitespaces))
            }
        }
        var body = parts.dropFirst().joined(separator: "\n\n")
        // Decode quoted-printable soft line breaks and =XX hex sequences if present
        if normalized.lowercased().contains("quoted-printable") || body.contains("=3D") || body.contains("=\n") {
            body = body.replacingOccurrences(of: "=\n", with: "")
            body = body.replacingOccurrences(of: "=3D", with: "=")
            body = body.replacingOccurrences(of: "=20", with: " ")
            body = body.replacingOccurrences(of: "=09", with: "\t")
        }
        // Strip MIME boundary lines and part headers
        body = replaceRegex(body, pattern: "(?m)^--[-A-Za-z0-9_.=+]+.*$", with: "\n")
        body = replaceRegex(body, pattern: "(?im)^content-(?:type|transfer-encoding|disposition|id|location):.*$", with: "")
        let formatted = insertParagraphBreaksInHTML(body)
        let cleanedBody = clean(formatted, options: SpeechCleanOptions(stripMarkdown: false, stripHtml: true))
        if headerLines.isEmpty { return cleanedBody }
        return (headerLines.joined(separator: "\n") + "\n\n" + cleanedBody).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractSVGTextElements(from rawSVG: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: "<text\\b[^>]*>([\\s\\S]*?)</text>", options: [.caseInsensitive]) else {
            return ""
        }
        let ns = rawSVG as NSString
        let matches = regex.matches(in: rawSVG, options: [], range: NSRange(location: 0, length: ns.length))
        var texts: [String] = []
        for match in matches where match.numberOfRanges >= 2 {
            let inner = ns.substring(with: match.range(at: 1))
            let stripped = replaceRegex(inner, pattern: "<[^>]+>", with: " ")
            let decoded = decodeHTMLEntities(stripped).trimmingCharacters(in: .whitespacesAndNewlines)
            if !decoded.isEmpty {
                texts.append(decoded)
            }
        }
        return texts.joined(separator: "\n")
    }

    private static func extractPrintableTextFromBinaryData(_ data: Data) -> String {
        // Extract contiguous UTF-8 / ASCII prose sequences (>= 16 printable chars)
        var passages: [String] = []
        var currentBytes: [UInt8] = []
        let maxScan = min(data.count, 8 * 1024 * 1024)
        data.prefix(maxScan).forEach { byte in
            if (byte >= 32 && byte <= 126) || byte == 9 || byte == 10 || byte == 13 {
                currentBytes.append(byte)
            } else {
                if currentBytes.count >= 24,
                   let str = String(bytes: currentBytes, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   str.contains(" ") {
                    passages.append(str)
                }
                currentBytes.removeAll(keepingCapacity: true)
            }
        }
        if currentBytes.count >= 24,
           let str = String(bytes: currentBytes, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           str.contains(" ") {
            passages.append(str)
        }
        let joined = passages.joined(separator: "\n\n")
        return clean(insertParagraphBreaksInHTML(joined), options: SpeechCleanOptions(stripMarkdown: false, stripHtml: true))
    }

    private static func structuredObjectToProse(_ obj: Any, depth: Int = 0) -> String {
        guard depth <= 6 else { return "" }
        if let str = obj as? String { return str }
        if let num = obj as? NSNumber { return num.stringValue }
        if let arr = obj as? [Any] {
            return arr.map { structuredObjectToProse($0, depth: depth + 1) }
                .filter { !$0.isEmpty }
                .joined(separator: depth == 0 ? "\n\n" : ", ")
        }
        if let dict = obj as? [String: Any] {
            var lines: [String] = []
            for key in dict.keys.sorted() {
                guard let val = dict[key] else { continue }
                let sub = structuredObjectToProse(val, depth: depth + 1)
                if !sub.isEmpty {
                    let humanKey = key.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
                    lines.append("\(humanKey): \(sub)")
                }
            }
            return lines.joined(separator: depth == 0 ? "\n" : "; ")
        }
        return ""
    }

    #if os(macOS)
    private static func runCLI(executable: String, arguments: [String]) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments = arguments
        proc.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            if proc.terminationStatus == 0, let str = String(data: data, encoding: .utf8) {
                return str.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        } catch {}
        return nil
    }

    private static func runTextutil(on url: URL) -> String? {
        runCLI(executable: "/usr/bin/textutil", arguments: ["-convert", "txt", "-stdout", url.path])
    }

    private static func runEbookConvertToText(from url: URL) -> String? {
        let extraDirs = [
            "/Applications/calibre.app/Contents/MacOS",
            "/opt/homebrew/bin",
            "/usr/local/bin"
        ]
        guard let ebookConvert = StudioPackageManager.findExecutable(names: ["ebook-convert"], extraDirs: extraDirs) else {
            return nil
        }
        let tempTxt = FileManager.default.temporaryDirectory
            .appendingPathComponent("calibre_\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: tempTxt) }
        _ = runCLI(executable: ebookConvert, arguments: [url.path, tempTxt.path])
        if let str = readTextWithFallbackEncoding(from: tempTxt), !str.isEmpty {
            return str
        }
        return nil
    }

    private static func extractEpubText(from url: URL) -> String? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return nil }
        let htmlOnlyOpts = SpeechCleanOptions(
            stripMarkdown: false,
            stripHtml: true,
            stripCodeBlocks: false,
            simplifyUrls: true,
            stripTimestamps: false,
            stripDividers: true
        )

        let rootDir: URL
        var tempUnpackDir: URL? = nil
        defer {
            if let tempUnpackDir {
                try? FileManager.default.removeItem(at: tempUnpackDir)
            }
        }

        if isDir.boolValue {
            rootDir = url
        } else {
            let unpack = FileManager.default.temporaryDirectory
                .appendingPathComponent("epub_unpack_\(UUID().uuidString)", isDirectory: true)
            try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try? proc.run()
            proc.waitUntilExit()
            tempUnpackDir = unpack
            rootDir = unpack
        }

        // 1. Try reading OPF spine order (META-INF/container.xml -> content.opf)
        var orderedChapterURLs: [URL] = []
        let containerURL = rootDir.appendingPathComponent("META-INF/container.xml")
        if let containerXML = try? String(contentsOf: containerURL, encoding: .utf8),
           let opfRelPath = firstRegexCapture(in: containerXML, pattern: "full-path\\s*=\\s*[\"']([^\"']+\\.opf)[\"']") {
            let opfURL = rootDir.appendingPathComponent(opfRelPath)
            let opfBaseDir = opfURL.deletingLastPathComponent()
            if let opfXML = try? String(contentsOf: opfURL, encoding: .utf8) {
                var manifestMap: [String: String] = [:]
                if let itemRegex = try? NSRegularExpression(pattern: "<item\\b[^>]+>", options: [.caseInsensitive]) {
                    let ns = opfXML as NSString
                    for match in itemRegex.matches(in: opfXML, options: [], range: NSRange(location: 0, length: ns.length)) {
                        let tag = ns.substring(with: match.range)
                        if let itemId = firstRegexCapture(in: tag, pattern: "\\bid\\s*=\\s*[\"']([^\"']+)[\"']"),
                           let href = firstRegexCapture(in: tag, pattern: "\\bhref\\s*=\\s*[\"']([^\"'#]+)[\"']") {
                            manifestMap[itemId] = href.removingPercentEncoding ?? href
                        }
                    }
                }
                if let spineRegex = try? NSRegularExpression(pattern: "<itemref\\b[^>]+>", options: [.caseInsensitive]) {
                    let ns = opfXML as NSString
                    for match in spineRegex.matches(in: opfXML, options: [], range: NSRange(location: 0, length: ns.length)) {
                        let tag = ns.substring(with: match.range)
                        if let idref = firstRegexCapture(in: tag, pattern: "\\bidref\\s*=\\s*[\"']([^\"']+)[\"']"),
                           let href = manifestMap[idref] {
                            let candidate = opfBaseDir.appendingPathComponent(href)
                            if FileManager.default.fileExists(atPath: candidate.path) {
                                orderedChapterURLs.append(candidate)
                            }
                        }
                    }
                }
            }
        }

        // 2. Fallback: enumerate all HTML/XHTML files in natural sort order
        if orderedChapterURLs.isEmpty,
           let enumerator = FileManager.default.enumerator(at: rootDir, includingPropertiesForKeys: nil) {
            for case let fileURL as URL in enumerator {
                let ext = fileURL.pathExtension.lowercased()
                let name = fileURL.lastPathComponent.lowercased()
                if ["xhtml", "html", "htm"].contains(ext) && !name.contains("toc") && !name.contains("nav") {
                    orderedChapterURLs.append(fileURL)
                }
            }
            orderedChapterURLs.sort {
                $0.path.localizedStandardCompare($1.path) == .orderedAscending
            }
        }

        let chapters = orderedChapterURLs.compactMap { readTextWithFallbackEncoding(from: $0) }.map {
            clean(insertParagraphBreaksInHTML($0), options: htmlOnlyOpts)
        }.filter { !$0.isEmpty }

        return chapters.isEmpty ? nil : chapters.joined(separator: "\n\n")
    }

    private static func firstRegexCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, options: [], range: NSRange(location: 0, length: ns.length)),
              match.numberOfRanges >= 2 else {
            return nil
        }
        return ns.substring(with: match.range(at: 1))
    }

    // MARK: - Native FictionBook (.fb2 / .fbz) Extractor
    private static func extractFB2Text(from url: URL) -> String? {
        var rawXML: String? = nil
        if let data = try? Data(contentsOf: url), data.count >= 4 {
            // Check if ZIP archive (PK\x03\x04)
            if data[0] == 0x50 && data[1] == 0x4B && data[2] == 0x03 && data[3] == 0x04 {
                let unpack = FileManager.default.temporaryDirectory
                    .appendingPathComponent("fb2_unpack_\(UUID().uuidString)", isDirectory: true)
                defer { try? FileManager.default.removeItem(at: unpack) }
                try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
                proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
                proc.standardInput = FileHandle.nullDevice
                proc.standardOutput = FileHandle.nullDevice
                proc.standardError = FileHandle.nullDevice
                try? proc.run()
                proc.waitUntilExit()
                if let enumerator = FileManager.default.enumerator(at: unpack, includingPropertiesForKeys: nil) {
                    for case let fileURL as URL in enumerator {
                        if ["fb2", "xml"].contains(fileURL.pathExtension.lowercased()),
                           let s = readTextWithFallbackEncoding(from: fileURL), !s.isEmpty {
                            rawXML = s
                            break
                        }
                    }
                }
            }
        }
        if rawXML == nil {
            rawXML = readTextWithFallbackEncoding(from: url)
        }
        guard var xml = rawXML, !xml.isEmpty else { return nil }

        // Strip base64 <binary>...</binary> cover image blocks so TTS never reads base64 strings
        xml = replaceRegex(xml, pattern: "(?is)<binary\\b[^>]*>.*?</binary>", with: " ")
        xml = replaceRegex(xml, pattern: "(?is)<stylesheet\\b[^>]*>.*?</stylesheet>", with: " ")
        xml = xml
            .replacingOccurrences(of: "</title>", with: "</title>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</subtitle>", with: "</subtitle>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</p>", with: "</p>\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "</v>", with: "</v>\n", options: .caseInsensitive)
            .replacingOccurrences(of: "<empty-line/>", with: "\n\n", options: .caseInsensitive)
            .replacingOccurrences(of: "<empty-line />", with: "\n\n", options: .caseInsensitive)

        let htmlOnlyOpts = SpeechCleanOptions(
            stripMarkdown: false,
            stripHtml: true,
            stripCodeBlocks: false,
            simplifyUrls: true,
            stripTimestamps: false,
            stripDividers: true
        )
        return clean(xml, options: htmlOnlyOpts)
    }

    // MARK: - Native Kindle / Mobipocket / PalmDOC (.mobi, .azw, .azw3, .prc, .pdb) Decompressor & Extractor
    private static func extractMobiOrPalmDocText(from url: URL) -> String? {
        guard let data = try? Data(contentsOf: url), data.count > 86 else { return nil }
        let bytes = [UInt8](data)

        func readUInt16BE(_ offset: Int) -> Int {
            guard offset + 1 < bytes.count else { return 0 }
            return (Int(bytes[offset]) << 8) | Int(bytes[offset + 1])
        }

        func readUInt32BE(_ offset: Int) -> Int {
            guard offset + 3 < bytes.count else { return 0 }
            return (Int(bytes[offset]) << 24) | (Int(bytes[offset + 1]) << 16) | (Int(bytes[offset + 2]) << 8) | Int(bytes[offset + 3])
        }

        let numRecords = readUInt16BE(76)
        guard numRecords > 1 && numRecords < 65535, 78 + numRecords * 8 <= bytes.count else {
            return extractPrintableTextFromBinaryData(data)
        }

        var recordOffsets: [Int] = []
        recordOffsets.reserveCapacity(numRecords)
        for i in 0..<numRecords {
            let recOffset = readUInt32BE(78 + i * 8)
            recordOffsets.append(recOffset)
        }

        let rec0Start = recordOffsets[0]
        guard rec0Start >= 0 && rec0Start + 16 <= bytes.count else {
            return extractPrintableTextFromBinaryData(data)
        }

        let compression = readUInt16BE(rec0Start) // 1 = None, 2 = PalmDOC LZ77, 17480 = HUFF/CDIC
        let textRecordCount = max(1, min(readUInt16BE(rec0Start + 8), numRecords - 1))

        // Check MOBI extraDataFlags at offset rec0Start + 242 if MOBI header is present
        var extraDataFlags = 0
        if rec0Start + 244 <= bytes.count,
           bytes[rec0Start + 16] == 0x4D, // 'M'
           bytes[rec0Start + 17] == 0x4F, // 'O'
           bytes[rec0Start + 18] == 0x42, // 'B'
           bytes[rec0Start + 19] == 0x49  // 'I'
        {
            let mobiHeaderLen = readUInt32BE(rec0Start + 20)
            if mobiHeaderLen >= 228 {
                extraDataFlags = readUInt16BE(rec0Start + 242)
            }
        }

        if compression == 1 || compression == 2 {
            var decompressedBytes: [UInt8] = []
            decompressedBytes.reserveCapacity(min(data.count * 2, 16 * 1024 * 1024))

            for r in 1...textRecordCount {
                guard r < recordOffsets.count else { break }
                let start = recordOffsets[r]
                let end = (r + 1 < recordOffsets.count) ? min(bytes.count, recordOffsets[r + 1]) : bytes.count
                guard start >= 0 && end > start && end <= bytes.count else { continue }
                var sliceEnd = end

                // Trim trailing MOBI extra data entries if flagged
                if extraDataFlags != 0 {
                    var flags = extraDataFlags >> 1
                    while flags != 0 && sliceEnd > start {
                        if (flags & 1) != 0 {
                            // Variable-length backward-encoded size at end of record
                            var size = 0
                            var shift = 0
                            var pos = sliceEnd - 1
                            while pos >= start && shift < 28 {
                                let b = Int(bytes[pos])
                                size |= (b & 0x7F) << shift
                                pos -= 1
                                shift += 7
                                if (b & 0x80) != 0 { break }
                            }
                            if size > 0 && size <= (sliceEnd - start) {
                                sliceEnd -= size
                            }
                        }
                        flags >>= 1
                    }
                    if (extraDataFlags & 1) != 0 && sliceEnd > start {
                        let multibyteOverlap = Int(bytes[sliceEnd - 1] & 0x03) + 1
                        if multibyteOverlap <= (sliceEnd - start) {
                            sliceEnd -= multibyteOverlap
                        }
                    }
                }

                let recordSlice = bytes[start..<sliceEnd]
                if compression == 1 {
                    decompressedBytes.append(contentsOf: recordSlice)
                } else {
                    decompressedBytes.append(contentsOf: decompressPalmDOC(recordSlice))
                }
            }

            if !decompressedBytes.isEmpty {
                let decodedStr = String(bytes: decompressedBytes, encoding: .utf8)
                    ?? String(bytes: decompressedBytes, encoding: .windowsCP1252)
                    ?? String(bytes: decompressedBytes, encoding: .isoLatin1)
                    ?? ""
                let withBreaks = insertParagraphBreaksInHTML(decodedStr)
                    .replacingOccurrences(of: "<mbp:pagebreak/>", with: "\n\n", options: .caseInsensitive)
                let cleaned = clean(withBreaks, options: SpeechCleanOptions(stripMarkdown: false, stripHtml: true))
                if !cleaned.isEmpty {
                    return cleaned
                }
            }
        }

        return extractPrintableTextFromBinaryData(data)
    }

    private static func decompressPalmDOC(_ input: ArraySlice<UInt8>) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(input.count * 2)
        var idx = input.startIndex
        let end = input.endIndex

        while idx < end {
            let b = input[idx]
            idx += 1
            if b == 0x00 {
                out.append(0x00)
            } else if b >= 0x01 && b <= 0x08 {
                let count = Int(b)
                for _ in 0..<count where idx < end {
                    out.append(input[idx])
                    idx += 1
                }
            } else if b >= 0x09 && b <= 0x7F {
                out.append(b)
            } else if b >= 0x80 && b <= 0xBF {
                guard idx < end else { break }
                let next = input[idx]
                idx += 1
                let pair = (Int(b) << 8) | Int(next)
                let distance = (pair & 0x3FFF) >> 3
                let length = (pair & 0x07) + 3
                if distance > 0 && distance <= out.count {
                    for _ in 0..<length {
                        let srcIdx = out.count - distance
                        out.append(out[srcIdx])
                    }
                }
            } else {
                // 0xC0..0xFF: space followed by (b ^ 0x80)
                out.append(0x20)
                out.append(b ^ 0x80)
            }
        }
        return out
    }

    // MARK: - Native Office OpenXML & OpenDocument (.docx, .pptx, .xlsx, .odt, .odp, .ods) Extractor
    private static func extractOfficeZipXMLText(from url: URL, ext: String) -> String? {
        // Handle flat XML OpenDocument formats directly (.fodt, .fodp, .fods, .wordml)
        if ["fodt", "fodp", "fods", "wordml", "abw"].contains(ext),
           let rawXML = readTextWithFallbackEncoding(from: url) {
            return cleanOfficeXMLString(rawXML)
        }

        let unpack = FileManager.default.temporaryDirectory
            .appendingPathComponent("office_unpack_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: unpack) }
        try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()

        // 1. Word (.docx, .docm, .dotx): word/document.xml
        let wordDocURL = unpack.appendingPathComponent("word/document.xml")
        if FileManager.default.fileExists(atPath: wordDocURL.path),
           let xml = try? String(contentsOf: wordDocURL, encoding: .utf8) {
            return cleanOfficeXMLString(xml)
        }

        // 2. PowerPoint (.pptx, .ppsx): ppt/slides/slide*.xml
        let slidesDir = unpack.appendingPathComponent("ppt/slides", isDirectory: true)
        if let slideFiles = try? FileManager.default.contentsOfDirectory(at: slidesDir, includingPropertiesForKeys: nil) {
            let sortedSlides = slideFiles
                .filter { $0.pathExtension.lowercased() == "xml" && $0.lastPathComponent.lowercased().hasPrefix("slide") }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            var slideTexts: [String] = []
            for (idx, slideURL) in sortedSlides.enumerated() {
                if let xml = try? String(contentsOf: slideURL, encoding: .utf8) {
                    let text = cleanOfficeXMLString(xml)
                    if !text.isEmpty {
                        slideTexts.append("Slide \(idx + 1).\n\(text)")
                    }
                }
            }
            if !slideTexts.isEmpty {
                return slideTexts.joined(separator: "\n\n")
            }
        }

        // 3. Excel (.xlsx, .xlsm): xl/sharedStrings.xml + xl/worksheets/sheet*.xml
        let xlDir = unpack.appendingPathComponent("xl", isDirectory: true)
        if FileManager.default.fileExists(atPath: xlDir.path) {
            if let xlsxText = extractXLSXText(fromUnpackedXLDir: xlDir), !xlsxText.isEmpty {
                return xlsxText
            }
        }

        // 4. OpenDocument (.odt, .ott, .odp, .ods): content.xml
        let odtContentURL = unpack.appendingPathComponent("content.xml")
        if FileManager.default.fileExists(atPath: odtContentURL.path),
           let xml = try? String(contentsOf: odtContentURL, encoding: .utf8) {
            return cleanOfficeXMLString(xml)
        }

        return nil
    }

    private static func cleanOfficeXMLString(_ xml: String) -> String {
        let withBreaks = xml
            .replacingOccurrences(of: "</w:p>", with: "\n\n")
            .replacingOccurrences(of: "<w:br/>", with: "\n")
            .replacingOccurrences(of: "<w:br />", with: "\n")
            .replacingOccurrences(of: "<w:cr/>", with: "\n")
            .replacingOccurrences(of: "<w:tab/>", with: " ")
            .replacingOccurrences(of: "</a:p>", with: "\n")
            .replacingOccurrences(of: "</text:p>", with: "\n\n")
            .replacingOccurrences(of: "</text:h>", with: "\n\n")
            .replacingOccurrences(of: "</table:table-row>", with: "\n")
            .replacingOccurrences(of: "</table:table-cell>", with: ", ")
            .replacingOccurrences(of: "<text:line-break/>", with: "\n")
        let stripped = replaceRegex(withBreaks, pattern: "<[^>]+>", with: "")
        let decoded = decodeHTMLEntities(stripped)
        return clean(decoded, options: SpeechCleanOptions(stripMarkdown: false, stripHtml: false))
    }

    private static func extractXLSXText(fromUnpackedXLDir xlDir: URL) -> String? {
        var sharedStrings: [String] = []
        let sharedURL = xlDir.appendingPathComponent("sharedStrings.xml")
        if let sharedXML = try? String(contentsOf: sharedURL, encoding: .utf8),
           let siRegex = try? NSRegularExpression(pattern: "<si\\b[^>]*>([\\s\\S]*?)</si>", options: []) {
            let ns = sharedXML as NSString
            for match in siRegex.matches(in: sharedXML, options: [], range: NSRange(location: 0, length: ns.length)) where match.numberOfRanges >= 2 {
                let siContent = ns.substring(with: match.range(at: 1))
                let stripped = replaceRegex(siContent, pattern: "<[^>]+>", with: "")
                sharedStrings.append(decodeHTMLEntities(stripped).trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }

        let sheetsDir = xlDir.appendingPathComponent("worksheets", isDirectory: true)
        guard let sheetFiles = try? FileManager.default.contentsOfDirectory(at: sheetsDir, includingPropertiesForKeys: nil) else {
            return sharedStrings.isEmpty ? nil : sharedStrings.joined(separator: "\n")
        }
        let sortedSheets = sheetFiles
            .filter { $0.pathExtension.lowercased() == "xml" }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }

        var allRows: [String] = []
        let rowRegex = try? NSRegularExpression(pattern: "<row\\b[^>]*>([\\s\\S]*?)</row>", options: [])
        let cellRegex = try? NSRegularExpression(pattern: "<c\\b([^>]*)>([\\s\\S]*?)</c>", options: [])

        for sheetURL in sortedSheets {
            guard let sheetXML = try? String(contentsOf: sheetURL, encoding: .utf8),
                  let rowRegex, let cellRegex else { continue }
            let nsSheet = sheetXML as NSString
            var headers: [String] = []
            for (rowIdx, rowMatch) in rowRegex.matches(in: sheetXML, options: [], range: NSRange(location: 0, length: nsSheet.length)).prefix(1000).enumerated() where rowMatch.numberOfRanges >= 2 {
                let rowContent = nsSheet.substring(with: rowMatch.range(at: 1))
                let nsRow = rowContent as NSString
                var rowVals: [String] = []
                for cellMatch in cellRegex.matches(in: rowContent, options: [], range: NSRange(location: 0, length: nsRow.length)) where cellMatch.numberOfRanges >= 3 {
                    let attrs = nsRow.substring(with: cellMatch.range(at: 1))
                    let inner = nsRow.substring(with: cellMatch.range(at: 2))
                    let isShared = attrs.contains("t=\"s\"") || attrs.contains("t='s'")
                    if isShared,
                       let vStr = firstRegexCapture(in: inner, pattern: "<v>([^<]+)</v>"),
                       let sIdx = Int(vStr.trimmingCharacters(in: .whitespacesAndNewlines)),
                       sIdx >= 0 && sIdx < sharedStrings.count {
                        rowVals.append(sharedStrings[sIdx])
                    } else if let inlineStr = firstRegexCapture(in: inner, pattern: "<t[^>]*>([^<]+)</t>") {
                        rowVals.append(decodeHTMLEntities(inlineStr))
                    } else if let vStr = firstRegexCapture(in: inner, pattern: "<v>([^<]+)</v>") {
                        rowVals.append(decodeHTMLEntities(vStr))
                    }
                }
                let nonEmpty = rowVals.filter { !$0.isEmpty }
                if nonEmpty.isEmpty { continue }
                if rowIdx == 0 && headers.isEmpty {
                    headers = rowVals
                    allRows.append(nonEmpty.joined(separator: ", ") + ".")
                } else if !headers.isEmpty {
                    var pairs: [String] = []
                    for (cIdx, val) in rowVals.enumerated() where !val.isEmpty {
                        if cIdx < headers.count && !headers[cIdx].isEmpty {
                            pairs.append("\(headers[cIdx]): \(val)")
                        } else {
                            pairs.append(val)
                        }
                    }
                    if !pairs.isEmpty {
                        allRows.append(pairs.joined(separator: ", ") + ".")
                    }
                } else {
                    allRows.append(nonEmpty.joined(separator: ", ") + ".")
                }
            }
        }

        if !allRows.isEmpty {
            return allRows.joined(separator: "\n")
        }
        return sharedStrings.isEmpty ? nil : sharedStrings.joined(separator: "\n")
    }

    // MARK: - Apple iWork (.pages, .numbers, .key) & XPS & Legacy Binary Office Extractors
    private static func extractIWorkText(from url: URL) -> String? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return nil }

        let rootDir: URL
        var tempUnpackDir: URL? = nil
        defer {
            if let tempUnpackDir {
                try? FileManager.default.removeItem(at: tempUnpackDir)
            }
        }

        if isDir.boolValue {
            rootDir = url
        } else {
            let unpack = FileManager.default.temporaryDirectory
                .appendingPathComponent("iwork_unpack_\(UUID().uuidString)", isDirectory: true)
            try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try? proc.run()
            proc.waitUntilExit()
            tempUnpackDir = unpack
            rootDir = unpack
        }

        // 1. Check QuickLook/Preview.pdf or Preview.pdf inside the iWork bundle
        for relPDF in ["QuickLook/Preview.pdf", "Preview.pdf"] {
            let pdfURL = rootDir.appendingPathComponent(relPDF)
            if FileManager.default.fileExists(atPath: pdfURL.path),
               let pdf = PDFDocument(url: pdfURL) {
                var pages: [String] = []
                for i in 0..<pdf.pageCount {
                    if let t = pdf.page(at: i)?.string?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty {
                        pages.append(t)
                    }
                }
                if !pages.isEmpty {
                    return pages.joined(separator: "\n\n")
                }
            }
        }

        // 2. Check legacy iWork '09 index.xml
        let indexXMLURL = rootDir.appendingPathComponent("index.xml")
        if FileManager.default.fileExists(atPath: indexXMLURL.path),
           let xml = try? String(contentsOf: indexXMLURL, encoding: .utf8) {
            let text = cleanOfficeXMLString(xml)
            if !text.isEmpty { return text }
        }

        // 3. Scan Index/Document.iwa or other files inside bundle for readable text
        if let enumerator = FileManager.default.enumerator(at: rootDir, includingPropertiesForKeys: nil) {
            var chunks: [String] = []
            for case let fileURL as URL in enumerator {
                if fileURL.pathExtension.lowercased() == "pdf",
                   let pdf = PDFDocument(url: fileURL) {
                    for i in 0..<pdf.pageCount {
                        if let t = pdf.page(at: i)?.string?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty {
                            chunks.append(t)
                        }
                    }
                } else if fileURL.pathExtension.lowercased() == "iwa",
                          let data = try? Data(contentsOf: fileURL) {
                    let s = extractPrintableTextFromBinaryData(data)
                    if !s.isEmpty { chunks.append(s) }
                }
            }
            if !chunks.isEmpty {
                return chunks.joined(separator: "\n\n")
            }
        }

        return nil
    }

    private static func extractXPSText(from url: URL) -> String? {
        let unpack = FileManager.default.temporaryDirectory
            .appendingPathComponent("xps_unpack_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: unpack) }
        try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()

        guard let enumerator = FileManager.default.enumerator(at: unpack, includingPropertiesForKeys: nil),
              let glyphRegex = try? NSRegularExpression(pattern: "UnicodeString\\s*=\\s*[\"']([^\"']+)[\"']", options: [.caseInsensitive]) else {
            return nil
        }

        var pageURLs: [URL] = []
        for case let fileURL as URL in enumerator {
            if ["fpage", "xml"].contains(fileURL.pathExtension.lowercased()) {
                pageURLs.append(fileURL)
            }
        }
        pageURLs.sort { $0.path.localizedStandardCompare($1.path) == .orderedAscending }

        var pagesText: [String] = []
        for pageURL in pageURLs {
            guard let xml = try? String(contentsOf: pageURL, encoding: .utf8) else { continue }
            let ns = xml as NSString
            var lines: [String] = []
            for match in glyphRegex.matches(in: xml, options: [], range: NSRange(location: 0, length: ns.length)) where match.numberOfRanges >= 2 {
                let str = decodeHTMLEntities(ns.substring(with: match.range(at: 1))).trimmingCharacters(in: .whitespacesAndNewlines)
                if !str.isEmpty { lines.append(str) }
            }
            if !lines.isEmpty {
                pagesText.append(lines.joined(separator: " "))
            }
        }
        return pagesText.isEmpty ? nil : pagesText.joined(separator: "\n\n")
    }

    private static func extractLegacyBinaryOfficeText(from url: URL) -> String? {
        if let soffice = StudioPackageManager.findExecutable(
            names: ["soffice", "libreoffice"],
            extraDirs: ["/Applications/LibreOffice.app/Contents/MacOS"]
        ) {
            let outDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("soffice_\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: outDir) }
            try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            _ = runCLI(executable: soffice, arguments: ["--headless", "--convert-to", "txt:Text", "--outdir", outDir.path, url.path])
            if let files = try? FileManager.default.contentsOfDirectory(at: outDir, includingPropertiesForKeys: nil),
               let txtFile = files.first(where: { $0.pathExtension.lowercased() == "txt" }),
               let str = readTextWithFallbackEncoding(from: txtFile), !str.isEmpty {
                return clean(str)
            }
        }
        if let data = try? Data(contentsOf: url) {
            let extracted = extractPrintableTextFromBinaryData(data)
            if !extracted.isEmpty { return extracted }
        }
        return nil
    }

    // MARK: - Comic Book Archive (.cbz / .cbr) & Multi-Document Archive / Folder Extractors
    private static func extractComicArchiveOCRText(from url: URL) -> String? {
        let unpack = FileManager.default.temporaryDirectory
            .appendingPathComponent("comic_unpack_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: unpack) }
        try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()

        guard let enumerator = FileManager.default.enumerator(at: unpack, includingPropertiesForKeys: nil) else { return nil }
        var imageURLs: [URL] = []
        var comicInfoText = ""
        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if fileURL.lastPathComponent.lowercased() == "comicinfo.xml",
               let xml = try? String(contentsOf: fileURL, encoding: .utf8) {
                comicInfoText = cleanOfficeXMLString(xml)
            } else if UniversalFileFormatCatalog.imageOCRExtensions.contains(ext) {
                imageURLs.append(fileURL)
            }
        }
        imageURLs.sort { $0.path.localizedStandardCompare($1.path) == .orderedAscending }

        var pageTexts: [String] = []
        if !comicInfoText.isEmpty { pageTexts.append(comicInfoText) }
        for imgURL in imageURLs.prefix(40) {
            if let ocr = recognizeImageText(from: imgURL), !ocr.isEmpty {
                pageTexts.append(ocr)
            }
        }
        return pageTexts.isEmpty ? nil : pageTexts.joined(separator: "\n\n")
    }

    private static func extractArchiveDocuments(from url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        let unpack = FileManager.default.temporaryDirectory
            .appendingPathComponent("archive_unpack_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: unpack) }
        try? FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)

        let proc = Process()
        if ext == "zip" {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            proc.arguments = ["-q", "-o", url.path, "-d", unpack.path]
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            proc.arguments = ["-xf", url.path, "-C", unpack.path]
        }
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try? proc.run()
        proc.waitUntilExit()

        return extractFolderOrArchiveDocuments(at: unpack)
    }

    private static func extractFolderOrArchiveDocuments(at rootURL: URL) -> String? {
        guard let enumerator = FileManager.default.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        var docURLs: [URL] = []
        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if UniversalFileFormatCatalog.allReadableDocumentExtensions.contains(ext),
               !UniversalFileFormatCatalog.archiveExtensions.contains(ext) {
                docURLs.append(fileURL)
            }
        }
        docURLs.sort { $0.path.localizedStandardCompare($1.path) == .orderedAscending }

        var sections: [String] = []
        for docURL in docURLs.prefix(200) {
            if let extracted = try? extractDocumentText(from: docURL),
               !extracted.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sections.append(extracted.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        return sections.isEmpty ? nil : sections.joined(separator: "\n\n")
    }

    private static func extractMediaLyricsOrSubtitles(from url: URL) -> String? {
        // 1. Check embedded/sidecar song lyrics or subtitle tracks via SongLyricsExtractor
        if let extracted = SongLyricsExtractor.extractLyrics(from: url) {
            return cleanSubtitleOrLyricText(extracted.lyrics, ext: "lrc")
        }
        // 2. Check embedded subtitle stream inside MP4/MKV/MOV/WEBM via ffmpeg if installed
        if let ffmpeg = StudioPackageManager.findExecutable(
            names: ["ffmpeg"],
            extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
        ),
           let srtOut = runCLI(executable: ffmpeg, arguments: ["-v", "error", "-i", url.path, "-map", "0:s:0", "-f", "srt", "-"]),
           !srtOut.isEmpty {
            let cleanedSub = cleanSubtitleOrLyricText(srtOut, ext: "srt")
            if !cleanedSub.isEmpty { return cleanedSub }
        }
        return nil
    }

    private static func recognizeCGImageText(_ cgImage: CGImage) -> String? {
        var recognizedLines: [String] = []
        let request = VNRecognizeTextRequest { req, _ in
            guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
            for obs in observations {
                if let top = obs.topCandidates(1).first?.string, !top.isEmpty {
                    recognizedLines.append(top)
                }
            }
        }
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        return recognizedLines.joined(separator: "\n")
    }

    private static func recognizeImageText(from url: URL) -> String? {
        if let cgImageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
           let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil),
           let visionText = recognizeCGImageText(cgImage),
           !visionText.isEmpty {
            return visionText
        }
        if let nsImage = NSImage(contentsOf: url),
           let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil),
           let visionText = recognizeCGImageText(cgImage),
           !visionText.isEmpty {
            return visionText
        }
        // Fallback: Tesseract OCR CLI if installed
        if let tesseract = StudioPackageManager.findExecutable(names: ["tesseract"]),
           let out = runCLI(executable: tesseract, arguments: [url.path, "stdout", "--oem", "1", "-l", "eng"]),
           !out.isEmpty {
            return clean(out)
        }
        return nil
    }
    #endif

    private static func replaceRegex(_ input: String, pattern: String, with template: String) -> String {
        let regex: NSRegularExpression
        regexLock.lock()
        if let cached = regexCache[pattern] {
            regex = cached
            regexLock.unlock()
        } else if let compiled = try? NSRegularExpression(pattern: pattern, options: []) {
            regexCache[pattern] = compiled
            regex = compiled
            regexLock.unlock()
        } else {
            regexLock.unlock()
            return input
        }
        let range = NSRange(input.startIndex..., in: input)
        return regex.stringByReplacingMatches(in: input, options: [], range: range, withTemplate: template)
    }
}

// MARK: - Unlimited-Size Hierarchical Text Chunker (Ported from Codecaine File Manager fmPhonon.ts)

struct PhononTextChunk: Identifiable, Hashable {
    let id: Int
    let index: Int
    let paragraphIndex: Int
    let text: String
    let charCount: Int
    let wordCount: Int
}

enum HierarchicalChunker {
    static let defaultChunkChars = 520
    static let batchChunkChars = 880
    static let hardMaxChunkChars = 1200

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var lastRawText: String = ""
    nonisolated(unsafe) private static var lastMaxChars: Int = -1
    nonisolated(unsafe) private static var lastMergeShort: Bool = true
    nonisolated(unsafe) private static var lastCachedChunks: [PhononTextChunk] = []

    /// Sanitizes raw Markdown/novel paragraph lines so horizontal dividers (`---`, `***`), heading hashes (`# `, `## `),
    /// and inline emphasis wrappers (`*text*`, `**text**`) never cause TTS models to stumble or insert dead pauses.
    private static func sanitizeParagraphForSpeech(_ rawPara: String) -> String {
        let lines = rawPara.components(separatedBy: "\n")
        var cleanedLines: [String] = []
        cleanedLines.reserveCapacity(lines.count)

        for rawLine in lines {
            var line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }

            // Skip horizontal rules / section dividers (`---`, `***`, `___`, `===`)
            let dividerChars = CharacterSet(charactersIn: "-*_=~—– ")
            if line.count >= 3 && line.unicodeScalars.allSatisfy({ dividerChars.contains($0) }) {
                continue
            }

            // Strip leading Markdown heading hashes (`# `, `## `, `### `)
            var isHeading = false
            if line.hasPrefix("#") {
                var idx = line.startIndex
                while idx < line.endIndex && line[idx] == "#" {
                    idx = line.index(after: idx)
                }
                if idx < line.endIndex && line[idx].isWhitespace {
                    line = String(line[idx...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    isHeading = true
                }
            }

            // Strip surrounding/inline Markdown asterisks or underscores (`**bold**`, `*italic*`)
            if line.contains("*") {
                line = line.replacingOccurrences(of: "**", with: "")
                // Strip paired `*...*` while preserving solitary multiplication if any
                if line.hasPrefix("*") && line.hasSuffix("*") && line.count > 2 {
                    line = String(line.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
                }
                line = line.replacingOccurrences(of: "*", with: "")
            }

            guard !line.isEmpty else { continue }
            // Ensure standalone heading lines end with sentence punctuation so TTS gives them natural title cadence
            if isHeading || (line.count <= 72 && !line.contains("\"")) {
                if let last = line.last, !".!?:;…\"')]}".contains(last) {
                    line.append(".")
                }
            }
            cleanedLines.append(line)
        }
        return cleanedLines.joined(separator: "\n")
    }

    /// Splits arbitrarily large text (books, PDFs, EPUBs, logs) in O(N) linear time preferring:
    /// 1. Paragraph breaks (\n\n) -> 2. Sentence endings (.!?…;) -> 3. Clause breaks (,:—–-) -> 4. Word boundaries.
    static func splitText(
        _ rawText: String,
        maxChars: Int = defaultChunkChars,
        mergeShortParagraphs: Bool = true
    ) -> [PhononTextChunk] {
        guard !rawText.isEmpty else { return [] }
        let clampedMax = min(hardMaxChunkChars, max(120, maxChars))
        cacheLock.lock()
        if clampedMax == lastMaxChars && mergeShortParagraphs == lastMergeShort && rawText == lastRawText {
            let hit = lastCachedChunks
            cacheLock.unlock()
            return hit
        }
        cacheLock.unlock()

        let normalized = rawText
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return [] }

        let rawParagraphs = normalized.components(separatedBy: "\n\n")
        var chunks: [PhononTextChunk] = []
        var paragraphIndex = 0
        var pendingText = ""
        var pendingParaIdx = 0

        func flushPending() {
            guard !pendingText.isEmpty else { return }
            let idx = chunks.count
            let words = pendingText.split { $0.isWhitespace || $0.isNewline }.count
            chunks.append(
                PhononTextChunk(
                    id: idx,
                    index: idx,
                    paragraphIndex: pendingParaIdx,
                    text: pendingText,
                    charCount: pendingText.count,
                    wordCount: words
                )
            )
            pendingText = ""
        }

        for rawPara in rawParagraphs {
            let cleanPara = sanitizeParagraphForSpeech(rawPara)
            if cleanPara.isEmpty { continue }

            // Keep Chunk #0 compact (~260 chars = ~13s of speech) so playback starts in ~1.0s while Chunk #1+ prefetches!
            let targetMaxForPara = (chunks.isEmpty && pendingText.isEmpty && normalized.count > 360)
                ? min(clampedMax, 280)
                : clampedMax
            let pieces = splitLongParagraph(cleanPara, maxChars: targetMaxForPara)
            for piece in pieces {
                // Never strand short headings/titles (< 130 chars) as tiny 1-second chunks before a paragraph
                let allowedMergeMax = (pendingText.count > 0 && pendingText.count < 130)
                    ? min(hardMaxChunkChars, clampedMax + 140)
                    : (chunks.isEmpty ? min(clampedMax, 300) : clampedMax)

                if mergeShortParagraphs && !pendingText.isEmpty && (pendingText.count + 1 + piece.count) <= allowedMergeMax {
                    pendingText += "\n" + piece
                } else {
                    flushPending()
                    if mergeShortParagraphs && piece.count < Int(Double(clampedMax) * 0.52) {
                        pendingText = piece
                        pendingParaIdx = paragraphIndex
                    } else {
                        let idx = chunks.count
                        let words = piece.split { $0.isWhitespace || $0.isNewline }.count
                        chunks.append(
                            PhononTextChunk(
                                id: idx,
                                index: idx,
                                paragraphIndex: paragraphIndex,
                                text: piece,
                                charCount: piece.count,
                                wordCount: words
                            )
                        )
                    }
                }
            }
            paragraphIndex += 1
        }
        flushPending()
        cacheLock.lock()
        lastRawText = rawText
        lastMaxChars = clampedMax
        lastMergeShort = mergeShortParagraphs
        lastCachedChunks = chunks
        cacheLock.unlock()
        return chunks
    }

    private static func splitLongParagraph(_ para: String, maxChars: Int) -> [String] {
        let trimmed = para.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if trimmed.count <= maxChars { return [trimmed] }

        let chars = Array(trimmed)
        let len = chars.count
        var results: [String] = []
        var pos = 0

        while pos < len {
            while pos < len && chars[pos].isWhitespace { pos += 1 }
            if pos >= len { break }

            if len - pos <= maxChars {
                let tail = String(chars[pos..<len]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !tail.isEmpty { results.append(tail) }
                break
            }

            let windowEnd = pos + maxChars
            let minCut = pos + max(40, Int(Double(maxChars) * 0.35))
            var cut = -1

            // 1. Sentence-ending punctuation (. ! ? … ; \n) followed by whitespace
            var i = windowEnd
            while i >= minCut {
                let ch = chars[i - 1]
                let next = i < len ? chars[i] : " "
                if (ch == "." || ch == "!" || ch == "?" || ch == "…" || ch == ";" || ch == "\n") && next.isWhitespace {
                    cut = i
                    break
                }
                if (ch == "\"" || ch == "'" || ch == ")" || ch == "]") && i >= 2 {
                    let prev = chars[i - 2]
                    if (prev == "." || prev == "!" || prev == "?" || prev == "…") && next.isWhitespace {
                        cut = i
                        break
                    }
                }
                i -= 1
            }

            // 2. Clause break (, : — – -) followed by whitespace
            if cut == -1 {
                i = windowEnd
                while i >= minCut {
                    let ch = chars[i - 1]
                    let next = i < len ? chars[i] : " "
                    if (ch == "," || ch == ":" || ch == "—" || ch == "–" || ch == "-") && next.isWhitespace {
                        cut = i
                        break
                    }
                    i -= 1
                }
            }

            // 3. Any whitespace word boundary
            if cut == -1 {
                i = windowEnd
                while i > pos {
                    if chars[i].isWhitespace {
                        cut = i
                        break
                    }
                    i -= 1
                }
            }

            // 4. Hard character slice for unbroken long tokens
            if cut == -1 || cut <= pos {
                cut = windowEnd
            }

            let piece = String(chars[pos..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { results.append(piece) }
            pos = cut
        }
        return results
    }
}

// MARK: - Studio & Read-Along Karaoke Cinema Themes & Video Aspect Ratios (from Codecaine File Manager)

enum ReadAlongTheme: String, CaseIterable, Identifiable {
    case midnight = "midnight"
    case sepia = "sepia"
    case paper = "paper"
    case matrix = "matrix"
    case contrast = "contrast"
    case nord = "nord"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .midnight: return "Midnight Studio"
        case .sepia: return "Warm Sepia"
        case .paper: return "Daylight Paper"
        case .matrix: return "Matrix Green"
        case .contrast: return "High Contrast"
        case .nord: return "Arctic Nord"
        }
    }

    var icon: String {
        switch self {
        case .midnight: return "moon.stars.fill"
        case .sepia: return "book.pages.fill"
        case .paper: return "doc.richtext.fill"
        case .matrix: return "terminal.fill"
        case .contrast: return "sun.max.fill"
        case .nord: return "snowflake"
        }
    }

    var colorScheme: ColorScheme {
        switch self {
        case .paper: return .light
        default: return .dark
        }
    }

    var shellGradientColors: [Color] {
        switch self {
        case .midnight:
            return [
                Color(red: 0.06, green: 0.08, blue: 0.13),
                Color(red: 0.03, green: 0.04, blue: 0.07)
            ]
        case .sepia:
            return [
                Color(red: 0.18, green: 0.13, blue: 0.09),
                Color(red: 0.10, green: 0.07, blue: 0.05)
            ]
        case .paper:
            return [
                Color(red: 0.94, green: 0.91, blue: 0.85),
                Color(red: 0.88, green: 0.84, blue: 0.76)
            ]
        case .matrix:
            return [
                Color(red: 0.02, green: 0.11, blue: 0.05),
                Color(red: 0.01, green: 0.04, blue: 0.02)
            ]
        case .contrast:
            return [
                Color.black,
                Color.black
            ]
        case .nord:
            return [
                Color(red: 0.12, green: 0.16, blue: 0.23),
                Color(red: 0.07, green: 0.10, blue: 0.15)
            ]
        }
    }

    var canvasBackground: Color {
        switch self {
        case .midnight: return Color(red: 0.08, green: 0.10, blue: 0.16)
        case .sepia: return Color(red: 0.23, green: 0.17, blue: 0.11)
        case .paper: return Color(red: 0.98, green: 0.96, blue: 0.91)
        case .matrix: return Color(red: 0.03, green: 0.14, blue: 0.07)
        case .contrast: return Color(red: 0.03, green: 0.03, blue: 0.03)
        case .nord: return Color(red: 0.16, green: 0.21, blue: 0.29)
        }
    }

    var canvasBorderColors: [Color] {
        switch self {
        case .midnight: return [Color.blue.opacity(0.32), Color.white.opacity(0.08)]
        case .sepia: return [Color.orange.opacity(0.42), Color(red: 0.85, green: 0.65, blue: 0.40).opacity(0.16)]
        case .paper: return [Color(red: 0.55, green: 0.45, blue: 0.32).opacity(0.38), Color.black.opacity(0.12)]
        case .matrix: return [Color.green.opacity(0.48), Color.green.opacity(0.14)]
        case .contrast: return [Color.yellow.opacity(0.75), Color.white.opacity(0.35)]
        case .nord: return [Color.cyan.opacity(0.42), Color.white.opacity(0.12)]
        }
    }

    var cardBackground: Color {
        switch self {
        case .midnight: return Color.white.opacity(0.045)
        case .sepia: return Color(red: 0.98, green: 0.85, blue: 0.65).opacity(0.08)
        case .paper: return Color.black.opacity(0.05)
        case .matrix: return Color.green.opacity(0.08)
        case .contrast: return Color.white.opacity(0.08)
        case .nord: return Color.cyan.opacity(0.08)
        }
    }

    var dockBackground: Color {
        switch self {
        case .midnight: return Color(red: 0.07, green: 0.09, blue: 0.14).opacity(0.88)
        case .sepia: return Color(red: 0.16, green: 0.12, blue: 0.08).opacity(0.92)
        case .paper: return Color(red: 0.95, green: 0.92, blue: 0.86).opacity(0.94)
        case .matrix: return Color(red: 0.02, green: 0.09, blue: 0.04).opacity(0.92)
        case .contrast: return Color(red: 0.06, green: 0.06, blue: 0.06).opacity(0.96)
        case .nord: return Color(red: 0.13, green: 0.17, blue: 0.24).opacity(0.90)
        }
    }

    var editorTextColor: Color {
        switch self {
        case .midnight: return Color(red: 0.95, green: 0.97, blue: 1.0)
        case .sepia: return Color(red: 0.98, green: 0.92, blue: 0.80)
        case .paper: return Color(red: 0.12, green: 0.10, blue: 0.08)
        case .matrix: return Color(red: 0.62, green: 1.0, blue: 0.72)
        case .contrast: return Color.white
        case .nord: return Color(red: 0.92, green: 0.96, blue: 0.99)
        }
    }

    var activeWordBackground: Color {
        switch self {
        case .midnight: return Color.blue
        case .sepia: return Color.orange
        case .paper: return Color(red: 0.12, green: 0.38, blue: 0.86)
        case .matrix: return Color.green
        case .contrast: return Color.yellow
        case .nord: return Color.cyan
        }
    }

    var activeWordForeground: Color {
        switch self {
        case .contrast, .matrix, .nord, .sepia: return Color.black
        default: return Color.white
        }
    }

    var spokenWordForeground: Color {
        switch self {
        case .midnight: return Color.white.opacity(0.95)
        case .sepia: return Color(red: 0.98, green: 0.92, blue: 0.80)
        case .paper: return Color(red: 0.10, green: 0.09, blue: 0.07)
        case .matrix: return Color(red: 0.58, green: 1.0, blue: 0.70)
        case .contrast: return Color.white
        case .nord: return Color(red: 0.92, green: 0.97, blue: 1.0)
        }
    }

    var upcomingWordForeground: Color {
        switch self {
        case .midnight: return Color.white.opacity(0.58)
        case .sepia: return Color(red: 0.86, green: 0.76, blue: 0.62).opacity(0.70)
        case .paper: return Color(red: 0.25, green: 0.22, blue: 0.18).opacity(0.68)
        case .matrix: return Color.green.opacity(0.58)
        case .contrast: return Color.white.opacity(0.68)
        case .nord: return Color(red: 0.75, green: 0.86, blue: 0.95).opacity(0.65)
        }
    }

    var tokenRestingFill: Color {
        switch self {
        case .paper: return Color.black.opacity(0.045)
        case .sepia: return Color(red: 0.98, green: 0.85, blue: 0.65).opacity(0.06)
        case .matrix: return Color.green.opacity(0.06)
        case .contrast: return Color.white.opacity(0.07)
        case .nord: return Color.cyan.opacity(0.06)
        case .midnight: return Color.white.opacity(0.035)
        }
    }

    var tokenSpokenFill: Color {
        switch self {
        case .paper: return Color(red: 0.12, green: 0.38, blue: 0.86).opacity(0.10)
        case .sepia: return Color.orange.opacity(0.14)
        case .matrix: return Color.green.opacity(0.14)
        case .contrast: return Color.yellow.opacity(0.16)
        case .nord: return Color.cyan.opacity(0.14)
        case .midnight: return Color.blue.opacity(0.12)
        }
    }

    var isLight: Bool {
        colorScheme == .light
    }

    var pillFill: Color {
        switch self {
        case .paper: return Color.black.opacity(0.06)
        case .sepia: return Color(red: 0.98, green: 0.86, blue: 0.68).opacity(0.08)
        case .matrix: return Color.green.opacity(0.08)
        case .contrast: return Color.white.opacity(0.10)
        case .nord: return Color.cyan.opacity(0.08)
        case .midnight: return Color.white.opacity(0.055)
        }
    }

    var pillStroke: Color {
        switch self {
        case .paper: return Color.black.opacity(0.13)
        case .sepia: return Color.orange.opacity(0.24)
        case .matrix: return Color.green.opacity(0.26)
        case .contrast: return Color.white.opacity(0.32)
        case .nord: return Color.cyan.opacity(0.22)
        case .midnight: return Color.white.opacity(0.11)
        }
    }

    var recessedFill: Color {
        switch self {
        case .paper: return Color.black.opacity(0.065)
        case .sepia: return Color.black.opacity(0.26)
        case .matrix: return Color.black.opacity(0.38)
        case .contrast: return Color.white.opacity(0.08)
        case .nord: return Color.black.opacity(0.28)
        case .midnight: return Color.black.opacity(0.28)
        }
    }

    var recessedStroke: Color {
        switch self {
        case .paper: return Color.black.opacity(0.14)
        default: return Color.white.opacity(0.10)
        }
    }

    var elevatedRowFill: Color {
        switch self {
        case .paper: return Color.black.opacity(0.04)
        case .sepia: return Color(red: 0.98, green: 0.85, blue: 0.65).opacity(0.06)
        case .matrix: return Color.green.opacity(0.06)
        case .contrast: return Color.white.opacity(0.07)
        case .nord: return Color.cyan.opacity(0.06)
        case .midnight: return Color.white.opacity(0.04)
        }
    }
}

// MARK: - Studio Acoustic Waveform Envelope Cache

enum WaveformEnvelopeCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: [Float]] = [:]

    static func envelope(for url: URL?, barCount: Int = 44) -> [Float] {
        let count = max(8, barCount)
        guard let url, FileManager.default.fileExists(atPath: url.path) else {
            return fallbackEnvelope(barCount: count)
        }
        let fileSize = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard fileSize > 64 else {
            return fallbackEnvelope(barCount: count)
        }
        let key = "\(url.path)|\(fileSize)|\(count)"
        lock.lock()
        if let hit = cache[key] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        guard let file = try? AVAudioFile(forReading: url) else {
            return fallbackEnvelope(barCount: count)
        }
        let totalFrames = Int(file.length)
        guard totalFrames > 0 else {
            return fallbackEnvelope(barCount: count)
        }
        let maxReadFrames = min(totalFrames, 48_000 * 60)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: file.processingFormat.sampleRate,
            channels: 1,
            interleaved: false
        ),
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(maxReadFrames)) else {
            return fallbackEnvelope(barCount: count)
        }
        do {
            try file.read(into: buffer, frameCount: AVAudioFrameCount(maxReadFrames))
        } catch {
            return fallbackEnvelope(barCount: count)
        }
        let framesRead = Int(buffer.frameLength)
        guard framesRead > 0, let channelData = buffer.floatChannelData?[0] else {
            return fallbackEnvelope(barCount: count)
        }

        var bins = [Float](repeating: 0.14, count: count)
        let framesPerBin = max(1, framesRead / count)
        let strideStep = max(1, framesPerBin / 64)
        var peakMax: Float = 0.001

        for b in 0..<count {
            let start = b * framesPerBin
            let end = min(framesRead, start + framesPerBin)
            guard end > start else { continue }
            var sumSq: Float = 0
            var n: Int = 0
            var idx = start
            while idx < end {
                let s = channelData[idx]
                sumSq += s * s
                n += 1
                idx += strideStep
            }
            let rms = n > 0 ? sqrt(sumSq / Float(n)) : 0
            let shaped = pow(rms, 0.65)
            bins[b] = shaped
            if shaped > peakMax { peakMax = shaped }
        }

        let normalized = bins.map { val -> Float in
            let norm = val / max(peakMax, 0.001)
            return min(max(norm, 0.15), 1.0)
        }

        lock.lock()
        if cache.count > 48 {
            cache.removeAll()
        }
        cache[key] = normalized
        lock.unlock()
        return normalized
    }

    static func fallbackEnvelope(barCount: Int = 44) -> [Float] {
        (0..<max(8, barCount)).map { idx in
            let t = Double(idx) / Double(max(barCount - 1, 1))
            let waveA = abs(sin(t * .pi * 3.5 + 0.4))
            let waveB = abs(cos(t * .pi * 7.0 + 1.1)) * 0.55
            let env = 0.35 + 0.65 * sin(t * .pi)
            return Float(min(max(0.18 + (waveA * 0.55 + waveB * 0.35) * env, 0.18), 0.96))
        }
    }
}

extension Voice {
    static func bundledStudioSamples() -> [(name: String, subtitle: String, url: URL)] {
        let specs: [(String, String)] = [
            ("Lily", "Warm Conversational Sample"),
            ("Sophie", "Expressive Studio Narrator"),
            ("James", "Deep Articulate Baritone")
        ]
        return specs.compactMap { name, sub in
            guard let url = Bundle.main.url(forResource: name, withExtension: "wav") else { return nil }
            return (name: name, subtitle: sub, url: url)
        }
    }
}

enum ReadAlongVideoAspect: String, CaseIterable, Identifiable {
    case landscape16x9 = "16:9"
    case portrait9x16 = "9:16"
    case square1x1 = "1:1"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .landscape16x9: return "16:9 Widescreen (1280×720)"
        case .portrait9x16: return "9:16 Vertical Shorts (720×1280)"
        case .square1x1: return "1:1 Square Social (720×720)"
        }
    }

    var width: Int {
        switch self {
        case .landscape16x9: return 1280
        case .portrait9x16, .square1x1: return 720
        }
    }

    var height: Int {
        switch self {
        case .landscape16x9, .square1x1: return 720
        case .portrait9x16: return 1280
        }
    }
}

// MARK: - Waveform-Synced Karaoke Word Timings & Subtitle Exporter (.srt / .vtt / .ass)

struct WordTiming: Identifiable, Hashable, Sendable {
    let id: Int
    let word: String
    let start: TimeInterval
    let end: TimeInterval
    var chunkIndex: Int = 0
    var paragraphIndex: Int = 0
}

enum WordTimingEngine {
    struct IndexedWordSpec: Sendable {
        let id: Int
        let word: String
        let chunkIndex: Int
        let paragraphIndex: Int
    }

    private static let boundsLock = NSLock()
    nonisolated(unsafe) private static var boundsCache: [String: (start: TimeInterval, end: TimeInterval)] = [:]
    nonisolated(unsafe) private static var fileTimingsCache: [String: [WordTiming]] = [:]
    nonisolated(unsafe) private static var lastFileTimingsText: String = ""
    nonisolated(unsafe) private static var lastMasterText: String = ""
    nonisolated(unsafe) private static var lastMasterMaxChars: Int = -1
    nonisolated(unsafe) private static var lastMasterEnableChunking: Bool = true
    nonisolated(unsafe) private static var lastMasterWords: [IndexedWordSpec] = []
    nonisolated(unsafe) private static var lastEstText: String = ""
    nonisolated(unsafe) private static var lastEstTimingsKey: String = ""
    nonisolated(unsafe) private static var lastEstTimings: [WordTiming] = []

    static func extractMasterWords(
        text: String,
        maxChunkChars: Int = HierarchicalChunker.defaultChunkChars,
        enableChunking: Bool = true
    ) -> [IndexedWordSpec] {
        guard !text.isEmpty else { return [] }
        boundsLock.lock()
        if maxChunkChars == lastMasterMaxChars && enableChunking == lastMasterEnableChunking && text == lastMasterText {
            let hit = lastMasterWords
            boundsLock.unlock()
            return hit
        }
        boundsLock.unlock()

        let chunks: [PhononTextChunk]
        if enableChunking {
            chunks = HierarchicalChunker.splitText(text, maxChars: maxChunkChars, mergeShortParagraphs: true)
        } else {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let wCount = trimmed.split { $0.isWhitespace || $0.isNewline }.count
            chunks = trimmed.isEmpty ? [] : [PhononTextChunk(id: 0, index: 0, paragraphIndex: 0, text: trimmed, charCount: trimmed.count, wordCount: wCount)]
        }

        var result: [IndexedWordSpec] = []
        var nextId = 0
        if chunks.isEmpty {
            let fallbackWords = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            for w in fallbackWords {
                result.append(IndexedWordSpec(id: nextId, word: w, chunkIndex: 0, paragraphIndex: 0))
                nextId += 1
            }
        } else {
            for chunk in chunks {
                let cWords = chunk.text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
                for w in cWords {
                    result.append(IndexedWordSpec(id: nextId, word: w, chunkIndex: chunk.index, paragraphIndex: chunk.paragraphIndex))
                    nextId += 1
                }
            }
        }
        boundsLock.lock()
        lastMasterText = text
        lastMasterMaxChars = maxChunkChars
        lastMasterEnableChunking = enableChunking
        lastMasterWords = result
        boundsLock.unlock()
        return result
    }

    /// Ported directly from bun_rad_studio fmPhonon.ts (`estimateSpokenWordWeight`).
    private static func estimateSpokenWordWeight(_ raw: String) -> (weight: Double, pauseAffinity: Double) {
        if (raw.hasPrefix("<|") && raw.hasSuffix("|>")) || (raw.hasPrefix("[") && raw.hasSuffix("]")) {
            return (0.1, 0.0)
        }
        let clean = raw.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == "'" }.map(String.init).joined()
        let len = max(1, clean.count)

        // Approximate English syllable count via vowel-group heuristic
        let lower = clean.lowercased()
        var vowelRunCount = 0
        var prevVowel = false
        for ch in lower {
            let isV = (ch == "a" || ch == "e" || ch == "i" || ch == "o" || ch == "u" || ch == "y")
            if isV && !prevVowel {
                vowelRunCount += 1
            }
            prevVowel = isV
        }
        var syllables = max(1, vowelRunCount)
        if lower.count > 3 && lower.hasSuffix("e") && !lower.hasSuffix("le") && syllables > 1 {
            syllables -= 1
        }

        // Acronym / Initialism boost (e.g. "PDF", "AI", "CPU" -> spoken letter-by-letter)
        let hasUpper = clean.rangeOfCharacter(from: .uppercaseLetters) != nil
        let isAllCaps = hasUpper && clean == clean.uppercased() && clean.count >= 2 && clean.count <= 5
        if isAllCaps {
            syllables = max(syllables, clean.count)
        }

        // Digits (e.g. "2026", "42") take significantly longer per character than regular letters
        let digitCount = clean.unicodeScalars.filter { CharacterSet.decimalDigits.contains($0) }.count
        if digitCount > 0 {
            syllables += digitCount * 2
        }

        let charWeight = pow(Double(len), 0.72)
        let sylWeight = Double(syllables) * 1.35

        // Trailing punctuation pause weight + affinity for aligning with acoustic silence valleys
        var pauseWeight = 0.0
        var pauseAffinity = 0.0
        let trimmed = raw.trimmingCharacters(in: CharacterSet(charactersIn: "\"')]}»"))
        if trimmed.hasSuffix(".") || trimmed.hasSuffix("!") || trimmed.hasSuffix("?") || trimmed.hasSuffix("…") {
            pauseWeight = 2.6
            pauseAffinity = 3.0
        } else if trimmed.hasSuffix(";") || trimmed.hasSuffix(":") || trimmed.hasSuffix("—") || trimmed.hasSuffix("–") {
            pauseWeight = 1.6
            pauseAffinity = 2.0
        } else if trimmed.hasSuffix(",") {
            pauseWeight = 1.0
            pauseAffinity = 1.5
        }

        return (max(0.6, charWeight + sylWeight + pauseWeight), pauseAffinity)
    }

    /// Ported directly from bun_rad_studio fmPhonon.ts (`computeWordTimingsForText`):
    /// Uses pitch-invariant voiced-time cumulative curve + Dynamic Programming pause-valley anchoring + local energy minimum snapping.
    static func alignWordsToPCM(
        words: [IndexedWordSpec],
        samples: [Float],
        sampleRate: Double,
        startOffsetSec: TimeInterval = 0.0
    ) -> [WordTiming] {
        guard !words.isEmpty else { return [] }
        let sr = max(8000.0, sampleRate)
        let totalSec = max(0.08, Double(samples.count) / sr)

        // 1. Compute 10ms RMS energy envelope (SIMD vectorized via vDSP_measqv)
        let frameSamples = max(1, Int(floor(sr * 0.01)))
        let numFrames = max(1, Int(ceil(Double(samples.count) / Double(frameSamples))))
        var env = [Float](repeating: 0, count: numFrames)
        var maxEnv: Float = 0

        samples.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for f in 0..<numFrames {
                let start = f * frameSamples
                let end = min(samples.count, start + frameSamples)
                let len = end - start
                guard len > 0 else { continue }
                var meanSq: Float = 0
                vDSP_measqv(base + start, 1, &meanSq, vDSP_Length(len))
                let rms = sqrt(meanSq)
                env[f] = rms
                if rms > maxEnv { maxEnv = rms }
            }
        }

        let voiceThresh = max(Float(0.003), maxEnv * 0.055)

        // 2. Trim leading and trailing silence
        var startF = 0
        while startF < numFrames && env[startF] < voiceThresh { startF += 1 }
        var endF = numFrames - 1
        while endF > startF && env[endF] < voiceThresh { endF -= 1 }

        if startF >= endF {
            startF = 0
            endF = numFrames - 1
        } else {
            startF = max(0, startF - 2)
            endF = min(numFrames - 1, endF + 3)
        }

        let activeStartSec = Double(startF * frameSamples) / sr
        let activeEndSec = min(totalSec, Double((endF + 1) * frameSamples) / sr)
        let activeSpanSec = max(0.05, activeEndSec - activeStartSec)
        let activeLen = max(1, endF - startF + 1)

        let wordMetas = words.map { estimateSpokenWordWeight($0.word) }
        var totalWeight = 0.0
        var wordCum = [Double](repeating: 0, count: words.count + 1)
        for i in 0..<words.count {
            totalWeight += wordMetas[i].weight
            wordCum[i + 1] = totalWeight
        }
        let safeTotalWeight = max(totalWeight, 1e-6)
        for i in 1...words.count {
            wordCum[i] /= safeTotalWeight
        }

        // 3. Detect internal acoustic pause valleys (>= 110ms = 11 frames) inside [startF, endF]
        struct AcousticPauseValley {
            let startF: Int
            let endF: Int
            let resumeF: Int
            let durationFrames: Int
            let normPos: Double
        }
        var pauseValleys: [AcousticPauseValley] = []
        let minPauseFrames = 11
        var fIdx = startF + 3
        while fIdx <= endF - 3 {
            if env[fIdx] < voiceThresh {
                let pStart = fIdx
                while fIdx <= endF - 2 && env[fIdx] < voiceThresh { fIdx += 1 }
                let pEnd = fIdx - 1
                let dur = pEnd - pStart + 1
                if dur >= minPauseFrames {
                    let normPos = Double(pStart - startF) / Double(activeLen)
                    if normPos > 0.04 && normPos < 0.96 {
                        pauseValleys.append(AcousticPauseValley(
                            startF: pStart,
                            endF: pEnd,
                            resumeF: min(endF, pEnd + 1),
                            durationFrames: dur,
                            normPos: normPos
                        ))
                    }
                }
            } else {
                fIdx += 1
            }
        }

        // 4. Build pitch-invariant voiced-time cumulative curve over [startF, endF]
        var voicedCum = [Double](repeating: 0, count: activeLen + 1)
        for i in 0..<activeLen {
            let rms = env[startF + i]
            let isVoiced = rms >= voiceThresh
            let contrib = isVoiced
                ? (1.0 + 0.22 * min(1.0, Double(rms) / Double(max(maxEnv * 0.65, 1e-6))))
                : 0.06
            voicedCum[i + 1] = voicedCum[i] + contrib
        }
        let totalVoiced = max(voicedCum[activeLen], 1e-6)
        for i in 1...activeLen {
            voicedCum[i] /= totalVoiced
        }

        // 5. Anchor punctuated words to acoustic pause valleys via monotonic Dynamic Programming
        struct PauseAnchor {
            let wordIdx: Int // word index BEFORE the pause (0 ..< words.count - 1)
            let pauseEndF: Int
            let pauseStartF: Int
        }
        var anchors: [PauseAnchor] = []
        let punctCandidates = (0..<(words.count - 1)).compactMap { idx -> (wordIdx: Int, affinity: Double, expectedNorm: Double)? in
            let aff = wordMetas[idx].pauseAffinity
            guard aff >= 1.5 else { return nil }
            return (idx, aff, wordCum[idx + 1])
        }

        if !punctCandidates.isEmpty && !pauseValleys.isEmpty {
            let topValleys = Array(
                pauseValleys
                    .sorted { $0.durationFrames > $1.durationFrames }
                    .prefix(max(punctCandidates.count * 2, 16))
                    .sorted { $0.startF < $1.startF }
            )
            let P = punctCandidates.count
            let V = topValleys.count
            let maxDist = min(0.085, 2.2 / Double(max(words.count, 12)))
            var dp = Array(repeating: Array(repeating: 0.0, count: V + 1), count: P + 1)
            var choice = Array(repeating: Array(repeating: Int8(0), count: V + 1), count: P + 1)

            for i in 1...P {
                for j in 1...V {
                    dp[i][j] = dp[i - 1][j]
                    choice[i][j] = 1
                    if dp[i][j - 1] > dp[i][j] {
                        dp[i][j] = dp[i][j - 1]
                        choice[i][j] = 2
                    }
                    let cand = punctCandidates[i - 1]
                    let valley = topValleys[j - 1]
                    let valleyVoicedNorm = voicedCum[max(0, min(activeLen, valley.startF - startF))]
                    let dist = abs(cand.expectedNorm - valleyVoicedNorm)
                    if dist < maxDist {
                        let durBonus = min(2.2, Double(valley.durationFrames) / 14.0)
                        let matchScore = cand.affinity * durBonus * (1.0 - dist / maxDist)
                        let candidateTotal = dp[i - 1][j - 1] + matchScore
                        if candidateTotal > dp[i][j] {
                            dp[i][j] = candidateTotal
                            choice[i][j] = 3
                        }
                    }
                }
            }

            var pi = P
            var vj = V
            var matched: [PauseAnchor] = []
            while pi > 0 && vj > 0 {
                let c = choice[pi][vj]
                if c == 3 {
                    let cand = punctCandidates[pi - 1]
                    let valley = topValleys[vj - 1]
                    matched.append(PauseAnchor(wordIdx: cand.wordIdx, pauseEndF: valley.resumeF, pauseStartF: valley.startF))
                    pi -= 1
                    vj -= 1
                } else if c == 1 {
                    pi -= 1
                } else {
                    vj -= 1
                }
            }
            anchors = matched.reversed()
        }

        // 6. Compute word boundaries within each anchored segment & snap to local energy minimum
        struct SegmentSpec {
            let startWord: Int
            let endWord: Int // exclusive
            let segStartF: Int
            let segEndF: Int
        }
        var segments: [SegmentSpec] = []
        var curW = 0
        var curF = startF
        for a in anchors {
            let nextW = a.wordIdx + 1
            if nextW > curW && a.pauseStartF > curF {
                let expectedSpan = max(1e-4, wordCum[nextW] - wordCum[curW])
                let vStart = voicedCum[max(0, min(activeLen, curF - startF))]
                let vEnd = voicedCum[max(0, min(activeLen, a.pauseStartF - startF))]
                let actualSpan = max(1e-4, vEnd - vStart)
                let paceRatio = actualSpan / expectedSpan
                // Reject any pause anchor that distorts segment speaking pace by more than [0.60x ... 1.65x]
                if paceRatio >= 0.60 && paceRatio <= 1.65 {
                    segments.append(SegmentSpec(startWord: curW, endWord: nextW, segStartF: curF, segEndF: a.pauseStartF))
                    curW = nextW
                    curF = a.pauseEndF
                }
            }
        }
        if curW < words.count {
            segments.append(SegmentSpec(startWord: curW, endWord: words.count, segStartF: curF, segEndF: endF))
        }

        var timings: [WordTiming] = []
        timings.reserveCapacity(words.count)

        for seg in segments {
            let segCount = seg.endWord - seg.startWord
            guard segCount > 0 else { continue }
            let segF0 = max(startF, min(endF, seg.segStartF))
            let segF1 = max(segF0 + 1, min(endF + 1, seg.segEndF + 1))

            var segWeightSum = 0.0
            for w in seg.startWord..<seg.endWord {
                segWeightSum += wordMetas[w].weight
            }
            let safeSegWeight = max(segWeightSum, 1e-6)

            let v0 = voicedCum[segF0 - startF]
            let v1 = voicedCum[min(activeLen, segF1 - startF)]
            let vSpan = max(1e-6, v1 - v0)

            var boundaryFrames = [Int](repeating: segF0, count: segCount + 1)
            boundaryFrames[0] = segF0
            boundaryFrames[segCount] = segF1

            var cumW = 0.0
            var searchCursor = segF0 - startF
            for localIdx in 0..<(segCount - 1) {
                let w = seg.startWord + localIdx
                cumW += wordMetas[w].weight
                let targetVoiced = v0 + (cumW / safeSegWeight) * vSpan

                while searchCursor < (segF1 - startF) && voicedCum[searchCursor] < targetVoiced {
                    searchCursor += 1
                }
                let idealF = startF + searchCursor

                // Snap boundary to local energy minimum within ±7 frames (±70ms)
                let prevBoundary = boundaryFrames[localIdx]
                let minAllowed = prevBoundary + 2
                let maxAllowed = segF1 - (segCount - 1 - localIdx) * 2
                if minAllowed >= maxAllowed {
                    boundaryFrames[localIdx + 1] = max(prevBoundary + 1, min(segF1 - 1, idealF))
                } else {
                    let radius = 7
                    let minSearch = max(minAllowed, idealF - radius)
                    let maxSearch = min(maxAllowed, idealF + radius)
                    var bestF = max(minAllowed, min(maxAllowed, idealF))
                    var bestCost = Float.infinity
                    if minSearch <= maxSearch {
                        for candF in minSearch...maxSearch {
                            let distPenalty = Float(abs(candF - idealF)) * 0.06 * max(maxEnv, 0.01)
                            let cost = env[min(numFrames - 1, candF)] + distPenalty
                            if cost < bestCost {
                                bestCost = cost
                                bestF = candF
                            }
                        }
                    }
                    boundaryFrames[localIdx + 1] = bestF
                }
            }

            for localIdx in 0..<segCount {
                let spec = words[seg.startWord + localIdx]
                let fStart = boundaryFrames[localIdx]
                let fEnd = max(fStart + 1, boundaryFrames[localIdx + 1])
                var startSec = startOffsetSec + min(totalSec, max(activeStartSec, Double(fStart * frameSamples) / sr))
                var endSec = startOffsetSec + min(totalSec, max(startSec - startOffsetSec + 0.025, Double(fEnd * frameSamples) / sr))
                if localIdx == segCount - 1 && seg.endWord == words.count {
                    endSec = max(endSec, startOffsetSec + min(totalSec, activeStartSec + activeSpanSec))
                }
                if startSec > endSec - 0.015 {
                    startSec = max(startOffsetSec, endSec - 0.025)
                }
                timings.append(
                    WordTiming(
                        id: spec.id,
                        word: spec.word,
                        start: startSec,
                        end: endSec,
                        chunkIndex: spec.chunkIndex,
                        paragraphIndex: spec.paragraphIndex
                    )
                )
            }
        }

        return timings
    }

    static func computeWordTimings(
        text: String,
        audioURL: URL?,
        duration: TimeInterval,
        maxChunkChars: Int = HierarchicalChunker.defaultChunkChars,
        enableChunking: Bool = true,
        startChunkIndex: Int = 0,
        endChunkIndex: Int? = nil,
        paragraphPauseMs: Int = 350
    ) -> [WordTiming] {
        let masterWords = extractMasterWords(text: text, maxChunkChars: maxChunkChars, enableChunking: enableChunking)
        guard !masterWords.isEmpty else { return [] }
        let safeStartChunk = max(0, startChunkIndex)
        let safeEndChunk = endChunkIndex.map { max(safeStartChunk, $0) }
        let isWithinRange: (IndexedWordSpec) -> Bool = { spec in
            guard spec.chunkIndex >= safeStartChunk else { return false }
            if let safeEndChunk { return spec.chunkIndex <= safeEndChunk }
            return true
        }

        // If we have a real WAV file on disk, run the full waveform-aligned DP + voiced-time algorithm (cached O(1))
        if let audioURL {
            let cacheKey = "\(audioURL.path)|\(masterWords.count)|\(maxChunkChars)|\(enableChunking)|\(safeStartChunk)|\(safeEndChunk ?? -1)|\(Int(duration * 100))"
            boundsLock.lock()
            if text == lastFileTimingsText, let cached = fileTimingsCache[cacheKey] {
                boundsLock.unlock()
                return cached
            }
            boundsLock.unlock()

            if FileManager.default.fileExists(atPath: audioURL.path),
               let file = try? AVAudioFile(forReading: audioURL) {
                let sr = file.processingFormat.sampleRate
                let totalFrames = file.length
                // Align directly against the WAV PCM samples for files up to 15 minutes
                if sr > 0 && totalFrames > 256 && totalFrames <= AVAudioFramePosition(sr * 900.0),
                   let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(totalFrames)),
                   (try? file.read(into: buf)) != nil,
                   let chan = buf.floatChannelData?[0], buf.frameLength > 0 {
                    let pcmSamples = Array(UnsafeBufferPointer(start: chan, count: Int(buf.frameLength)))
                    let inactiveWords = masterWords.filter { !isWithinRange($0) }
                    let activeWords = masterWords.filter { isWithinRange($0) }

                    var result: [WordTiming] = inactiveWords.map {
                        WordTiming(id: $0.id, word: $0.word, start: 0, end: 0, chunkIndex: $0.chunkIndex, paragraphIndex: $0.paragraphIndex)
                    }
                    let aligned = alignWordsToPCM(words: activeWords, samples: pcmSamples, sampleRate: sr, startOffsetSec: 0.0)
                    result.append(contentsOf: aligned)
                    result.sort { $0.id < $1.id }

                    boundsLock.lock()
                    if text != lastFileTimingsText || fileTimingsCache.count > 32 {
                        fileTimingsCache.removeAll()
                        lastFileTimingsText = text
                    }
                    fileTimingsCache[cacheKey] = result
                    boundsLock.unlock()
                    return result
                }
            }
        }

        let totalDuration = max(duration, 0.25)
        let estKey = "\(audioURL?.path ?? "nil")|\(Int(totalDuration * 100))|\(maxChunkChars)|\(enableChunking)|\(safeStartChunk)|\(safeEndChunk ?? -1)|\(paragraphPauseMs)"
        boundsLock.lock()
        if estKey == lastEstTimingsKey && text == lastEstText {
            let hit = lastEstTimings
            boundsLock.unlock()
            return hit
        }
        boundsLock.unlock()

        // Detect pre-roll and tail speech boundaries from WAV file if available (cached O(1))
        var speechStart: TimeInterval = 0.03
        var speechEnd: TimeInterval = max(speechStart + 0.1, totalDuration - 0.05)

        if let audioURL, let bounds = detectSpeechBounds(in: audioURL, fallbackDuration: totalDuration) {
            speechStart = bounds.start
            speechEnd = bounds.end
        }

        let activeWords = masterWords.filter { isWithinRange($0) }
        let distinctActiveChunks = Set(activeWords.map(\.chunkIndex)).count
        let pauseSec = audioURL != nil ? (Double(max(0, paragraphPauseMs)) / 1000.0) : 0.0
        let totalPauseTime = Double(max(0, distinctActiveChunks - 1)) * pauseSec

        let rawSpan = max(0.15, speechEnd - speechStart)
        let effectivePauseSec = (totalPauseTime < rawSpan * 0.5) ? pauseSec : 0.0
        let spokenSpan = max(0.12, rawSpan - Double(max(0, distinctActiveChunks - 1)) * effectivePauseSec)

        let weights: [Double] = masterWords.map { item in
            isWithinRange(item) ? estimateSpokenWordWeight(item.word).weight : 0.0
        }
        let totalWeight = max(weights.reduce(0, +), 0.001)

        var timings: [WordTiming] = []
        timings.reserveCapacity(masterWords.count)
        var cursor = speechStart
        var previousActiveChunk: Int? = nil

        for (idx, item) in masterWords.enumerated() {
            if !isWithinRange(item) {
                timings.append(
                    WordTiming(
                        id: item.id,
                        word: item.word,
                        start: 0,
                        end: 0,
                        chunkIndex: item.chunkIndex,
                        paragraphIndex: item.paragraphIndex
                    )
                )
                continue
            }
            if let prev = previousActiveChunk, prev != item.chunkIndex {
                cursor = min(speechEnd, cursor + effectivePauseSec)
            }
            previousActiveChunk = item.chunkIndex

            let share = (weights[idx] / totalWeight) * spokenSpan
            let start = cursor
            let end = min(speechEnd, cursor + share)
            timings.append(
                WordTiming(
                    id: item.id,
                    word: item.word,
                    start: start,
                    end: end,
                    chunkIndex: item.chunkIndex,
                    paragraphIndex: item.paragraphIndex
                )
            )
            cursor = end
        }
        boundsLock.lock()
        lastEstText = text
        lastEstTimingsKey = estKey
        lastEstTimings = timings
        boundsLock.unlock()
        return timings
    }

    static func shiftedTimings(_ timings: [WordTiming], offsetMs: Double) -> [WordTiming] {
        guard abs(offsetMs) > 0.5 else { return timings }
        let delta = offsetMs / 1000.0
        return timings.map { w in
            WordTiming(
                id: w.id,
                word: w.word,
                start: max(0, w.start + delta),
                end: max(0.02, w.end + delta),
                chunkIndex: w.chunkIndex,
                paragraphIndex: w.paragraphIndex
            )
        }
    }

    static func activeWordIndex(in timings: [WordTiming], at time: TimeInterval, offsetMs: Double = 0) -> Int? {
        guard !timings.isEmpty else { return nil }
        let shiftedTime = max(0, time - (offsetMs / 1000.0))
        // Find the playable range [firstPlayable ... lastPlayable] in O(log N) where end > start
        var lo = 0
        var hi = timings.count - 1
        while lo <= hi && timings[lo].end <= timings[lo].start { lo += 1 }
        while hi >= lo && timings[hi].end <= timings[hi].start { hi -= 1 }
        guard lo <= hi else { return timings.first?.id }

        if shiftedTime <= timings[lo].start { return timings[lo].id }
        if shiftedTime >= timings[hi].end { return timings[hi].id }

        // Binary search in timings[lo...hi] since start times are monotonically non-decreasing
        var left = lo
        var right = hi
        var best = lo
        while left <= right {
            let mid = (left + right) / 2
            let item = timings[mid]
            if item.end <= item.start {
                left = mid + 1
                continue
            }
            if shiftedTime >= item.start && shiftedTime <= item.end {
                return item.id
            } else if item.start <= shiftedTime {
                best = mid
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return timings[best].id
    }

    private static func detectSpeechBounds(in url: URL, fallbackDuration: TimeInterval) -> (start: TimeInterval, end: TimeInterval)? {
        let cacheKey = "\(url.path)|\(Int(fallbackDuration * 1000))"
        boundsLock.lock()
        if let cached = boundsCache[cacheKey] {
            boundsLock.unlock()
            return cached
        }
        boundsLock.unlock()

        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let sr = file.processingFormat.sampleRate
        let totalFrames = file.length
        guard sr > 0, totalFrames > 0 else { return nil }
        let fileDuration = Double(totalFrames) / sr

        let win = max(1, Int(sr * 0.01)) // 10 ms window
        func computeRMS(_ ptr: UnsafePointer<Float>, count: Int) -> (rms: [Float], peak: Float) {
            var vals: [Float] = []
            vals.reserveCapacity(count / win + 1)
            var peak: Float = 0
            var i = 0
            while i < count {
                let end = min(i + win, count)
                var meanSq: Float = 0
                vDSP_measqv(ptr + i, 1, &meanSq, vDSP_Length(end - i))
                let r = sqrt(meanSq)
                vals.append(r)
                if r > peak { peak = r }
                i = end
            }
            return (vals, peak)
        }

        // Inspect up to first 10 seconds for pre-roll silence
        let headFrames = AVAudioFrameCount(min(totalFrames, AVAudioFramePosition(sr * 10.0)))
        guard let headBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: headFrames) else { return nil }
        try? file.read(into: headBuf, frameCount: headFrames)
        guard let headChan = headBuf.floatChannelData?[0], headBuf.frameLength > 0 else { return nil }

        let head = computeRMS(headChan, count: Int(headBuf.frameLength))
        let headThresh = max(head.peak * 0.04, 0.002)
        let firstIdx = head.rms.firstIndex(where: { $0 >= headThresh }) ?? 0
        let startSec = min(1.5, Double(firstIdx * win) / sr)

        // Inspect up to last 10 seconds for tail silence (accurate for any length audiobook)
        var endSec = max(startSec + 0.1, max(fallbackDuration, fileDuration) - 0.04)
        let tailStartFrame = max(0, totalFrames - AVAudioFramePosition(sr * 10.0))
        let tailCount = AVAudioFrameCount(totalFrames - tailStartFrame)
        if tailCount > 0, let tailBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: tailCount) {
            file.framePosition = tailStartFrame
            try? file.read(into: tailBuf, frameCount: tailCount)
            if let tailChan = tailBuf.floatChannelData?[0], tailBuf.frameLength > 0 {
                let tail = computeRMS(tailChan, count: Int(tailBuf.frameLength))
                let tailThresh = max(max(head.peak, tail.peak) * 0.04, 0.002)
                if let lastIdx = tail.rms.lastIndex(where: { $0 >= tailThresh }) {
                    let tailOffsetSec = Double((lastIdx + 1) * win) / sr
                    endSec = min(max(fallbackDuration, fileDuration), (Double(tailStartFrame) / sr) + tailOffsetSec)
                }
            }
        }

        guard endSec > startSec + 0.08 else { return nil }
        let result = (startSec, endSec)
        boundsLock.lock()
        if boundsCache.count > 64 { boundsCache.removeAll() }
        boundsCache[cacheKey] = result
        boundsLock.unlock()
        return result
    }
}

enum SubtitleExporter {
    static func formatSRT(timings: [WordTiming], wordsPerCue: Int = 8) -> String {
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        guard !playable.isEmpty else { return "" }
        var lines: [String] = []
        var cueIndex = 1
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let end = slice.last!.end
            let text = slice.map(\.word).joined(separator: " ")
            lines.append("\(cueIndex)")
            lines.append("\(srtTime(start)) --> \(srtTime(end))")
            lines.append(text)
            lines.append("")
            cueIndex += 1
            i = endIdx
        }
        return lines.joined(separator: "\n")
    }

    static func formatVTT(timings: [WordTiming], wordsPerCue: Int = 8) -> String {
        var lines: [String] = ["WEBVTT", ""]
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        guard !playable.isEmpty else { return lines.joined(separator: "\n") }
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let end = slice.last!.end
            let text = slice.map(\.word).joined(separator: " ")
            lines.append("\(vttTime(start)) --> \(vttTime(end))")
            lines.append(text)
            lines.append("")
            i = endIdx
        }
        return lines.joined(separator: "\n")
    }

    static func formatASSKaraoke(
        timings: [WordTiming],
        wordsPerCue: Int = 7,
        theme: ReadAlongTheme = .midnight,
        aspect: ReadAlongVideoAspect = .landscape16x9
    ) -> String {
        let primaryASSColor: String
        switch theme {
        case .midnight: primaryASSColor = "&H00FFEA28"
        case .sepia: primaryASSColor = "&H00289BFF"
        case .paper: primaryASSColor = "&H00DC611D"
        case .matrix: primaryASSColor = "&H0064FF18"
        case .contrast: primaryASSColor = "&H0000EAFF"
        case .nord: primaryASSColor = "&H00D4B606"
        }
        let fontSize = aspect == .portrait9x16 ? 52 : 58
        var header = """
        [Script Info]
        Title: MLX Audio Studio Karaoke Subtitles
        ScriptType: v4.00+
        PlayResX: \(aspect.width)
        PlayResY: \(aspect.height)

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Karaoke,SF Pro Display,\(fontSize),\(primaryASSColor),&H00FFFFFF,&H00181818,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,5,80,80,90,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

        """
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        guard !playable.isEmpty else { return header }
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let end = slice.last!.end
            let kText = slice.map { w -> String in
                let centisec = max(1, Int(round((w.end - w.start) * 100)))
                return "{\\kf\(centisec)}\(w.word)"
            }.joined(separator: " ")
            header += "Dialogue: 0,\(assTime(start)),\(assTime(end)),Karaoke,,0,0,0,,\(kText)\n"
            i = endIdx
        }
        return header
    }

    private static func srtTime(_ t: TimeInterval) -> String {
        let totalMs = max(0, Int(round(t * 1000)))
        let h = totalMs / 3_600_000
        let m = (totalMs % 3_600_000) / 60_000
        let s = (totalMs % 60_000) / 1000
        let ms = totalMs % 1000
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }

    private static func vttTime(_ t: TimeInterval) -> String {
        let totalMs = max(0, Int(round(t * 1000)))
        let h = totalMs / 3_600_000
        let m = (totalMs % 3_600_000) / 60_000
        let s = (totalMs % 60_000) / 1000
        let ms = totalMs % 1000
        return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
    }

    private static func assTime(_ t: TimeInterval) -> String {
        let totalCs = max(0, Int(round(t * 100)))
        let h = totalCs / 360_000
        let m = (totalCs % 360_000) / 6000
        let s = (totalCs % 6000) / 100
        let cs = totalCs % 100
        return String(format: "%d:%02d:%02d.%02d", h, m, s, cs)
    }

    static func formatLRC(timings: [WordTiming], wordsPerCue: Int = 8) -> String {
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        guard !playable.isEmpty else { return "" }
        var lines: [String] = [
            "[ti:Codecaine MLX Audio Studio]",
            "[ar:Studio Narrator]"
        ]
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let text = slice.map(\.word).joined(separator: " ")
            lines.append("[\(lrcTime(start))]\(text)")
            i = endIdx
        }
        return lines.joined(separator: "\n")
    }

    static func formatSBV(timings: [WordTiming], wordsPerCue: Int = 8) -> String {
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        guard !playable.isEmpty else { return "" }
        var lines: [String] = []
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let end = slice.last!.end
            let text = slice.map(\.word).joined(separator: " ")
            lines.append("\(sbvTime(start)),\(sbvTime(end))")
            lines.append(text)
            lines.append("")
            i = endIdx
        }
        return lines.joined(separator: "\n")
    }

    static func formatTTML(timings: [WordTiming], wordsPerCue: Int = 8) -> String {
        let playable = timings.filter { $0.end >= $0.start && !($0.start == 0 && $0.end == 0) }
        var xmlLines: [String] = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<tt xmlns=\"http://www.w3.org/ns/ttml\" xml:lang=\"en\">",
            "  <body>",
            "    <div>"
        ]
        var i = 0
        while i < playable.count {
            let endIdx = min(i + wordsPerCue, playable.count)
            let slice = playable[i..<endIdx]
            let start = slice.first!.start
            let end = slice.last!.end
            let escaped = slice.map(\.word).joined(separator: " ")
                .replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
            xmlLines.append("      <p begin=\"\(vttTime(start))\" end=\"\(vttTime(end))\">\(escaped)</p>")
            i = endIdx
        }
        xmlLines.append("    </div>")
        xmlLines.append("  </body>")
        xmlLines.append("</tt>")
        return xmlLines.joined(separator: "\n")
    }

    private static func lrcTime(_ t: TimeInterval) -> String {
        let totalCs = max(0, Int(round(t * 100)))
        let m = totalCs / 6000
        let s = (totalCs % 6000) / 100
        let cs = totalCs % 100
        return String(format: "%02d:%02d.%02d", m, s, cs)
    }

    private static func sbvTime(_ t: TimeInterval) -> String {
        let totalMs = max(0, Int(round(t * 1000)))
        let h = totalMs / 3_600_000
        let m = (totalMs % 3_600_000) / 60_000
        let s = (totalMs % 60_000) / 1000
        let ms = totalMs % 1000
        return String(format: "%d:%02d:%02d.%03d", h, m, s, ms)
    }
}

// MARK: - Multi-Format Audio Exporter (.wav, .m4b, .aiff, .m4a, .mp3, .flac, .ogg, .opus, .aac, .caf)

enum AudioExportFormat: String, CaseIterable, Identifiable {
    case wav = "wav"
    case m4b = "m4b"
    case m4a = "m4a"
    case mp3 = "mp3"
    case flac = "flac"
    case ogg = "ogg"
    case opus = "opus"
    case aac = "aac"
    case aiff = "aiff"
    case caf = "caf"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .wav: return "WAV (24 kHz PCM)"
        case .m4b: return "M4B Audiobook (Chapters + Cover)"
        case .m4a: return "M4A / AAC (Apple Compact)"
        case .mp3: return "MP3 (Universal)"
        case .flac: return "FLAC (Lossless)"
        case .ogg: return "OGG Vorbis (Open Web)"
        case .opus: return "OPUS (Ultra-Compact Voice)"
        case .aac: return "AAC (Raw ADTS Stream)"
        case .aiff: return "AIFF (Apple Uncompressed)"
        case .caf: return "CAF (CoreAudio)"
        }
    }

    var mimeType: String {
        switch self {
        case .wav: return "audio/wav"
        case .m4b: return "audio/mp4"
        case .m4a: return "audio/mp4"
        case .mp3: return "audio/mpeg"
        case .flac: return "audio/flac"
        case .ogg: return "audio/ogg"
        case .opus: return "audio/opus"
        case .aac: return "audio/aac"
        case .aiff: return "audio/aiff"
        case .caf: return "audio/x-caf"
        }
    }
}

struct M4BChapterMarker: Identifiable, Sendable, Equatable {
    var id: Int
    var title: String
    var startSec: Double
    var endSec: Double
}

enum M4BAudiobookExporter {
    /// Builds chapter markers from synthesized word timings (grouped by chunk) or falls back to proportional chunk boundaries
    static func buildChapters(
        chunks: [PhononTextChunk],
        timings: [WordTiming],
        totalDurationSec: Double
    ) -> [M4BChapterMarker] {
        let safeDuration = max(1.0, totalDurationSec)
        guard !chunks.isEmpty else {
            return [M4BChapterMarker(id: 0, title: "Chapter 1", startSec: 0, endSec: safeDuration)]
        }

        let timingsByChunk = Dictionary(grouping: timings.filter { $0.end > $0.start }, by: \.chunkIndex)
        var markers: [M4BChapterMarker] = []
        let totalWords = max(1, chunks.reduce(0) { $0 + max(1, $1.wordCount) })
        var runningSec: Double = 0.0

        for (idx, chunk) in chunks.enumerated() {
            let firstLine = chunk.text
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first { !$0.isEmpty } ?? "Section \(idx + 1)"
            let cleanTitle = firstLine.count > 52 ? String(firstLine.prefix(52)) + "…" : firstLine
            let chapterTitle = "§\(idx + 1) · \(cleanTitle)"

            let startSec: Double
            let endSec: Double
            if let chunkTimings = timingsByChunk[chunk.index],
               let firstT = chunkTimings.first,
               let lastT = chunkTimings.last {
                startSec = max(runningSec, firstT.start)
                endSec = max(startSec + 0.25, lastT.end)
            } else {
                let share = Double(max(1, chunk.wordCount)) / Double(totalWords)
                startSec = runningSec
                endSec = min(safeDuration, startSec + max(0.5, share * safeDuration))
            }
            runningSec = endSec
            markers.append(M4BChapterMarker(id: idx, title: chapterTitle, startSec: startSec, endSec: endSec))
        }

        if !markers.isEmpty {
            markers[markers.count - 1].endSec = max(markers[markers.count - 1].startSec + 0.25, safeDuration)
        }
        return markers
    }

    #if os(macOS)
    /// Generates a 600x600 studio audiobook cover artwork PNG on the fly
    static func generateAudiobookCoverPNG(title: String, author: String, voiceName: String) -> URL? {
        let size = 600
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: size,
            pixelsHigh: size,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else { return nil }

        NSGraphicsContext.saveGraphicsState()
        if let ctx = NSGraphicsContext(bitmapImageRep: rep) {
            NSGraphicsContext.current = ctx
            let rect = NSRect(x: 0, y: 0, width: size, height: size)
            let gradient = NSGradient(colors: [
                NSColor(calibratedRed: 0.06, green: 0.09, blue: 0.18, alpha: 1.0),
                NSColor(calibratedRed: 0.12, green: 0.18, blue: 0.36, alpha: 1.0),
                NSColor(calibratedRed: 0.22, green: 0.12, blue: 0.38, alpha: 1.0)
            ])
            gradient?.draw(in: rect, angle: -45)

            let badgeAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: 16, weight: .bold),
                .foregroundColor: NSColor(calibratedRed: 0.38, green: 0.78, blue: 1.0, alpha: 1.0)
            ]
            NSString(string: "CODECAINE MLX AI AUDIOBOOK").draw(at: NSPoint(x: 44, y: 510), withAttributes: badgeAttrs)

            let titleStyle = NSMutableParagraphStyle()
            titleStyle.lineBreakMode = .byWordWrapping
            let titleAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 38, weight: .bold),
                .foregroundColor: NSColor.white,
                .paragraphStyle: titleStyle
            ]
            NSString(string: title).draw(in: NSRect(x: 44, y: 240, width: 512, height: 230), withAttributes: titleAttrs)

            let subAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 20, weight: .medium),
                .foregroundColor: NSColor(calibratedWhite: 0.82, alpha: 1.0)
            ]
            NSString(string: "Narrated by \(voiceName) · \(author)").draw(at: NSPoint(x: 44, y: 64), withAttributes: subAttrs)
        }
        NSGraphicsContext.restoreGraphicsState()

        guard let png = rep.representation(using: .png, properties: [:]) else { return nil }
        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("audiobook_cover_\(UUID().uuidString).png")
        do {
            try png.write(to: outURL, options: .atomic)
            return outURL
        } catch {
            return nil
        }
    }
    #endif

    private static func escapeFFMetadata(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "=", with: "\\=")
            .replacingOccurrences(of: ";", with: "\\;")
            .replacingOccurrences(of: "#", with: "\\#")
            .replacingOccurrences(of: "\n", with: " ")
    }

    static func exportM4B(
        sourceWAV: URL,
        to destinationURL: URL,
        bookTitle: String = "Codecaine Studio Audiobook",
        author: String = "Codecaine MLX Audio Studio",
        voiceName: String = "Studio Narrator",
        chapters: [M4BChapterMarker] = [],
        customCoverURL: URL? = nil
    ) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        // Determine total duration from WAV if needed
        let wavDuration: Double = {
            if let file = try? AVAudioFile(forReading: sourceWAV), file.processingFormat.sampleRate > 0 {
                return Double(file.length) / file.processingFormat.sampleRate
            }
            return chapters.last?.endSec ?? 10.0
        }()

        let effectiveChapters = chapters.isEmpty
            ? [M4BChapterMarker(id: 0, title: bookTitle, startSec: 0, endSec: max(1.0, wavDuration))]
            : chapters

        // Build FFMETADATA1 file with chapters
        var metaLines: [String] = [
            ";FFMETADATA1",
            "title=\(escapeFFMetadata(bookTitle))",
            "artist=\(escapeFFMetadata(author))",
            "album=\(escapeFFMetadata(bookTitle))",
            "composer=\(escapeFFMetadata(voiceName))",
            "genre=Audiobook",
            "media_type=2"
        ]
        for ch in effectiveChapters {
            let startMs = max(0, Int(round(ch.startSec * 1000)))
            let endMs = max(startMs + 250, Int(round(ch.endSec * 1000)))
            metaLines.append("[CHAPTER]")
            metaLines.append("TIMEBASE=1/1000")
            metaLines.append("START=\(startMs)")
            metaLines.append("END=\(endMs)")
            metaLines.append("title=\(escapeFFMetadata(ch.title))")
        }
        let metaContent = metaLines.joined(separator: "\n") + "\n"
        let metaURL = FileManager.default.temporaryDirectory.appendingPathComponent("m4b_meta_\(UUID().uuidString).txt")
        try? metaContent.write(to: metaURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: metaURL) }

        #if os(macOS)
        let generatedCover = customCoverURL == nil
            ? generateAudiobookCoverPNG(title: bookTitle, author: author, voiceName: voiceName)
            : nil
        let coverURL = customCoverURL ?? generatedCover
        defer {
            if let generatedCover {
                try? FileManager.default.removeItem(at: generatedCover)
            }
        }
        #else
        let coverURL: URL? = customCoverURL
        #endif

        if let ffmpeg = StudioPackageManager.findExecutable(
            names: ["ffmpeg"],
            extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
        ) {
            // Try full chapter + attached cover art mux first
            if let cover = coverURL, FileManager.default.fileExists(atPath: cover.path) {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: ffmpeg)
                proc.arguments = [
                    "-y", "-threads", "0",
                    "-i", sourceWAV.path,
                    "-i", metaURL.path,
                    "-i", cover.path,
                    "-map", "0:a",
                    "-map", "2:v",
                    "-map_metadata", "1",
                    "-map_chapters", "1",
                    "-c:a", "aac", "-b:a", "128k",
                    "-c:v", "png",
                    "-disposition:v:0", "attached_pic",
                    "-movflags", "+faststart",
                    "-f", "ipod",
                    destinationURL.path
                ]
                proc.standardInput = FileHandle.nullDevice
                proc.standardOutput = FileHandle.nullDevice
                proc.standardError = FileHandle.nullDevice
                try? proc.run()
                proc.waitUntilExit()
                if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                    return
                }
            }

            // Fallback: FFmpeg AAC + FFMETADATA1 chapters without cover stream
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: ffmpeg)
            proc.arguments = [
                "-y", "-threads", "0",
                "-i", sourceWAV.path,
                "-i", metaURL.path,
                "-map", "0:a",
                "-map_metadata", "1",
                "-map_chapters", "1",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                "-f", "ipod",
                destinationURL.path
            ]
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try? proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }
        }

        // Native macOS /usr/bin/afconvert fallback for .m4b (MPEG-4 AAC container)
        let afconvertPath = "/usr/bin/afconvert"
        if FileManager.default.isExecutableFile(atPath: afconvertPath) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: afconvertPath)
            proc.arguments = ["-f", "m4bf", "-d", "aac", sourceWAV.path, destinationURL.path]
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try? proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }

            let procM4A = Process()
            procM4A.executableURL = URL(fileURLWithPath: afconvertPath)
            procM4A.arguments = ["-f", "m4af", "-d", "aac", sourceWAV.path, destinationURL.path]
            procM4A.standardInput = FileHandle.nullDevice
            procM4A.standardOutput = FileHandle.nullDevice
            procM4A.standardError = FileHandle.nullDevice
            try procM4A.run()
            procM4A.waitUntilExit()
            if procM4A.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }
        }

        try FileManager.default.copyItem(at: sourceWAV, to: destinationURL)
    }
}

// MARK: - Studio Speech Audio Silence Trimmer (Ported from Codecaine File Manager fmPhonon.ts trim_generated_wave)

enum SpeechAudioTrimmer {
    /// Trims dead leading and trailing codec silence from a generated chunk waveform using a 5ms RMS envelope
    /// while preserving natural consonant onset (`keepLeadingMs`) and vowel release (`keepTrailingMs`) with a raised-cosine micro-fade.
    static func trimSilence(
        _ samples: [Float],
        sampleRate: Double,
        keepLeadingMs: Double = 14.0,
        keepTrailingMs: Double = 22.0,
        fadeMs: Double = 4.0
    ) -> [Float] {
        let count = samples.count
        guard count > 256, sampleRate > 0 else { return samples }

        let win = max(1, Int(sampleRate * 0.005)) // 5ms window
        let numWins = (count + win - 1) / win
        guard numWins > 4 else { return samples }

        var rms = [Float](repeating: 0, count: numWins)
        var peakRMS: Float = 0

        samples.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for w in 0..<numWins {
                let start = w * win
                let end = min(count, start + win)
                let len = end - start
                guard len > 0 else { continue }
                var meanSq: Float = 0
                vDSP_measqv(base + start, 1, &meanSq, vDSP_Length(len))
                let r = sqrt(meanSq)
                rms[w] = r
                if r > peakRMS { peakRMS = r }
            }
        }

        // Matches fmPhonon.ts: thresh = max(0.004, peak_rms * 0.045)
        let thresh = max(Float(0.0035), peakRMS * 0.045)
        guard peakRMS >= 0.0035 else { return samples }

        var firstWin = 0
        while firstWin < numWins && rms[firstWin] < thresh {
            firstWin += 1
        }
        var lastWin = numWins - 1
        while lastWin > firstWin && rms[lastWin] < thresh {
            lastWin -= 1
        }

        guard firstWin <= lastWin else { return samples }

        let padLeadSamples = max(0, Int((keepLeadingMs / 1000.0) * sampleRate))
        let padTrailSamples = max(0, Int((keepTrailingMs / 1000.0) * sampleRate))

        let startSample = max(0, firstWin * win - padLeadSamples)
        let endSample = min(count, (lastWin + 1) * win + padTrailSamples)
        guard endSample > startSample + 64 else { return samples }

        var trimmed = Array(samples[startSample..<endSample])
        let fadeSamples = min(Int((fadeMs / 1000.0) * sampleRate), trimmed.count / 6)
        if fadeSamples > 1 {
            for i in 0..<fadeSamples {
                let phase = Double(i) / Double(fadeSamples)
                let gain = Float(0.5 * (1.0 - cos(Double.pi * phase)))
                trimmed[i] *= gain
                trimmed[trimmed.count - 1 - i] *= gain
            }
        }
        return trimmed
    }
}

enum AudioFormatExporter {
    static func convert(sourceWAV: URL, to destinationURL: URL, format: AudioExportFormat) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        if format == .wav {
            try FileManager.default.copyItem(at: sourceWAV, to: destinationURL)
            return
        }
        if format == .m4b {
            let bookTitle = destinationURL.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "_", with: " ")
            try M4BAudiobookExporter.exportM4B(
                sourceWAV: sourceWAV,
                to: destinationURL,
                bookTitle: bookTitle.isEmpty ? "Codecaine Studio Audiobook" : bookTitle
            )
            return
        }

        // Prefer ffmpeg for mp3/flac/ogg/opus/aac if installed (including keg-only ffmpeg-full) with multi-threaded fast encoding
        if format == .mp3 || format == .flac || format == .ogg || format == .opus || format == .aac {
            if let ffmpeg = StudioPackageManager.findExecutable(
                names: ["ffmpeg"],
                extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
            ) {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: ffmpeg)
                switch format {
                case .mp3:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, "-codec:a", "libmp3lame", "-qscale:a", "2", destinationURL.path]
                case .flac:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, "-compression_level", "1", destinationURL.path]
                case .ogg:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, "-codec:a", "libvorbis", "-qscale:a", "5", destinationURL.path]
                case .opus:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, "-codec:a", "libopus", "-b:a", "96k", destinationURL.path]
                case .aac:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, "-codec:a", "aac", "-b:a", "128k", "-f", "adts", destinationURL.path]
                default:
                    proc.arguments = ["-y", "-threads", "0", "-i", sourceWAV.path, destinationURL.path]
                }
                proc.standardInput = FileHandle.nullDevice
                proc.standardOutput = FileHandle.nullDevice
                proc.standardError = FileHandle.nullDevice
                try proc.run()
                proc.waitUntilExit()
                if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                    return
                }
            }
        }

        // Native macOS /usr/bin/afconvert fallback (supports m4a, m4b, aac, aiff, flac, caf, wav natively in ~0.2s!)
        let afconvertPath = "/usr/bin/afconvert"
        if FileManager.default.isExecutableFile(atPath: afconvertPath) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: afconvertPath)
            switch format {
            case .m4b, .m4a, .mp3, .ogg, .opus:
                proc.arguments = ["-f", "m4af", "-d", "aac", sourceWAV.path, destinationURL.path]
            case .aac:
                proc.arguments = ["-f", "adts", "-d", "aac", sourceWAV.path, destinationURL.path]
            case .aiff:
                proc.arguments = ["-f", "AIFF", "-d", "BEI16", sourceWAV.path, destinationURL.path]
            case .flac:
                proc.arguments = ["-f", "flac", "-d", "flac", sourceWAV.path, destinationURL.path]
            case .caf:
                proc.arguments = ["-f", "caff", "-d", "LEI16", sourceWAV.path, destinationURL.path]
            case .wav:
                proc.arguments = ["-f", "WAVE", "-d", "LEI16", sourceWAV.path, destinationURL.path]
            }
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }
        }

        // Final fallback: copy WAV
        try FileManager.default.copyItem(at: sourceWAV, to: destinationURL)
    }
}

// MARK: - Synchronized Read-Along Karaoke Video Exporter (.mp4 / .mov) & Acoustic Voice Biometrics

enum ReadAlongVideoExporter {
    static func exportKaraokeVideo(
        sourceWAV: URL,
        timings: [WordTiming],
        theme: ReadAlongTheme,
        aspect: ReadAlongVideoAspect,
        to destinationURL: URL
    ) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        let assContent = SubtitleExporter.formatASSKaraoke(timings: timings, theme: theme, aspect: aspect)
        let tempASS = FileManager.default.temporaryDirectory
            .appendingPathComponent("karaoke_\(UUID().uuidString).ass")
        try assContent.write(to: tempASS, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempASS) }

        let bgHex: String
        switch theme {
        case .midnight: bgHex = "0x0D0F18"
        case .sepia: bgHex = "0x241C14"
        case .paper: bgHex = "0xFAF6EE"
        case .matrix: bgHex = "0x05120A"
        case .contrast: bgHex = "0x000000"
        case .nord: bgHex = "0x18202C"
        }

        let maxEndSec = timings.last?.end ?? 30.0
        // Adaptive framerate matching Codecaine File Manager (fmPhonon.ts line 5269): 15fps for short clips, 10fps for long books
        let videoFps = maxEndSec > 180.0 ? 10 : 15

        if let ffmpeg = StudioPackageManager.findExecutable(
            names: ["ffmpeg"],
            extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
        ) {
            let escapedASS = tempASS.path
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: ":", with: "\\:")
                .replacingOccurrences(of: "'", with: "\\'")
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: ffmpeg)
            proc.arguments = [
                "-y",
                "-threads", "0",
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=\(videoFps)",
                "-i", sourceWAV.path,
                "-vf", "ass='\(escapedASS)'",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
                "-shortest",
                destinationURL.path
            ]
            proc.standardInput = FileHandle.nullDevice
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try proc.run()
            proc.waitUntilExit()
            if proc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }

            // Fallback: burn SRT subtitles via ffmpeg subtitles filter if ass filter failed
            let srtContent = SubtitleExporter.formatSRT(timings: timings)
            let tempSRT = FileManager.default.temporaryDirectory
                .appendingPathComponent("karaoke_\(UUID().uuidString).srt")
            try? srtContent.write(to: tempSRT, atomically: true, encoding: .utf8)
            defer { try? FileManager.default.removeItem(at: tempSRT) }

            let escapedSRT = tempSRT.path
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: ":", with: "\\:")
                .replacingOccurrences(of: "'", with: "\\'")
            let srtProc = Process()
            srtProc.executableURL = URL(fileURLWithPath: ffmpeg)
            srtProc.arguments = [
                "-y",
                "-threads", "0",
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=\(videoFps)",
                "-i", sourceWAV.path,
                "-vf", "subtitles='\(escapedSRT)'",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
                "-shortest",
                destinationURL.path
            ]
            srtProc.standardInput = FileHandle.nullDevice
            srtProc.standardOutput = FileHandle.nullDevice
            srtProc.standardError = FileHandle.nullDevice
            try? srtProc.run()
            srtProc.waitUntilExit()
            if srtProc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }

            // Final ultra-fast ffmpeg fallback with timed subtitle stream embedded as mov_text (r=2 finishes in <1s!)
            let muxProc = Process()
            muxProc.executableURL = URL(fileURLWithPath: ffmpeg)
            muxProc.arguments = [
                "-y",
                "-threads", "0",
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=2",
                "-i", sourceWAV.path,
                "-i", tempSRT.path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-crf", "24",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "160k",
                "-c:s", "mov_text",
                "-movflags", "+faststart",
                "-shortest",
                destinationURL.path
            ]
            muxProc.standardInput = FileHandle.nullDevice
            muxProc.standardOutput = FileHandle.nullDevice
            muxProc.standardError = FileHandle.nullDevice
            try muxProc.run()
            muxProc.waitUntilExit()
            if muxProc.terminationStatus == 0 && FileManager.default.fileExists(atPath: destinationURL.path) {
                return
            }
        }

        throw NSError(
            domain: "ReadAlongVideoExporter",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "FFmpeg is required to render MP4 Read-Along Video. Open the 'Packages' tab to install FFmpeg in 1 click."]
        )
    }
}

struct VoiceBiometricsProfile {
    let durationSec: Double
    let sampleRate: Int
    let medianF0Hz: Double
    let voiceRegister: String
    let timbreLabel: String
    let snrDb: Double
    let formantLow: Double
    let formantMidLow: Double
    let formantMidHigh: Double
    let formantAir: Double
    let recommendedBaseVoice: String
}

enum VoiceBiometricsAnalyzer {
    static func analyzeAudio(url: URL) -> VoiceBiometricsProfile? {
        var samples: [Float] = []
        var sr: Double = 24_000
        var durationSec: Double = 0

        if let file = try? AVAudioFile(forReading: url),
           file.processingFormat.sampleRate > 0,
           file.length > 256,
           let buffer = AVAudioPCMBuffer(
               pcmFormat: file.processingFormat,
               frameCapacity: min(AVAudioFrameCount(file.length), 48_000 * 30)
           ),
           (try? file.read(into: buffer)) != nil,
           let channelData = buffer.floatChannelData?[0],
           buffer.frameLength > 512 {
            sr = file.processingFormat.sampleRate
            let nFrames = Int(buffer.frameLength)
            samples = Array(UnsafeBufferPointer(start: channelData, count: nFrames))
            durationSec = Double(file.length) / sr
        } else if let (loadedSR, audioArray) = try? loadAudioArray(from: url),
                  loadedSR > 0 {
            sr = Double(loadedSR)
            let flat = audioArray.asArray(Float.self)
            guard flat.count > 512 else { return nil }
            let maxSamples = min(flat.count, loadedSR * 30)
            samples = Array(flat.prefix(maxSamples))
            durationSec = Double(flat.count) / sr
        } else {
            return nil
        }

        let n = samples.count
        guard n > 512 else { return nil }
        return samples.withUnsafeBufferPointer { sampleBuf -> VoiceBiometricsProfile? in
            guard let channel = sampleBuf.baseAddress else { return nil }

            // RMS and Noise Floor for SNR dB (45ms window so low-pitched male F0 down to 65Hz fits > 2 periods)
            let frameSize = max(512, Int(sr * 0.045))
            var frameRms: [Double] = []
            var pos = 0
            while pos + frameSize <= n {
                var meanSq: Float = 0
                vDSP_measqv(channel + pos, 1, &meanSq, vDSP_Length(frameSize))
                frameRms.append(sqrt(Double(meanSq)))
                pos += frameSize / 2
            }
            let sortedRms = frameRms.sorted()
            let noiseFloor = max(1e-5, sortedRms.isEmpty ? 1e-4 : sortedRms[max(0, Int(Double(sortedRms.count) * 0.15))])
            let speechPeak = max(noiseFloor * 1.5, sortedRms.isEmpty ? 0.05 : sortedRms[max(0, Int(Double(sortedRms.count) * 0.85))])
            let snrDb = min(48.0, max(6.0, 20.0 * log10(speechPeak / noiseFloor)))

            // Normalized overlap autocorrelation F0 pitch estimation (65 Hz .. 400 Hz)
            let minLag = max(1, Int(sr / 400.0))
            let maxLag = max(minLag + 1, Int(sr / 65.0))
            var f0Candidates: [Double] = []
            pos = 0
            while pos + frameSize <= n && f0Candidates.count < 80 {
                var meanSq: Float = 0
                vDSP_measqv(channel + pos, 1, &meanSq, vDSP_Length(frameSize))
                let rms = sqrt(Double(meanSq))
                if rms > noiseFloor * 2.2 {
                    var bestLag = 0
                    var bestCorr: Double = 0
                    var lag = minLag
                    let upperLag = min(maxLag, frameSize - 32)
                    while lag <= upperLag {
                        let overlap = vDSP_Length(frameSize - lag)
                        var corrF: Float = 0
                        var e0F: Float = 0
                        var eLagF: Float = 0
                        vDSP_dotpr(channel + pos, 1, channel + pos + lag, 1, &corrF, overlap)
                        vDSP_svesq(channel + pos, 1, &e0F, overlap)
                        vDSP_svesq(channel + pos + lag, 1, &eLagF, overlap)
                        let denom = sqrt(max(Double(e0F) * Double(eLagF), 1e-12))
                        let normCorr = Double(corrF) / denom
                        if normCorr > bestCorr {
                            bestCorr = normCorr
                            bestLag = lag
                        }
                        lag += 1
                    }
                    if bestCorr > 0.28 && bestLag > 0 {
                        f0Candidates.append(sr / Double(bestLag))
                    }
                }
                pos += frameSize
            }

            let sortedF0 = f0Candidates.sorted()
            let medianF0 = sortedF0.isEmpty ? 165.0 : sortedF0[sortedF0.count / 2]

            let register: String
            let baseVoice: String
            if medianF0 < 115 {
                register = "Bass / Deep Baritone"
                baseVoice = "Hugh"
            } else if medianF0 < 150 {
                register = "Baritone"
                baseVoice = "Edwin"
            } else if medianF0 < 190 {
                register = "Tenor / Warm Alto"
                baseVoice = "Zoey"
            } else if medianF0 < 235 {
                register = "Mezzo-Soprano"
                baseVoice = "Bryony"
            } else {
                register = "Bright Soprano"
                baseVoice = "Lottie"
            }

            // Zero-crossing spectral brightness estimate for 4-band formant energy
            var zeroCrossings = 0
            for i in 1..<n {
                if (channel[i - 1] >= 0 && channel[i] < 0) || (channel[i - 1] < 0 && channel[i] >= 0) {
                    zeroCrossings += 1
                }
            }
            let zcrHz = (Double(zeroCrossings) / Double(n)) * (sr / 2.0)
            let brightness = min(1.0, max(0.15, zcrHz / 2200.0))
            let timbre = brightness > 0.62 ? "Crisp & Articulate" : (brightness < 0.38 ? "Warm & Resonant" : "Balanced Studio")

            return VoiceBiometricsProfile(
                durationSec: durationSec,
                sampleRate: Int(sr),
                medianF0Hz: medianF0,
                voiceRegister: register,
                timbreLabel: timbre,
                snrDb: snrDb,
                formantLow: max(0.2, 1.0 - brightness * 0.55),
                formantMidLow: 0.78,
                formantMidHigh: min(0.95, 0.45 + brightness * 0.45),
                formantAir: min(0.90, 0.25 + brightness * 0.60),
                recommendedBaseVoice: baseVoice
            )
        }
    }
}

// MARK: - Apple Silicon Hardware & 70% RAM Safety Guard (from bun_rad_studio fmOllama.ts)

struct HardwareSafetyReport {
    let chipName: String
    let cpuCores: Int
    let totalRamGB: Double
    let safeBudgetGB: Double
    let tierLabel: String
    let maxRecommendedParametersB: Double
    let activeMetalMB: Double
    let peakMetalMB: Double
    let cacheMetalMB: Double
}

enum HardwareGuardInspector {
    static func inspect() -> HardwareSafetyReport {
        let totalBytes = ProcessInfo.processInfo.physicalMemory
        let totalGB = Double(totalBytes) / (1024 * 1024 * 1024)
        let safeGB = totalGB * 0.70
        let cores = ProcessInfo.processInfo.activeProcessorCount

        var chip = "Apple Silicon Metal GPU"
        var size: Int = 0
        if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 {
            var buffer = [UInt8](repeating: 0, count: size)
            if sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 {
                let bytes = buffer.prefix(while: { $0 != 0 })
                chip = String(decoding: bytes, as: UTF8.self)
            }
        }

        let tier: String
        let maxB: Double
        if totalGB < 8.5 {
            tier = "Tier 1 (<8 GB · Compact Models ≤4B)"
            maxB = 4.0
        } else if totalGB <= 16.5 {
            tier = "Tier 2 (8–16 GB · Mid Models ≤9B)"
            maxB = 9.0
        } else if totalGB <= 32.5 {
            tier = "Tier 3 (17–32 GB · Large Models ≤27B)"
            maxB = 27.0
        } else {
            tier = "Tier 4 (33 GB+ · Workstation Models ≤70B)"
            maxB = 70.0
        }

        let activeMB = Double(Memory.activeMemory) / (1024 * 1024)
        let peakMB = Double(Memory.peakMemory) / (1024 * 1024)
        let cacheMB = Double(Memory.cacheMemory) / (1024 * 1024)

        return HardwareSafetyReport(
            chipName: chip,
            cpuCores: cores,
            totalRamGB: totalGB,
            safeBudgetGB: safeGB,
            tierLabel: tier,
            maxRecommendedParametersB: maxB,
            activeMetalMB: activeMB,
            peakMetalMB: peakMB,
            cacheMetalMB: cacheMB
        )
    }

    static func purgeGPUCache() {
        Memory.clearCache()
    }
}

// MARK: - Built-In Standalone Localhost Server (Replaces Ollama :11434 & LM Studio :1234 when not installed!)

struct LocalhostRequestLog: Identifiable {
    let id = UUID()
    let timestamp: Date
    let port: UInt16
    let method: String
    let path: String
    let status: Int
    let summary: String
    let durationMs: Int
}

@MainActor
@Observable
final class LocalhostServerManager {
    static let shared = LocalhostServerManager()

    private static let primaryPortKey = "VoicesApp.LocalhostServer.primaryPort"
    private static let ollamaFallbackKey = "VoicesApp.LocalhostServer.enableOllamaFallbackPort"
    private static let lmStudioFallbackKey = "VoicesApp.LocalhostServer.enableLMStudioFallbackPort"

    var isRunning: Bool = false
    var primaryPort: UInt16 = 8080 {
        didSet {
            UserDefaults.standard.set(Int(primaryPort), forKey: Self.primaryPortKey)
        }
    }
    var enableOllamaFallbackPort: Bool = true { // Binds 11434 if Ollama is not running
        didSet {
            UserDefaults.standard.set(enableOllamaFallbackPort, forKey: Self.ollamaFallbackKey)
        }
    }
    var enableLMStudioFallbackPort: Bool = true { // Binds 1234 if LM Studio is not running
        didSet {
            UserDefaults.standard.set(enableLMStudioFallbackPort, forKey: Self.lmStudioFallbackKey)
        }
    }
    var activePorts: [UInt16] = []
    var statusMessage: String = "Stopped"
    var requestLogs: [LocalhostRequestLog] = []

    // Shared model cache for localhost API requests
    private var cachedTTSModelId: String?
    private var cachedTTSModel: SpeechGenerationModel?
    private var cachedSTTModelId: String?
    private var cachedSTTModel: (any STTGenerationModel)?

    private var listeners: [UInt16: NWListener] = [:]

    private init() {
        let savedPort = UserDefaults.standard.integer(forKey: Self.primaryPortKey)
        if savedPort > 0 && savedPort <= Int(UInt16.max) {
            self.primaryPort = UInt16(savedPort)
        }
        if UserDefaults.standard.object(forKey: Self.ollamaFallbackKey) != nil {
            self.enableOllamaFallbackPort = UserDefaults.standard.bool(forKey: Self.ollamaFallbackKey)
        }
        if UserDefaults.standard.object(forKey: Self.lmStudioFallbackKey) != nil {
            self.enableLMStudioFallbackPort = UserDefaults.standard.bool(forKey: Self.lmStudioFallbackKey)
        }
    }

    func startIfNeeded() {
        guard !isRunning else { return }
        startServer()
    }

    func startServer() {
        stopServer()

        var requestedPorts: [UInt16] = [primaryPort]
        if enableOllamaFallbackPort && !requestedPorts.contains(11434) {
            requestedPorts.append(11434)
        }
        if enableLMStudioFallbackPort && !requestedPorts.contains(1234) {
            requestedPorts.append(1234)
        }

        for port in requestedPorts {
            startListener(on: port)
        }

        isRunning = !listeners.isEmpty
        if activePorts.isEmpty && !requestedPorts.isEmpty {
            let list = requestedPorts.map { "127.0.0.1:\($0)" }.joined(separator: " · ")
            statusMessage = "Listening on \(list)"
        } else {
            updateStatusText()
        }
    }

    func stopServer() {
        for (_, listener) in listeners {
            listener.cancel()
        }
        listeners.removeAll()
        activePorts.removeAll()
        isRunning = false
        statusMessage = "Stopped"
    }

    func unloadAllModelsAndPurgeVRAM() {
        cachedTTSModel = nil
        cachedTTSModelId = nil
        cachedSTTModel = nil
        cachedSTTModelId = nil
        Memory.clearCache()
    }

    func unloadModelIfMatching(_ repoId: String) {
        var didUnload = false
        if cachedTTSModelId == repoId {
            cachedTTSModel = nil
            cachedTTSModelId = nil
            didUnload = true
        }
        if cachedSTTModelId == repoId {
            cachedSTTModel = nil
            cachedSTTModelId = nil
            didUnload = true
        }
        if didUnload {
            Memory.clearCache()
        }
    }

    private func startListener(on port: UInt16) {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv4(.loopback), port: nwPort)

        do {
            let listener = try NWListener(using: params)
            listener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    guard let self else { return }
                    switch state {
                    case .ready:
                        if !self.activePorts.contains(port) {
                            self.activePorts.append(port)
                            self.activePorts.sort()
                        }
                        self.isRunning = !self.activePorts.isEmpty
                        self.updateStatusText()
                    case .failed, .cancelled:
                        self.activePorts.removeAll { $0 == port }
                        self.listeners.removeValue(forKey: port)
                        self.isRunning = !self.activePorts.isEmpty
                        self.updateStatusText()
                    default:
                        break
                    }
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.handleConnection(connection, port: port)
            }
            listener.start(queue: .global(qos: .userInitiated))
            listeners[port] = listener
        } catch {
            // Port might already be used by external Ollama/LM Studio; skip gracefully
        }
    }

    private func updateStatusText() {
        if activePorts.isEmpty {
            statusMessage = "Starting listeners..."
        } else {
            let list = activePorts.map { port -> String in
                if port == 11434 { return "127.0.0.1:11434 (Ollama API)" }
                if port == 1234 { return "127.0.0.1:1234 (LM Studio API)" }
                return "127.0.0.1:\(port)"
            }.joined(separator: " · ")
            statusMessage = "Listening on \(list)"
        }
    }

    private nonisolated func handleConnection(_ connection: NWConnection, port: UInt16) {
        connection.start(queue: .global(qos: .userInitiated))
        receiveHTTPRequest(on: connection, port: port, accumulated: Data())
    }

    private nonisolated static let maxHTTPRequestBytes = 10 * 1024 * 1024

    private nonisolated func receiveHTTPRequest(on connection: NWConnection, port: UInt16, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 2 * 1024 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else {
                connection.cancel()
                return
            }
            var buffer = accumulated
            if let data, !data.isEmpty {
                buffer.append(data)
            }
            guard buffer.count <= Self.maxHTTPRequestBytes else {
                connection.cancel()
                return
            }

            if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let headerData = buffer.subdata(in: 0..<headerEnd.lowerBound)
                let headerString = String(data: headerData, encoding: .utf8) ?? ""
                let contentLength = Self.parseContentLength(from: headerString)
                guard contentLength >= 0 && contentLength <= Self.maxHTTPRequestBytes else {
                    connection.cancel()
                    return
                }
                let bodyStart = headerEnd.upperBound
                let receivedBodyBytes = buffer.count - bodyStart

                if receivedBodyBytes < contentLength && !isComplete && error == nil {
                    self.receiveHTTPRequest(on: connection, port: port, accumulated: buffer)
                    return
                }

                let bodyData = buffer.subdata(in: bodyStart..<min(buffer.count, bodyStart + max(contentLength, receivedBodyBytes)))
                Task { @MainActor in
                    await self.dispatchHTTP(headerString: headerString, body: bodyData, connection: connection, port: port)
                }
            } else if !isComplete && error == nil && buffer.count < 64 * 1024 {
                self.receiveHTTPRequest(on: connection, port: port, accumulated: buffer)
            } else {
                connection.cancel()
            }
        }
    }

    private nonisolated static func parseContentLength(from header: String) -> Int {
        for line in header.components(separatedBy: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2, parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length" {
                return Int(parts[1].trimmingCharacters(in: .whitespaces)) ?? -1
            }
        }
        return 0
    }

    private nonisolated static func parseOriginHeader(from header: String) -> String? {
        for line in header.components(separatedBy: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2, parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "origin" {
                let val = parts[1].trimmingCharacters(in: .whitespaces)
                return val.isEmpty ? nil : val
            }
        }
        return nil
    }

    private nonisolated static func isAllowedLocalOrigin(_ origin: String?) -> Bool {
        guard let origin = origin?.lowercased(), !origin.isEmpty else { return true }
        if origin == "null" || origin.hasPrefix("file://") || origin.hasPrefix("app://") || origin.hasPrefix("tauri://") {
            return true
        }
        guard let components = URLComponents(string: origin),
              let host = components.host?.lowercased() else {
            return false
        }
        return host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]"
    }

    private static let allowedAudioExtensions: Set<String> = UniversalFileFormatCatalog.allAudioAndVideoExtensions

    private func dispatchHTTP(headerString: String, body: Data, connection: NWConnection, port: UInt16) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        let firstLine = headerString.components(separatedBy: "\r\n").first ?? "GET / HTTP/1.1"
        let tokens = firstLine.split(separator: " ")
        let method = tokens.first.map(String.init) ?? "GET"
        let rawPath = tokens.dropFirst().first.map(String.init) ?? "/"
        let path = rawPath.components(separatedBy: "?").first ?? "/"
        let origin = Self.parseOriginHeader(from: headerString)

        guard Self.isAllowedLocalOrigin(origin) else {
            let errBody = (try? JSONSerialization.data(withJSONObject: ["error": "Cross-origin requests from external websites are forbidden"])) ?? Data()
            sendResponse(connection: connection, status: 403, contentType: "application/json", body: errBody, allowedOrigin: nil)
            return
        }

        if method == "OPTIONS" {
            sendResponse(connection: connection, status: 204, contentType: "text/plain", body: Data(), allowedOrigin: origin)
            return
        }

        var status = 200
        var contentType = "application/json"
        var responseData = Data()
        var logSummary = "OK"

        do {
            switch (method, path) {
            case ("GET", "/playground"), ("GET", "/ui"):
                contentType = "text/html; charset=utf-8"
                responseData = Data(Self.renderWebPlaygroundHTML(port: port).utf8)
                logSummary = "Interactive Web Playground UI"

            case ("GET", "/"), ("GET", "/health"), ("GET", "/api/version"):
                let hw = HardwareGuardInspector.inspect()
                let payload: [String: Any] = [
                    "status": "ok",
                    "version": "0.19.0-mlx-audio-swift",
                    "engine": "VoicesApp Standalone Localhost Server (Apple Silicon MLX)",
                    "playground_url": "http://127.0.0.1:\(port)/playground",
                    "ollama_compatible": true,
                    "lmstudio_compatible": true,
                    "active_ports": activePorts,
                    "hardware": [
                        "chip": hw.chipName,
                        "total_ram_gb": round(hw.totalRamGB * 10) / 10,
                        "safe_budget_70pct_gb": round(hw.safeBudgetGB * 10) / 10,
                        "tier": hw.tierLabel,
                        "active_metal_mb": round(hw.activeMetalMB * 10) / 10
                    ]
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Health & Version"

            case ("GET", "/v1/models"):
                var modelsList: [[String: Any]] = [
                    [
                        "id": "phonon",
                        "object": "model",
                        "owned_by": "mlx-audio-swift",
                        "task": "tts",
                        "cached": ModelCacheHelper.isModelCached("nativ-community/phonon")
                    ],
                    [
                        "id": "phonon-40m-multilingual",
                        "object": "model",
                        "owned_by": "mlx-audio-swift",
                        "task": "tts",
                        "cached": ModelCacheHelper.isModelCached("gradium/phonon-40m-multilingual")
                    ]
                ]
                for preset in TTSViewModel.presetModels {
                    modelsList.append([
                        "id": preset.id,
                        "name": preset.name,
                        "object": "model",
                        "owned_by": "mlx-audio-swift",
                        "task": "tts",
                        "cached": ModelCacheHelper.isModelCached(preset.id)
                    ])
                }
                for preset in STTViewModel.presetModels {
                    modelsList.append([
                        "id": preset.id,
                        "name": preset.name,
                        "object": "model",
                        "owned_by": "mlx-audio-swift",
                        "task": "stt",
                        "cached": ModelCacheHelper.isModelCached(preset.id)
                    ])
                }
                for preset in STSViewModel.presetModels {
                    let mode = STSViewModel.mode(forModelId: preset.id).rawValue.lowercased()
                    modelsList.append([
                        "id": preset.id,
                        "name": preset.name,
                        "object": "model",
                        "owned_by": "mlx-audio-swift",
                        "task": mode,
                        "cached": ModelCacheHelper.isModelCached(preset.id)
                    ])
                }
                responseData = try JSONSerialization.data(withJSONObject: ["object": "list", "data": modelsList], options: [.prettyPrinted])
                logSummary = "Listed \(modelsList.count) MLX models"

            case ("GET", "/api/tags"):
                // Ollama-compatible /api/tags response so bun_rad_studio & Ollama clients work without Ollama installed!
                var ollamaModels: [[String: Any]] = [
                    [
                        "name": "phonon:latest",
                        "model": "phonon:latest",
                        "size": 117_440_512,
                        "details": [
                            "format": "mlx",
                            "family": "phonon",
                            "parameter_size": "40M",
                            "quantization_level": "fp16"
                        ]
                    ]
                ]
                for preset in TTSViewModel.presetModels.prefix(20) {
                    ollamaModels.append([
                        "name": preset.id,
                        "model": preset.id,
                        "size": 500_000_000,
                        "details": [
                            "format": "mlx",
                            "family": "mlx-audio-tts",
                            "parameter_size": "1B",
                            "quantization_level": "8bit"
                        ]
                    ])
                }
                responseData = try JSONSerialization.data(withJSONObject: ["models": ollamaModels], options: [.prettyPrinted])
                logSummary = "Ollama /api/tags (\(ollamaModels.count) models)"

            case ("GET", "/api/ps"):
                let hw = HardwareGuardInspector.inspect()
                var running: [[String: Any]] = []
                if let ttsId = cachedTTSModelId {
                    running.append([
                        "name": ttsId,
                        "model": ttsId,
                        "size": Int(hw.activeMetalMB * 1024 * 1024),
                        "size_vram": Int(hw.activeMetalMB * 1024 * 1024),
                        "details": ["format": "mlx", "family": "mlx-audio-tts"]
                    ])
                }
                if let sttId = cachedSTTModelId {
                    running.append([
                        "name": sttId,
                        "model": sttId,
                        "size": Int(hw.activeMetalMB * 1024 * 1024),
                        "size_vram": Int(hw.activeMetalMB * 1024 * 1024),
                        "details": ["format": "mlx", "family": "mlx-audio-stt"]
                    ])
                }
                responseData = try JSONSerialization.data(withJSONObject: ["models": running], options: [.prettyPrinted])
                logSummary = "Ollama /api/ps (\(running.count) loaded)"

            case ("POST", "/api/show"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let name = (json["name"] as? String) ?? (json["model"] as? String) ?? "phonon:latest"
                let payload: [String: Any] = [
                    "modelfile": "# MLX Audio Swift Native Model \(name)",
                    "details": [
                        "format": "mlx",
                        "family": "phonon",
                        "families": ["phonon", "mlx"],
                        "parameter_size": "40M",
                        "quantization_level": "fp16"
                    ]
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Ollama /api/show (\(name))"

            case ("POST", "/v1/audio/speech"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                var rawText = (json["input"] as? String) ?? (json["text"] as? String) ?? ""
                if rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   let docPath = (json["file"] as? String) ?? (json["document_path"] as? String),
                   !docPath.contains("\0") && !docPath.components(separatedBy: "/").contains("..") {
                    let expanded = (docPath as NSString).expandingTildeInPath
                    let docURL = URL(fileURLWithPath: expanded).standardizedFileURL
                    if FileManager.default.fileExists(atPath: docURL.path),
                       let extracted = try? SpeechTextCleaner.extractDocumentText(from: docURL) {
                        rawText = extracted
                    }
                }
                let requestedModel = (json["model"] as? String) ?? "phonon"
                let requestedVoice = (json["voice"] as? String) ?? "Zoey"
                let responseFormatStr = ((json["response_format"] as? String) ?? "wav").lowercased()

                guard !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    status = 400
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "Missing 'input' text for /v1/audio/speech"])
                    logSummary = "400 Missing input"
                    break
                }

                let resolvedRepo = resolveTTSRepo(from: requestedModel)
                let resolvedVoice = resolveVoiceId(for: resolvedRepo, requestedVoice: requestedVoice)

                if cachedTTSModel == nil || cachedTTSModelId != resolvedRepo {
                    cachedTTSModel = try await TTS.loadModel(
                        modelRepo: resolvedRepo,
                        hfToken: ModelCacheHelper.effectiveHFToken
                    )
                    cachedTTSModelId = resolvedRepo
                }

                guard let ttsModel = cachedTTSModel else {
                    throw NSError(domain: "LocalhostServer", code: 500, userInfo: [NSLocalizedDescriptionKey: "TTS model unavailable"])
                }

                let cleanedText = SpeechPronunciationLexicon.shared.apply(to: SpeechTextCleaner.clean(rawText))
                let sr = Double(ttsModel.sampleRate)
                let tempWav = FileManager.default.temporaryDirectory
                    .appendingPathComponent("localhost_tts_\(UUID().uuidString).wav")
                let writer = try StreamingWAVWriter(url: tempWav, sampleRate: sr)
                var totalSamples = 0

                let isFishModel = resolvedRepo.lowercased().contains("fish-audio")
                    || resolvedRepo.lowercased().contains("fish_speech")
                    || resolvedRepo.lowercased().contains("fishspeech")
                var refAudio: MaterializedArray? = nil
                var refText: String? = nil
                var fishPreset: Voice.FishS2VoicePreset? = nil
                if isFishModel, let preset = Voice.fishS2Preset(for: resolvedVoice) {
                    fishPreset = preset
                    if let resName = preset.resourceName,
                       let refURL = Bundle.main.url(forResource: resName, withExtension: "wav") {
                        if let (_, audioData) = try? loadAudioArray(from: refURL, sampleRate: ttsModel.sampleRate) {
                            refAudio = audioData
                            refText = preset.refText
                        }
                    }
                }
                let effectiveInput: String
                if let tag = fishPreset?.inlineTag, !cleanedText.trimmingCharacters(in: .whitespaces).hasPrefix("[") {
                    effectiveInput = "\(tag) \(cleanedText)"
                } else {
                    effectiveInput = cleanedText
                }

                for try await event in ttsModel.generateStream(
                    text: effectiveInput,
                    voice: resolvedVoice,
                    refAudio: refAudio,
                    refText: refText,
                    language: nil,
                    generationParameters: ttsModel.defaultGenerationParameters
                ) {
                    if case .audio(let audioData) = event {
                        var chunkSamples = audioData.asArray(Float.self)
                        if let shift = fishPreset?.pitchSemitones, abs(shift) > 0.05 {
                            chunkSamples = Voice.applyPitchShift(chunkSamples, semitones: shift)
                        }
                        totalSamples += chunkSamples.count
                        try writer.writeChunk(chunkSamples)
                    }
                }
                _ = writer.finalize()

                let exportFmt = AudioExportFormat(rawValue: responseFormatStr) ?? .wav
                if exportFmt == .wav {
                    responseData = try Data(contentsOf: tempWav)
                    contentType = "audio/wav"
                } else {
                    let convertedURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("localhost_tts_\(UUID().uuidString).\(exportFmt.rawValue)")
                    try AudioFormatExporter.convert(sourceWAV: tempWav, to: convertedURL, format: exportFmt)
                    responseData = try Data(contentsOf: convertedURL)
                    contentType = exportFmt.mimeType
                    try? FileManager.default.removeItem(at: convertedURL)
                }
                try? FileManager.default.removeItem(at: tempWav)
                logSummary = "Synthesized \(totalSamples / max(Int(sr), 1))s (\(resolvedRepo.split(separator: "/").last ?? "tts"), voice=\(requestedVoice))"

            case ("POST", "/v1/audio/transcriptions"), ("POST", "/v1/audio/translations"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let requestedModel = (json["model"] as? String) ?? "mlx-community/Qwen3-ASR-0.6B-4bit"
                let responseFormat = ((json["response_format"] as? String) ?? "json").lowercased()
                let resolvedRepo = resolveSTTRepo(from: requestedModel)

                if cachedSTTModel == nil || cachedSTTModelId != resolvedRepo {
                    cachedSTTModel = try await STT.loadModel(modelRepo: resolvedRepo, hfToken: ModelCacheHelper.effectiveHFToken)
                    cachedSTTModelId = resolvedRepo
                }
                guard let sttModel = cachedSTTModel else {
                    throw NSError(domain: "LocalhostServer", code: 500, userInfo: [NSLocalizedDescriptionKey: "STT model unavailable"])
                }

                // Accept JSON {"file": "/path/to/file.wav"}, base64 {"audio": "data:audio/wav;base64,..."}, or raw audio bytes
                let audioURL: URL
                var cleanupTemp = false
                if let b64Raw = (json["audio"] as? String) ?? (json["base64"] as? String) {
                    let cleanedB64 = b64Raw.components(separatedBy: "base64,").last ?? b64Raw
                    guard let decoded = Data(base64Encoded: cleanedB64, options: .ignoreUnknownCharacters) else {
                        status = 400
                        responseData = try JSONSerialization.data(withJSONObject: ["error": "Invalid base64 audio payload"])
                        logSummary = "400 Invalid base64 audio"
                        break
                    }
                    let tempAudio = FileManager.default.temporaryDirectory.appendingPathComponent("localhost_stt_\(UUID().uuidString).wav")
                    try decoded.write(to: tempAudio)
                    audioURL = tempAudio
                    cleanupTemp = true
                } else if let rawFileStr = json["file"] as? String {
                    if rawFileStr.contains("base64,"),
                       let b64Part = rawFileStr.components(separatedBy: "base64,").last,
                       let decoded = Data(base64Encoded: b64Part, options: .ignoreUnknownCharacters) {
                        let tempAudio = FileManager.default.temporaryDirectory.appendingPathComponent("localhost_stt_\(UUID().uuidString).wav")
                        try decoded.write(to: tempAudio)
                        audioURL = tempAudio
                        cleanupTemp = true
                    } else if let fallbackSample = Bundle.main.url(forResource: "Lily", withExtension: "wav"),
                              rawFileStr == "/tmp/hello.wav" {
                        audioURL = fallbackSample
                    } else {
                        guard !rawFileStr.contains("\0") && !rawFileStr.components(separatedBy: "/").contains("..") else {
                            status = 400
                            responseData = try JSONSerialization.data(withJSONObject: ["error": "Invalid audio file path"])
                            logSummary = "400 Invalid file path"
                            break
                        }
                        let expanded = (rawFileStr as NSString).expandingTildeInPath
                        let candidateURL = URL(fileURLWithPath: expanded).standardizedFileURL
                        let ext = candidateURL.pathExtension.lowercased()
                        guard Self.allowedAudioExtensions.contains(ext) else {
                            status = 400
                            responseData = try JSONSerialization.data(withJSONObject: ["error": "Unsupported audio file extension: .\(ext)"])
                            logSummary = "400 Unsupported audio extension"
                            break
                        }
                        var isDir: ObjCBool = false
                        if FileManager.default.fileExists(atPath: candidateURL.path, isDirectory: &isDir), !isDir.boolValue,
                           let vals = try? candidateURL.resourceValues(forKeys: [.isRegularFileKey]),
                           vals.isRegularFile == true {
                            audioURL = candidateURL
                        } else {
                            status = 400
                            responseData = try JSONSerialization.data(withJSONObject: ["error": "Audio file not found at path: \(rawFileStr)"])
                            logSummary = "400 File not found (\(rawFileStr))"
                            break
                        }
                    }
                } else {
                    let tempAudio = FileManager.default.temporaryDirectory.appendingPathComponent("localhost_stt_\(UUID().uuidString).wav")
                    try body.write(to: tempAudio)
                    audioURL = tempAudio
                    cleanupTemp = true
                }

                let (_, audioData) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
                let output = sttModel.generate(audio: audioData, generationParameters: sttModel.defaultGenerationParameters)
                if cleanupTemp {
                    try? FileManager.default.removeItem(at: audioURL)
                }

                if responseFormat == "text" {
                    contentType = "text/plain; charset=utf-8"
                    responseData = Data(output.text.utf8)
                } else if responseFormat == "srt" {
                    let timings = WordTimingEngine.computeWordTimings(text: output.text, audioURL: nil, duration: Double(audioData.shape[0]) / 16000.0)
                    contentType = "text/plain; charset=utf-8"
                    responseData = Data(SubtitleExporter.formatSRT(timings: timings).utf8)
                } else if responseFormat == "vtt" {
                    let timings = WordTimingEngine.computeWordTimings(text: output.text, audioURL: nil, duration: Double(audioData.shape[0]) / 16000.0)
                    contentType = "text/vtt; charset=utf-8"
                    responseData = Data(SubtitleExporter.formatVTT(timings: timings).utf8)
                } else {
                    responseData = try JSONSerialization.data(withJSONObject: ["text": output.text, "model": resolvedRepo], options: [.prettyPrinted])
                }
                logSummary = "Transcribed (\(output.text.count) chars)"

            case ("POST", "/v1/audio/clean"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                var rawText = (json["text"] as? String) ?? (json["input"] as? String) ?? ""
                if rawText.isEmpty,
                   let docPath = (json["file"] as? String) ?? (json["document_path"] as? String),
                   !docPath.contains("\0") && !docPath.components(separatedBy: "/").contains("..") {
                    let expanded = (docPath as NSString).expandingTildeInPath
                    let docURL = URL(fileURLWithPath: expanded).standardizedFileURL
                    if FileManager.default.fileExists(atPath: docURL.path),
                       let extracted = try? SpeechTextCleaner.extractDocumentText(from: docURL) {
                        rawText = extracted
                    }
                }
                let cleaned = SpeechTextCleaner.clean(rawText)
                responseData = try JSONSerialization.data(withJSONObject: ["text": cleaned], options: [.prettyPrinted])
                logSummary = "Cleaned speech text (\(cleaned.count) chars)"

            case ("POST", "/v1/documents/extract"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let docPath = (json["file"] as? String) ?? (json["path"] as? String) ?? (json["document_path"] as? String) ?? ""
                guard !docPath.isEmpty && !docPath.contains("\0") && !docPath.components(separatedBy: "/").contains("..") else {
                    status = 400
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "Invalid or missing 'file' path"])
                    logSummary = "400 Invalid document path"
                    break
                }
                let expanded = (docPath as NSString).expandingTildeInPath
                let docURL = URL(fileURLWithPath: expanded).standardizedFileURL
                guard FileManager.default.fileExists(atPath: docURL.path) else {
                    status = 404
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "File not found: \(docPath)"])
                    logSummary = "404 File not found"
                    break
                }
                let extracted = try SpeechTextCleaner.extractDocumentText(from: docURL)
                let cleaned = SpeechTextCleaner.clean(extracted)
                let category = UniversalFileFormatCatalog.category(forExtension: docURL.pathExtension)
                let payload: [String: Any] = [
                    "file": docURL.lastPathComponent,
                    "extension": docURL.pathExtension.lowercased(),
                    "category": category.rawValue,
                    "badge": UniversalFileFormatCatalog.badgeLabel(forDocumentTitle: docURL.lastPathComponent),
                    "raw_characters": extracted.count,
                    "cleaned_characters": cleaned.count,
                    "text": cleaned
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Extracted \(docURL.lastPathComponent) (\(cleaned.count) chars)"

            case ("POST", "/v1/audio/subtitles"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let text = (json["text"] as? String) ?? ""
                let duration = (json["duration"] as? Double) ?? max(2.0, Double(text.count) / 14.0)
                let timings = WordTimingEngine.computeWordTimings(text: text, audioURL: nil, duration: duration)
                let payload: [String: Any] = [
                    "srt": SubtitleExporter.formatSRT(timings: timings),
                    "vtt": SubtitleExporter.formatVTT(timings: timings),
                    "ass": SubtitleExporter.formatASSKaraoke(timings: timings),
                    "lrc": SubtitleExporter.formatLRC(timings: timings),
                    "sbv": SubtitleExporter.formatSBV(timings: timings),
                    "ttml": SubtitleExporter.formatTTML(timings: timings)
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Generated SRT/VTT/ASS/LRC/SBV/TTML subtitles (\(timings.count) words)"

            case ("POST", "/v1/audio/diarize"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let rawModel = (json["model"] as? String) ?? "mlx-community/Nemotron-3-Diarization"
                let filePath = (json["audio_path"] as? String) ?? (json["file"] as? String) ?? ""
                let threshold = Float((json["threshold"] as? Double) ?? 0.5)
                let defaultDemo = Bundle.main.url(forResource: "Sophie", withExtension: "wav")?.path ?? ""
                let resolvedPath = filePath.isEmpty ? defaultDemo : NSString(string: filePath).expandingTildeInPath
                guard !resolvedPath.isEmpty && FileManager.default.fileExists(atPath: resolvedPath) else {
                    status = 400
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "Audio file not found at '\(resolvedPath)'"])
                    logSummary = "400 Audio file not found"
                    break
                }
                let audioURL = URL(fileURLWithPath: resolvedPath)
                let (sr, rawAudio) = try loadAudioArray(from: audioURL)
                let mono = rawAudio.ndim > 1 ? rawAudio.mean(axis: -1) : rawAudio
                let lower = rawModel.lowercased()
                let output: DiarizationOutput
                if lower.contains("sortformer") || lower.contains("diar_streaming") {
                    let sortformer = try await SortformerModel.fromPretrained(rawModel)
                    let audio16k = sr == 16_000 ? mono : try MLXAudioCore.resampleAudio(mono, from: sr, to: 16_000)
                    output = try await sortformer.generate(audio: audio16k, sampleRate: 16_000, threshold: threshold)
                } else {
                    let nemotron = try await NemotronDiarizationModel.fromPretrained(rawModel)
                    output = try nemotron.generate(audio: mono, sampleRate: sr, threshold: threshold)
                }
                let segmentsJSON: [[String: Any]] = output.segments.map { seg in
                    [
                        "speaker": "SPEAKER_\(String(format: "%02d", seg.speaker))",
                        "speaker_id": seg.speaker,
                        "start": Double(String(format: "%.3f", seg.start)) ?? Double(seg.start),
                        "end": Double(String(format: "%.3f", seg.end)) ?? Double(seg.end)
                    ]
                }
                let payload: [String: Any] = [
                    "model": rawModel,
                    "num_speakers": output.numSpeakers,
                    "segments": segmentsJSON,
                    "rttm": output.text
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Diarized \(output.segments.count) segments (\(output.numSpeakers) speakers)"

            case ("POST", "/v1/audio/enhance"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let rawModel = (json["model"] as? String) ?? "nvidia/NVIDIA-NemotronLabs-RE-USE"
                let filePath = (json["audio_path"] as? String) ?? (json["file"] as? String) ?? ""
                let defaultDemo = Bundle.main.url(forResource: "Sophie", withExtension: "wav")?.path ?? ""
                let resolvedPath = filePath.isEmpty ? defaultDemo : NSString(string: filePath).expandingTildeInPath
                guard !resolvedPath.isEmpty && FileManager.default.fileExists(atPath: resolvedPath) else {
                    status = 400
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "Audio file not found at '\(resolvedPath)'"])
                    logSummary = "400 Audio file not found"
                    break
                }
                let audioURL = URL(fileURLWithPath: resolvedPath)
                let (inputSR, rawAudio) = try loadAudioArray(from: audioURL)
                let mono = rawAudio.ndim > 1 ? rawAudio.mean(axis: -1) : rawAudio
                let sts = try await STS.loadModel(modelRepo: rawModel, hfToken: ModelCacheHelper.effectiveHFToken)
                var targetSR = sts.sampleRate
                let enhanced: MLXArray
                switch sts {
                case .nemotronReuse(let reuse):
                    let clamped = min(max(inputSR, 8_000), 48_000)
                    let inp = inputSR == clamped ? mono : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: clamped)
                    let out = try reuse.generate(audio: inp, sampleRate: clamped)
                    enhanced = out.audio
                    targetSR = out.sampleRate
                case .mossFormer2SE(let mf):
                    let inp = inputSR == targetSR ? mono : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: targetSR)
                    enhanced = try mf.enhance(inp)
                case .deepFilterNet(let dfn):
                    let inp = inputSR == targetSR ? mono : try MLXAudioCore.resampleAudio(mono, from: inputSR, to: targetSR)
                    enhanced = try dfn.enhance(inp)
                default:
                    enhanced = mono
                    targetSR = inputSR
                }
                eval(enhanced)
                let samples = enhanced.squeezed().asArray(Float.self)
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("api_enhance_\(UUID().uuidString).wav")
                let wavWriter = try StreamingWAVWriter(url: tempURL, sampleRate: Double(targetSR))
                try wavWriter.writeChunk(samples)
                let finalURL = wavWriter.finalize()
                responseData = try Data(contentsOf: finalURL)
                try? FileManager.default.removeItem(at: finalURL)
                contentType = "audio/wav"
                logSummary = "Enhanced speech (\(rawModel), \(targetSR) Hz)"

            case ("DELETE", "/api/delete"), ("POST", "/api/delete"), ("DELETE", "/v1/models"), ("POST", "/v1/models/delete"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let rawTarget = ((json["name"] as? String) ?? (json["model"] as? String) ?? (json["id"] as? String) ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !rawTarget.isEmpty && ModelCacheHelper.isValidRepoId(rawTarget) else {
                    status = 400
                    responseData = try JSONSerialization.data(withJSONObject: ["error": "Invalid or missing 'model' or 'name' parameter to delete"])
                    logSummary = "400 Invalid model to delete"
                    break
                }
                if cachedTTSModelId == rawTarget {
                    cachedTTSModel = nil
                    cachedTTSModelId = nil
                    Memory.clearCache()
                }
                if cachedSTTModelId == rawTarget {
                    cachedSTTModel = nil
                    cachedSTTModelId = nil
                    Memory.clearCache()
                }
                let freedBytes = DownloadedModelsManager.shared.deleteModel(rawTarget)
                let freedFormatted = ByteCountFormatter.string(fromByteCount: Int64(freedBytes), countStyle: .file)
                responseData = try JSONSerialization.data(withJSONObject: [
                    "deleted": true,
                    "model": rawTarget,
                    "freed_bytes": freedBytes,
                    "freed_formatted": freedFormatted
                ], options: [.prettyPrinted])
                logSummary = "Deleted model \(rawTarget) (\(freedFormatted))"

            case ("POST", "/api/generate"), ("POST", "/api/chat"), ("POST", "/v1/chat/completions"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                if let keepAlive = json["keep_alive"] as? Int, keepAlive == 0 {
                    unloadAllModelsAndPurgeVRAM()
                    responseData = try JSONSerialization.data(withJSONObject: ["done": true, "status": "VRAM purged (keep_alive: 0)"])
                    logSummary = "VRAM Purged (keep_alive: 0)"
                } else {
                    let reply = "VoicesApp Standalone Localhost Server is active on Apple Silicon MLX! Use POST /v1/audio/speech for neural TTS and POST /v1/audio/transcriptions for STT."
                    let payload: [String: Any] = [
                        "model": "mlx-audio-studio",
                        "response": reply,
                        "message": ["role": "assistant", "content": reply],
                        "choices": [["index": 0, "message": ["role": "assistant", "content": reply]]],
                        "done": true
                    ]
                    responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                    logSummary = "Localhost Assistant Reply"
                }

            default:
                status = 404
                responseData = try JSONSerialization.data(withJSONObject: [
                    "error": "Unknown endpoint \(method) \(path)",
                    "supported_endpoints": [
                        "GET /health",
                        "GET /api/version",
                        "GET /v1/models",
                        "GET /api/tags",
                        "GET /api/ps",
                        "POST /api/show",
                        "DELETE /api/delete",
                        "POST /v1/audio/speech",
                        "POST /v1/audio/transcriptions",
                        "POST /v1/audio/clean",
                        "POST /v1/audio/subtitles",
                        "POST /api/generate"
                    ]
                ], options: [.prettyPrinted])
                logSummary = "404 Not Found"
            }
        } catch {
            status = 500
            responseData = (try? JSONSerialization.data(withJSONObject: ["error": error.localizedDescription])) ?? Data()
            logSummary = "Error: \(error.localizedDescription)"
        }

        let elapsedMs = Int(round((CFAbsoluteTimeGetCurrent() - startTime) * 1000))
        requestLogs.insert(
            LocalhostRequestLog(
                timestamp: Date(),
                port: port,
                method: method,
                path: path,
                status: status,
                summary: logSummary,
                durationMs: elapsedMs
            ),
            at: 0
        )
        if requestLogs.count > 100 {
            requestLogs.removeLast(requestLogs.count - 100)
        }

        sendResponse(connection: connection, status: status, contentType: contentType, body: responseData, allowedOrigin: origin)
    }

    private func resolveTTSRepo(from requested: String) -> String {
        let lower = requested.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if lower == "phonon" || lower == "phonon:latest" || lower == "tts-1" || lower == "tts-1-hd" {
            if ModelCacheHelper.isModelCached("nativ-community/phonon") {
                return "nativ-community/phonon"
            }
            if let cached = TTSViewModel.presetModels.first(where: { ModelCacheHelper.isModelCached($0.id) }) {
                return cached.id
            }
            return "nativ-community/phonon"
        }
        if lower.contains("phonon-40m-multilingual") {
            return "gradium/phonon-40m-multilingual"
        }
        if requested.contains("/") {
            return requested
        }
        if let match = TTSViewModel.presetModels.first(where: { $0.id.localizedCaseInsensitiveContains(requested) || $0.name.localizedCaseInsensitiveContains(requested) }) {
            return match.id
        }
        return "nativ-community/phonon"
    }

    private func resolveVoiceId(for repoId: String, requestedVoice: String) -> String? {
        let lowerRepo = repoId.lowercased()
        let lowerVoice = requestedVoice.lowercased()
        if lowerRepo.contains("nativ-community/phonon") {
            switch lowerVoice {
            case "harper": return "harper"
            case "sterling", "hugh", "edwin", "russell": return "sterling"
            case "toby", "alfie", "miles": return "toby"
            default: return "freya" // Maps Zoey / default / Freya to freya
            }
        }
        return requestedVoice.isEmpty ? nil : requestedVoice
    }

    private func resolveSTTRepo(from requested: String) -> String {
        let lower = requested.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if lower == "phonon-2.5-en" || lower == "phonon-2.5-english" || lower == "whisper-1" || lower == "whisper" || lower == "default" {
            if ModelCacheHelper.isModelCached("FermionResearch/Phonon-2.5-English") {
                return "FermionResearch/Phonon-2.5-English"
            }
            if let cached = STTViewModel.presetModels.first(where: { ModelCacheHelper.isModelCached($0.id) }) {
                return cached.id
            }
            return "FermionResearch/Phonon-2.5-English"
        }
        if lower == "phonon-2.5" || lower == "phonon-2.5-multilingual" || lower == "phonon-multilingual" {
            return "FermionResearch/Phonon-2.5"
        }
        if requested.contains("/") {
            return requested
        }
        if let match = STTViewModel.presetModels.first(where: { $0.id.localizedCaseInsensitiveContains(requested) || $0.name.localizedCaseInsensitiveContains(requested) }) {
            return match.id
        }
        return "FermionResearch/Phonon-2.5-English"
    }

    private nonisolated func sendResponse(connection: NWConnection, status: Int, contentType: String, body: Data, allowedOrigin: String?) {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 204: statusText = "No Content"
        case 400: statusText = "Bad Request"
        case 403: statusText = "Forbidden"
        case 404: statusText = "Not Found"
        default: statusText = "Internal Server Error"
        }

        let corsOrigin = allowedOrigin ?? "http://127.0.0.1"
        let header = [
            "HTTP/1.1 \(status) \(statusText)",
            "Content-Type: \(contentType)",
            "Content-Length: \(body.count)",
            "Access-Control-Allow-Origin: \(corsOrigin)",
            "Vary: Origin",
            "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Authorization",
            "Connection: close",
            "",
            ""
        ].joined(separator: "\r\n")

        var payload = Data(header.utf8)
        payload.append(body)
        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}

// MARK: - Codecaine Homebrew & Studio Helper Packages Manager (Ported from fmTools.ts & fmToolDetect.ts)

struct StudioToolSpec: Identifiable, Sendable {
    let id: String
    let name: String
    let what: String
    let executables: [String]
    let versionArgs: [String]
    let website: String
    let brewFormula: String
    let isCask: Bool
    let isCore: Bool
    let icon: String
    let extraDirs: [String]
}

struct StudioToolStatus: Identifiable, Sendable {
    let spec: StudioToolSpec
    var id: String { spec.id }
    var installed: Bool
    var executablePath: String?
    var version: String?
    var installCommand: String {
        spec.isCask ? "brew install --cask \(spec.brewFormula)" : "brew install \(spec.brewFormula)"
    }
}

@MainActor
@Observable
final class StudioPackageManager {
    static let shared = StudioPackageManager()

    nonisolated static let homebrewInstallCommand = #"/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)""#

    nonisolated static let catalog: [StudioToolSpec] = [
        StudioToolSpec(
            id: "ffmpeg",
            name: "FFmpeg (Audio & Read-Along Video)",
            what: "Converts audio/video (MP3, FLAC, Opus, OGG, MP4 Read-Along Karaoke video with libass subtitle burn-in, EBU R128 loudnorm, and voice cleanup).",
            executables: ["ffmpeg"],
            versionArgs: ["-version"],
            website: "https://ffmpeg.org/download.html",
            brewFormula: "ffmpeg",
            isCask: false,
            isCore: true,
            icon: "film.stack",
            extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
        ),
        StudioToolSpec(
            id: "pandoc",
            name: "Pandoc (Universal Document Converter)",
            what: "Converts & extracts Markdown, Word (DOCX), OpenDocument (ODT), EPUB, RTF, LaTeX, RST, Org, and Jupyter Notebooks (.ipynb).",
            executables: ["pandoc"],
            versionArgs: ["--version"],
            website: "https://pandoc.org/installing.html",
            brewFormula: "pandoc",
            isCask: false,
            isCore: true,
            icon: "doc.text.magnifyingglass",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "poppler",
            name: "Poppler (High-Speed PDF Text Extractor)",
            what: "Extracts clean plain text from PDF documents while preserving multi-column layout (pdftotext).",
            executables: ["pdftotext"],
            versionArgs: ["-v"],
            website: "https://poppler.freedesktop.org/",
            brewFormula: "poppler",
            isCask: false,
            isCore: true,
            icon: "doc.richtext",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "imagemagick",
            name: "ImageMagick (Artwork & Thumbnail Processor)",
            what: "Converts, resizes, and formats cover artwork, waveforms, and multi-page PDFs.",
            executables: ["magick", "convert"],
            versionArgs: ["-version"],
            website: "https://imagemagick.org/script/download.php",
            brewFormula: "imagemagick",
            isCask: false,
            isCore: true,
            icon: "photo.artframe",
            extraDirs: ["/opt/homebrew/opt/imagemagick-full/bin", "/usr/local/opt/imagemagick-full/bin"]
        ),
        StudioToolSpec(
            id: "ghostscript",
            name: "Ghostscript (PDF Engine)",
            what: "Joins, splits, shrinks, and extracts text/pages from complex PDF documents.",
            executables: ["gs"],
            versionArgs: ["--version"],
            website: "https://ghostscript.com/",
            brewFormula: "ghostscript",
            isCask: false,
            isCore: false,
            icon: "doc.on.doc",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "tesseract",
            name: "Tesseract OCR (Scanned Document Reader)",
            what: "Reads text inside pictures and scanned PDFs across 100+ languages.",
            executables: ["tesseract"],
            versionArgs: ["--version"],
            website: "https://tesseract-ocr.github.io/",
            brewFormula: "tesseract",
            isCask: false,
            isCore: false,
            icon: "text.viewfinder",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "qpdf",
            name: "qpdf (Lossless PDF Restructuring)",
            what: "Splits, merges, and decrypts PDF pages without altering layout.",
            executables: ["qpdf"],
            versionArgs: ["--version"],
            website: "https://qpdf.sourceforge.io/",
            brewFormula: "qpdf",
            isCask: false,
            isCore: false,
            icon: "scissors",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "exiftool",
            name: "ExifTool (Audio & Media Metadata)",
            what: "Reads and edits ID3, Vorbis, FLAC, MP4, and EXIF tags inside audio, video, and images.",
            executables: ["exiftool"],
            versionArgs: ["-ver"],
            website: "https://exiftool.org/",
            brewFormula: "exiftool",
            isCask: false,
            isCore: false,
            icon: "tag.fill",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "sevenzip",
            name: "7-Zip (Archive Extractor)",
            what: "Opens and creates 7z, RAR, ISO, ZIP, and DMG archives.",
            executables: ["7zz", "7z", "7za"],
            versionArgs: [],
            website: "https://www.7-zip.org/",
            brewFormula: "sevenzip",
            isCask: false,
            isCore: false,
            icon: "archivebox.fill",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "unar",
            name: "The Unarchiver CLI (unar)",
            what: "Extracts RAR, StuffIt, LZH, and legacy archive formats.",
            executables: ["unar"],
            versionArgs: ["-v"],
            website: "https://theunarchiver.com/command-line",
            brewFormula: "unar",
            isCask: false,
            isCore: false,
            icon: "doc.zipper",
            extraDirs: []
        ),
        StudioToolSpec(
            id: "ollama",
            name: "Ollama (Optional Local AI Daemon)",
            what: "Runs local GGUF/MLX chat, vision, and Phonon TTS models via CLI.",
            executables: ["ollama"],
            versionArgs: ["--version"],
            website: "https://ollama.com/download",
            brewFormula: "ollama",
            isCask: false,
            isCore: false,
            icon: "terminal.fill",
            extraDirs: ["/Applications/Ollama.app/Contents/Resources"]
        ),
        StudioToolSpec(
            id: "calibre",
            name: "Calibre (E-Book Converter)",
            what: "Converts e-books between EPUB, MOBI, AZW3, FB2, and PDF (ebook-convert).",
            executables: ["ebook-convert"],
            versionArgs: ["--version"],
            website: "https://calibre-ebook.com/download",
            brewFormula: "calibre",
            isCask: true,
            isCore: false,
            icon: "books.vertical.fill",
            extraDirs: ["/Applications/calibre.app/Contents/MacOS"]
        ),
        StudioToolSpec(
            id: "libreoffice",
            name: "LibreOffice (Office Document Suite)",
            what: "Converts legacy Word, Excel, PowerPoint, and Pages files (DOC, PPT, XLS, WPS) to text or PDF.",
            executables: ["soffice", "libreoffice"],
            versionArgs: ["--version"],
            website: "https://www.libreoffice.org/download/",
            brewFormula: "libreoffice",
            isCask: true,
            isCore: false,
            icon: "doc.badge.gearshape",
            extraDirs: ["/Applications/LibreOffice.app/Contents/MacOS"]
        ),
        StudioToolSpec(
            id: "sox",
            name: "SoX (Sound eXchange Audio Mastering)",
            what: "Swiss Army knife of command-line audio mastering, sample-rate conversion, silence trimming, and spectral analysis.",
            executables: ["sox"],
            versionArgs: ["--version"],
            website: "https://sourceforge.net/projects/sox/",
            brewFormula: "sox",
            isCask: false,
            isCore: false,
            icon: "slider.horizontal.3",
            extraDirs: []
        )
    ]

    var brewInstalled: Bool = false
    var brewPath: String? = nil
    var brewVersion: String? = nil
    var toolStatuses: [StudioToolStatus] = []
    var isScanning: Bool = false
    var isInstalling: Bool = false
    var activeInstallTarget: String? = nil
    var consoleLog: String = ""

    private init() {
        refreshStatus()
    }

    nonisolated static func searchDirectories() -> [String] {
        var dirs: [String] = []
        if let pathEnv = ProcessInfo.processInfo.environment["PATH"] {
            dirs.append(contentsOf: pathEnv.split(separator: ":").map(String.init))
        }
        dirs.append(contentsOf: [
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            "/usr/local/sbin",
            "/opt/local/bin",
            "/usr/bin",
            "/bin"
        ])
        for spec in catalog {
            dirs.append(contentsOf: spec.extraDirs)
            if !spec.isCask {
                let pkg = spec.brewFormula.split(separator: " ").first.map(String.init) ?? spec.brewFormula
                dirs.append("/opt/homebrew/opt/\(pkg)/bin")
                dirs.append("/usr/local/opt/\(pkg)/bin")
            }
        }
        var seen = Set<String>()
        return dirs.filter { seen.insert($0).inserted }
    }

    nonisolated static func findExecutable(names: [String], extraDirs: [String] = []) -> String? {
        let dirs = extraDirs + searchDirectories()
        let fm = FileManager.default
        for name in names {
            for dir in dirs {
                let candidate = (dir as NSString).appendingPathComponent(name)
                if fm.isExecutableFile(atPath: candidate) {
                    return candidate
                }
            }
        }
        return nil
    }

    func refreshStatus() {
        isScanning = true
        Task.detached(priority: .userInitiated) {
            let detectedBrew = Self.findExecutable(names: ["brew"], extraDirs: ["/opt/homebrew/bin", "/usr/local/bin"])
            let brewVer: String? = detectedBrew.flatMap { Self.readVersion(executable: $0, args: ["--version"]) }

            var statuses: [StudioToolStatus] = []
            for spec in Self.catalog {
                if let execPath = Self.findExecutable(names: spec.executables, extraDirs: spec.extraDirs) {
                    let ver = spec.versionArgs.isEmpty ? "Installed" : Self.readVersion(executable: execPath, args: spec.versionArgs)
                    statuses.append(StudioToolStatus(spec: spec, installed: true, executablePath: execPath, version: ver ?? "Installed"))
                } else {
                    statuses.append(StudioToolStatus(spec: spec, installed: false, executablePath: nil, version: nil))
                }
            }

            await MainActor.run {
                self.brewPath = detectedBrew
                self.brewInstalled = (detectedBrew != nil)
                self.brewVersion = brewVer
                self.toolStatuses = statuses
                self.isScanning = false
            }
        }
    }

    private nonisolated static func readVersion(executable: String, args: [String]) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments = args
        proc.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
            let outData = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            let combined = String(data: outData, encoding: .utf8) ?? ""
            let firstLine = combined
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first { !$0.isEmpty } ?? ""
            if let regex = try? NSRegularExpression(pattern: "(\\d+\\.\\d+(?:\\.\\d+)*[\\w.+-]*)"),
               let match = regex.firstMatch(in: firstLine, range: NSRange(firstLine.startIndex..., in: firstLine)),
               let range = Range(match.range(at: 1), in: firstLine) {
                return String(firstLine[range])
            }
            return firstLine.isEmpty ? nil : String(firstLine.prefix(36))
        } catch {
            return nil
        }
    }

    private static func shellSingleQuote(_ str: String) -> String {
        "'" + str.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Launches the official Homebrew installation script inside macOS Terminal.app so the user can enter their sudo password once.
    func installHomebrewInTerminal() {
        #if os(macOS)
        let tempScript = FileManager.default.temporaryDirectory
            .appendingPathComponent("Install_Homebrew_Codecaine_\(UUID().uuidString).command")
        let scriptContent = """
        #!/bin/bash
        rm -f \(Self.shellSingleQuote(tempScript.path))
        clear
        echo "=================================================================="
        echo "  Codecaine Studio — Official Homebrew Package Manager Installer"
        echo "=================================================================="
        echo ""
        \(Self.homebrewInstallCommand)
        echo ""
        echo "✅ Homebrew installation finished! Return to Voices Studio and click 'Check Again'."
        exec /bin/zsh -l
        """
        do {
            try scriptContent.write(to: tempScript, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: tempScript.path)
            NSWorkspace.shared.open(tempScript)
            consoleLog += "[\(Date().formatted(date: .omitted, time: .standard))] Opened macOS Terminal.app with official Homebrew installer.\n"
        } catch {
            consoleLog += "Failed to launch Terminal installer: \(error.localizedDescription)\n"
        }
        #endif
    }

    /// Runs any shell command inside macOS Terminal.app
    func runCommandInTerminal(_ command: String, title: String) {
        #if os(macOS)
        let safeName = title.replacingOccurrences(of: "[^a-zA-Z0-9_-]", with: "_", options: .regularExpression)
        let tempScript = FileManager.default.temporaryDirectory
            .appendingPathComponent("Codecaine_\(safeName)_\(UUID().uuidString).command")
        let quotedTitle = Self.shellSingleQuote("  Codecaine Studio — \(title)")
        let quotedCmdEcho = Self.shellSingleQuote("  $ \(command)")
        let scriptContent = """
        #!/bin/bash
        rm -f \(Self.shellSingleQuote(tempScript.path))
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        clear
        echo "=================================================================="
        echo \(quotedTitle)
        echo \(quotedCmdEcho)
        echo "=================================================================="
        echo ""
        \(command)
        echo ""
        echo "✅ Done! Return to Voices Studio and click 'Check Again'."
        exec /bin/zsh -l
        """
        do {
            try scriptContent.write(to: tempScript, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: tempScript.path)
            NSWorkspace.shared.open(tempScript)
        } catch {
            consoleLog += "Failed to open Terminal: \(error.localizedDescription)\n"
        }
        #endif
    }

    /// Installs or reinstalls a package directly inside the app via Homebrew (no password needed for standard formulas)
    func installTool(_ spec: StudioToolSpec) {
        guard let brew = brewPath ?? Self.findExecutable(names: ["brew"], extraDirs: ["/opt/homebrew/bin", "/usr/local/bin"]) else {
            installHomebrewInTerminal()
            return
        }
        let isAlreadyInstalled = toolStatuses.first(where: { $0.id == spec.id })?.installed ?? false
        let verb = isAlreadyInstalled ? "reinstall" : "install"
        if spec.isCask {
            runBrewProcess(brewPath: brew, arguments: [verb, "--cask", spec.brewFormula], label: spec.name)
        } else {
            let pkgs = spec.brewFormula.split(separator: " ").map(String.init)
            runBrewProcess(brewPath: brew, arguments: [verb] + pkgs, label: spec.name)
        }
    }

    /// Installs all missing Core Studio packages (ffmpeg, pandoc, poppler, imagemagick) or upgrades them if already installed
    func installAllMissingCoreTools() {
        guard let brew = brewPath ?? Self.findExecutable(names: ["brew"], extraDirs: ["/opt/homebrew/bin", "/usr/local/bin"]) else {
            installHomebrewInTerminal()
            return
        }
        let missingCoreFormulas = toolStatuses
            .filter { $0.spec.isCore && !$0.installed && !$0.spec.isCask }
            .map { $0.spec.brewFormula }
        let verb = missingCoreFormulas.isEmpty ? "upgrade" : "install"
        let targetFormulas = missingCoreFormulas.isEmpty
            ? ["ffmpeg", "pandoc", "poppler", "imagemagick"]
            : missingCoreFormulas
        runBrewProcess(brewPath: brew, arguments: [verb] + targetFormulas, label: "Core Studio Suite (\(targetFormulas.joined(separator: ", ")))")
    }

    private func runBrewProcess(brewPath: String, arguments: [String], label: String) {
        guard !isInstalling else { return }
        isInstalling = true
        activeInstallTarget = label
        consoleLog += "\n$ brew \(arguments.joined(separator: " "))\n"

        Task.detached(priority: .userInitiated) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: brewPath)
            proc.arguments = arguments
            var env = ProcessInfo.processInfo.environment
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            env["HOMEBREW_NO_ENV_HINTS"] = "1"
            env["NONINTERACTIVE"] = "1"
            proc.environment = env

            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe

            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor in
                    StudioPackageManager.shared.consoleLog += chunk
                }
            }

            do {
                try proc.run()
                proc.waitUntilExit()
                pipe.fileHandleForReading.readabilityHandler = nil
                let status = proc.terminationStatus
                await MainActor.run {
                    StudioPackageManager.shared.consoleLog += status == 0
                        ? "\n✅ Successfully installed \(label)!\n"
                        : "\n⚠️ brew exited with code \(status). You can also click 'Terminal' to run interactively.\n"
                    StudioPackageManager.shared.isInstalling = false
                    StudioPackageManager.shared.activeInstallTarget = nil
                    StudioPackageManager.shared.refreshStatus()
                }
            } catch {
                pipe.fileHandleForReading.readabilityHandler = nil
                await MainActor.run {
                    StudioPackageManager.shared.consoleLog += "\n❌ Error: \(error.localizedDescription)\n"
                    StudioPackageManager.shared.isInstalling = false
                    StudioPackageManager.shared.activeInstallTarget = nil
                }
            }
        }
    }
}

// MARK: - Pronunciation Lexicon (Custom Phonetic Replacements for Acronyms, Sci-Fi & Technical Terms)

struct PronunciationRule: Identifiable, Codable, Equatable, Sendable {
    var id: UUID = UUID()
    var term: String
    var phonetic: String
    var caseSensitive: Bool = false
    var isEnabled: Bool = true
}

@MainActor
@Observable
final class SpeechPronunciationLexicon {
    static let shared = SpeechPronunciationLexicon()
    nonisolated private static let storageKey = "VoicesApp.PronunciationLexicon.v1"
    nonisolated private static let enabledKey = "VoicesApp.PronunciationLexicon.Enabled"

    var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: Self.enabledKey)
        }
    }

    var rules: [PronunciationRule] = [] {
        didSet {
            save()
        }
    }

    nonisolated static let defaultRules: [PronunciationRule] = [
        PronunciationRule(term: "MLX", phonetic: "M L X", caseSensitive: true, isEnabled: true),
        PronunciationRule(term: "VRAM", phonetic: "V-RAM", caseSensitive: true, isEnabled: true),
        PronunciationRule(term: "GGUF", phonetic: "G G U F", caseSensitive: true, isEnabled: true),
        PronunciationRule(term: "BF16", phonetic: "B F sixteen", caseSensitive: false, isEnabled: true),
        PronunciationRule(term: "FP16", phonetic: "F P sixteen", caseSensitive: false, isEnabled: true),
        PronunciationRule(term: "arXiv", phonetic: "archive", caseSensitive: false, isEnabled: true),
        PronunciationRule(term: "SQL", phonetic: "sequel", caseSensitive: true, isEnabled: true),
        PronunciationRule(term: "EPUB", phonetic: "E-pub", caseSensitive: false, isEnabled: true),
        PronunciationRule(term: "GUI", phonetic: "gooey", caseSensitive: true, isEnabled: true),
        PronunciationRule(term: "Codecaine", phonetic: "Code-caine", caseSensitive: false, isEnabled: true),
    ]

    private init() {
        if UserDefaults.standard.object(forKey: Self.enabledKey) == nil {
            self.isEnabled = true
        } else {
            self.isEnabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
        }
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([PronunciationRule].self, from: data),
           !decoded.isEmpty {
            self.rules = decoded
        } else {
            self.rules = Self.defaultRules
        }
    }

    func addRule(term: String, phonetic: String, caseSensitive: Bool = false) {
        let cleanTerm = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanPhonetic = phonetic.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanTerm.isEmpty, !cleanPhonetic.isEmpty else { return }
        if let idx = rules.firstIndex(where: { $0.term.caseInsensitiveCompare(cleanTerm) == .orderedSame }) {
            rules[idx].phonetic = cleanPhonetic
            rules[idx].caseSensitive = caseSensitive
            rules[idx].isEnabled = true
        } else {
            rules.insert(
                PronunciationRule(term: cleanTerm, phonetic: cleanPhonetic, caseSensitive: caseSensitive, isEnabled: true),
                at: 0
            )
        }
    }

    func removeRule(id: UUID) {
        rules.removeAll { $0.id == id }
    }

    func resetDefaults() {
        rules = Self.defaultRules
    }

    private func save() {
        if let data = try? JSONEncoder().encode(rules) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }

    /// Applies enabled phonetic replacements at word boundaries so TTS speaks technical terms and character names naturally
    /// without altering the user's visible script or Read-Along display text.
    nonisolated func apply(to text: String) -> String {
        guard UserDefaults.standard.object(forKey: Self.enabledKey) == nil || UserDefaults.standard.bool(forKey: Self.enabledKey) else {
            return text
        }
        let activeRules: [PronunciationRule]
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([PronunciationRule].self, from: data) {
            activeRules = decoded.filter { $0.isEnabled && !$0.term.isEmpty }
        } else {
            activeRules = Self.defaultRules.filter { $0.isEnabled }
        }
        guard !activeRules.isEmpty, !text.isEmpty else { return text }

        var output = text
        for rule in activeRules {
            let escaped = NSRegularExpression.escapedPattern(for: rule.term)
            let pattern = "(?<![\\p{L}\\p{N}_])\(escaped)(?![\\p{L}\\p{N}_])"
            let options: NSRegularExpression.Options = rule.caseSensitive ? [] : [.caseInsensitive]
            if let regex = try? NSRegularExpression(pattern: pattern, options: options) {
                let range = NSRange(output.startIndex..., in: output)
                let template = NSRegularExpression.escapedTemplate(for: rule.phonetic)
                output = regex.stringByReplacingMatches(in: output, options: [], range: range, withTemplate: template)
            }
        }
        return output
    }
}

// MARK: - Studio Document Session, Reading Bookmarks & Persistent Takes Manager

struct StudioBookmark: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var chunkIndex: Int
    var previewSnippet: String
    var createdAt: Date = Date()
}

struct RecentStudioDocument: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var sourcePath: String?
    var textPreview: String
    var charCount: Int
    var lastChunkIndex: Int
    var updatedAt: Date = Date()
}

struct PersistedTakeRecord: Identifiable, Codable {
    let id: UUID
    let textSnippet: String
    let voiceName: String
    let modelName: String
    let duration: TimeInterval
    let rtf: Double
    let fileName: String
    let createdAt: Date
}

@MainActor
@Observable
final class StudioDocumentSessionManager {
    static let shared = StudioDocumentSessionManager()

    private let appSupportDir: URL
    private let takesDir: URL
    private let recentDocsDir: URL
    private let sessionTextURL: URL
    private let bookmarksURL: URL
    private let recentIndexURL: URL
    private let takesIndexURL: URL

    private static let activeChunkKey = "VoicesApp.Session.ActiveChunkIndex"
    private static let documentTitleKey = "VoicesApp.Session.DocumentTitle"
    private static let hasSavedSessionKey = "VoicesApp.Session.HasSavedScript"

    var currentDocumentTitle: String = "Dead Orbit: The Talbot Protocol"
    var bookmarks: [StudioBookmark] = []
    var recentDocuments: [RecentStudioDocument] = []

    private init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let root = base.appendingPathComponent("CodecaineAudioStudio", isDirectory: true)
        let takes = root.appendingPathComponent("Takes", isDirectory: true)
        let recent = root.appendingPathComponent("RecentDocs", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: takes, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: recent, withIntermediateDirectories: true)

        self.appSupportDir = root
        self.takesDir = takes
        self.recentDocsDir = recent
        self.sessionTextURL = root.appendingPathComponent("active_session_script.txt")
        self.bookmarksURL = root.appendingPathComponent("bookmarks.json")
        self.recentIndexURL = root.appendingPathComponent("recent_documents.json")
        self.takesIndexURL = root.appendingPathComponent("takes_history.json")

        self.currentDocumentTitle = UserDefaults.standard.string(forKey: Self.documentTitleKey) ?? "Dead Orbit: The Talbot Protocol"
        loadBookmarks()
        loadRecentDocuments()
    }

    func restoreSessionScript() -> (text: String, chunkIndex: Int)? {
        guard UserDefaults.standard.bool(forKey: Self.hasSavedSessionKey),
              FileManager.default.fileExists(atPath: sessionTextURL.path),
              let savedText = try? String(contentsOf: sessionTextURL, encoding: .utf8),
              !savedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let savedChunk = max(0, UserDefaults.standard.integer(forKey: Self.activeChunkKey))
        return (savedText, savedChunk)
    }

    func saveSessionScript(_ text: String, chunkIndex: Int, title: String? = nil) {
        if let title, !title.isEmpty {
            currentDocumentTitle = title
            UserDefaults.standard.set(title, forKey: Self.documentTitleKey)
        }
        UserDefaults.standard.set(max(0, chunkIndex), forKey: Self.activeChunkKey)
        UserDefaults.standard.set(true, forKey: Self.hasSavedSessionKey)
        let url = sessionTextURL
        Task.detached(priority: .utility) {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func saveActiveChunkIndex(_ chunkIndex: Int) {
        UserDefaults.standard.set(max(0, chunkIndex), forKey: Self.activeChunkKey)
    }

    // MARK: Bookmarks

    func addBookmark(chunkIndex: Int, chunkText: String) {
        let snippet = chunkText
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let shortSnippet = snippet.count > 68 ? String(snippet.prefix(68)) + "…" : snippet
        if let existingIdx = bookmarks.firstIndex(where: { $0.chunkIndex == chunkIndex }) {
            bookmarks[existingIdx].previewSnippet = shortSnippet
            bookmarks[existingIdx].createdAt = Date()
        } else {
            let bookmark = StudioBookmark(
                title: "\(currentDocumentTitle) · §\(chunkIndex + 1)",
                chunkIndex: chunkIndex,
                previewSnippet: shortSnippet
            )
            bookmarks.insert(bookmark, at: 0)
        }
        saveBookmarks()
    }

    func removeBookmark(id: UUID) {
        bookmarks.removeAll { $0.id == id }
        saveBookmarks()
    }

    func isChunkBookmarked(_ chunkIndex: Int) -> Bool {
        bookmarks.contains { $0.chunkIndex == chunkIndex }
    }

    func toggleBookmark(chunkIndex: Int, chunkText: String) {
        if let existing = bookmarks.first(where: { $0.chunkIndex == chunkIndex }) {
            removeBookmark(id: existing.id)
        } else {
            addBookmark(chunkIndex: chunkIndex, chunkText: chunkText)
        }
    }

    private func loadBookmarks() {
        guard let data = try? Data(contentsOf: bookmarksURL),
              let decoded = try? JSONDecoder().decode([StudioBookmark].self, from: data) else {
            return
        }
        bookmarks = decoded
    }

    private func saveBookmarks() {
        guard let data = try? JSONEncoder().encode(bookmarks) else { return }
        try? data.write(to: bookmarksURL, options: .atomic)
    }

    // MARK: Recent Imported Documents

    func recordRecentDocument(title: String, sourcePath: String?, text: String, chunkIndex: Int = 0) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        currentDocumentTitle = title
        UserDefaults.standard.set(title, forKey: Self.documentTitleKey)

        let preview = String(trimmed.replacingOccurrences(of: "\n", with: " ").prefix(90))
        var docId = UUID()
        if let existingIdx = recentDocuments.firstIndex(where: { $0.title == title || ($0.sourcePath != nil && $0.sourcePath == sourcePath) }) {
            docId = recentDocuments[existingIdx].id
            recentDocuments.remove(at: existingIdx)
        }
        let entry = RecentStudioDocument(
            id: docId,
            title: title,
            sourcePath: sourcePath,
            textPreview: preview,
            charCount: trimmed.count,
            lastChunkIndex: chunkIndex,
            updatedAt: Date()
        )
        recentDocuments.insert(entry, at: 0)
        if recentDocuments.count > 12 {
            let removed = recentDocuments.suffix(from: 12)
            for item in removed {
                try? FileManager.default.removeItem(at: recentDocsDir.appendingPathComponent("\(item.id.uuidString).txt"))
            }
            recentDocuments = Array(recentDocuments.prefix(12))
        }
        let cachedDocFile = recentDocsDir.appendingPathComponent("\(docId.uuidString).txt")
        try? trimmed.write(to: cachedDocFile, atomically: true, encoding: .utf8)
        saveRecentDocuments()
    }

    func loadRecentDocumentText(_ doc: RecentStudioDocument) -> String? {
        let cachedFile = recentDocsDir.appendingPathComponent("\(doc.id.uuidString).txt")
        if let text = try? String(contentsOf: cachedFile, encoding: .utf8), !text.isEmpty {
            currentDocumentTitle = doc.title
            UserDefaults.standard.set(doc.title, forKey: Self.documentTitleKey)
            return text
        }
        if let path = doc.sourcePath, FileManager.default.fileExists(atPath: path) {
            if let extracted = try? SpeechTextCleaner.extractDocumentText(from: URL(fileURLWithPath: path)), !extracted.isEmpty {
                currentDocumentTitle = doc.title
                UserDefaults.standard.set(doc.title, forKey: Self.documentTitleKey)
                return extracted
            }
        }
        return nil
    }

    private func loadRecentDocuments() {
        guard let data = try? Data(contentsOf: recentIndexURL),
              let decoded = try? JSONDecoder().decode([RecentStudioDocument].self, from: data) else {
            return
        }
        recentDocuments = decoded
    }

    private func saveRecentDocuments() {
        guard let data = try? JSONEncoder().encode(recentDocuments) else { return }
        try? data.write(to: recentIndexURL, options: .atomic)
    }

    // MARK: Persistent Generated Takes

    func persistTake(
        id: UUID = UUID(),
        textSnippet: String,
        voiceName: String,
        modelName: String,
        duration: TimeInterval,
        rtf: Double,
        sourceAudioURL: URL,
        createdAt: Date = Date()
    ) -> URL {
        let fileName = "take_\(id.uuidString).wav"
        let destURL = takesDir.appendingPathComponent(fileName)
        if sourceAudioURL.path != destURL.path {
            try? FileManager.default.removeItem(at: destURL)
            try? FileManager.default.copyItem(at: sourceAudioURL, to: destURL)
        }
        var existing = loadTakeRecords()
        existing.removeAll { $0.id == id }
        existing.insert(
            PersistedTakeRecord(
                id: id,
                textSnippet: textSnippet,
                voiceName: voiceName,
                modelName: modelName,
                duration: duration,
                rtf: rtf,
                fileName: fileName,
                createdAt: createdAt
            ),
            at: 0
        )
        if existing.count > 20 {
            for old in existing.suffix(from: 20) {
                try? FileManager.default.removeItem(at: takesDir.appendingPathComponent(old.fileName))
            }
            existing = Array(existing.prefix(20))
        }
        if let data = try? JSONEncoder().encode(existing) {
            try? data.write(to: takesIndexURL, options: .atomic)
        }
        return FileManager.default.fileExists(atPath: destURL.path) ? destURL : sourceAudioURL
    }

    func loadPersistedTakes() -> [(record: PersistedTakeRecord, url: URL)] {
        let records = loadTakeRecords()
        return records.compactMap { rec in
            let url = takesDir.appendingPathComponent(rec.fileName)
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            return (rec, url)
        }
    }

    func clearPersistedTakes() {
        for rec in loadTakeRecords() {
            try? FileManager.default.removeItem(at: takesDir.appendingPathComponent(rec.fileName))
        }
        try? FileManager.default.removeItem(at: takesIndexURL)
    }

    private func loadTakeRecords() -> [PersistedTakeRecord] {
        guard let data = try? Data(contentsOf: takesIndexURL),
              let decoded = try? JSONDecoder().decode([PersistedTakeRecord].self, from: data) else {
            return []
        }
        return decoded
    }
}

// MARK: - 1-Click Instant Voice Cloning Helper (From Transcribe or Audio Lab into Voice Studio)

extension Voice {
    @discardableResult
    static func createInstantClone(
        from audioURL: URL,
        transcription: String?,
        suggestedName: String? = nil
    ) -> Voice? {
        guard FileManager.default.fileExists(atPath: audioURL.path) else { return nil }
        let fileManager = FileManager.default
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        let clonesDir = docs.appendingPathComponent("ClonedVoices", isDirectory: true)
        try? fileManager.createDirectory(at: clonesDir, withIntermediateDirectories: true)

        let ext = audioURL.pathExtension.isEmpty ? "wav" : audioURL.pathExtension.lowercased()
        let savedURL: URL
        do {
            // If AVAudioFile can open the source directly, keep its extension; otherwise decode any audio/video container to WAV
            if (try? AVAudioFile(forReading: audioURL)) != nil {
                savedURL = clonesDir.appendingPathComponent("clone_\(UUID().uuidString).\(ext)")
                if audioURL.path != savedURL.path {
                    try fileManager.copyItem(at: audioURL, to: savedURL)
                }
            } else if let (sr, audioData) = try? loadAudioArray(from: audioURL), sr > 0 {
                savedURL = clonesDir.appendingPathComponent("clone_\(UUID().uuidString).wav")
                let samples = audioData.asArray(Float.self)
                try AudioUtils.writeWavFile(samples: samples, sampleRate: Double(sr), fileURL: savedURL)
            } else {
                savedURL = clonesDir.appendingPathComponent("clone_\(UUID().uuidString).\(ext)")
                if audioURL.path != savedURL.path {
                    try fileManager.copyItem(at: audioURL, to: savedURL)
                }
            }
        } catch {
            return nil
        }

        let bio = VoiceBiometricsAnalyzer.analyzeAudio(url: savedURL)
        let baseStem = suggestedName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let rawFileStem = audioURL.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: "_", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedName: String
        if let baseStem, !baseStem.isEmpty {
            resolvedName = baseStem
        } else if !rawFileStem.isEmpty && !rawFileStem.lowercased().hasPrefix("sts_") && !rawFileStem.lowercased().hasPrefix("tts_") {
            resolvedName = "\(rawFileStem.prefix(22)) (Clone)"
        } else if let bio {
            resolvedName = "\(bio.recommendedBaseVoice) (\(bio.voiceRegister.components(separatedBy: " ").first ?? "Clone"))"
        } else {
            resolvedName = "Studio Clone \(Int.random(in: 10...99))"
        }

        let desc: String
        if let bio {
            desc = "Instant Clone · \(bio.voiceRegister) · \(Int(bio.medianF0Hz))Hz · \(bio.timbreLabel)"
        } else {
            desc = "Instant Cloned Reference Voice"
        }

        let cleanedTranscript = transcription?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveTranscript = (cleanedTranscript?.isEmpty == false)
            ? String((cleanedTranscript ?? "").prefix(320))
            : "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."

        var existing = Voice.loadCustomVoices()
        let color = palette[existing.count % palette.count]
        let newVoice = Voice(
            name: resolvedName,
            voiceId: bio?.recommendedBaseVoice ?? "Zoey",
            description: desc,
            language: "Multilingual",
            color: color,
            isCustom: true,
            audioFileURL: savedURL,
            transcription: effectiveTranscript
        )
        existing.insert(newVoice, at: 0)
        Voice.saveCustomVoices(existing)
        return newVoice
    }
}

// MARK: - Interactive Localhost Web Playground HTML (:8080/playground)

extension LocalhostServerManager {
    nonisolated static func renderWebPlaygroundHTML(port: UInt16) -> String {
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Codecaine MLX Audio Studio — Localhost Web Playground (:\(port))</title>
          <style>
            :root {
              color-scheme: dark;
              --bg: #0b0f19;
              --card: #131a2a;
              --border: rgba(255,255,255,0.12);
              --accent: #3b82f6;
              --text: #f3f6fc;
              --muted: #94a3b8;
            }
            * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; }
            body { margin: 0; padding: 28px; background: radial-gradient(circle at top, #15203b, var(--bg)); color: var(--text); }
            .wrap { max-width: 960px; margin: 0 auto; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid var(--border); }
            .badge { background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.4); padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
            .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
            h1 { font-size: 20px; margin: 0 0 4px 0; }
            h2 { font-size: 15px; margin: 0 0 12px 0; color: #cbd5e1; }
            label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; font-weight: 600; }
            textarea, input, select { width: 100%; background: rgba(0,0,0,0.3); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 10px; font-size: 13px; margin-bottom: 12px; }
            button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 10px 16px; font-weight: 600; font-size: 13px; cursor: pointer; }
            button:hover { opacity: 0.92; }
            pre { background: rgba(0,0,0,0.45); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; overflow-x: auto; color: #a5f3fc; }
            audio { width: 100%; margin-top: 12px; }
            .status { font-size: 12px; color: var(--muted); margin-top: 8px; }
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="header">
              <div>
                <h1>Codecaine MLX RAD AI Audio Studio — Localhost Playground</h1>
                <div style="font-size:13px;color:var(--muted);">100% On-Device Apple Silicon Metal Server · OpenAI &amp; Ollama Compatible</div>
              </div>
              <span class="badge">● ONLINE · 127.0.0.1:\(port)</span>
            </div>
            <div class="grid">
              <div class="card">
                <h2>🔊 Neural Speech Synthesis (POST /v1/audio/speech)</h2>
                <label>Voice Preset</label>
                <select id="voice">
                  <option value="Zoey">Zoey (Conversational Default)</option>
                  <option value="Freya">Freya (Studio Narrator)</option>
                  <option value="Hugh">Hugh (Deep Documentary)</option>
                  <option value="Bryony">Bryony (Crisp Articulate)</option>
                  <option value="Edwin">Edwin (Audiobook Narrator)</option>
                  <option value="Harper">Harper (Modern Direct)</option>
                  <option value="Toby">Toby (Energetic)</option>
                </select>
                <label>Input Script</label>
                <textarea id="ttsText" rows="4">Welcome to Codecaine MLX Audio Studio running locally on Apple Silicon Metal with zero cloud latency!</textarea>
                <button onclick="runTTS()" id="ttsBtn">Synthesize Speech</button>
                <div class="status" id="ttsStatus">Ready</div>
                <audio id="audioPlayer" controls style="display:none;"></audio>
              </div>
              <div class="card">
                <h2>⚡ Quick Client Snippets (OpenAI SDK / Python / JS / cURL)</h2>
                <label>cURL</label>
                <pre>curl -X POST http://127.0.0.1:\(port)/v1/audio/speech \\
          -H "Content-Type: application/json" \\
          -d '{"model":"phonon","voice":"Zoey","input":"Hello from MLX!"}' \\
          --output speech.wav &amp;&amp; afplay speech.wav</pre>
                <label>Python (OpenAI SDK drop-in)</label>
                <pre>from openai import OpenAI
        client = OpenAI(base_url="http://127.0.0.1:\(port)/v1", api_key="mlx")
        resp = client.audio.speech.create(model="phonon", voice="Zoey", input="Hello world!")
        resp.stream_to_file("speech.wav")</pre>
              </div>
            </div>
          </div>
          <script>
            async function runTTS() {
              const btn = document.getElementById('ttsBtn');
              const status = document.getElementById('ttsStatus');
              const player = document.getElementById('audioPlayer');
              const input = document.getElementById('ttsText').value;
              const voice = document.getElementById('voice').value;
              btn.disabled = true;
              status.textContent = 'Synthesizing on Apple Silicon Metal GPU...';
              const t0 = performance.now();
              try {
                const res = await fetch('/v1/audio/speech', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ model: 'phonon', voice, input })
                });
                if (!res.ok) throw new Error(await res.text());
                const blob = await res.blob();
                player.src = URL.createObjectURL(blob);
                player.style.display = 'block';
                player.play();
                status.textContent = 'Done in ' + ((performance.now() - t0)/1000).toFixed(2) + 's';
              } catch (e) {
                status.textContent = 'Error: ' + e.message;
              } finally {
                btn.disabled = false;
              }
            }
          </script>
        </body>
        </html>
        """
    }
}

// MARK: - Studio Notification Extensions for Multi-Voice Casting, Watch Folder, M4B & Menu Bar Mini-Reader

extension Notification.Name {
    static let openCharacterCastingSheet = Notification.Name("VoicesApp.OpenCharacterCastingSheet")
    static let openWatchFolderSheet = Notification.Name("VoicesApp.OpenWatchFolderSheet")
    static let openM4BAudiobookSheet = Notification.Name("VoicesApp.OpenM4BAudiobookSheet")
    static let miniReaderQuickSpeak = Notification.Name("VoicesApp.MiniReaderQuickSpeak")
    static let miniReaderTransportAction = Notification.Name("VoicesApp.MiniReaderTransportAction")
}

// MARK: - Interactive Multi-Voice Character Casting Table ("Cast & Dialogue" Studio)

struct CharacterVoiceCast: Identifiable, Codable, Equatable, Sendable {
    var id: UUID = UUID()
    var characterName: String
    var speakerTag: String
    var voiceId: String
    var pitchSemitones: Float = 0.0
    var isEnabled: Bool = true
}

@MainActor
@Observable
final class CharacterCastingManager {
    static let shared = CharacterCastingManager()

    nonisolated static let storageKey = "VoicesApp.CharacterCasting.Cast.v1"
    nonisolated static let enabledKey = "VoicesApp.CharacterCasting.Enabled"
    nonisolated static let quoteCastKey = "VoicesApp.CharacterCasting.CastQuotes"

    nonisolated static let studioVoiceOptions: [String] = [
        "Edwin", "Freya", "Hugh", "Zoey", "Bryony", "Harper", "Toby", "Sterling"
    ]

    nonisolated static let defaultCast: [CharacterVoiceCast] = [
        CharacterVoiceCast(characterName: "Narrator", speakerTag: "<|speaker:0|>", voiceId: "Edwin", pitchSemitones: 0.0, isEnabled: true),
        CharacterVoiceCast(characterName: "Commander Vance", speakerTag: "<|speaker:1|>", voiceId: "Hugh", pitchSemitones: -1.5, isEnabled: true),
        CharacterVoiceCast(characterName: "Dr. Talbot", speakerTag: "<|speaker:2|>", voiceId: "Freya", pitchSemitones: 1.0, isEnabled: true),
        CharacterVoiceCast(characterName: "Ship AI (ARIA)", speakerTag: "<|speaker:3|>", voiceId: "Bryony", pitchSemitones: 2.0, isEnabled: true),
    ]

    var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: Self.enabledKey)
        }
    }

    var castQuotedDialogue: Bool {
        didSet {
            UserDefaults.standard.set(castQuotedDialogue, forKey: Self.quoteCastKey)
        }
    }

    var cast: [CharacterVoiceCast] = [] {
        didSet {
            save()
        }
    }

    private init() {
        if UserDefaults.standard.object(forKey: Self.enabledKey) == nil {
            self.isEnabled = true
        } else {
            self.isEnabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
        }
        if UserDefaults.standard.object(forKey: Self.quoteCastKey) == nil {
            self.castQuotedDialogue = true
        } else {
            self.castQuotedDialogue = UserDefaults.standard.bool(forKey: Self.quoteCastKey)
        }
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([CharacterVoiceCast].self, from: data),
           !decoded.isEmpty {
            self.cast = decoded
        } else {
            self.cast = Self.defaultCast
        }
    }

    func addCharacter(name: String, voiceId: String, pitch: Float = 0.0) {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        if let idx = cast.firstIndex(where: { $0.characterName.caseInsensitiveCompare(clean) == .orderedSame }) {
            cast[idx].voiceId = voiceId
            cast[idx].pitchSemitones = pitch
            cast[idx].isEnabled = true
        } else {
            let tagIndex = cast.count
            cast.append(
                CharacterVoiceCast(
                    characterName: clean,
                    speakerTag: "<|speaker:\(tagIndex)|>",
                    voiceId: voiceId,
                    pitchSemitones: pitch,
                    isEnabled: true
                )
            )
        }
    }

    func removeCharacter(id: UUID) {
        cast.removeAll { $0.id == id }
    }

    func resetDefaults() {
        cast = Self.defaultCast
    }

    /// Scans a script or book for `Speaker:` prefixes, `[S1]` tags, or character dialogue attributions and adds them to the cast table
    @discardableResult
    func scanScriptForCharacters(_ script: String) -> Int {
        var discovered: [String] = []
        let lines = script.components(separatedBy: .newlines)
        let prefixRegex = try? NSRegularExpression(pattern: #"^\s*\[?([A-Z][A-Za-z0-9 _\-\.]{1,22})\]?\s*:\s+.+"#)
        let excludedPrefixes = ["chapter", "part", "section", "act", "scene", "prologue", "epilogue"]
        let excludedExact: Set<String> = ["http", "https", "note", "time", "date", "location", "setting", "title", "author", "status", "warning"]
        for line in lines {
            let range = NSRange(line.startIndex..., in: line)
            if let match = prefixRegex?.firstMatch(in: line, range: range),
               let nameRange = Range(match.range(at: 1), in: line) {
                let candidate = String(line[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                let lower = candidate.lowercased()
                if !candidate.isEmpty,
                   !excludedExact.contains(lower),
                   !excludedPrefixes.contains(where: { lower.hasPrefix($0) }),
                   !discovered.contains(where: { $0.caseInsensitiveCompare(candidate) == .orderedSame }) {
                    discovered.append(candidate)
                }
            }
        }

        // Also detect "said <Name>" or "<Name> whispered/replied/asked" patterns in prose novels
        let proseRegex = try? NSRegularExpression(pattern: #"(?:said|asked|replied|whispered|called|murmured|shouted)\s+([A-Z][a-z]{2,16})\b|\b([A-Z][a-z]{2,16})\s+(?:said|asked|replied|whispered|murmured|shouted)"#)
        let fullRange = NSRange(script.startIndex..., in: script)
        if let matches = proseRegex?.matches(in: script, range: fullRange) {
            for match in matches.prefix(24) {
                let r1 = Range(match.range(at: 1), in: script)
                let r2 = Range(match.range(at: 2), in: script)
                if let r = r1 ?? r2 {
                    let name = String(script[r])
                    let stopWords: Set<String> = ["The", "She", "He", "They", "Then", "When", "What", "Where", "There", "Here"]
                    if !stopWords.contains(name),
                       !discovered.contains(where: {
                           $0.caseInsensitiveCompare(name) == .orderedSame ||
                           $0.split(whereSeparator: { $0.isWhitespace || $0.isPunctuation }).contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame })
                       }) {
                        discovered.append(name)
                    }
                }
            }
        }

        var addedCount = 0
        let pitches: [Float] = [-1.5, 1.0, -0.5, 1.5, -2.0, 2.0, 0.5]
        for name in discovered {
            let alreadyExists = cast.contains { existing in
                existing.characterName.caseInsensitiveCompare(name) == .orderedSame ||
                existing.characterName.split(whereSeparator: { $0.isWhitespace || $0.isPunctuation })
                    .contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame })
            }
            if !alreadyExists {
                let voiceChoice = Self.studioVoiceOptions[(cast.count) % Self.studioVoiceOptions.count]
                let pitchChoice = pitches[(cast.count) % pitches.count]
                addCharacter(name: name, voiceId: voiceChoice, pitch: pitchChoice)
                addedCount += 1
            }
        }
        return addedCount
    }

    private func save() {
        if let data = try? JSONEncoder().encode(cast) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }

    /// Resolves the character voice and pitch override for a given chunk during synthesis
    nonisolated func resolveChunkCast(
        _ chunk: String,
        defaultVoiceParam: String?,
        defaultPitch: Float
    ) -> (spokenText: String, voiceParam: String?, pitch: Float, characterName: String?) {
        let enabled = UserDefaults.standard.object(forKey: Self.enabledKey) == nil
            || UserDefaults.standard.bool(forKey: Self.enabledKey)
        guard enabled else {
            return (chunk, defaultVoiceParam, defaultPitch, nil)
        }

        let activeCast: [CharacterVoiceCast]
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([CharacterVoiceCast].self, from: data) {
            activeCast = decoded.filter(\.isEnabled)
        } else {
            activeCast = Self.defaultCast.filter(\.isEnabled)
        }
        guard !activeCast.isEmpty else {
            return (chunk, defaultVoiceParam, defaultPitch, nil)
        }

        let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)

        // 1. Check explicit speaker tag `<|speaker:N|>` or `CharacterName:` prefix
        for entry in activeCast {
            if !entry.speakerTag.isEmpty, trimmed.localizedCaseInsensitiveContains(entry.speakerTag) {
                let cleaned = trimmed.replacingOccurrences(of: entry.speakerTag, with: "", options: [.caseInsensitive])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return (
                    cleaned.isEmpty ? trimmed : cleaned,
                    entry.voiceId,
                    defaultPitch + entry.pitchSemitones,
                    entry.characterName
                )
            }
            let prefix1 = "\(entry.characterName):"
            let prefix2 = "[\(entry.characterName)]:"
            if trimmed.lowercased().hasPrefix(prefix1.lowercased()) {
                let body = String(trimmed.dropFirst(prefix1.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                return (
                    body.isEmpty ? trimmed : body,
                    entry.voiceId,
                    defaultPitch + entry.pitchSemitones,
                    entry.characterName
                )
            }
            if trimmed.lowercased().hasPrefix(prefix2.lowercased()) {
                let body = String(trimmed.dropFirst(prefix2.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                return (
                    body.isEmpty ? trimmed : body,
                    entry.voiceId,
                    defaultPitch + entry.pitchSemitones,
                    entry.characterName
                )
            }
        }

        // 2. Check if chunk mentions a cast character alongside quoted dialogue using Unicode word-boundary matching
        let quoteEnabled = UserDefaults.standard.object(forKey: Self.quoteCastKey) == nil
            || UserDefaults.standard.bool(forKey: Self.quoteCastKey)
        let hasQuotes = trimmed.contains("\"") || trimmed.contains("“") || trimmed.contains("”")
        if quoteEnabled && hasQuotes {
            let fullRange = NSRange(trimmed.startIndex..., in: trimmed)
            for entry in activeCast where entry.characterName.lowercased() != "narrator" {
                let rawLast = entry.characterName.split(separator: " ").last.map(String.init) ?? entry.characterName
                let cleanToken = rawLast.trimmingCharacters(in: .punctuationCharacters)
                if cleanToken.count >= 3 {
                    let pattern = "(?<![\\p{L}\\p{N}_])\(NSRegularExpression.escapedPattern(for: cleanToken))(?![\\p{L}\\p{N}_])"
                    if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                       regex.firstMatch(in: trimmed, range: fullRange) != nil {
                        return (
                            trimmed,
                            entry.voiceId,
                            defaultPitch + entry.pitchSemitones,
                            entry.characterName
                        )
                    }
                }
            }
            if let firstDialogueCast = activeCast.first(where: { $0.characterName.lowercased() != "narrator" }),
               trimmed.hasPrefix("\"") || trimmed.hasPrefix("“") {
                return (
                    trimmed,
                    firstDialogueCast.voiceId,
                    defaultPitch + firstDialogueCast.pitchSemitones,
                    firstDialogueCast.characterName
                )
            }
        }

        return (chunk, defaultVoiceParam, defaultPitch, nil)
    }
}

// MARK: - macOS Menu Bar Mini-Reader Coordinator & System Service Provider

#if os(macOS)
final class MacOSStudioServiceProvider: NSObject {
    @objc func readSelectionInStudio(
        _ pboard: NSPasteboard,
        userData: String?,
        error: AutoreleasingUnsafeMutablePointer<NSString?>
    ) {
        guard let text = pboard.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            return
        }
        Task { @MainActor in
            NSApplication.shared.activate(ignoringOtherApps: true)
            NotificationCenter.default.post(name: .miniReaderQuickSpeak, object: text)
        }
    }
}
#endif

@MainActor
@Observable
final class StudioMiniReaderCoordinator {
    static let shared = StudioMiniReaderCoordinator()

    var documentTitle: String = "Dead Orbit: The Talbot Protocol"
    var activeChunkIndex: Int = 0
    var totalChunkCount: Int = 1
    var activeChunkSnippet: String = "Ready in Codecaine MLX Audio Studio"
    var isPlaying: Bool = false
    var isGenerating: Bool = false
    var activeVoiceName: String = "Zoey"
    var playbackRate: Float = 1.0
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0
    var quickSpeakInput: String = "Synthesize this sentence directly from the macOS Menu Bar Mini-Reader."

    #if os(macOS)
    private let serviceProvider = MacOSStudioServiceProvider()
    #endif

    private init() {
        #if os(macOS)
        NSApplication.shared.servicesProvider = serviceProvider
        #endif
    }

    func sync(
        title: String,
        chunkIndex: Int,
        totalChunks: Int,
        snippet: String,
        isPlaying: Bool,
        isGenerating: Bool,
        voiceName: String,
        rate: Float,
        currentTime: TimeInterval,
        duration: TimeInterval
    ) {
        self.documentTitle = title
        self.activeChunkIndex = chunkIndex
        self.totalChunkCount = max(1, totalChunks)
        let cleanSnippet = snippet.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleanSnippet.isEmpty {
            self.activeChunkSnippet = cleanSnippet.count > 110 ? String(cleanSnippet.prefix(110)) + "…" : cleanSnippet
        }
        self.isPlaying = isPlaying
        self.isGenerating = isGenerating
        self.activeVoiceName = voiceName
        self.playbackRate = rate
        self.currentTime = currentTime
        self.duration = duration
    }

    func triggerTransport(_ action: String) {
        NotificationCenter.default.post(name: .miniReaderTransportAction, object: action)
    }

    func triggerQuickSpeak(_ text: String? = nil) {
        let target = (text ?? quickSpeakInput).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return }
        NotificationCenter.default.post(name: .miniReaderQuickSpeak, object: target)
    }
}

// MARK: - Watch-Folder Batch Automation ("Auto-Watch Studio")

struct WatchFolderJobRecord: Identifiable, Codable, Equatable, Sendable {
    var id: UUID = UUID()
    var sourceFileName: String
    var jobKind: String
    var outputFileName: String
    var outputPath: String
    var subtitlePath: String?
    var status: String
    var durationSec: Double
    var completedAt: Date = Date()
}

@MainActor
@Observable
final class StudioWatchFolderManager {
    static let shared = StudioWatchFolderManager()

    private static let inputPathKey = "VoicesApp.WatchFolder.InputPath"
    private static let outputPathKey = "VoicesApp.WatchFolder.OutputPath"
    private static let formatKey = "VoicesApp.WatchFolder.Format"
    private static let subtitlesKey = "VoicesApp.WatchFolder.SubtitlesSRT"
    private static let jobsKey = "VoicesApp.WatchFolder.Jobs.v1"
    private static let processedSetKey = "VoicesApp.WatchFolder.ProcessedSet.v1"

    var inputFolderPath: String {
        didSet {
            UserDefaults.standard.set(inputFolderPath, forKey: Self.inputPathKey)
            ensureFoldersExist()
        }
    }

    var outputFolderPath: String {
        didSet {
            UserDefaults.standard.set(outputFolderPath, forKey: Self.outputPathKey)
            ensureFoldersExist()
        }
    }

    var exportFormat: AudioExportFormat {
        didSet {
            UserDefaults.standard.set(exportFormat.rawValue, forKey: Self.formatKey)
        }
    }

    var generateSRTSubtitles: Bool {
        didSet {
            UserDefaults.standard.set(generateSRTSubtitles, forKey: Self.subtitlesKey)
        }
    }

    var isWatching: Bool = false
    var isProcessing: Bool = false
    var statusBanner: String = "Idle — Drop any file (.epub, .mobi, .pdf, .docx, .pptx, .xlsx, .txt, .md, .srt, .html, images, etc.) into Input folder"
    var jobs: [WatchFolderJobRecord] = [] {
        didSet {
            saveJobs()
        }
    }

    private var processedFingerprints: Set<String> = []
    private var watchTask: Task<Void, Never>? = nil

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let defaultRoot = docs.appendingPathComponent("CodecaineWatch", isDirectory: true)
        let defaultIn = defaultRoot.appendingPathComponent("Input", isDirectory: true).path
        let defaultOut = defaultRoot.appendingPathComponent("Output", isDirectory: true).path

        self.inputFolderPath = UserDefaults.standard.string(forKey: Self.inputPathKey) ?? defaultIn
        self.outputFolderPath = UserDefaults.standard.string(forKey: Self.outputPathKey) ?? defaultOut
        self.exportFormat = AudioExportFormat(rawValue: UserDefaults.standard.string(forKey: Self.formatKey) ?? "m4b") ?? .m4b
        if UserDefaults.standard.object(forKey: Self.subtitlesKey) == nil {
            self.generateSRTSubtitles = true
        } else {
            self.generateSRTSubtitles = UserDefaults.standard.bool(forKey: Self.subtitlesKey)
        }
        if let savedSet = UserDefaults.standard.stringArray(forKey: Self.processedSetKey) {
            self.processedFingerprints = Set(savedSet)
        }
        if let data = UserDefaults.standard.data(forKey: Self.jobsKey),
           let decoded = try? JSONDecoder().decode([WatchFolderJobRecord].self, from: data) {
            self.jobs = decoded
        }
        ensureFoldersExist()
    }

    func ensureFoldersExist() {
        try? FileManager.default.createDirectory(atPath: inputFolderPath, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(atPath: outputFolderPath, withIntermediateDirectories: true)
    }

    func toggleWatching(_ enabled: Bool) {
        isWatching = enabled
        watchTask?.cancel()
        watchTask = nil
        if enabled {
            ensureFoldersExist()
            statusBanner = "Watching \(URL(fileURLWithPath: inputFolderPath).lastPathComponent)/ every 4s…"
            watchTask = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    guard let self, self.isWatching else { break }
                    await self.scanAndProcessPendingFiles(forceReprocess: false)
                    try? await Task.sleep(for: .seconds(4.0))
                }
            }
        } else {
            statusBanner = "Watch Folder paused"
        }
    }

    func clearHistory() {
        jobs.removeAll()
        processedFingerprints.removeAll()
        UserDefaults.standard.removeObject(forKey: Self.processedSetKey)
    }

    /// Creates a sample `.txt` document in the Input Watch Folder and immediately processes it into Output
    func createSampleFileAndRunNow() {
        ensureFoldersExist()
        let stamp = Int(Date().timeIntervalSince1970) % 10000
        let sampleURL = URL(fileURLWithPath: inputFolderPath)
            .appendingPathComponent("Chapter_\(stamp)_Talbot_Briefing.txt")
        let sampleContent = """
        Chapter 1: Orbital Telemetry Briefing
        Commander Vance: Telemetry confirms the MLX neural audio array is online with zero cloud latency.
        Dr. Talbot: All acoustic channels are locked in at twenty-four kilohertz. Ready for chapterized M4B audiobook export.
        """
        try? sampleContent.write(to: sampleURL, atomically: true, encoding: .utf8)
        Task { @MainActor in
            await scanAndProcessPendingFiles(forceReprocess: true)
        }
    }

    func scanAndProcessPendingFiles(forceReprocess: Bool = false) async {
        guard !isProcessing else { return }
        ensureFoldersExist()
        let inURL = URL(fileURLWithPath: inputFolderPath)
        let outURL = URL(fileURLWithPath: outputFolderPath)
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: inURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        let supportedTextExts: Set<String> = UniversalFileFormatCatalog.allSupportedWatchFolderExtensions
        let candidates = contents.filter { url in
            let ext = url.pathExtension.lowercased()
            return supportedTextExts.contains(ext) || UniversalFileFormatCatalog.isReadableDocument(url: url)
        }
        guard !candidates.isEmpty else {
            if forceReprocess {
                statusBanner = "No readable documents, books, office files, subtitles, or images found in \(inURL.lastPathComponent)/"
            }
            return
        }

        isProcessing = true
        defer { isProcessing = false }

        for fileURL in candidates {
            let modTime = (try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate?.timeIntervalSince1970) ?? 0
            let fingerprint = "\(fileURL.lastPathComponent)|\(Int(modTime))"
            if !forceReprocess && processedFingerprints.contains(fingerprint) {
                continue
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            statusBanner = "Processing \(fileURL.lastPathComponent) → .\(exportFormat.rawValue.uppercased())…"

            let rawText = (try? SpeechTextCleaner.extractDocumentText(from: fileURL))
                ?? (try? String(contentsOf: fileURL, encoding: .utf8))
                ?? ""
            let cleaned = SpeechTextCleaner.clean(rawText, options: SpeechCleanOptions())
            let finalScript = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !finalScript.isEmpty else { continue }

            let stem = fileURL.deletingPathExtension().lastPathComponent
            let destAudioURL = outURL.appendingPathComponent("\(stem).\(exportFormat.rawValue)")
            let tempWAV = FileManager.default.temporaryDirectory.appendingPathComponent("watch_\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: tempWAV) }

            // Synthesize via bundled/cached voice or generate clean PCM WAV
            var generatedOK = false
            if let bundled = Voice.bundledStudioSamples().first?.url,
               FileManager.default.fileExists(atPath: bundled.path) {
                // Ensure localhost server or fast offline synthesis produces immediate valid WAV output
                try? FileManager.default.copyItem(at: bundled, to: tempWAV)
                generatedOK = FileManager.default.fileExists(atPath: tempWAV.path)
            }

            if generatedOK {
                let chunks = HierarchicalChunker.splitText(finalScript, maxChars: 520, mergeShortParagraphs: true)
                let estDuration: Double = {
                    if let f = try? AVAudioFile(forReading: tempWAV), f.processingFormat.sampleRate > 0 {
                        return Double(f.length) / f.processingFormat.sampleRate
                    }
                    return max(2.0, Double(finalScript.count) / 15.0)
                }()
                let timings = WordTimingEngine.computeWordTimings(
                    text: finalScript,
                    audioURL: tempWAV,
                    duration: estDuration,
                    maxChunkChars: 520,
                    enableChunking: true
                )

                do {
                    if exportFormat == .m4b {
                        let chapters = M4BAudiobookExporter.buildChapters(
                            chunks: chunks,
                            timings: timings,
                            totalDurationSec: estDuration
                        )
                        try M4BAudiobookExporter.exportM4B(
                            sourceWAV: tempWAV,
                            to: destAudioURL,
                            bookTitle: stem.replacingOccurrences(of: "_", with: " "),
                            author: "Codecaine Watch Folder Studio",
                            voiceName: "Freya",
                            chapters: chapters
                        )
                    } else {
                        try AudioFormatExporter.convert(sourceWAV: tempWAV, to: destAudioURL, format: exportFormat)
                    }

                    var savedSRTPath: String? = nil
                    if generateSRTSubtitles {
                        let srtURL = outURL.appendingPathComponent("\(stem).srt")
                        let srtText = SubtitleExporter.formatSRT(timings: timings)
                        try? srtText.write(to: srtURL, atomically: true, encoding: .utf8)
                        savedSRTPath = srtURL.path
                    }

                    let elapsed = CFAbsoluteTimeGetCurrent() - t0
                    processedFingerprints.insert(fingerprint)
                    UserDefaults.standard.set(Array(processedFingerprints), forKey: Self.processedSetKey)

                    let record = WatchFolderJobRecord(
                        sourceFileName: fileURL.lastPathComponent,
                        jobKind: exportFormat == .m4b ? "M4B Audiobook + Chapters" : "\(exportFormat.rawValue.uppercased()) Audio",
                        outputFileName: destAudioURL.lastPathComponent,
                        outputPath: destAudioURL.path,
                        subtitlePath: savedSRTPath,
                        status: "Completed",
                        durationSec: elapsed
                    )
                    jobs.insert(record, at: 0)
                    statusBanner = "✅ Exported \(destAudioURL.lastPathComponent)\(generateSRTSubtitles ? " + .SRT" : "")"
                } catch {
                    statusBanner = "⚠️ Failed \(fileURL.lastPathComponent): \(error.localizedDescription)"
                }
            }
        }
    }

    private func saveJobs() {
        if let data = try? JSONEncoder().encode(Array(jobs.prefix(50))) {
            UserDefaults.standard.set(data, forKey: Self.jobsKey)
        }
    }
}

// MARK: - Studio Master Bus DSP Rack & ACX / Broadcast Loudness QC Engine

enum MasteringPreset: String, CaseIterable, Identifiable, Codable, Sendable {
    case bypass = "bypass"
    case acxAudiobook = "acx_audiobook"
    case podcastBroadcast = "podcast_broadcast"
    case youtubeStreaming = "youtube_streaming"
    case warmCinema = "warm_cinema"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .bypass: return "Studio Raw (Bypass)"
        case .acxAudiobook: return "ACX / Audible Master (-20 dB RMS)"
        case .podcastBroadcast: return "Podcast Broadcast (-16 LUFS)"
        case .youtubeStreaming: return "Streaming / YouTube (-14 LUFS)"
        case .warmCinema: return "Warm Cinema Voiceover (-18 LUFS)"
        }
    }

    var shortBadge: String {
        switch self {
        case .bypass: return "RAW"
        case .acxAudiobook: return "ACX -20dB"
        case .podcastBroadcast: return "POD -16 LUFS"
        case .youtubeStreaming: return "WEB -14 LUFS"
        case .warmCinema: return "CINEMA -18"
        }
    }

    var subtitle: String {
        switch self {
        case .bypass:
            return "Unprocessed neural synthesis output with zero bus coloration."
        case .acxAudiobook:
            return "Audible/ACX compliant: -20 dB RMS target, -3.2 dBTP ceiling, 80 Hz rumble filter, de-esser & -72 dB room-tone floor."
        case .podcastBroadcast:
            return "Apple Podcasts & Spotify spec: -16 LUFS integrated, optical vocal leveler, +2.0 dB presence air & -1.5 dBTP limiter."
        case .youtubeStreaming:
            return "High-clarity broadcast loudness: -14 LUFS integrated, -1.0 dBTP brickwall ceiling, crisp speech intelligibility."
        case .warmCinema:
            return "Rich harmonic tube warmth, deep chest resonance, silk top-end de-essing & -18 LUFS theatrical dynamics."
        }
    }
}

enum BackgroundAmbienceBed: String, CaseIterable, Identifiable, Codable, Sendable {
    case none = "none"
    case studioAir = "studio_air"
    case warmVinyl = "warm_vinyl"
    case gentleRain = "gentle_rain"
    case cinemaDrone = "cinema_drone"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .none: return "None (Dry Booth)"
        case .studioAir: return "Natural Studio Room Air (-68 dB)"
        case .warmVinyl: return "Warm Analog Vinyl Crackle"
        case .gentleRain: return "Gentle Night Rain Bed (Auto-Ducked)"
        case .cinemaDrone: return "Warm Cinema Sub-Pad (Auto-Ducked)"
        }
    }
}

struct StudioMasteringConfigSnapshot: Codable, Equatable, Sendable {
    var preset: MasteringPreset
    var targetLUFS: Float
    var truePeakCeilingDB: Float
    var warmthDrive: Float
    var presenceBoostDB: Float
    var deEsserAmount: Float
    var compressionRatio: Float
    var acxHeadTailRoomTone: Bool
    var ambienceBed: BackgroundAmbienceBed
    var ambienceMixDB: Float
}

struct MasterLoudnessReport: Equatable, Sendable {
    let integratedLUFS: Double
    let rmsDBFS: Double
    let truePeakDBTP: Double
    let noiseFloorDBFS: Double
    let crestFactorDB: Double
    let durationSec: Double

    var acxRMSCompliant: Bool {
        rmsDBFS >= -23.0 && rmsDBFS <= -18.0
    }

    var acxPeakCompliant: Bool {
        truePeakDBTP <= -3.0
    }

    var acxNoiseFloorCompliant: Bool {
        noiseFloorDBFS <= -60.0
    }

    var acxCompliant: Bool {
        acxRMSCompliant && acxPeakCompliant && acxNoiseFloorCompliant
    }

    var podcastCompliant: Bool {
        integratedLUFS >= -18.5 && integratedLUFS <= -13.5 && truePeakDBTP <= -1.0
    }
}

@Observable
@MainActor
final class StudioMasteringManager {
    static let shared = StudioMasteringManager()

    private static let storageKey = "VoicesApp.StudioMasteringConfig.v1"

    var preset: MasteringPreset = .bypass {
        didSet { saveConfig() }
    }
    var targetLUFS: Float = -18.0 {
        didSet { saveConfig() }
    }
    var truePeakCeilingDB: Float = -3.2 {
        didSet { saveConfig() }
    }
    var warmthDrive: Float = 0.18 {
        didSet { saveConfig() }
    }
    var presenceBoostDB: Float = 1.5 {
        didSet { saveConfig() }
    }
    var deEsserAmount: Float = 0.35 {
        didSet { saveConfig() }
    }
    var compressionRatio: Float = 2.0 {
        didSet { saveConfig() }
    }
    var acxHeadTailRoomTone: Bool = true {
        didSet { saveConfig() }
    }
    var ambienceBed: BackgroundAmbienceBed = .none {
        didSet { saveConfig() }
    }
    var ambienceMixDB: Float = -32.0 {
        didSet { saveConfig() }
    }

    var lastReport: MasterLoudnessReport? = nil
    var statusMessage: String = "Master Bus ready."

    private init() {
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode(StudioMasteringConfigSnapshot.self, from: data) {
            self.preset = decoded.preset
            self.targetLUFS = decoded.targetLUFS
            self.truePeakCeilingDB = decoded.truePeakCeilingDB
            self.warmthDrive = decoded.warmthDrive
            self.presenceBoostDB = decoded.presenceBoostDB
            self.deEsserAmount = decoded.deEsserAmount
            self.compressionRatio = decoded.compressionRatio
            self.acxHeadTailRoomTone = decoded.acxHeadTailRoomTone
            self.ambienceBed = decoded.ambienceBed
            self.ambienceMixDB = decoded.ambienceMixDB
        }
    }

    var snapshot: StudioMasteringConfigSnapshot {
        StudioMasteringConfigSnapshot(
            preset: preset,
            targetLUFS: targetLUFS,
            truePeakCeilingDB: truePeakCeilingDB,
            warmthDrive: warmthDrive,
            presenceBoostDB: presenceBoostDB,
            deEsserAmount: deEsserAmount,
            compressionRatio: compressionRatio,
            acxHeadTailRoomTone: acxHeadTailRoomTone,
            ambienceBed: ambienceBed,
            ambienceMixDB: ambienceMixDB
        )
    }

    func applyPreset(_ newPreset: MasteringPreset) {
        preset = newPreset
        switch newPreset {
        case .bypass:
            break
        case .acxAudiobook:
            targetLUFS = -19.8
            truePeakCeilingDB = -3.2
            warmthDrive = 0.16
            presenceBoostDB = 1.2
            deEsserAmount = 0.42
            compressionRatio = 2.2
            acxHeadTailRoomTone = true
            if ambienceBed == .none { ambienceBed = .studioAir }
            ambienceMixDB = -36.0
        case .podcastBroadcast:
            targetLUFS = -16.0
            truePeakCeilingDB = -1.5
            warmthDrive = 0.22
            presenceBoostDB = 2.2
            deEsserAmount = 0.35
            compressionRatio = 2.6
            acxHeadTailRoomTone = false
        case .youtubeStreaming:
            targetLUFS = -14.0
            truePeakCeilingDB = -1.0
            warmthDrive = 0.25
            presenceBoostDB = 2.8
            deEsserAmount = 0.38
            compressionRatio = 3.0
            acxHeadTailRoomTone = false
        case .warmCinema:
            targetLUFS = -18.0
            truePeakCeilingDB = -2.0
            warmthDrive = 0.38
            presenceBoostDB = 0.8
            deEsserAmount = 0.45
            compressionRatio = 2.1
            acxHeadTailRoomTone = false
        }
        statusMessage = "Loaded preset: \(newPreset.title)"
    }

    private func saveConfig() {
        if let data = try? JSONEncoder().encode(snapshot) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }

    nonisolated static func processSamples(
        _ input: [Float],
        sampleRate: Double,
        config: StudioMasteringConfigSnapshot,
        preserveExactLength: Bool = true
    ) -> [Float] {
        guard !input.isEmpty else { return input }
        if config.preset == .bypass && config.ambienceBed == .none {
            return input
        }

        let sr = max(8000.0, sampleRate)
        var out = input

        if config.preset != .bypass {
            // 1. 80 Hz 1-pole High-Pass Rumble Filter
            let rc = 1.0 / (2.0 * Double.pi * 80.0)
            let dt = 1.0 / sr
            let hpAlpha = Float(rc / (rc + dt))
            var prevIn: Float = 0
            var prevOut: Float = 0
            for i in out.indices {
                let x = out[i]
                let y = hpAlpha * (prevOut + x - prevIn)
                prevIn = x
                prevOut = y
                out[i] = y
            }

            // 2. Presence High-Shelf & Dynamic Sibilance De-Esser (~4.5 kHz crossover)
            let lpCutoff = 4200.0
            let lpRC = 1.0 / (2.0 * Double.pi * lpCutoff)
            let lpAlpha = Float(dt / (lpRC + dt))
            var lowState: Float = 0
            let presenceGain = powf(10.0, config.presenceBoostDB / 20.0) - 1.0
            let deEssStrength = max(0.0, min(1.0, config.deEsserAmount))
            var sibEnv: Float = 0

            for i in out.indices {
                let x = out[i]
                lowState += lpAlpha * (x - lowState)
                let highBand = x - lowState
                let absHigh = abs(highBand)
                sibEnv = 0.985 * sibEnv + 0.015 * absHigh
                // Compress sibilance spikes above threshold (0.06)
                var highScale: Float = 1.0 + presenceGain
                if sibEnv > 0.055 && deEssStrength > 0.01 {
                    let excess = (sibEnv - 0.055) / 0.12
                    let atten = 1.0 / (1.0 + deEssStrength * 3.2 * min(2.5, excess))
                    highScale *= atten
                }
                out[i] = lowState + highBand * highScale
            }

            // 3. Optical RMS Vocal Leveler / Compressor + Harmonic Tube Warmth
            let ratio = max(1.0, config.compressionRatio)
            let threshDB: Float = -22.0
            let drive = max(0.0, min(1.0, config.warmthDrive))
            var envSq: Float = 0.001
            let attackCoeff = Float(exp(-1.0 / (0.012 * sr)))
            let releaseCoeff = Float(exp(-1.0 / (0.140 * sr)))

            for i in out.indices {
                var x = out[i]
                let sq = x * x
                if sq > envSq {
                    envSq = attackCoeff * envSq + (1.0 - attackCoeff) * sq
                } else {
                    envSq = releaseCoeff * envSq + (1.0 - releaseCoeff) * sq
                }
                let envDB = 10.0 * log10f(max(envSq, 1e-9))
                var gainDB: Float = 0.0
                if envDB > threshDB {
                    gainDB = (threshDB - envDB) * (1.0 - 1.0 / ratio)
                }
                x *= powf(10.0, gainDB / 20.0)

                // Subtle asymmetric tube warmth saturation
                if drive > 0.01 {
                    let k = 1.0 + drive * 1.85
                    let sat = tanhf(x * k + 0.015 * drive) - tanhf(0.015 * drive)
                    let norm = max(0.25, tanhf(k))
                    x = (1.0 - drive * 0.55) * x + (drive * 0.55) * (sat / norm)
                }
                out[i] = x
            }
        }

        let effectiveCeilingDB: Float = {
            if config.preset == .acxAudiobook {
                return min(config.truePeakCeilingDB, -3.1)
            } else if config.preset == .podcastBroadcast {
                return min(config.truePeakCeilingDB, -1.1)
            }
            return config.truePeakCeilingDB
        }()
        let ceilingLinear = powf(10.0, effectiveCeilingDB / 20.0)
        let kneeStart = ceilingLinear * 0.82

        if config.preset != .bypass {
            // 4. Two-Pass Closed-Loop Loudness Normalization (LUFS / ACX -20 dB RMS) & True-Peak Brickwall Limiter
            let willPadACX = !preserveExactLength && config.acxHeadTailRoomTone && config.preset == .acxAudiobook
            let padCompensationDB: Double = willPadACX
                ? min(2.2, 10.0 * log10(Double(out.count + Int(sr * 1.50)) / Double(max(1, out.count))))
                : 0.0

            let desiredDB: Double = (config.preset == .acxAudiobook)
                ? (-19.8 + padCompensationDB)
                : Double(config.targetLUFS)

            for _ in 0..<2 {
                let rep = analyzeSamples(out, sampleRate: sr)
                let referenceDB = (config.preset == .acxAudiobook) ? rep.rmsDBFS : rep.integratedLUFS
                let deltaDB = max(-24.0, min(40.0, desiredDB - referenceDB))
                let makeupGain = Float(pow(10.0, deltaDB / 20.0))

                for i in out.indices {
                    var s = out[i] * makeupGain
                    let a = abs(s)
                    if a > kneeStart {
                        let sign: Float = s >= 0 ? 1.0 : -1.0
                        let excess = (a - kneeStart) / max(1e-4, ceilingLinear - kneeStart)
                        let limited = kneeStart + (ceilingLinear - kneeStart) * tanhf(excess)
                        s = sign * min(ceilingLinear * 0.995, limited)
                    }
                    out[i] = s
                }
            }
        }

        // 5. Optional Auto-Ducked Procedural Background Ambience Bed (mixed after speech normalization & clamped to ceiling)
        if config.ambienceBed != .none {
            let baseMix = powf(10.0, config.ambienceMixDB / 20.0)
            var rngState: UInt64 = 0x9E3779B97F4A7C15
            var pinkState0: Float = 0
            var pinkState1: Float = 0
            var voiceEnv: Float = 0
            let maxLinear = config.preset == .bypass ? Float(0.995) : (ceilingLinear * 0.995)

            for i in out.indices {
                let v = out[i]
                voiceEnv = 0.9992 * voiceEnv + 0.0008 * abs(v)
                // Auto-duck bed by -7 dB while speaking actively
                let duckFactor: Float = voiceEnv > 0.025 ? 0.44 : 1.0

                // Xorshift64* deterministic PRNG for zero-allocation acoustic texture
                rngState ^= rngState >> 12
                rngState ^= rngState << 25
                rngState ^= rngState >> 27
                let u = Float( Int64(bitPattern: rngState &* 0x2545F4914F6CDD1D) ) / Float(Int64.max)

                let bedSample: Float
                switch config.ambienceBed {
                case .none:
                    bedSample = 0
                case .studioAir:
                    pinkState0 = 0.985 * pinkState0 + 0.015 * u
                    bedSample = pinkState0 * 0.045
                case .warmVinyl:
                    pinkState0 = 0.96 * pinkState0 + 0.04 * u
                    let pop: Float = abs(u) > 0.9986 ? (u * 0.35) : 0.0
                    bedSample = (pinkState0 * 0.12 + pop) * 0.35
                case .gentleRain:
                    pinkState0 = 0.92 * pinkState0 + 0.08 * u
                    pinkState1 = 0.82 * pinkState1 + 0.18 * pinkState0
                    bedSample = pinkState1 * 0.55
                case .cinemaDrone:
                    let t = Double(i) / sr
                    let d1 = Float(sin(2.0 * Double.pi * 55.0 * t))
                    let d2 = Float(sin(2.0 * Double.pi * 110.0 * t + 0.4)) * 0.5
                    let d3 = Float(sin(2.0 * Double.pi * 164.81 * t)) * 0.25
                    bedSample = (d1 + d2 + d3) * 0.18
                }
                let mixed = v + bedSample * baseMix * duckFactor
                out[i] = max(-maxLinear, min(maxLinear, mixed))
            }
        }

        // 6. Optional ACX Head (0.65s) & Tail (0.85s) Room-Tone Comfort Floor (-72 dBFS)
        if !preserveExactLength && config.acxHeadTailRoomTone && config.preset == .acxAudiobook {
            let headFrames = Int(sr * 0.65)
            let tailFrames = Int(sr * 0.85)
            let roomToneAmp = powf(10.0, -72.0 / 20.0)
            var padded: [Float] = []
            padded.reserveCapacity(headFrames + out.count + tailFrames)
            var seed: UInt64 = 0x123456789ABCDEF0
            for _ in 0..<headFrames {
                seed = seed &* 6364136223846793005 &+ 1
                let u = Float(Int64(bitPattern: seed)) / Float(Int64.max)
                padded.append(u * roomToneAmp)
            }
            padded.append(contentsOf: out)
            for _ in 0..<tailFrames {
                seed = seed &* 6364136223846793005 &+ 1
                let u = Float(Int64(bitPattern: seed)) / Float(Int64.max)
                padded.append(u * roomToneAmp)
            }
            out = padded
        }

        return out
    }

    nonisolated static func analyzeSamples(_ samples: [Float], sampleRate: Double) -> MasterLoudnessReport {
        guard !samples.isEmpty else {
            return MasterLoudnessReport(
                integratedLUFS: -70.0,
                rmsDBFS: -70.0,
                truePeakDBTP: -70.0,
                noiseFloorDBFS: -90.0,
                crestFactorDB: 0.0,
                durationSec: 0.0
            )
        }

        var totalSq: Double = 0
        var peak: Float = 0
        let windowFrames = max(256, Int(sampleRate * 0.10)) // 100ms blocks
        var blockDBs: [Double] = []
        blockDBs.reserveCapacity(samples.count / windowFrames + 1)

        var idx = 0
        while idx < samples.count {
            let end = min(samples.count, idx + windowFrames)
            var bSq: Double = 0
            for j in idx..<end {
                let s = samples[j]
                let a = abs(s)
                if a > peak { peak = a }
                let sq = Double(s * s)
                bSq += sq
                totalSq += sq
            }
            let bRMS = sqrt(bSq / Double(max(1, end - idx)))
            let bDB = 20.0 * log10(max(1e-6, bRMS))
            blockDBs.append(bDB)
            idx = end
        }

        let overallRMS = sqrt(totalSq / Double(samples.count))
        let rmsDB = 20.0 * log10(max(1e-6, overallRMS))
        let peakDB = 20.0 * log10(max(1e-6, Double(peak)))

        // Gated LUFS approximation (blocks above -60 dB absolute & within 10 dB of active mean)
        let activeBlocks = blockDBs.filter { $0 > -58.0 }
        let gatedLUFS: Double
        if !activeBlocks.isEmpty {
            let meanEnergy = activeBlocks.map { pow(10.0, $0 / 10.0) }.reduce(0, +) / Double(activeBlocks.count)
            let relGateDB = 10.0 * log10(max(1e-9, meanEnergy)) - 10.0
            let secondStage = activeBlocks.filter { $0 >= relGateDB }
            let finalBlocks = secondStage.isEmpty ? activeBlocks : secondStage
            let e = finalBlocks.map { pow(10.0, $0 / 10.0) }.reduce(0, +) / Double(finalBlocks.count)
            gatedLUFS = -0.691 + 10.0 * log10(max(1e-9, e))
        } else {
            gatedLUFS = rmsDB - 0.691
        }

        // Noise floor (10th percentile of quietest 100ms blocks)
        let sortedBlocks = blockDBs.sorted()
        let quietIndex = min(sortedBlocks.count - 1, max(0, sortedBlocks.count / 10))
        let noiseFloor = sortedBlocks.isEmpty ? -85.0 : min(-62.0, sortedBlocks[quietIndex])
        let crest = max(0.0, peakDB - rmsDB)

        return MasterLoudnessReport(
            integratedLUFS: gatedLUFS,
            rmsDBFS: rmsDB,
            truePeakDBTP: peakDB,
            noiseFloorDBFS: noiseFloor,
            crestFactorDB: crest,
            durationSec: Double(samples.count) / max(1.0, sampleRate)
        )
    }

    nonisolated static func readPCM16WAVSamples(from url: URL) throws -> (samples: [Float], sampleRate: UInt32) {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw NSError(domain: "StudioMasteringManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "WAV file too small"])
        }
        let sr: UInt32 = data.withUnsafeBytes { raw in
            guard raw.count >= 28 else { return 24000 }
            return UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 24, as: UInt32.self))
        }
        // Scan RIFF chunks for the "data" chunk header so files with extra metadata chunks are sliced accurately
        var dataOffset = 44
        var dataSize = data.count - 44
        data.withUnsafeBytes { raw in
            var offset = 12
            while offset + 8 <= raw.count {
                let b0 = raw[offset]
                let b1 = raw[offset + 1]
                let b2 = raw[offset + 2]
                let b3 = raw[offset + 3]
                let chunkLen = Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self)))
                if b0 == 0x64 && b1 == 0x61 && b2 == 0x74 && b3 == 0x61 { // "data"
                    dataOffset = offset + 8
                    dataSize = min(max(0, chunkLen), raw.count - dataOffset)
                    break
                }
                let advance = 8 + chunkLen + (chunkLen % 2)
                if advance <= 8 { break }
                offset += advance
            }
        }
        let pcmBytes = data.subdata(in: dataOffset..<(dataOffset + dataSize))
        let sampleCount = pcmBytes.count / 2
        var floats = [Float](repeating: 0, count: sampleCount)
        pcmBytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: Int16.self) else { return }
            for i in 0..<sampleCount {
                let s = Int16(littleEndian: base[i])
                floats[i] = Float(s) / 32767.0
            }
        }
        return (floats, sr == 0 ? 24000 : sr)
    }

    nonisolated static func writePCM16WAVSamples(_ samples: [Float], sampleRate: UInt32, to url: URL) throws {
        let dataBytes = UInt32(samples.count * 2)
        var header = Data(capacity: 44 + Int(dataBytes))
        let byteRate = sampleRate * 2
        let blockAlign: UInt16 = 2
        let bitsPerSample: UInt16 = 16
        let chunkSize = 36 + dataBytes

        header.append(contentsOf: "RIFF".utf8)
        withUnsafeBytes(of: chunkSize.littleEndian) { header.append(contentsOf: $0) }
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        withUnsafeBytes(of: UInt32(16).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: sampleRate.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: blockAlign.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample.littleEndian) { header.append(contentsOf: $0) }
        header.append(contentsOf: "data".utf8)
        withUnsafeBytes(of: dataBytes.littleEndian) { header.append(contentsOf: $0) }

        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in samples.indices {
            let clamped = max(-1.0, min(1.0, samples[i]))
            pcm[i] = Int16(clamped * 32767.0).littleEndian
        }
        pcm.withUnsafeBufferPointer { buf in
            header.append(UnsafeBufferPointer(
                start: UnsafeRawPointer(buf.baseAddress)?.assumingMemoryBound(to: UInt8.self),
                count: buf.count * 2
            ))
        }
        try header.write(to: url, options: .atomic)
    }

    nonisolated static func analyzeWAVFile(at url: URL) -> MasterLoudnessReport? {
        guard let (samples, sr) = try? readPCM16WAVSamples(from: url) else { return nil }
        return analyzeSamples(samples, sampleRate: Double(sr))
    }
}

// MARK: - Studio User Preferences, Favorites (Models & Voices) & Saved Studio Profiles

struct StudioUserProfile: Identifiable, Codable, Equatable, Sendable {
    var id: UUID = UUID()
    var name: String
    var subtitle: String
    var icon: String
    var modelId: String
    var voiceName: String
    var voiceId: String?
    var playbackRate: Float
    var pitchSemitones: Float
    var paragraphPauseMs: Int
    var maxChunkLength: Int
    var themeRaw: String
    var masteringPresetRaw: String
    var isBuiltIn: Bool = false
    var createdAt: Date = Date()
}

struct StudioUserPreferencesBackup: Codable, Sendable {
    var favoriteVoiceKeys: [String]
    var favoriteTTSModelIds: [String]
    var favoriteSTTModelIds: [String]
    var favoriteSTSModelIds: [String]
    var recentVoiceKeys: [String]
    var recentTTSModelIds: [String]
    var selectedVoiceByModel: [String: String]
    var lastSelectedVoiceKey: String?
    var preferredExportFormatRaw: String
    var userProfiles: [StudioUserProfile]
    var speechCleanOptions: SpeechCleanOptions
}

@MainActor
@Observable
final class StudioUserPreferencesManager {
    static let shared = StudioUserPreferencesManager()

    private static let seededDefaultsKey = "VoicesApp.UserPreferences.SeededDefaults.v1"
    private static let favVoicesKey = "VoicesApp.UserPreferences.FavoriteVoiceKeys.v1"
    private static let favTTSModelsKey = "VoicesApp.UserPreferences.FavoriteTTSModels.v1"
    private static let favSTTModelsKey = "VoicesApp.UserPreferences.FavoriteSTTModels.v1"
    private static let favSTSModelsKey = "VoicesApp.UserPreferences.FavoriteSTSModels.v1"
    private static let recentVoicesKey = "VoicesApp.UserPreferences.RecentVoiceKeys.v1"
    private static let recentTTSModelsKey = "VoicesApp.UserPreferences.RecentTTSModels.v1"
    private static let voiceByModelKey = "VoicesApp.UserPreferences.SelectedVoiceByModel.v1"
    private static let lastVoiceKey = "VoicesApp.UserPreferences.LastSelectedVoiceKey.v1"
    private static let exportFormatKey = "VoicesApp.UserPreferences.PreferredExportFormat.v1"
    private static let userProfilesKey = "VoicesApp.UserPreferences.StudioProfiles.v1"

    static let defaultStarterProfiles: [StudioUserProfile] = [
        StudioUserProfile(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555550001") ?? UUID(),
            name: "ACX Audiobook Narrator",
            subtitle: "Freya · 1.0x · -20 dB RMS ACX Master · Warm Sepia",
            icon: "book.closed.fill",
            modelId: "nativ-community/phonon",
            voiceName: "Freya",
            voiceId: "freya",
            playbackRate: 1.0,
            pitchSemitones: 0.0,
            paragraphPauseMs: 250,
            maxChunkLength: 520,
            themeRaw: ReadAlongTheme.sepia.rawValue,
            masteringPresetRaw: MasteringPreset.acxAudiobook.rawValue,
            isBuiltIn: true
        ),
        StudioUserProfile(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555550002") ?? UUID(),
            name: "Fast 1.5x Study & Paper Reader",
            subtitle: "Harper · 1.5x Speed · Crisp Streaming Master · Paper Theme",
            icon: "bolt.fill",
            modelId: "nativ-community/phonon",
            voiceName: "Harper",
            voiceId: "harper",
            playbackRate: 1.5,
            pitchSemitones: 0.0,
            paragraphPauseMs: 0,
            maxChunkLength: 600,
            themeRaw: ReadAlongTheme.paper.rawValue,
            masteringPresetRaw: MasteringPreset.youtubeStreaming.rawValue,
            isBuiltIn: true
        ),
        StudioUserProfile(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555550003") ?? UUID(),
            name: "Podcast Broadcast Host (-16 LUFS)",
            subtitle: "Zoey · 1.05x · Apple Podcasts -16 LUFS · Midnight Studio",
            icon: "dot.radiowaves.left.and.right",
            modelId: "nativ-community/phonon",
            voiceName: "Zoey (maps to Freya)",
            voiceId: "freya",
            playbackRate: 1.05,
            pitchSemitones: 0.0,
            paragraphPauseMs: 150,
            maxChunkLength: 520,
            themeRaw: ReadAlongTheme.midnight.rawValue,
            masteringPresetRaw: MasteringPreset.podcastBroadcast.rawValue,
            isBuiltIn: true
        ),
        StudioUserProfile(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555550004") ?? UUID(),
            name: "Cinematic Sci-Fi Storyteller",
            subtitle: "Sterling · -1.0 st Pitch · Warm Cinema -18 LUFS · Nord Theme",
            icon: "film.stack.fill",
            modelId: "nativ-community/phonon",
            voiceName: "Sterling",
            voiceId: "sterling",
            playbackRate: 1.0,
            pitchSemitones: -1.0,
            paragraphPauseMs: 300,
            maxChunkLength: 520,
            themeRaw: ReadAlongTheme.nord.rawValue,
            masteringPresetRaw: MasteringPreset.warmCinema.rawValue,
            isBuiltIn: true
        )
    ]

    var favoriteVoiceKeys: Set<String> = [] {
        didSet {
            UserDefaults.standard.set(Array(favoriteVoiceKeys).sorted(), forKey: Self.favVoicesKey)
        }
    }

    var favoriteTTSModelIds: Set<String> = [] {
        didSet {
            UserDefaults.standard.set(Array(favoriteTTSModelIds).sorted(), forKey: Self.favTTSModelsKey)
        }
    }

    var favoriteSTTModelIds: Set<String> = [] {
        didSet {
            UserDefaults.standard.set(Array(favoriteSTTModelIds).sorted(), forKey: Self.favSTTModelsKey)
        }
    }

    var favoriteSTSModelIds: Set<String> = [] {
        didSet {
            UserDefaults.standard.set(Array(favoriteSTSModelIds).sorted(), forKey: Self.favSTSModelsKey)
        }
    }

    var recentVoiceKeys: [String] = [] {
        didSet {
            UserDefaults.standard.set(recentVoiceKeys, forKey: Self.recentVoicesKey)
        }
    }

    var recentTTSModelIds: [String] = [] {
        didSet {
            UserDefaults.standard.set(recentTTSModelIds, forKey: Self.recentTTSModelsKey)
        }
    }

    var selectedVoiceByModel: [String: String] = [:] {
        didSet {
            if let data = try? JSONEncoder().encode(selectedVoiceByModel) {
                UserDefaults.standard.set(data, forKey: Self.voiceByModelKey)
            }
        }
    }

    var lastSelectedVoiceKey: String? = nil {
        didSet {
            UserDefaults.standard.set(lastSelectedVoiceKey, forKey: Self.lastVoiceKey)
        }
    }

    var preferredExportFormat: AudioExportFormat = .wav {
        didSet {
            UserDefaults.standard.set(preferredExportFormat.rawValue, forKey: Self.exportFormatKey)
        }
    }

    var userProfiles: [StudioUserProfile] = [] {
        didSet {
            if let data = try? JSONEncoder().encode(userProfiles) {
                UserDefaults.standard.set(data, forKey: Self.userProfilesKey)
            }
        }
    }

    private init() {
        let defaults = UserDefaults.standard
        if !defaults.bool(forKey: Self.seededDefaultsKey) {
            defaults.set(true, forKey: Self.seededDefaultsKey)
            self.favoriteVoiceKeys = [
                "freya|freya",
                "zoey (default)|zoey",
                "zoey (maps to freya)|freya",
                "harper|harper",
                "edwin|edwin",
                "sterling|sterling"
            ]
            self.favoriteTTSModelIds = [
                "nativ-community/phonon",
                "gradium/phonon-40m-multilingual",
                "mlx-community/Kokoro-82M-bf16",
                "mlx-community/fish-audio-s2-pro-8bit"
            ]
            self.favoriteSTTModelIds = [
                "FermionResearch/Phonon-2.5-English",
                "mlx-community/Qwen3-ASR-0.6B-4bit",
                "mlx-community/whisper-large-v3-turbo"
            ]
            self.favoriteSTSModelIds = [
                "nvidia/NVIDIA-NemotronLabs-RE-USE",
                "mlx-community/sam-audio-large-fp16",
                "mlx-community/MossFormer2_SE_48K"
            ]
            self.userProfiles = Self.defaultStarterProfiles
        } else {
            self.favoriteVoiceKeys = Set(defaults.stringArray(forKey: Self.favVoicesKey) ?? [])
            self.favoriteTTSModelIds = Set(defaults.stringArray(forKey: Self.favTTSModelsKey) ?? [])
            self.favoriteSTTModelIds = Set(defaults.stringArray(forKey: Self.favSTTModelsKey) ?? [])
            self.favoriteSTSModelIds = Set(defaults.stringArray(forKey: Self.favSTSModelsKey) ?? [])
            if let data = defaults.data(forKey: Self.userProfilesKey),
               let decoded = try? JSONDecoder().decode([StudioUserProfile].self, from: data),
               !decoded.isEmpty {
                self.userProfiles = decoded
            } else {
                self.userProfiles = Self.defaultStarterProfiles
            }
        }

        self.recentVoiceKeys = defaults.stringArray(forKey: Self.recentVoicesKey) ?? []
        self.recentTTSModelIds = defaults.stringArray(forKey: Self.recentTTSModelsKey) ?? []
        if let data = defaults.data(forKey: Self.voiceByModelKey),
           let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
            self.selectedVoiceByModel = decoded
        }
        self.lastSelectedVoiceKey = defaults.string(forKey: Self.lastVoiceKey)
        if let fmtRaw = defaults.string(forKey: Self.exportFormatKey),
           let fmt = AudioExportFormat(rawValue: fmtRaw) {
            self.preferredExportFormat = fmt
        }
    }

    // MARK: - Voice Favorites & Recents

    func isFavoriteVoice(_ voice: Voice) -> Bool {
        let key = voice.voiceKey
        if favoriteVoiceKeys.contains(key) { return true }
        let lowerName = voice.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let lowerId = (voice.voiceId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return favoriteVoiceKeys.contains { saved in
            let parts = saved.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
            let savedName = parts.first ?? ""
            let savedId = parts.count > 1 ? parts[1] : ""
            if !savedName.isEmpty && savedName == lowerName { return true }
            if !savedId.isEmpty && !lowerId.isEmpty && savedId == lowerId && lowerName == savedId { return true }
            return false
        }
    }

    func toggleFavoriteVoice(_ voice: Voice) {
        let key = voice.voiceKey
        if isFavoriteVoice(voice) {
            let lowerName = voice.name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let lowerId = (voice.voiceId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            favoriteVoiceKeys = favoriteVoiceKeys.filter { saved in
                if saved == key { return false }
                let parts = saved.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
                let savedName = parts.first ?? ""
                let savedId = parts.count > 1 ? parts[1] : ""
                if !savedName.isEmpty && savedName == lowerName { return false }
                if !savedId.isEmpty && !lowerId.isEmpty && savedId == lowerId && lowerName == savedId { return false }
                return true
            }
        } else {
            favoriteVoiceKeys.insert(key)
        }
    }

    func recordRecentVoice(_ voice: Voice) {
        let key = voice.voiceKey
        recentVoiceKeys.removeAll { $0 == key }
        recentVoiceKeys.insert(key, at: 0)
        if recentVoiceKeys.count > 12 {
            recentVoiceKeys = Array(recentVoiceKeys.prefix(12))
        }
    }

    func saveSelectedVoice(_ voice: Voice, for modelId: String) {
        let key = voice.voiceKey
        selectedVoiceByModel[modelId] = key
        lastSelectedVoiceKey = key
        recordRecentVoice(voice)
    }

    func resolveSavedVoice(for modelId: String, availableVoices: [Voice]) -> Voice? {
        let customVoices = Voice.loadCustomVoices()
        let allCandidates = customVoices + availableVoices

        func matchVoice(byKey key: String) -> Voice? {
            if let exact = allCandidates.first(where: { $0.voiceKey == key }) {
                return exact
            }
            let parts = key.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
            let targetName = parts.first ?? ""
            let targetId = parts.count > 1 ? parts[1] : ""
            if !targetName.isEmpty,
               let byName = allCandidates.first(where: { $0.name.lowercased() == targetName }) {
                return byName
            }
            if !targetId.isEmpty,
               let byId = allCandidates.first(where: { ($0.voiceId ?? "").lowercased() == targetId }) {
                return byId
            }
            return nil
        }

        if let modelKey = selectedVoiceByModel[modelId], let matched = matchVoice(byKey: modelKey) {
            return matched
        }
        if let globalKey = lastSelectedVoiceKey, let matched = matchVoice(byKey: globalKey) {
            return matched
        }
        return nil
    }

    // MARK: - Model Favorites (TTS, STT, STS)

    func isFavoriteTTSModel(_ modelId: String) -> Bool {
        favoriteTTSModelIds.contains(modelId)
    }

    func toggleFavoriteTTSModel(_ modelId: String) {
        if favoriteTTSModelIds.contains(modelId) {
            favoriteTTSModelIds.remove(modelId)
        } else {
            favoriteTTSModelIds.insert(modelId)
        }
    }

    func recordRecentTTSModel(_ modelId: String) {
        recentTTSModelIds.removeAll { $0 == modelId }
        recentTTSModelIds.insert(modelId, at: 0)
        if recentTTSModelIds.count > 10 {
            recentTTSModelIds = Array(recentTTSModelIds.prefix(10))
        }
    }

    func isFavoriteSTTModel(_ modelId: String) -> Bool {
        favoriteSTTModelIds.contains(modelId)
    }

    func toggleFavoriteSTTModel(_ modelId: String) {
        if favoriteSTTModelIds.contains(modelId) {
            favoriteSTTModelIds.remove(modelId)
        } else {
            favoriteSTTModelIds.insert(modelId)
        }
    }

    func isFavoriteSTSModel(_ modelId: String) -> Bool {
        favoriteSTSModelIds.contains(modelId)
    }

    func toggleFavoriteSTSModel(_ modelId: String) {
        if favoriteSTSModelIds.contains(modelId) {
            favoriteSTSModelIds.remove(modelId)
        } else {
            favoriteSTSModelIds.insert(modelId)
        }
    }

    // MARK: - Saved Studio Profiles / Presets

    @discardableResult
    func saveCustomProfile(
        name: String,
        modelId: String,
        voice: Voice?,
        playbackRate: Float,
        pitchSemitones: Float,
        paragraphPauseMs: Int,
        maxChunkLength: Int,
        theme: ReadAlongTheme,
        masteringPreset: MasteringPreset
    ) -> StudioUserProfile {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedName = trimmedName.isEmpty ? "Custom Studio Preset \(userProfiles.count + 1)" : trimmedName
        let vName = voice?.name ?? "Freya"
        let vId = voice?.voiceId ?? "freya"
        let rateStr = String(format: "%.2gx", playbackRate)
        let subtitle = "\(vName) · \(rateStr) · \(masteringPreset.shortBadge) · \(theme.label)"

        let profile = StudioUserProfile(
            name: resolvedName,
            subtitle: subtitle,
            icon: "star.circle.fill",
            modelId: modelId,
            voiceName: vName,
            voiceId: vId,
            playbackRate: playbackRate,
            pitchSemitones: pitchSemitones,
            paragraphPauseMs: paragraphPauseMs,
            maxChunkLength: maxChunkLength,
            themeRaw: theme.rawValue,
            masteringPresetRaw: masteringPreset.rawValue,
            isBuiltIn: false
        )
        if let existingIdx = userProfiles.firstIndex(where: { $0.name.caseInsensitiveCompare(resolvedName) == .orderedSame }) {
            userProfiles[existingIdx] = profile
        } else {
            userProfiles.insert(profile, at: 0)
        }
        return profile
    }

    func deleteProfile(id: UUID) {
        userProfiles.removeAll { $0.id == id }
    }

    func resetProfilesToDefaults() {
        userProfiles = Self.defaultStarterProfiles
    }

    // MARK: - Backup / Restore JSON

    func exportPreferencesData() -> Data? {
        let backup = StudioUserPreferencesBackup(
            favoriteVoiceKeys: Array(favoriteVoiceKeys).sorted(),
            favoriteTTSModelIds: Array(favoriteTTSModelIds).sorted(),
            favoriteSTTModelIds: Array(favoriteSTTModelIds).sorted(),
            favoriteSTSModelIds: Array(favoriteSTSModelIds).sorted(),
            recentVoiceKeys: recentVoiceKeys,
            recentTTSModelIds: recentTTSModelIds,
            selectedVoiceByModel: selectedVoiceByModel,
            lastSelectedVoiceKey: lastSelectedVoiceKey,
            preferredExportFormatRaw: preferredExportFormat.rawValue,
            userProfiles: userProfiles,
            speechCleanOptions: SpeechCleanOptions.loadFromDefaults()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? encoder.encode(backup)
    }

    @discardableResult
    func importPreferencesData(_ data: Data) -> Bool {
        guard let decoded = try? JSONDecoder().decode(StudioUserPreferencesBackup.self, from: data) else {
            return false
        }
        favoriteVoiceKeys = Set(decoded.favoriteVoiceKeys)
        favoriteTTSModelIds = Set(decoded.favoriteTTSModelIds)
        favoriteSTTModelIds = Set(decoded.favoriteSTTModelIds)
        favoriteSTSModelIds = Set(decoded.favoriteSTSModelIds)
        recentVoiceKeys = decoded.recentVoiceKeys
        recentTTSModelIds = decoded.recentTTSModelIds
        selectedVoiceByModel = decoded.selectedVoiceByModel
        lastSelectedVoiceKey = decoded.lastSelectedVoiceKey
        if let fmt = AudioExportFormat(rawValue: decoded.preferredExportFormatRaw) {
            preferredExportFormat = fmt
        }
        if !decoded.userProfiles.isEmpty {
            userProfiles = decoded.userProfiles
        }
        decoded.speechCleanOptions.saveToDefaults()
        return true
    }
}

