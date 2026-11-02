import AVFoundation
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS
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

    static func modelDirectory(for repoId: String) -> URL? {
        let subdir = repoId.replacingOccurrences(of: "/", with: "_")
        let mlxDir = mlxAudioCacheRoot.appendingPathComponent(subdir)
        if directoryHasWeights(mlxDir) {
            return mlxDir
        }
        let hubSubdir = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        let hubSnapshotsDir = hubCacheRoot.appendingPathComponent(hubSubdir).appendingPathComponent("snapshots")
        if let snapshots = try? FileManager.default.contentsOfDirectory(at: hubSnapshotsDir, includingPropertiesForKeys: nil),
           let first = snapshots.first,
           directoryHasWeights(first) {
            return first
        }
        return nil
    }

    private static func directoryHasWeights(_ dir: URL) -> Bool {
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
            if ["safetensors", "npz", "bin", "gguf", "pt", "onnx"].contains(ext) {
                let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                if size > 0 { return true }
            }
        }
        return false
    }

    static func isModelCached(_ repoId: String) -> Bool {
        modelDirectory(for: repoId) != nil
    }

    static func cachedModelSizeString(_ repoId: String) -> String? {
        guard let dir = modelDirectory(for: repoId) else { return nil }
        let bytes = directoryByteSize(dir)
        guard bytes > 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    static func directoryByteSize(_ dir: URL) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: []
        ) else { return 0 }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
               values.isRegularFile == true,
               let size = values.fileSize {
                total += UInt64(max(size, 0))
            }
        }
        return total
    }

    static func revealCacheInFinder() {
        #if os(macOS)
        let target = FileManager.default.fileExists(atPath: mlxAudioCacheRoot.path) ? mlxAudioCacheRoot : hubCacheRoot
        NSWorkspace.shared.open(target)
        #endif
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
            || lower.contains("vyvotts") {
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
        } else if lower.contains("gradium/phonon") {
            let lang: String
            if lower.hasSuffix("-fr") { lang = "French" }
            else if lower.hasSuffix("-de") { lang = "German" }
            else if lower.hasSuffix("-es") { lang = "Spanish" }
            else if lower.hasSuffix("-pt") { lang = "Portuguese" }
            else if lower.contains("multilingual") { lang = "Multilingual" }
            else { lang = "English" }
            specs = [
                ("Zoey (Default)", "Zoey", "Conversational · Default", lang),
                ("default", "default", "Conversational · Built-in", lang),
                ("Alfie", "Alfie", "Conversational", lang),
                ("Bryony", "Bryony", "Conversational", lang),
                ("Delia", "Delia", "Conversational", lang),
                ("Easton", "Easton", "Conversational", lang),
                ("Everly", "Everly", "Conversational", lang),
                ("Hadley", "Hadley", "Conversational", lang),
                ("Harper", "Harper", "Conversational", lang),
                ("Juniper", "Juniper", "Conversational", lang),
                ("Lottie", "Lottie", "Conversational", lang),
                ("Micah", "Micah", "Conversational", lang),
                ("Miles", "Miles", "Conversational", lang),
                ("Pippa", "Pippa", "Conversational", lang),
                ("Spencer", "Spencer", "Conversational", lang),
                ("Teddy", "Teddy", "Conversational", lang),
                ("Toby", "Toby", "Conversational", lang),
                ("Edwin", "Edwin", "Narrator", lang),
                ("Elodie-Rose", "Elodie-Rose", "Narrator", lang),
                ("Freya", "Freya", "Narrator", lang),
                ("Hugh", "Hugh", "Narrator", lang),
                ("Imogen", "Imogen", "Narrator", lang),
                ("Maeve", "Maeve", "Narrator", lang),
                ("Preston", "Preston", "Narrator", lang),
                ("Russell", "Russell", "Narrator", lang),
                ("Trevor", "Trevor", "Narrator", lang),
                ("Captain Silas Blackwater", "Captain Silas Blackwater", "Character · Pirate Captain", lang),
                ("Fairy Godmother", "Fairy Godmother", "Character · Magical", lang),
                ("Fennick the Young Elf", "Fennick the Young Elf", "Character · Fantasy", lang),
                ("Grub the Ogre", "Grub the Ogre", "Character · Deep Ogre", lang),
                ("Master Aldric", "Master Aldric", "Character · Wise Scholar", lang),
                ("Morwenna of the Deep", "Morwenna of the Deep", "Character · Mystic", lang),
                ("Old Mortlake", "Old Mortlake", "Character · Elderly", lang),
                ("Sir Bramwell", "Sir Bramwell", "Character · Knight", lang),
                ("The Snow Queen", "The Snow Queen", "Character · Regal", lang),
                ("The Thing Under the Stairs", "The Thing Under the Stairs", "Character · Eerie", lang),
                ("Vyrmathrax the Ancient", "Vyrmathrax the Ancient", "Character · Dragon", lang),
                ("Azgorath", "Azgorath", "Character · Dark Lord", lang),
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

                if expectedContPos + corrLen < resampledLen && searchEnd > searchStart {
                    var bestScore: Float = -.infinity
                    let step = 2
                    var cand = searchStart
                    while cand <= searchEnd {
                        var dot: Float = 0
                        var normA: Float = 1e-9
                        var normB: Float = 1e-9
                        var k = 0
                        while k < corrLen {
                            let a = resampled[expectedContPos + k]
                            let b = resampled[cand + k]
                            dot += a * b
                            normA += a * a
                            normB += b * b
                            k += step
                        }
                        let score = dot / sqrt(normA * normB)
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
                for i in 0..<copyLen {
                    let w = hann[i]
                    output[outPos + i] += resampled[bestInPos + i] * w
                    norm[outPos + i] += w
                }
            }

            prevInPos = bestInPos
            outPos += hopOut
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

    static let promptTemplates: [PromptTemplateItem] = [
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
}

// MARK: - Smart Speech Text Cleaner & Universal Document Extractor (from bun_rad_studio fmPhonon.ts)

struct SpeechCleanOptions: Equatable {
    var stripMarkdown: Bool = true
    var stripHtml: Bool = true
    var stripCodeBlocks: Bool = true
    var simplifyUrls: Bool = true
    var stripTimestamps: Bool = true
    var stripDividers: Bool = true
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
            s = s
                .replacingOccurrences(of: "&nbsp;", with: " ")
                .replacingOccurrences(of: "&#160;", with: " ")
                .replacingOccurrences(of: "&amp;", with: " and ")
                .replacingOccurrences(of: "&lt;", with: " less than ")
                .replacingOccurrences(of: "&gt;", with: " greater than ")
                .replacingOccurrences(of: "&quot;", with: "\"")
                .replacingOccurrences(of: "&#39;", with: "'")
                .replacingOccurrences(of: "&apos;", with: "'")
                .replacingOccurrences(of: "&mdash;", with: " — ")
                .replacingOccurrences(of: "&ndash;", with: " – ")
        }

        if options.stripTimestamps {
            s = replaceRegex(s, pattern: "(?m)^\\d+\\s*$", with: "")
            s = replaceRegex(
                s,
                pattern: "(?m)^\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d{1,3}\\s*-->\\s*\\d{1,2}:\\d{2}(?::\\d{2})?[.,]\\d{1,3}.*$",
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
        }

        if options.stripDividers {
            s = replaceRegex(s, pattern: "(?m)^\\s*[-=*_]{3,}\\s*$", with: "")
        }

        s = replaceRegex(s, pattern: "[ \\t]+", with: " ")
        s = replaceRegex(s, pattern: "\\n{3,}", with: "\n\n")
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func extractDocumentText(from url: URL) throws -> String {
        let ext = url.pathExtension.lowercased()
        #if os(macOS)
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
                let maxPages = min(pdf.pageCount, 30)
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
        // EPUB (zip archive, unpacked macOS .epub bundle, or Pandoc)
        if ext == "epub" {
            let epubCleanOpts = SpeechCleanOptions(
                stripMarkdown: false,
                stripHtml: true,
                stripCodeBlocks: false,
                simplifyUrls: true,
                stripTimestamps: false,
                stripDividers: true
            )
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out, options: epubCleanOpts)
            }
            if let epubText = extractEpubText(from: url), !epubText.isEmpty {
                return epubText
            }
        }
        // DOCX / ODT / DOC / RTF / WEBARCHIVE / TEX / RST / ORG via Pandoc or native macOS /usr/bin/textutil
        if ["docx", "odt", "doc", "rtf", "rtfd", "webarchive", "wordml", "tex", "rst", "org"].contains(ext) {
            if let pandoc = StudioPackageManager.findExecutable(names: ["pandoc"]),
               let out = runCLI(executable: pandoc, arguments: ["-t", "plain", url.path]),
               !out.isEmpty {
                return clean(out)
            }
            if let textutilOut = runTextutil(on: url), !textutilOut.isEmpty {
                return clean(textutilOut)
            }
            if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil) {
                return attr.string
            }
        }
        // Images via macOS Vision On-Device OCR (VNRecognizeTextRequest)
        if ["png", "jpg", "jpeg", "webp", "tiff", "tif", "bmp", "heic"].contains(ext) {
            if let ocrText = recognizeImageText(from: url), !ocrText.isEmpty {
                return ocrText
            }
        }
        if ext == "html" || ext == "htm" {
            if let attr = try? NSAttributedString(url: url, options: [:], documentAttributes: nil) {
                return attr.string
            }
        }
        #endif

        var detectedEncoding: String.Encoding = .utf8
        let raw: String
        if let utf8Str = try? String(contentsOf: url, encoding: .utf8) {
            raw = utf8Str
        } else if let detectedStr = try? String(contentsOf: url, usedEncoding: &detectedEncoding) {
            raw = detectedStr
        } else {
            raw = try String(contentsOf: url, encoding: .isoLatin1)
        }
        // Jupyter Notebook (.ipynb)
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
        // CSV / TSV structured table -> spoken prose
        if ext == "csv" || ext == "tsv" {
            let delimiter: Character = (ext == "tsv") ? "\t" : ","
            let lines = raw.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            if lines.count >= 2 {
                let headers = lines[0].split(separator: delimiter).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                var spokenRows: [String] = []
                for rowLine in lines.dropFirst().prefix(500) {
                    let cols = rowLine.split(separator: delimiter, omittingEmptySubsequences: false).map {
                        $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
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
                if !spokenRows.isEmpty {
                    return spokenRows.joined(separator: "\n")
                }
            }
        }
        // JSON structured object -> spoken prose
        if ext == "json", let data = raw.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) {
            let prose = structuredObjectToProse(obj)
            if !prose.isEmpty { return prose }
        }
        if ["md", "markdown", "srt", "vtt", "lrc", "html", "htm", "xml"].contains(ext) {
            return clean(raw)
        }
        return raw
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
        if isDir.boolValue {
            guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) else { return nil }
            var htmlFiles: [URL] = []
            for case let fileURL as URL in enumerator {
                let ext = fileURL.pathExtension.lowercased()
                if ["xhtml", "html", "htm"].contains(ext) {
                    htmlFiles.append(fileURL)
                }
            }
            htmlFiles.sort { $0.path < $1.path }
            let chapters = htmlFiles.compactMap { try? String(contentsOf: $0, encoding: .utf8) }.map {
                let withParagraphBreaks = $0
                    .replacingOccurrences(of: "</p>", with: "</p>\n\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "</div>", with: "</div>\n\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "</h1>", with: "</h1>\n\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "</h2>", with: "</h2>\n\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "</h3>", with: "</h3>\n\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "<br>", with: "\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "<br/>", with: "\n", options: .caseInsensitive)
                    .replacingOccurrences(of: "<br />", with: "\n", options: .caseInsensitive)
                return clean(withParagraphBreaks, options: htmlOnlyOpts)
            }
            return chapters.filter { !$0.isEmpty }.joined(separator: "\n\n")
        } else {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            proc.arguments = ["-p", url.path, "*.xhtml", "*.html", "*.htm"]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice
            do {
                try proc.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                if let str = String(data: data, encoding: .utf8), !str.isEmpty {
                    let withParagraphBreaks = str
                        .replacingOccurrences(of: "</p>", with: "</p>\n\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "</div>", with: "</div>\n\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "</h1>", with: "</h1>\n\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "</h2>", with: "</h2>\n\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "</h3>", with: "</h3>\n\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "<br>", with: "\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "<br/>", with: "\n", options: .caseInsensitive)
                        .replacingOccurrences(of: "<br />", with: "\n", options: .caseInsensitive)
                    return clean(withParagraphBreaks, options: htmlOnlyOpts)
                }
            } catch {}
            return nil
        }
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
        guard let cgImageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil) else {
            return nil
        }
        return recognizeCGImageText(cgImage)
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
    static let hardMaxChunkChars = 1200

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var lastCacheKey: String = ""
    nonisolated(unsafe) private static var lastCachedChunks: [PhononTextChunk] = []

    /// Splits arbitrarily large text (books, PDFs, EPUBs, logs) in O(N) linear time preferring:
    /// 1. Paragraph breaks (\n\n) -> 2. Sentence endings (.!?…;) -> 3. Clause breaks (,:—–-) -> 4. Word boundaries.
    static func splitText(
        _ rawText: String,
        maxChars: Int = defaultChunkChars,
        mergeShortParagraphs: Bool = true
    ) -> [PhononTextChunk] {
        guard !rawText.isEmpty else { return [] }
        let clampedMax = min(hardMaxChunkChars, max(120, maxChars))
        let cacheKey = "\(rawText.count)|\(rawText.hashValue)|\(clampedMax)|\(mergeShortParagraphs)"
        cacheLock.lock()
        if cacheKey == lastCacheKey {
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
            let cleanPara = rawPara.trimmingCharacters(in: .whitespacesAndNewlines)
            if cleanPara.isEmpty { continue }

            let pieces = splitLongParagraph(cleanPara, maxChars: clampedMax)
            for piece in pieces {
                if mergeShortParagraphs && !pendingText.isEmpty && (pendingText.count + 1 + piece.count) <= Int(Double(clampedMax) * 0.75) {
                    pendingText += "\n" + piece
                } else {
                    flushPending()
                    if mergeShortParagraphs && piece.count < Int(Double(clampedMax) * 0.40) {
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
        lastCacheKey = cacheKey
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
        case .contrast, .matrix, .nord: return Color.black
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

struct WordTiming: Identifiable, Hashable {
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
    nonisolated(unsafe) private static var lastMasterWordsKey: String = ""
    nonisolated(unsafe) private static var lastMasterWords: [IndexedWordSpec] = []
    nonisolated(unsafe) private static var lastEstTimingsKey: String = ""
    nonisolated(unsafe) private static var lastEstTimings: [WordTiming] = []

    static func extractMasterWords(
        text: String,
        maxChunkChars: Int = HierarchicalChunker.defaultChunkChars,
        enableChunking: Bool = true
    ) -> [IndexedWordSpec] {
        guard !text.isEmpty else { return [] }
        let mKey = "\(text.count)|\(text.hashValue)|\(maxChunkChars)|\(enableChunking)"
        boundsLock.lock()
        if mKey == lastMasterWordsKey {
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
        lastMasterWordsKey = mKey
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

        // 1. Compute 10ms RMS energy envelope
        let frameSamples = max(1, Int(floor(sr * 0.01)))
        let numFrames = max(1, Int(ceil(Double(samples.count) / Double(frameSamples))))
        var env = [Float](repeating: 0, count: numFrames)
        var maxEnv: Float = 0

        for f in 0..<numFrames {
            let start = f * frameSamples
            let end = min(samples.count, start + frameSamples)
            var sumSq: Float = 0
            for i in start..<end {
                let s = samples[i]
                sumSq += s * s
            }
            let rms = sqrt(sumSq / Float(max(1, end - start)))
            env[f] = rms
            if rms > maxEnv { maxEnv = rms }
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
        if let audioURL, FileManager.default.fileExists(atPath: audioURL.path) {
            let cacheKey = "\(audioURL.path)|\(masterWords.count)|\(safeStartChunk)|\(safeEndChunk ?? -1)|\(Int(duration * 100))"
            boundsLock.lock()
            if let cached = fileTimingsCache[cacheKey] {
                boundsLock.unlock()
                return cached
            }
            boundsLock.unlock()

            if let file = try? AVAudioFile(forReading: audioURL) {
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
                    if fileTimingsCache.count > 32 { fileTimingsCache.removeAll() }
                    fileTimingsCache[cacheKey] = result
                    boundsLock.unlock()
                    return result
                }
            }
        }

        let totalDuration = max(duration, 0.25)
        let estKey = "\(text.count)|\(text.hashValue)|\(audioURL?.path ?? "nil")|\(Int(totalDuration * 100))|\(maxChunkChars)|\(enableChunking)|\(safeStartChunk)|\(safeEndChunk ?? -1)|\(paragraphPauseMs)"
        boundsLock.lock()
        if estKey == lastEstTimingsKey {
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
                var sum: Float = 0
                for j in i..<end {
                    let s = ptr[j]
                    sum += s * s
                }
                let r = sqrt(sum / Float(end - i))
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
}

// MARK: - Multi-Format Audio Exporter (.wav, .aiff, .m4a, .mp3, .flac, .caf)

enum AudioExportFormat: String, CaseIterable, Identifiable {
    case wav = "wav"
    case m4a = "m4a"
    case aiff = "aiff"
    case flac = "flac"
    case mp3 = "mp3"
    case caf = "caf"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .wav: return "WAV (24 kHz PCM)"
        case .m4a: return "M4A / AAC (Apple Compact)"
        case .aiff: return "AIFF (Apple Uncompressed)"
        case .flac: return "FLAC (Lossless)"
        case .mp3: return "MP3 (Universal)"
        case .caf: return "CAF (CoreAudio)"
        }
    }

    var mimeType: String {
        switch self {
        case .wav: return "audio/wav"
        case .m4a: return "audio/mp4"
        case .aiff: return "audio/aiff"
        case .flac: return "audio/flac"
        case .mp3: return "audio/mpeg"
        case .caf: return "audio/x-caf"
        }
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

        // Prefer ffmpeg for mp3/flac if installed (including keg-only ffmpeg-full)
        if format == .mp3 || format == .flac {
            if let ffmpeg = StudioPackageManager.findExecutable(
                names: ["ffmpeg"],
                extraDirs: ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin"]
            ) {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: ffmpeg)
                proc.arguments = ["-y", "-i", sourceWAV.path, destinationURL.path]
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

        // Native macOS /usr/bin/afconvert fallback (supports m4a, aiff, flac, caf, wav natively!)
        let afconvertPath = "/usr/bin/afconvert"
        if FileManager.default.isExecutableFile(atPath: afconvertPath) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: afconvertPath)
            switch format {
            case .m4a, .mp3:
                proc.arguments = ["-f", "m4af", "-d", "aac", sourceWAV.path, destinationURL.path]
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
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=24",
                "-i", sourceWAV.path,
                "-vf", "ass='\(escapedASS)'",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
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
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=24",
                "-i", sourceWAV.path,
                "-vf", "subtitles='\(escapedSRT)'",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
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

            // Final ffmpeg fallback with timed subtitle stream embedded as mov_text
            let muxProc = Process()
            muxProc.executableURL = URL(fileURLWithPath: ffmpeg)
            muxProc.arguments = [
                "-y",
                "-f", "lavfi",
                "-i", "color=c=\(bgHex):s=\(aspect.width)x\(aspect.height):r=24",
                "-i", sourceWAV.path,
                "-i", tempSRT.path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-c:s", "mov_text",
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
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let sr = file.processingFormat.sampleRate
        let totalFrames = AVAudioFrameCount(file.length)
        guard sr > 0, totalFrames > 256,
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: min(totalFrames, 48_000 * 30)) else {
            return nil
        }
        try? file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { return nil }
        let n = Int(buffer.frameLength)
        guard n > 512 else { return nil }

        let durationSec = Double(totalFrames) / sr

        // RMS and Noise Floor for SNR dB (45ms window so low-pitched male F0 down to 65Hz fits > 2 periods)
        let frameSize = max(512, Int(sr * 0.045))
        var frameRms: [Double] = []
        var pos = 0
        while pos + frameSize <= n {
            var sum: Double = 0
            for i in 0..<frameSize {
                let s = Double(channel[pos + i])
                sum += s * s
            }
            frameRms.append(sqrt(sum / Double(frameSize)))
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
            var energy: Double = 0
            for i in 0..<frameSize {
                let s = Double(channel[pos + i])
                energy += s * s
            }
            let rms = sqrt(energy / Double(frameSize))
            if rms > noiseFloor * 2.2 {
                var bestLag = 0
                var bestCorr: Double = 0
                var lag = minLag
                let upperLag = min(maxLag, frameSize - 32)
                while lag <= upperLag {
                    var corr: Double = 0
                    var e0: Double = 0
                    var eLag: Double = 0
                    let overlap = frameSize - lag
                    for i in 0..<overlap {
                        let a = Double(channel[pos + i])
                        let b = Double(channel[pos + i + lag])
                        corr += a * b
                        e0 += a * a
                        eLag += b * b
                    }
                    let denom = sqrt(max(e0 * eLag, 1e-12))
                    let normCorr = corr / denom
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

    var isRunning: Bool = false
    var primaryPort: UInt16 = 8080
    var enableOllamaFallbackPort: Bool = true // Binds 11434 if Ollama is not running
    var enableLMStudioFallbackPort: Bool = true // Binds 1234 if LM Studio is not running
    var activePorts: [UInt16] = []
    var statusMessage: String = "Stopped"
    var requestLogs: [LocalhostRequestLog] = []

    // Shared model cache for localhost API requests
    private var cachedTTSModelId: String?
    private var cachedTTSModel: SpeechGenerationModel?
    private var cachedSTTModelId: String?
    private var cachedSTTModel: (any STTGenerationModel)?

    private var listeners: [UInt16: NWListener] = [:]

    private init() {}

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

            if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let headerData = buffer.subdata(in: 0..<headerEnd.lowerBound)
                let headerString = String(data: headerData, encoding: .utf8) ?? ""
                let contentLength = Self.parseContentLength(from: headerString)
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
            } else if !isComplete && error == nil && buffer.count < 10 * 1024 * 1024 {
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
                return Int(parts[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        return 0
    }

    private func dispatchHTTP(headerString: String, body: Data, connection: NWConnection, port: UInt16) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        let firstLine = headerString.components(separatedBy: "\r\n").first ?? "GET / HTTP/1.1"
        let tokens = firstLine.split(separator: " ")
        let method = tokens.first.map(String.init) ?? "GET"
        let rawPath = tokens.dropFirst().first.map(String.init) ?? "/"
        let path = rawPath.components(separatedBy: "?").first ?? "/"

        if method == "OPTIONS" {
            sendResponse(connection: connection, status: 204, contentType: "text/plain", body: Data())
            return
        }

        var status = 200
        var contentType = "application/json"
        var responseData = Data()
        var logSummary = "OK"

        do {
            switch (method, path) {
            case ("GET", "/"), ("GET", "/health"), ("GET", "/api/version"):
                let hw = HardwareGuardInspector.inspect()
                let payload: [String: Any] = [
                    "status": "ok",
                    "version": "0.19.0-mlx-audio-swift",
                    "engine": "VoicesApp Standalone Localhost Server (Apple Silicon MLX)",
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
                let rawText = (json["input"] as? String) ?? (json["text"] as? String) ?? ""
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

                let cleanedText = SpeechTextCleaner.clean(rawText)
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
                    let expanded = (rawFileStr as NSString).expandingTildeInPath
                    if FileManager.default.fileExists(atPath: expanded) {
                        audioURL = URL(fileURLWithPath: expanded)
                    } else if rawFileStr.contains("base64,"),
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
                        status = 400
                        responseData = try JSONSerialization.data(withJSONObject: ["error": "Audio file not found at path: \(rawFileStr)"])
                        logSummary = "400 File not found (\(rawFileStr))"
                        break
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
                let rawText = (json["text"] as? String) ?? (json["input"] as? String) ?? ""
                let cleaned = SpeechTextCleaner.clean(rawText)
                responseData = try JSONSerialization.data(withJSONObject: ["text": cleaned], options: [.prettyPrinted])
                logSummary = "Cleaned speech text (\(cleaned.count) chars)"

            case ("POST", "/v1/audio/subtitles"):
                let json = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
                let text = (json["text"] as? String) ?? ""
                let duration = (json["duration"] as? Double) ?? max(2.0, Double(text.count) / 14.0)
                let timings = WordTimingEngine.computeWordTimings(text: text, audioURL: nil, duration: duration)
                let payload: [String: Any] = [
                    "srt": SubtitleExporter.formatSRT(timings: timings),
                    "vtt": SubtitleExporter.formatVTT(timings: timings),
                    "ass": SubtitleExporter.formatASSKaraoke(timings: timings)
                ]
                responseData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
                logSummary = "Generated SRT/VTT/ASS subtitles (\(timings.count) words)"

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

        sendResponse(connection: connection, status: status, contentType: contentType, body: responseData)
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

    private nonisolated func sendResponse(connection: NWConnection, status: Int, contentType: String, body: Data) {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 204: statusText = "No Content"
        case 400: statusText = "Bad Request"
        case 404: statusText = "Not Found"
        default: statusText = "Internal Server Error"
        }

        let header = [
            "HTTP/1.1 \(status) \(statusText)",
            "Content-Type: \(contentType)",
            "Content-Length: \(body.count)",
            "Access-Control-Allow-Origin: *",
            "Access-Control-Allow-Methods: GET, POST, OPTIONS",
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
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        do {
            try proc.run()
            let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
            let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            let combined = (String(data: outData, encoding: .utf8) ?? "") + "\n" + (String(data: errData, encoding: .utf8) ?? "")
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

    /// Launches the official Homebrew installation script inside macOS Terminal.app so the user can enter their sudo password once.
    func installHomebrewInTerminal() {
        #if os(macOS)
        let scriptContent = """
        #!/bin/bash
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
        let tempScript = FileManager.default.temporaryDirectory.appendingPathComponent("Install_Homebrew_Codecaine.command")
        do {
            try scriptContent.write(to: tempScript, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: tempScript.path)
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
        let scriptContent = """
        #!/bin/bash
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        clear
        echo "=================================================================="
        echo "  Codecaine Studio — \(title)"
        echo "  $ \(command)"
        echo "=================================================================="
        echo ""
        \(command)
        echo ""
        echo "✅ Done! Return to Voices Studio and click 'Check Again'."
        exec /bin/zsh -l
        """
        let safeName = title.replacingOccurrences(of: "[^a-zA-Z0-9_-]", with: "_", options: .regularExpression)
        let tempScript = FileManager.default.temporaryDirectory.appendingPathComponent("Codecaine_\(safeName).command")
        do {
            try scriptContent.write(to: tempScript, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: tempScript.path)
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


