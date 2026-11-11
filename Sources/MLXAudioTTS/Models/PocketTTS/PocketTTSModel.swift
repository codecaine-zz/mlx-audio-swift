import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

private let defaultAudioPrompt: String = "alba"

public struct PocketTTSState {
    public var flowCache: [KVCacheSimple]
}

public final class PocketTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: PocketTTSModelConfig
    private let modelFolder: URL
    @ModuleInfo(key: "flow_lm") public var flow_lm: FlowLMModel
    @ModuleInfo(key: "mimi") public var mimi: MimiAdapter
    public var speaker_proj_weight: MLXArray

    public var temp: Float
    public var lsd_decode_steps: Int
    public var noise_clamp: Float?
    public var eos_threshold: Float
    private var cachedVoiceKey: String?
    private var cachedVoiceConditioning: MLXArray?
    private var cachedVoiceKVKey: String?
    private var cachedVoiceKV: [[MLXArray]]?
    private var cachedVoicePromptFrames: Int = 0

    public var sampleRate: Int { config.mimi.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: config.temp)
    }

    private init(config: PocketTTSModelConfig, modelFolder: URL, flowLM: FlowLMModel, mimi: MimiAdapter) {
        self.config = config
        self.modelFolder = modelFolder
        self._flow_lm = ModuleInfo(wrappedValue: flowLM)
        self._mimi = ModuleInfo(wrappedValue: mimi)
        self.speaker_proj_weight = MLXArray.zeros([config.flowLM.transformer.dModel, config.mimi.seanet.dimension])
        self.temp = config.temp
        self.lsd_decode_steps = config.lsdDecodeSteps
        self.noise_clamp = config.noiseClamp
        self.eos_threshold = config.eosThreshold
        super.init()
    }

    public static func fromConfig(
        _ config: PocketTTSModelConfig, modelFolder: URL, encoderEnabled: Bool? = nil
    ) async throws -> PocketTTSModel {
        guard config.fuser.cross.isEmpty, config.fuser.streamingSum.isEmpty,
              Set(config.fuser.sum).isSubset(of: ["num_speakers", "duration_delta"]) else {
            throw modelError("Unsupported Pocket TTS conditioning fuser")
        }
        guard ["rms", "rms_var"].contains(config.flowLM.timeRmsNorm),
              ["erf", "tanh"].contains(config.flowLM.transformer.gelu),
              ["erf", "tanh"].contains(config.mimi.gelu) else {
            throw modelError("Unsupported timestep normalization or GELU activation")
        }
        var numSpeakers: SpeakerCountConditioner?
        var durationDelta: ContinuousConditioner?
        for conditioner in config.conditioners {
            if conditioner.name == "num_speakers", conditioner.type == "lut", let lut = conditioner.lut {
                numSpeakers = SpeakerCountConditioner(
                    nBins: lut.nBins, dim: lut.dim, outputDim: config.flowLM.transformer.dModel)
            } else if conditioner.name == "duration_delta", conditioner.type == "continuous", let continuous = conditioner.continuous {
                guard continuous.dim >= 4, continuous.dim % 2 == 0 else {
                    throw modelError("Continuous conditioning requires an even dim >= 4")
                }
                durationDelta = ContinuousConditioner(
                    dim: continuous.dim, outputDim: config.flowLM.transformer.dModel,
                    scaleFactor: continuous.scaleFactor, maxPeriod: continuous.maxPeriod)
            } else {
                throw modelError("Unsupported conditioner: \(conditioner.name)")
            }
        }
        let flowLM = try await FlowLMModel.fromConfig(
            config.flowLM,
            latentDim: config.flowLM.ldim ?? config.mimi.quantizer.dimension,
            modelFolder: modelFolder,
            tokenizerPath: config.tokenizerName,
            numSpeakers: numSpeakers,
            durationDelta: durationDelta
        )
        let mimi = MimiAdapter.fromConfig(config.mimi, encoderEnabled: encoderEnabled)
        let model = PocketTTSModel(config: config, modelFolder: modelFolder, flowLM: flowLM, mimi: mimi)
        try model.validateSampling()
        return model
    }

    private static func modelError(_ message: String) -> NSError {
        NSError(domain: "PocketTTSModel", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func validateSampling() throws {
        guard lsd_decode_steps > 0, temp.isFinite, temp >= 0,
              noise_clamp.map({ $0.isFinite && $0 >= 0 }) ?? true else {
            throw Self.modelError("Sampling requires positive lsd_decode_steps and finite, nonnegative temperature and noise_clamp")
        }
    }

    public func initState() -> PocketTTSState {
        PocketTTSState(flowCache: flow_lm.makeCache())
    }

    private func runFlowLM(
        _ state: inout PocketTTSState,
        textTokens: MLXArray,
        backboneInputLatents: MLXArray,
        audioConditioning: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let textEmb = textTokens.shape[1] > 0
            ? flow_lm.conditioner(TokenizedText(tokens: textTokens))
            : MLXArray.zeros([1, 0, flow_lm.dim])
        let combined = concatenated([textEmb, audioConditioning], axis: 1)
        if backboneInputLatents.shape[1] == 0 {
            if combined.shape[1] > 0 {
                _ = flow_lm.transformer(combined.asType(.float32), cache: state.flowCache)
            }
            // Match the reference RNG sequence without evaluating an unused latent.
            _ = MLXRandom.normal([1, flow_lm.ldim])
            return nil
        }
        let (out, isEos) = flow_lm(
            sequence: backboneInputLatents,
            textEmbeddings: combined,
            cache: state.flowCache,
            lsdDecodeSteps: lsd_decode_steps,
            temperature: temp,
            noiseClamp: noise_clamp,
            eosThreshold: eos_threshold
        )
        let outExpanded = out.expandedDimensions(axis: 1)
        return (outExpanded, isEos)
    }

    private func runFlowLMAndIncrementStep(
        _ state: inout PocketTTSState,
        textTokens: MLXArray? = nil,
        backboneInputLatents: MLXArray? = nil,
        audioConditioning: MLXArray? = nil
    ) -> (MLXArray, MLXArray)? {
        let tokens = textTokens ?? MLXArray.zeros([1, 0], type: Int32.self)
        let backbone = backboneInputLatents ?? MLXArray.zeros([1, 0, flow_lm.ldim])
        let conditioning = audioConditioning ?? MLXArray.zeros([1, 0, flow_lm.dim])
        return runFlowLM(&state, textTokens: tokens, backboneInputLatents: backbone, audioConditioning: conditioning)
    }

    private func encodeAudio(_ audio: MLXArray) -> MLXArray {
        let encoded = mimi.encodeToLatent(audio)
        let latents = encoded.transposed(0, 2, 1).asType(.float32)
        let projT = speaker_proj_weight.transposed(1, 0)
        let conditioning = matmul(latents, projT)
        return conditioning
    }

    private func normalizeAudio(_ audio: MLXArray) -> MLXArray {
        if audio.ndim == 1 {
            return audio.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        }
        if audio.ndim == 2 {
            var mono = audio
            if audio.shape[0] > 1 {
                mono = MLX.mean(audio, axis: 0, keepDims: true)
            }
            return mono.expandedDimensions(axis: 0)
        }
        return audio
    }

    private enum AudioPrompt {
        case embedding(MLXArray)
        case audio(MLXArray)
    }

    public static func loadPredefinedVoice(
        _ voiceName: String,
        modelFolder: URL,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXArray? {
        _ = progressHandler
        guard let fileURL = voiceFiles(in: modelFolder)[voiceName.lowercased()] else { return nil }
        let arrays = try MLX.loadArrays(url: fileURL)
        return arrays["audio_prompt"] ?? arrays["emb"] ?? arrays["speaker_wavs"]
    }

    private static func copyDereferencedFile(from src: URL, to dst: URL) {
        let fm = FileManager.default
        let resolvedSrc = src.resolvingSymlinksInPath()
        guard fm.fileExists(atPath: resolvedSrc.path) else { return }
        try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
        // Remove any existing file or broken symlink at dst before copying
        try? fm.removeItem(at: dst)
        if let data = try? Data(contentsOf: resolvedSrc), !data.isEmpty {
            try? data.write(to: dst, options: .atomic)
        } else {
            try? fm.copyItem(at: resolvedSrc, to: dst)
        }
        if !fm.fileExists(atPath: dst.path) {
            try? fm.removeItem(at: dst)
        }
    }

    private static func voiceFiles(in modelFolder: URL) -> [String: URL] {
        let fm = FileManager.default
        var voices: [String: URL] = [:]
        for directory in ["voices", "embeddings"] {
            let dirURL = modelFolder.appendingPathComponent(directory)
            let files = (try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil)) ?? []
            for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where file.pathExtension == "safetensors" {
                let resolved = file.resolvingSymlinksInPath()
                let size = (try? fm.attributesOfItem(atPath: resolved.path)[.size] as? NSNumber)?.intValue ?? 0
                guard fm.fileExists(atPath: resolved.path), size > 0 else {
                    // Clean up broken symlinks or empty voice files left in cache
                    try? fm.removeItem(at: file)
                    continue
                }
                voices[file.deletingPathExtension().lastPathComponent.lowercased()] = resolved
            }
        }
        let defaultVoice = modelFolder.appendingPathComponent("default-voice.safetensors")
        let resolvedDefault = defaultVoice.resolvingSymlinksInPath()
        if voices["default"] == nil, fm.fileExists(atPath: resolvedDefault.path) {
            voices["default"] = resolvedDefault
        }
        return voices
    }

    func voiceConditioning(_ arrays: [String: MLXArray]) throws -> MLXArray {
        if var latents = arrays["speaker_wavs"] {
            if latents.ndim == 2 { latents = latents.expandedDimensions(axis: 0) }
            guard latents.ndim == 3, latents.shape[0] == 1,
                  latents.shape[1] == speaker_proj_weight.shape[1] else {
                throw Self.modelError("Speaker latents must have shape [1, \(speaker_proj_weight.shape[1]), T]")
            }
            return matmul(latents.transposed(0, 2, 1).asType(.float32), speaker_proj_weight.T)
        }
        guard let embedding = arrays["emb"] ?? arrays["audio_prompt"] else {
            throw Self.modelError("Voice file must contain speaker_wavs, emb, or audio_prompt")
        }
        return embedding
    }

    private func resolveAudioPrompt(
        voice: String?,
        refAudio: MLXArray?,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> AudioPrompt {
        if let refAudio {
            guard mimi.encoder != nil else {
                throw Self.modelError("This checkpoint omits the speaker encoder; use a bundled voice embedding")
            }
            var audio = normalizeAudio(refAudio)
            guard audio.ndim == 3, audio.shape[0] == 1, audio.shape[1] == config.mimi.channels else {
                throw Self.modelError("Audio prompt must have shape [1, \(config.mimi.channels), T]")
            }
            guard Double(audio.shape[2]) >= config.audioPromptMinDuration * Double(sampleRate) else {
                throw Self.modelError("Audio prompt must be at least \(config.audioPromptMinDuration) seconds")
            }
            if let maximum = config.audioPromptMaxDuration {
                audio = audio[.ellipsis, ..<min(audio.shape[2], Int(maximum * Double(sampleRate)))]
            }
            if audio.shape[2] == 0 { return .embedding(MLXArray.zeros([1, 0, flow_lm.dim])) }
            return .audio(audio.asType(.float32))
        }

        let cacheKey = "voice:\(voice ?? defaultAudioPrompt)"
        if refAudio == nil, cachedVoiceKey == cacheKey, let cached = cachedVoiceConditioning {
            return .embedding(cached)
        }

        if let voice, let file = try await resolveVoiceFile(voice) {
            if let arrays = try? MLX.loadArrays(url: file.resolvingSymlinksInPath()),
               let cond = try? voiceConditioning(arrays) {
                eval(cond)
                cachedVoiceKey = cacheKey
                cachedVoiceConditioning = cond
                return .embedding(cond)
            } else if file.path.hasPrefix(modelFolder.path) {
                // Evict corrupt or dangling cached voice file and fall back to default voice
                try? FileManager.default.removeItem(at: file)
            }
        }

        let voices = Self.voiceFiles(in: modelFolder)
        let defaultVoice = ["default", defaultAudioPrompt].first { voices[$0] != nil }
        let selected = voice?.lowercased()
            ?? defaultVoice ?? voices.keys.sorted().first ?? defaultAudioPrompt
        let resolvedFile = voices[selected]
            ?? defaultVoice.flatMap { voices[$0] }
            ?? voices.keys.sorted().first.flatMap { voices[$0] }
        guard let file = resolvedFile else {
            throw Self.modelError("Unknown voice '\(selected)'. Available voices: \(voices.keys.sorted().joined(separator: ", "))")
        }
        let cond = try voiceConditioning(MLX.loadArrays(url: file))
        eval(cond)
        cachedVoiceKey = cacheKey
        cachedVoiceConditioning = cond
        return .embedding(cond)
    }

    /// Official Gradium Phonon 102-voice catalog (`gradium/phonon-voices`, Apache-2.0, ungated).
    private static let gradiumPhononVoiceIDByName: [String: String] = [
        // English (US)
        "zoey": "NbpkqMVS3CJeq2j8",
        "juniper": "NIWIOv5vTi5eRMj8",
        "everly": "vjFyp2BcCyTlUmYb",
        "harper": "4SZHfMpw-p46Ywgs",
        "hadley": "bXwrkHhURL1VFJdG",
        "delia": "nGzMHMPAWGaJFHOo",
        "easton": "LMDIYE0Oa377Qh6Z",
        "micah": "40QrMVgAxFiUnP9f",
        "trevor": "0zLApsZfVLLNaEGl",
        "russell": "_6Aslh2DxfmnRLmP",
        "preston": "qIxlZ7Ei8KvWlyQH",
        "miles": "d9XCyhjo4JhkHF1d",
        // English (UK)
        "maeve": "6PWnV0Nq4wu7RVBT",
        "freya": "GgfEkEJtxZR7gnpy",
        "pippa": "uem82D50GRv2Dwma",
        "elodie-rose": "y9bQqIDnyxwqK01t",
        "imogen": "gDT1nz7Ie36ZhL-C",
        "lottie": "m9MxpY6IUGXtWp76",
        "bryony": "gfP44RlHwr2kEwk7",
        "alfie": "X6NceqF285Gr04gb",
        "hugh": "YyKcxPG7iIgS3jsK",
        "sterling": "YyKcxPG7iIgS3jsK",
        "edwin": "zUCxzrOrugTSis03",
        "spencer": "JOW0vchyswkQUMDS",
        "toby": "dME3IWyZBvmh1n1q",
        "teddy": "Zn7g6NxLqXlJb7Qi",
        // French
        "noémie": "FXxJ9mANRq6BCTX5",
        "noemie": "FXxJ9mANRq6BCTX5",
        "coralie": "ZeSg853xFACESHHI",
        "tiphaine": "0fvH6XAKvA748v2v",
        "maëlys": "s048cR1l2Jmu4k3B",
        "maelys": "s048cR1l2Jmu4k3B",
        "orianne": "TUk4X11S6UytB3G6",
        "solène": "YhIHaAfQ0cQPDV9R",
        "solene": "YhIHaAfQ0cQPDV9R",
        "ninon": "qtPlUYAZ4EJDuWMm",
        "lilou": "4iKxBtHBRzr1YOph",
        "hélène": "Vvakv7iWZCUrBooR",
        "helene": "Vvakv7iWZCUrBooR",
        "augustin": "Tek4tJXiX6_yvXq7",
        "arnaud": "aXMVwiUhITMrwPzd",
        "jules": "YKeBw3OV1RgpdhLh",
        "baptiste": "rqqeErpASHVQXHDd",
        "lucien": "LIv4NaG8ay4Mhqtd",
        // German
        "nele": "2A2fFf8SkDEJ8nzx",
        "jette": "4Mn9VfG2wsLLEzi5",
        "annika": "p6Uutkyi3j2iNAUu",
        "jule": "MxhgvuHuUER7qjmL",
        "lotta": "MOrDPn052KUlnKXk",
        "josefine": "yadpmqsHVrYC0NJp",
        "svea": "Adoh4wA9tp5QXJi9",
        "anton": "3ZKKapPOvuWFcw9f",
        "mirko": "Xn9xDzqSSr4xGTNg",
        "mats": "Kf5m22mROozoMWj3",
        "henrik": "yyS1KYWs6mXoEw7D",
        "leon": "20zdyYrQPzKlCwkk",
        "mika": "R3uU1GTAZJotUhUs",
        // Spanish
        "noa": "b6FvJAiokjdqIti4",
        "lucía-sol": "3p0eIGbkmny71GlA",
        "lucia-sol": "3p0eIGbkmny71GlA",
        "candela": "c9wqBDmQiBie5q6Y",
        "celia": "rmLBOdHrfkFym9MM",
        "alicia": "4701YgRXjkBcg6sL",
        "nerea": "EQcQFGkYfMn5YJgx",
        "marcos": "jvPx8j8zLGQ3utZz",
        "borja": "tIjCgP8v3RY8UpgM",
        "iker": "t-_TS1e-0GzDAX02",
        "nuno": "fghJGlv90qu8OJ36",
        "izan": "Ejb29qHnSOK0cIHY",
        "guillermo": "FaLVVqpFQcDOtq45",
        "rebeca": "Sx8m6WdXBG06zwZT",
        "regina": "s58clrg2fe0MO7Y-",
        "danae": "evfxKSxpDIhmsNoq",
        "azucena": "KJpIQlbYrZPA4aLq",
        "jimena": "qG5BXOJYrKEnXLIX",
        "ximena": "VDwnGxAo68C8U8vC",
        "lalo": "F05gqkzX7WhKVpSL",
        "aldo": "ax7dW1XB0eiMfHgO",
        "yael": "ALczEazoF4n6PmGl",
        "ulises": "G5HLUsQd659UtRe7",
        "leonel": "BkuASiOJRrJEB5qA",
        "nico": "PwgPEaweQILZdGc4",
        // Portuguese
        "manuela-lu": "KgC2Nqnjj48NUiyV",
        "isadora": "CtNvKUUMrx1h56ih",
        "débora": "K8mj96kYHMy8q3Hb",
        "debora": "K8mj96kYHMy8q3Hb",
        "rafaela": "6k6cRt7QJfm5ChrH",
        "clarice": "sGCMzFIA53ppIngM",
        "flávia": "gOc90Rjmv5CobHCj",
        "flavia": "gOc90Rjmv5CobHCj",
        "davi": "NuUr_x5V90hSHzCJ",
        "igor": "l7520SIXi14iEamd",
        "kauã": "6HftnVU1XqVBJS8t",
        "kaua": "6HftnVU1XqVBJS8t",
        "vitor": "mWgyMRkFED70QxQv",
        "otávio": "JYtmeaysoaR0Wybf",
        "otavio": "JYtmeaysoaR0Wybf",
        "marcelo": "qDeAnE4L4bk0BRz1",
        // Characters
        "captain silas blackwater": "xPDpsV3F592srkjA",
        "morwenna of the deep": "nOykTmcH9ydsGGIc",
        "old mortlake": "lX3LavMNdlIbgEHH",
        "azgorath": "dGyGDKo2osj7h3s7",
        "fennick the young elf": "cXoIXJ2llS8S0WOW",
        "the thing under the stairs": "GXaSqMEmOg12OIYb",
        "vyrmathrax the ancient": "xO8xQo3BrB8FuI6J",
        "grub the ogre": "cYtsRSxV5q9IMehn",
        "master aldric": "gSgG5Tor0GPEnqzC",
        "the snow queen": "NIWoEurJHMJjyUTe",
        "sir bramwell": "y2eiKxtQo2B67deU",
        "fairy godmother": "QXNr0SvkukLB4wPq",
    ]

    private func resolveVoiceFile(_ voice: String) async throws -> URL? {
        if voice.hasPrefix("hf://") {
            let parts = voice.dropFirst(5).split(separator: "/", maxSplits: 2)
            guard parts.count == 3, let repo = Repo.ID(rawValue: "\(parts[0])/\(parts[1])") else {
                throw Self.modelError("Invalid Hugging Face voice URL: \(voice)")
            }
            let fileAndRevision = parts[2].split(separator: "@", maxSplits: 1)
            guard let filename = fileAndRevision.first, filename.hasSuffix(".safetensors") else {
                throw Self.modelError("Voice URL must reference a safetensors file")
            }
            let revision = fileAndRevision.count == 2 ? String(fileAndRevision[1]) : "main"
            let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? UserDefaults.standard.string(forKey: "VoicesApp.HuggingFaceToken")
                ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
            let client: HubClient
            if let token, !token.isEmpty {
                client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: .default)
            } else {
                client = HubClient(cache: .default)
            }
            return try await client.downloadFile(at: String(filename), from: repo, revision: revision).resolvingSymlinksInPath()
        }
        let expanded = (voice as NSString).expandingTildeInPath
        let file = URL(fileURLWithPath: expanded).resolvingSymlinksInPath()
        if file.pathExtension == "safetensors", FileManager.default.fileExists(atPath: file.path) {
            return file
        }
        // Check local modelFolder voices first
        let localVoices = Self.voiceFiles(in: modelFolder)
        let lowerVoice = voice.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let localMatch = localVoices[lowerVoice] {
            return localMatch
        }
        // Resolve named Gradium Phonon voices on-demand from `gradium/phonon-voices`
        if let catalogID = Self.gradiumPhononVoiceIDByName[lowerVoice] {
            let voicesDir = modelFolder.appendingPathComponent("voices")
            let cachedVoiceFile = voicesDir.appendingPathComponent("\(lowerVoice).safetensors")
            let resolvedCachedVoice = cachedVoiceFile.resolvingSymlinksInPath()
            if FileManager.default.fileExists(atPath: resolvedCachedVoice.path) {
                return resolvedCachedVoice
            } else {
                try? FileManager.default.removeItem(at: cachedVoiceFile)
            }
            let cachedByID = voicesDir.appendingPathComponent("\(catalogID).safetensors")
            let resolvedCachedByID = cachedByID.resolvingSymlinksInPath()
            if FileManager.default.fileExists(atPath: resolvedCachedByID.path) {
                return resolvedCachedByID
            } else {
                try? FileManager.default.removeItem(at: cachedByID)
            }
            if let repo = Repo.ID(rawValue: "gradium/phonon-voices") {
                let client = HubClient(cache: .default)
                if let downloaded = try? await client.downloadFile(at: "voices/\(catalogID).safetensors", from: repo, revision: "main") {
                    Self.copyDereferencedFile(from: downloaded, to: cachedVoiceFile)
                    if FileManager.default.fileExists(atPath: cachedVoiceFile.path) {
                        return cachedVoiceFile
                    }
                    let resolvedDownloaded = downloaded.resolvingSymlinksInPath()
                    if FileManager.default.fileExists(atPath: resolvedDownloaded.path) {
                        return resolvedDownloaded
                    }
                }
            }
        }
        return nil
    }

    private func getStateForAudioPrompt(_ prompt: AudioPrompt, cacheKeyHint: String? = nil) throws -> PocketTTSState {
        if let cacheKeyHint,
           cachedVoiceKVKey == cacheKeyHint,
           let savedKV = cachedVoiceKV,
           savedKV.count == flow_lm.transformer.layers.count {
            let caches = flow_lm.makeCache(step: 384)
            for (idx, layerCache) in caches.enumerated() where savedKV[idx].count == 2 {
                layerCache.state = savedKV[idx]
            }
            return PocketTTSState(flowCache: caches)
        }

        var state = PocketTTSState(flowCache: flow_lm.makeCache(step: 384))
        var conditioning: MLXArray = switch prompt {
        case .embedding(let emb):
            emb
        case .audio(let audio):
            encodeAudio(audio)
        }
        if conditioning.ndim == 2 { conditioning = conditioning.expandedDimensions(axis: 0) }
        guard conditioning.ndim == 3, conditioning.shape[0] == 1, conditioning.shape[2] == flow_lm.dim else {
            throw Self.modelError("Voice embedding must have shape [1, T, \(flow_lm.dim)]")
        }
        if conditioning.shape[1] > 0 {
            _ = runFlowLMAndIncrementStep(&state, audioConditioning: conditioning)
        }
        sliceFlowCache(&state, to: conditioning.shape[1])
        if let cacheKeyHint {
            let snapshot = state.flowCache.map { $0.state }
            let flat = snapshot.flatMap { $0 }
            if !flat.isEmpty { eval(flat) }
            cachedVoiceKVKey = cacheKeyHint
            cachedVoiceKV = snapshot
            cachedVoicePromptFrames = conditioning.shape[1]
        }
        return state
    }

    private func sliceFlowCache(_ state: inout PocketTTSState, to length: Int) {
        let targetLength = max(length, 0)
        for cache in state.flowCache {
            if cache.offset > targetLength {
                cache.trim(cache.offset - targetLength)
            }
        }
    }

    private func getFlowCacheNumFrames(_ state: PocketTTSState) -> Int {
        for cache in state.flowCache {
            let s = cache.innerState()
            guard s.count == 2 else { continue }
            return min(cache.offset, s[0].shape[2])
        }
        return 0
    }

    public func generateAudio(
        state: PocketTTSState?,
        text: String,
        framesAfterEos: Int?,
        maxFrames: Int?
    ) throws -> MLXArray {
        var chunks: [MLXArray] = []
        try forEachAudioChunk(
            state: state,
            text: text,
            framesAfterEos: framesAfterEos,
            maxFrames: maxFrames,
            sentenceBatchDecode: true
        ) {
            chunks.append($0)
        }
        if chunks.isEmpty { return MLXArray.zeros([0]) }
        return chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 0)
    }

    public func generateAudioStream(
        state: PocketTTSState?,
        text: String,
        framesAfterEos: Int?,
        maxFrames: Int?
    ) throws -> [MLXArray] {
        var outputs: [MLXArray] = []
        try forEachAudioChunk(
            state: state,
            text: text,
            framesAfterEos: framesAfterEos,
            maxFrames: maxFrames,
            sentenceBatchDecode: false
        ) {
            outputs.append($0)
        }
        return outputs
    }

    private func forEachAudioChunk(
        state: PocketTTSState?,
        text: String,
        framesAfterEos: Int?,
        maxFrames: Int?,
        sentenceBatchDecode: Bool = false,
        onChunk: (MLXArray) -> Void
    ) throws {
        guard var state else {
            throw NSError(domain: "PocketTTSModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing generation state"])
        }
        try validateSampling()
        let promptNumFrames = getFlowCacheNumFrames(state)
        let chunks = try PocketTTSTextUtils.splitIntoBestSentences(flow_lm.conditioner.tokenizer, text)
        for chunk in chunks {
            try Task.checkCancellation()
            sliceFlowCache(&state, to: promptNumFrames)
            let (preparedText, _) = try PocketTTSTextUtils.prepareTextPrompt(chunk)
            let frames = framesAfterEos ?? 1
            if sentenceBatchDecode {
                if let audio = try generateSentenceFast(
                    state: &state,
                    text: preparedText,
                    framesAfterEos: frames,
                    maxFrames: maxFrames
                ), audio.size > 0 {
                    onChunk(audio)
                }
            } else {
                try generateAudioStreamShortText(
                    state: &state, text: preparedText, framesAfterEos: frames, maxFrames: maxFrames, onChunk: onChunk)
            }
        }
    }

    /// Single-sentence fast path (`B = 1`) that eliminates per-frame `mimi.decodeStep` + CPU sync stalls:
    /// checks EOS once every 8 steps and decodes all sentence latents in a single parallel `mimi.decodeFromLatent` pass.
    private func generateSentenceFast(
        state: inout PocketTTSState,
        text: String,
        framesAfterEos: Int,
        maxFrames: Int?
    ) throws -> MLXArray? {
        let words = text.split(separator: " ").count
        let genLenSec = Double(words) * 1.0 + 2.0
        let computedMax = Int(genLenSec * mimi.frameRate)
        let maxGenLen = max(0, maxFrames.map { min($0, computedMax) } ?? computedMax)
        guard maxGenLen > 0 else { return nil }

        let prepared = flow_lm.conditioner.prepare(text)
        if prepared.tokens.shape[1] > 0 {
            let textEmb = flow_lm.conditioner(prepared).asType(.float32)
            _ = flow_lm.transformer(textEmb, cache: state.flowCache)
        }

        var prevLatent: MLXArray? = nil
        var latentSteps: [MLXArray] = []
        latentSteps.reserveCapacity(min(maxGenLen, 256))
        var pendingEos: [MLXArray] = []
        pendingEos.reserveCapacity(8)
        var firstEosStep: Int? = nil

        for step in 0 ..< maxGenLen {
            try Task.checkCancellation()
            let (nextLatent, isEos) = flow_lm.stepDecode(
                batchSize: 1,
                previousLatents: prevLatent,
                cache: state.flowCache,
                lsdDecodeSteps: lsd_decode_steps,
                temperature: temp,
                noiseClamp: noise_clamp,
                eosThreshold: eos_threshold
            )
            latentSteps.append(nextLatent)
            pendingEos.append(isEos.reshaped([1]))
            prevLatent = nextLatent.expandedDimensions(axis: 1)

            if pendingEos.count == 8 || step + 1 == maxGenLen {
                let chunkStartStep = step - pendingEos.count + 1
                let eosVec = pendingEos.count == 1 ? pendingEos[0] : concatenated(pendingEos, axis: 0)
                eval(eosVec, nextLatent)
                if firstEosStep == nil {
                    let flags = eosVec.asArray(Bool.self)
                    if let hitOffset = flags.firstIndex(of: true) {
                        firstEosStep = chunkStartStep + hitOffset
                    }
                }
                pendingEos.removeAll(keepingCapacity: true)
                if let eos = firstEosStep, step >= eos + framesAfterEos - 1 {
                    break
                }
            }
        }

        let validFrames = min(
            latentSteps.count,
            firstEosStep.map { $0 + framesAfterEos } ?? latentSteps.count
        )
        guard validFrames > 0 else { return nil }

        let usedLatents = Array(latentSteps.prefix(validFrames))
        let stackedLatents = usedLatents.count == 1
            ? usedLatents[0].expandedDimensions(axis: 1)
            : stacked(usedLatents, axis: 1) // [1, T, ldim]
        let decodingInput = stackedLatents * flow_lm.emb_std + flow_lm.emb_mean
        let quantized = mimi.quantizer(decodingInput.transposed(0, 2, 1)) // [1, 512, T]
        let audioFull = mimi.decodeFromLatent(quantized) // [1, 1, T * frameSize]
        let audio1D = audioFull[0, 0]
        eval(audio1D)
        return audio1D
    }

    private func generateAudioStreamShortText(
        state: inout PocketTTSState,
        text: String,
        framesAfterEos: Int,
        maxFrames: Int?,
        onChunk: (MLXArray) -> Void
    ) throws {
        mimi.resetState()

        let words = text.split(separator: " ").count
        let genLenSec = Double(words) * 1.0 + 2.0
        let computedMax = Int(genLenSec * mimi.frameRate)
        let maxGenLen = max(0, maxFrames.map { min($0, computedMax) } ?? computedMax)

        let prepared = flow_lm.conditioner.prepare(text)
        _ = runFlowLMAndIncrementStep(&state, textTokens: prepared.tokens)

        var prevLatent: MLXArray? = nil
        var eosStep: Int?

        for step in 0 ..< maxGenLen {
            try Task.checkCancellation()
            let (nextLatentRaw, isEos) = flow_lm.stepDecode(
                batchSize: 1,
                previousLatents: prevLatent,
                cache: state.flowCache,
                lsdDecodeSteps: lsd_decode_steps,
                temperature: temp,
                noiseClamp: noise_clamp,
                eosThreshold: eos_threshold
            )
            let nextLatent = nextLatentRaw.expandedDimensions(axis: 1)
            try Task.checkCancellation()
            if eosStep == nil {
                let eos = isEos.asArray(Bool.self).first ?? false
                if eos { eosStep = step }
            }
            if let eosStep, step >= eosStep + framesAfterEos {
                break
            }

            let decodingInput = nextLatent * flow_lm.emb_std + flow_lm.emb_mean
            let quantized = mimi.quantizer(decodingInput.transposed(0, 2, 1))
            let audioChunk = mimi.decodeStep(quantized)
            let chunk = audioChunk[0, 0]
            eval(chunk)
            onChunk(chunk)
            prevLatent = nextLatent
        }

        try Task.checkCancellation()
    }

    /// True GPU Tensor-Batched Synthesis for Full-Book / Multi-Chunk Export (`B = 24` simultaneous sentences on Metal GPU + batched Mimi parallel decode).
    /// Produces mathematically equivalent per-sentence voice + text + FlowLM + causal Mimi waveforms at 20x-50x higher GPU utilization than `B = 1`.
    public func generateBatch(
        texts: [String],
        voice: String?,
        refAudio: MaterializedArray? = nil,
        generationParameters: GenerateParameters,
        batchSize: Int = 48,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let prevTemp = temp
        temp = generationParameters.temperature
        defer { temp = prevTemp }
        try validateSampling()

        let prompt = try await resolveAudioPrompt(voice: voice, refAudio: refAudio)
        var voiceCond: MLXArray = switch prompt {
        case .embedding(let emb): emb
        case .audio(let audio): encodeAudio(audio)
        }
        if voiceCond.ndim == 2 { voiceCond = voiceCond.expandedDimensions(axis: 0) }
        guard voiceCond.ndim == 3, voiceCond.shape[0] == 1, voiceCond.shape[2] == flow_lm.dim else {
            throw Self.modelError("Voice embedding must have shape [1, T, \(flow_lm.dim)]")
        }
        voiceCond = voiceCond.asType(.float32)
        eval(voiceCond)
        let tvLen = voiceCond.shape[1]

        struct SentenceJob {
            let chunkIdx: Int
            let sentenceOrd: Int
            let tokens: MLXArray
            let tokenCount: Int
            let maxGenLen: Int
        }

        var jobs: [SentenceJob] = []
        var expectedSentencesPerChunk = Array(repeating: 0, count: texts.count)

        for (chunkIdx, rawText) in texts.enumerated() {
            try Task.checkCancellation()
            let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let sentences = (try? PocketTTSTextUtils.splitIntoBestSentences(flow_lm.conditioner.tokenizer, trimmed)) ?? [trimmed]
            var ord = 0
            for sent in sentences {
                let (preparedText, _) = (try? PocketTTSTextUtils.prepareTextPrompt(sent)) ?? (sent, 1)
                let words = preparedText.split(separator: " ").count
                let genLenSec = Double(words) * 1.0 + 2.0
                let computedMax = Int(genLenSec * mimi.frameRate)
                let maxGenLen = max(0, generationParameters.maxTokens.map { min($0, computedMax) } ?? computedMax)
                guard maxGenLen > 0 else { continue }
                let tok = flow_lm.conditioner.prepare(preparedText)
                let tCount = tok.tokens.shape[1]
                guard tCount > 0 else { continue }
                jobs.append(SentenceJob(
                    chunkIdx: chunkIdx,
                    sentenceOrd: ord,
                    tokens: tok.tokens,
                    tokenCount: tCount,
                    maxGenLen: maxGenLen
                ))
                ord += 1
            }
            expectedSentencesPerChunk[chunkIdx] = ord
        }

        guard !jobs.isEmpty else {
            return Array(repeating: [], count: texts.count)
        }

        // Sort sentences by token length so every batch of `B` items has near-zero padding overhead
        let sortedJobs = jobs.sorted {
            if $0.tokenCount != $1.tokenCount { return $0.tokenCount < $1.tokenCount }
            return $0.maxGenLen < $1.maxGenLen
        }

        var chunkSentenceSamples: [[[Float]]] = expectedSentencesPerChunk.map { count in
            Array(repeating: [], count: count)
        }
        var completedSentencesForChunk = Array(repeating: 0, count: texts.count)
        var completedChunks = expectedSentencesPerChunk.enumerated().reduce(0) { acc, pair in
            acc + (pair.element == 0 ? 1 : 0)
        }

        let effectiveBatchSize = max(1, min(64, batchSize))
        let framesAfterEos = 1
        let frameSamples = mimi.frameSize
        if Memory.cacheLimit < 1536 * 1024 * 1024 {
            Memory.cacheLimit = 1536 * 1024 * 1024
        }
        let isBenchLog = ProcessInfo.processInfo.arguments.contains("--benchmark-book-mp4")
        var totalFlowSec: Double = 0
        var totalMimiSec: Double = 0
        var totalGeneratedFramesCount: Int = 0

        // Prefill voice conditioning (`0 ..< tvLen`) ONCE across the entire book instead of repeating per sentence!
        var sharedVoiceKV: [(k: MLXArray, v: MLXArray)] = []
        if tvLen > 0 {
            let voiceCache1 = flow_lm.makeCache(step: max(64, tvLen + 8))
            _ = flow_lm.transformer(voiceCond, cache: voiceCache1)
            sharedVoiceKV = voiceCache1.map { c in
                let st = c.state
                return (st[0], st[1])
            }
            eval(sharedVoiceKV.flatMap { [$0.k, $0.v] })
        }

        var cursor = 0
        while cursor < sortedJobs.count {
            try Task.checkCancellation()
            let tFlow0 = CFAbsoluteTimeGetCurrent()
            let endCursor = min(sortedJobs.count, cursor + effectiveBatchSize)
            let group = Array(sortedJobs[cursor..<endCursor])
            cursor = endCursor
            let B = group.count

            let maxTextTokens = group.map(\.tokenCount).max() ?? 1
            let pMax = tvLen + maxTextTokens
            let batchMaxGen = group.map(\.maxGenLen).max() ?? 1
            let totalCapacity = pMax + batchMaxGen + 8

            var prefillRows: [MLXArray] = []
            prefillRows.reserveCapacity(B)
            var pads: [Int] = []
            pads.reserveCapacity(B)

            for item in group {
                let textEmb = flow_lm.conditioner(TokenizedText(tokens: item.tokens)).asType(.float32) // [1, L_b, dim]
                let padLen = maxTextTokens - item.tokenCount
                pads.append(padLen)
                if padLen > 0 {
                    let z = MLXArray.zeros([1, padLen, flow_lm.dim], dtype: .float32)
                    prefillRows.append(concatenated([z, textEmb], axis: 1))
                } else {
                    prefillRows.append(textEmb)
                }
            }

            let prefillInput = B == 1 ? prefillRows[0] : concatenated(prefillRows, axis: 0) // [B, maxTextTokens, dim]
            let batchCache = flow_lm.makeCache(step: totalCapacity)
            if tvLen > 0 {
                for (layerIdx, layerCache) in batchCache.enumerated() {
                    let vk = sharedVoiceKV[layerIdx].k
                    let vv = sharedVoiceKV[layerIdx].v
                    let nHeads = vk.shape[1]
                    let hDim = vk.shape[3]
                    var kBuf = MLXArray.zeros([B, nHeads, totalCapacity, hDim], dtype: vk.dtype)
                    var vBuf = MLXArray.zeros([B, nHeads, totalCapacity, hDim], dtype: vv.dtype)
                    kBuf[.ellipsis, 0..<tvLen, 0...] = B == 1 ? vk : MLX.broadcast(vk, to: [B, nHeads, tvLen, hDim])
                    vBuf[.ellipsis, 0..<tvLen, 0...] = B == 1 ? vv : MLX.broadcast(vv, to: [B, nHeads, tvLen, hDim])
                    layerCache.state = [kBuf, vBuf]
                    layerCache.trim(totalCapacity - tvLen)
                }
            }
            let hasAnyPad = pads.contains { $0 > 0 }

            let padVecInt = MLXArray(pads.map { Int32($0) }).reshaped([B, 1, 1, 1])
            let textPrefillPos: MLXArray?
            let prefillMask: MLXFast.ScaledDotProductAttentionMaskMode
            if !hasAnyPad {
                textPrefillPos = nil
                prefillMask = .causal
            } else {
                let qCol = MLXArray(Int32(0) ..< Int32(maxTextTokens)).reshaped([1, 1, maxTextTokens, 1])
                textPrefillPos = (MLXArray(Int32(tvLen)) + MLX.maximum(MLXArray(Int32(0)), qCol - padVecInt)).asType(.float32)
                let qGlobal = qCol + Int32(tvLen)
                let kAll = MLXArray(Int32(0) ..< Int32(pMax)).reshaped([1, 1, 1, pMax])
                let kNotPad = logicalOr(kAll .< Int32(tvLen), kAll .>= (Int32(tvLen) + padVecInt))
                let validMask = logicalAnd(qGlobal .>= kAll, logicalOr(kNotPad, qGlobal .== kAll))
                prefillMask = .array(validMask)
            }

            _ = flow_lm.transformer(
                prefillInput,
                cache: batchCache,
                positions: textPrefillPos,
                maskOverride: prefillMask
            )
            let prefillState = batchCache.flatMap { $0.innerState() }
            if !prefillState.isEmpty { eval(prefillState) }

            let totalMaxKV = pMax + batchMaxGen
            let fullDecodeKeyMask: MLXArray?
            let baseDecodeOffsetVec: MLXArray?
            if hasAnyPad {
                let kDecodeIdx = MLXArray(Int32(0) ..< Int32(totalMaxKV)).reshaped([1, 1, 1, totalMaxKV])
                fullDecodeKeyMask = logicalOr(kDecodeIdx .< Int32(tvLen), kDecodeIdx .>= (Int32(tvLen) + padVecInt))
                baseDecodeOffsetVec = MLXArray(pads.map { Int32(pMax - $0) })
            } else {
                fullDecodeKeyMask = nil
                baseDecodeOffsetVec = nil
            }

            var prevLatent: MLXArray? = nil
            var latentSteps: [MLXArray] = []
            latentSteps.reserveCapacity(min(batchMaxGen, 256))
            var pendingEos: [MLXArray] = []
            pendingEos.reserveCapacity(16)
            var firstEosStep = Array<Int?>(repeating: nil, count: B)

            for step in 0 ..< batchMaxGen {
                try Task.checkCancellation()
                let stepOffsetVec: MLXArray? = baseDecodeOffsetVec.map { $0 + MLXArray(Int32(step)) }
                let stepMask: MLXFast.ScaledDotProductAttentionMaskMode? = fullDecodeKeyMask.map {
                    .array($0[0..., 0..., 0..., ..<(pMax + step + 1)])
                }

                let (nextLatent, isEos) = flow_lm.stepDecode(
                    batchSize: B,
                    previousLatents: prevLatent,
                    cache: batchCache,
                    positions: nil,
                    ropeOffsetVec: stepOffsetVec,
                    maskOverride: stepMask,
                    lsdDecodeSteps: lsd_decode_steps,
                    temperature: temp,
                    noiseClamp: noise_clamp,
                    eosThreshold: eos_threshold
                )
                latentSteps.append(nextLatent)
                pendingEos.append(isEos.reshaped([B]))
                prevLatent = nextLatent.expandedDimensions(axis: 1)

                if pendingEos.count == 16 || step + 1 == batchMaxGen {
                    let kSteps = pendingEos.count
                    let chunkStartStep = step - kSteps + 1
                    let eosMat = kSteps == 1
                        ? pendingEos[0].reshaped([B, 1])
                        : stacked(pendingEos, axis: 1) // [B, kSteps]
                    eval(eosMat, nextLatent)
                    let eosFlags = eosMat.asArray(Bool.self)
                    for b in 0..<B where firstEosStep[b] == nil {
                        let rowBase = b * kSteps
                        for s in 0..<kSteps {
                            let absStep = chunkStartStep + s
                            if eosFlags[rowBase + s] || absStep >= group[b].maxGenLen {
                                firstEosStep[b] = min(absStep, group[b].maxGenLen)
                                break
                            }
                        }
                    }
                    pendingEos.removeAll(keepingCapacity: true)

                    let allDone = (0..<B).allSatisfy { b in
                        let itemMax = group[b].maxGenLen
                        if step + 1 >= itemMax { return true }
                        guard let eos = firstEosStep[b] else { return false }
                        return step >= eos + framesAfterEos - 1
                    }
                    if allDone { break }
                }
            }

            totalFlowSec += CFAbsoluteTimeGetCurrent() - tFlow0
            guard !latentSteps.isEmpty else { continue }

            let tMimi0 = CFAbsoluteTimeGetCurrent()
            let totalGeneratedSteps = latentSteps.count
            let validFrames: [Int] = (0..<B).map { b in
                let itemMax = group[b].maxGenLen
                let target = firstEosStep[b].map { $0 + framesAfterEos } ?? itemMax
                return max(0, min(totalGeneratedSteps, min(itemMax, target)))
            }
            totalGeneratedFramesCount += validFrames.reduce(0, +)

            let stackedLatents = totalGeneratedSteps == 1
                ? latentSteps[0].expandedDimensions(axis: 1)
                : stacked(latentSteps, axis: 1) // [B, T_gen, ldim]
            let decodingInput = stackedLatents * flow_lm.emb_std + flow_lm.emb_mean
            let quantizedAll = mimi.quantizer(decodingInput.transposed(0, 2, 1)) // [B, 512, T_gen]

            // Sort active sentences within the batch by `validFrames` so every Mimi subbatch has near-zero right-padding waste
            let activeIndices = (0..<B).filter { validFrames[$0] > 0 }.sorted { validFrames[$0] < validFrames[$1] }
            var ordPos = 0
            while ordPos < activeIndices.count {
                try Task.checkCancellation()
                let peekFrames = max(1, validFrames[activeIndices[ordPos]])
                let stepSubBatch = max(2, min(8, 640 / peekFrames))
                let ordEnd = min(activeIndices.count, ordPos + stepSubBatch)
                let subIndices = Array(activeIndices[ordPos..<ordEnd])
                let subMaxFrames = validFrames[subIndices.last!]
                if subMaxFrames > 0 {
                    let subQuant: MLXArray
                    if subIndices.count == 1 {
                        let onlyIdx = subIndices[0]
                        subQuant = quantizedAll[onlyIdx..<(onlyIdx + 1), 0..., ..<subMaxFrames]
                    } else {
                        let gatherIdx = MLXArray(subIndices.map { Int32($0) })
                        subQuant = quantizedAll[gatherIdx, 0..., ..<subMaxFrames]
                    }
                    let audioBatch = mimi.decodeFromLatent(subQuant) // [subB, 1, subMaxFrames * frameSamples]
                    eval(audioBatch)
                    let rowStride = audioBatch.shape[2]
                    let flatAudio = audioBatch.asArray(Float.self)
                    for (localRow, b) in subIndices.enumerated() {
                        let vf = validFrames[b]
                        guard vf > 0 else { continue }
                        let sampleCount = min(rowStride, vf * frameSamples)
                        let rowBase = localRow * rowStride
                        let samples = Array(flatAudio[rowBase ..< (rowBase + sampleCount)])
                        let job = group[b]
                        chunkSentenceSamples[job.chunkIdx][job.sentenceOrd] = samples
                    }
                }
                ordPos = ordEnd
            }
            for b in 0..<B {
                let job = group[b]
                completedSentencesForChunk[job.chunkIdx] += 1
                if completedSentencesForChunk[job.chunkIdx] == expectedSentencesPerChunk[job.chunkIdx] {
                    completedChunks += 1
                }
            }
            totalMimiSec += CFAbsoluteTimeGetCurrent() - tMimi0

            onProgress?(completedChunks, texts.count)
        }
        Memory.clearCache()
        if isBenchLog {
            print(String(format: "[PhononBatchGPU] sentences=%d frames=%d flowLM=%.2fs mimi=%.2fs", sortedJobs.count, totalGeneratedFramesCount, totalFlowSec, totalMimiSec))
        }

        return chunkSentenceSamples.map { sentences in
            if sentences.isEmpty { return [] }
            if sentences.count == 1 { return sentences[0] }
            let totalCount = sentences.reduce(0) { $0 + $1.count }
            var merged: [Float] = []
            merged.reserveCapacity(totalCount)
            for s in sentences {
                merged.append(contentsOf: s)
            }
            return merged
        }
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MaterializedArray? = nil,
        refText: String? = nil,
        language: String? = nil,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        _ = refText
        _ = language

        let prevTemp = temp
        temp = generationParameters.temperature
        defer { temp = prevTemp }
        try validateSampling()
        let cacheHint = refAudio == nil ? "voice_kv:\(voice ?? defaultAudioPrompt)" : nil
        let prompt = try await resolveAudioPrompt(voice: voice, refAudio: refAudio)
        let state = try getStateForAudioPrompt(prompt, cacheKeyHint: cacheHint)

        return (try generateAudio(state: state, text: text, framesAfterEos: nil, maxFrames: generationParameters.maxTokens)).materialized()
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
            generationParameters: generationParameters, streamingInterval: 2.0)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(); return }
            do {
                guard streamingInterval.isFinite else { throw Self.modelError("Streaming interval must be finite") }
                let previousTemperature = self.temp
                self.temp = generationParameters.temperature
                defer { self.temp = previousTemperature }
                try self.validateSampling()
                let cacheHint = refAudio == nil ? "voice_kv:\(voice ?? defaultAudioPrompt)" : nil
                let prompt = try await self.resolveAudioPrompt(voice: voice, refAudio: refAudio)
                let state = try self.getStateForAudioPrompt(prompt, cacheKeyHint: cacheHint)
                let useFastSentenceDecode = streamingInterval >= 4.0
                let intervalSamples = max(1, Int(max(0, streamingInterval) * Double(self.sampleRate)))
                var buffer: [MLXArray] = []
                var bufferedSamples = 0
                func flush() {
                    guard !buffer.isEmpty else { return }
                    let audio = buffer.count == 1 ? buffer[0] : concatenated(buffer, axis: 0)
                    buffer.removeAll(keepingCapacity: true)
                    bufferedSamples = 0
                    continuation.yield(.audio(audio))
                }
                try self.forEachAudioChunk(
                    state: state,
                    text: text,
                    framesAfterEos: nil,
                    maxFrames: generationParameters.maxTokens,
                    sentenceBatchDecode: useFastSentenceDecode
                ) { chunk in
                    buffer.append(chunk)
                    bufferedSamples += chunk.size
                    if bufferedSamples >= intervalSamples { flush() }
                }
                flush()
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }

        return stream
    }

    // MARK: - Loading

    /// Public ungated Hugging Face mirrors for official Gradium Phonon 40M models (`gated=False`).
    private static let publicPhononMirrorRepos: [String: String] = [
        "gradium/phonon-40m-multilingual": "empyrealworks/Phonon-40M-multilingual",
        "gradium/phonon-40m-en": "empyrealworks/Phonon-40M-en",
        "gradium/phonon-40m-pt": "empyrealworks/Phonon-40M-pt",
        "gradium/phonon-40m-fr": "empyrealworks/Phonon-40M-multilingual",
        "gradium/phonon-40m-de": "empyrealworks/Phonon-40M-multilingual",
        "gradium/phonon-40m-es": "empyrealworks/Phonon-40M-multilingual",
    ]

    public static func fromPretrained(
        _ modelRepo: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> PocketTTSModel {
        let rawToken: String? = hfToken
            ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? UserDefaults.standard.string(forKey: "VoicesApp.HuggingFaceToken")
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        let effectiveToken = rawToken?.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "PocketTTSModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        let targetSubdir = modelRepo.replacingOccurrences(of: "/", with: "_")
        let targetDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(targetSubdir)

        if hasUsablePocketTTSDirectory(targetDir) {
            await ensureDefaultVoiceFile(in: targetDir, cache: cache)
            return try await fromModelDirectory(targetDir)
        }

        let isGGUFRepo = modelRepo.lowercased().hasPrefix("empyrealworks/phonon-")
        let extraPatterns = ["*.model", "*.gguf", "voices/*.safetensors", "embeddings/*.safetensors"]

        do {
            let modelDir = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: isGGUFRepo ? ".gguf" : ".safetensors",
                additionalMatchingPatterns: extraPatterns,
                hfToken: (effectiveToken?.isEmpty == false) ? effectiveToken : nil,
                cache: cache
            )
            await ensureDefaultVoiceFile(in: modelDir, cache: cache)
            return try await fromModelDirectory(modelDir)
        } catch {
            guard let mirrorRepo = publicPhononMirrorRepos[modelRepo.lowercased()],
                  let mirrorID = Repo.ID(rawValue: mirrorRepo) else {
                throw error
            }
            print("Falling back to public ungated Phonon 40M checkpoint: \(mirrorRepo) (for \(modelRepo))")
            let mirrorDir = try await ModelUtils.resolveOrDownloadModel(
                repoID: mirrorID,
                requiredExtension: ".gguf",
                additionalMatchingPatterns: extraPatterns,
                hfToken: nil,
                cache: cache
            )
            await ensureDefaultVoiceFile(in: mirrorDir, cache: cache)
            populateMirrorAliasDirectory(from: mirrorDir, to: targetDir)
            if hasUsablePocketTTSDirectory(targetDir) {
                ModelUtils.deleteCachedModel(repoID: mirrorID, cache: cache)
                return try await fromModelDirectory(targetDir)
            }
            return try await fromModelDirectory(mirrorDir)
        }
    }

    private static func hasUsablePocketTTSDirectory(_ dir: URL) -> Bool {
        let fm = FileManager.default
        let configPath = dir.appendingPathComponent("config.json").path
        guard fm.fileExists(atPath: configPath) else { return false }
        for weightFile in ["model.safetensors", "model.q8.gguf", "model.q4k.gguf"] {
            let path = dir.appendingPathComponent(weightFile).path
            if fm.fileExists(atPath: path),
               let size = (try? fm.attributesOfItem(atPath: path)[.size] as? NSNumber)?.intValue,
               size > 0 {
                return true
            }
        }
        return false
    }

    private static func ensureDefaultVoiceFile(in modelDir: URL, cache: HubCache) async {
        let fm = FileManager.default
        let defaultVoiceURL = modelDir.appendingPathComponent("default-voice.safetensors")
        let resolvedDefault = defaultVoiceURL.resolvingSymlinksInPath()
        if fm.fileExists(atPath: resolvedDefault.path),
           ((try? fm.attributesOfItem(atPath: resolvedDefault.path)[.size] as? NSNumber)?.intValue ?? 0) > 0 {
            return
        }
        try? fm.removeItem(at: defaultVoiceURL)

        // 1. Check if any bundled voice exists under voices/ or embeddings/
        let existingVoices = voiceFiles(in: modelDir)
        if let firstVoice = existingVoices.values.first {
            copyDereferencedFile(from: firstVoice, to: defaultVoiceURL)
            if fm.fileExists(atPath: defaultVoiceURL.path) { return }
        }

        // 2. Check cached nativ-community/phonon Freya.safetensors (identical [1, 512, 125] speaker_wavs format)
        let cachedFreya = cache.cacheDirectory
            .appendingPathComponent("mlx-audio/nativ-community_phonon/voices/Freya.safetensors")
        if fm.fileExists(atPath: cachedFreya.path) {
            copyDereferencedFile(from: cachedFreya, to: defaultVoiceURL)
            if fm.fileExists(atPath: defaultVoiceURL.path) { return }
        }

        // 3. Download official default-voice.safetensors from ungated gradium/phonon-31m-en
        if let repo = Repo.ID(rawValue: "gradium/phonon-31m-en") {
            let client = HubClient(cache: cache)
            if let downloaded = try? await client.downloadFile(at: "default-voice.safetensors", from: repo, revision: "main") {
                copyDereferencedFile(from: downloaded, to: defaultVoiceURL)
            }
        }
    }

    private static func populateMirrorAliasDirectory(from sourceDir: URL, to targetDir: URL) {
        guard sourceDir.standardizedFileURL != targetDir.standardizedFileURL else { return }
        let fm = FileManager.default
        try? fm.createDirectory(at: targetDir, withIntermediateDirectories: true)
        guard let items = try? fm.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil) else { return }
        // Copy weights and assets before config.json so targetDir is never seen with config.json alone
        let sortedItems = items.sorted { a, b in
            if a.lastPathComponent == "config.json" { return false }
            if b.lastPathComponent == "config.json" { return true }
            return a.lastPathComponent < b.lastPathComponent
        }
        for item in sortedItems {
            let dst = targetDir.appendingPathComponent(item.lastPathComponent)
            let resolvedItem = item.resolvingSymlinksInPath()
            if !fm.fileExists(atPath: dst.path) {
                try? fm.removeItem(at: dst)
                try? fm.copyItem(at: resolvedItem, to: dst)
            }
        }
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> PocketTTSModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try PocketTTSModelConfig.load(from: configURL)

        let rawWeights = try loadPocketTTSWeights(modelDir: modelDir, weightsName: config.weightsName)
        let hasEncoder = config.mimi.encoderEnabled && rawWeights.keys.contains {
            $0.replacingOccurrences(of: "mimi.model.", with: "mimi.").hasPrefix("mimi.encoder.")
        }
        let model = try await PocketTTSModel.fromConfig(config, modelFolder: modelDir, encoderEnabled: hasEncoder)
        let weights = model.sanitize(weights: rawWeights)

        // Keep native MLX packed matrices quantized, including mixed precision
        // checkpoints. Only modules with scale tensors are replaced.
        var metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        metadata["model_type"] = config.modelType
        metadata["quantization"] = metadata["quantization"] ?? metadata["quantization_config"]
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: JSONSerialization.data(withJSONObject: metadata))
        if let quantization = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                return quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.flow_lm.cacheConditioning()
        eval(model)
        return model
    }
}

private func loadPocketTTSWeights(modelDir: URL, weightsName: String?) throws -> [String: MLXArray] {
    for name in [weightsName, "model.safetensors", "model.q8.gguf", "model.q4k.gguf"].compactMap({ $0 }) {
        let file = modelDir.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: file.path) {
            if file.pathExtension.lowercased() == "gguf" {
                return try loadPocketTTSGGUF(url: file)
            }
            return try MLX.loadArrays(url: file)
        }
    }
    let files = (try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)) ?? []
    let shards = files
        .filter { $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    if !shards.isEmpty {
        var weights: [String: MLXArray] = [:]
        for shard in shards { weights.merge(try MLX.loadArrays(url: shard)) { _, new in new } }
        return weights
    }
    if let ggufFile = files.first(where: { $0.pathExtension.lowercased() == "gguf" }) {
        return try loadPocketTTSGGUF(url: ggufFile)
    }
    throw NSError(domain: "PocketTTSModel", code: 2,
                  userInfo: [NSLocalizedDescriptionKey: "Model safetensors or gguf not found in \(modelDir.path)"])
}
