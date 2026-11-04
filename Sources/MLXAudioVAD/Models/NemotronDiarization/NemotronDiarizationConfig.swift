import Foundation
import MLXAudioCore

public enum NemotronDiarizationConfigError: Error, LocalizedError, Equatable {
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return message
        }
    }
}

public struct NemotronDiarizationEncoderConfig: Codable, Equatable, Sendable {
    public var featIn: Int
    public var dModel: Int
    public var nLayers: Int
    public var nHeads: Int
    public var subsamplingFactor: Int
    public var ffExpansion: Float
    public var qkvBias: Bool
    public var qkNorm: Bool
    public var preBlockNorm: Bool
    public var xscaling: Bool
    public var ropeBase: Float
    public var rotaryFraction: Float

    public init(
        featIn: Int = 128,
        dModel: Int = 512,
        nLayers: Int = 31,
        nHeads: Int = 8,
        subsamplingFactor: Int = 8,
        ffExpansion: Float = 4.0,
        qkvBias: Bool = false,
        qkNorm: Bool = false,
        preBlockNorm: Bool = true,
        xscaling: Bool = false,
        ropeBase: Float = 10000.0,
        rotaryFraction: Float = 1.0
    ) {
        self.featIn = featIn
        self.dModel = dModel
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.subsamplingFactor = subsamplingFactor
        self.ffExpansion = ffExpansion
        self.qkvBias = qkvBias
        self.qkNorm = qkNorm
        self.preBlockNorm = preBlockNorm
        self.xscaling = xscaling
        self.ropeBase = ropeBase
        self.rotaryFraction = rotaryFraction
    }

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case dModel = "d_model"
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case subsamplingFactor = "subsampling_factor"
        case ffExpansion = "ff_expansion"
        case qkvBias = "qkv_bias"
        case qkNorm = "qk_norm"
        case preBlockNorm = "pre_block_norm"
        case xscaling
        case ropeBase = "rope_base"
        case rotaryFraction = "rotary_fraction"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try c.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 512
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 31
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        ffExpansion = try c.decodeIfPresent(Float.self, forKey: .ffExpansion) ?? 4.0
        qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? false
        preBlockNorm = try c.decodeIfPresent(Bool.self, forKey: .preBlockNorm) ?? true
        xscaling = try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 10000.0
        rotaryFraction = try c.decodeIfPresent(Float.self, forKey: .rotaryFraction) ?? 1.0
    }
}

public struct NemotronDiarizationModulesConfig: Codable, Equatable, Sendable {
    public var numSpeakers: Int
    public var fcDModel: Int
    public var tfDModel: Int
    public var subsamplingFactor: Int
    public var chunkLen: Int
    public var fifoLen: Int
    public var spkcacheLen: Int
    public var spkcacheUpdatePeriod: Int
    public var chunkLeftContext: Int
    public var chunkRightContext: Int
    public var spkcacheSilFramesPerSpk: Int
    public var predScoreThreshold: Float
    public var maxIndex: Int
    public var scoresBoostLatest: Float
    public var silThreshold: Float
    public var strongBoostRate: Float
    public var weakBoostRate: Float
    public var minPosScoresRate: Float
    public var useAosc: Bool
    public var useLearnableSilEmb: Bool
    public var useActivityHead: Bool

    public init(
        numSpeakers: Int = 8,
        fcDModel: Int = 512,
        tfDModel: Int = 192,
        subsamplingFactor: Int = 8,
        chunkLen: Int = 340,
        fifoLen: Int = 40,
        spkcacheLen: Int = 264,
        spkcacheUpdatePeriod: Int = 300,
        chunkLeftContext: Int = 0,
        chunkRightContext: Int = 40,
        spkcacheSilFramesPerSpk: Int = 1,
        predScoreThreshold: Float = 0.25,
        maxIndex: Int = 99999,
        scoresBoostLatest: Float = 0.05,
        silThreshold: Float = 0.2,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        useAosc: Bool = true,
        useLearnableSilEmb: Bool = true,
        useActivityHead: Bool = true
    ) {
        self.numSpeakers = numSpeakers
        self.fcDModel = fcDModel
        self.tfDModel = tfDModel
        self.subsamplingFactor = subsamplingFactor
        self.chunkLen = chunkLen
        self.fifoLen = fifoLen
        self.spkcacheLen = spkcacheLen
        self.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        self.predScoreThreshold = predScoreThreshold
        self.maxIndex = maxIndex
        self.scoresBoostLatest = scoresBoostLatest
        self.silThreshold = silThreshold
        self.strongBoostRate = strongBoostRate
        self.weakBoostRate = weakBoostRate
        self.minPosScoresRate = minPosScoresRate
        self.useAosc = useAosc
        self.useLearnableSilEmb = useLearnableSilEmb
        self.useActivityHead = useActivityHead
    }

    enum CodingKeys: String, CodingKey {
        case numSpeakers = "num_speakers"
        case fcDModel = "fc_d_model"
        case tfDModel = "tf_d_model"
        case subsamplingFactor = "subsampling_factor"
        case chunkLen = "chunk_len"
        case fifoLen = "fifo_len"
        case spkcacheLen = "spkcache_len"
        case spkcacheUpdatePeriod = "spkcache_update_period"
        case chunkLeftContext = "chunk_left_context"
        case chunkRightContext = "chunk_right_context"
        case spkcacheSilFramesPerSpk = "spkcache_sil_frames_per_spk"
        case predScoreThreshold = "pred_score_threshold"
        case maxIndex = "max_index"
        case scoresBoostLatest = "scores_boost_latest"
        case silThreshold = "sil_threshold"
        case strongBoostRate = "strong_boost_rate"
        case weakBoostRate = "weak_boost_rate"
        case minPosScoresRate = "min_pos_scores_rate"
        case useAosc = "use_aosc"
        case useLearnableSilEmb = "use_learnable_sil_emb"
        case useActivityHead = "use_activity_head"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numSpeakers = try c.decodeIfPresent(Int.self, forKey: .numSpeakers) ?? 8
        fcDModel = try c.decodeIfPresent(Int.self, forKey: .fcDModel) ?? 512
        tfDModel = try c.decodeIfPresent(Int.self, forKey: .tfDModel) ?? 192
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        chunkLen = try c.decodeIfPresent(Int.self, forKey: .chunkLen) ?? 340
        fifoLen = try c.decodeIfPresent(Int.self, forKey: .fifoLen) ?? 40
        spkcacheLen = try c.decodeIfPresent(Int.self, forKey: .spkcacheLen) ?? 264
        spkcacheUpdatePeriod = try c.decodeIfPresent(Int.self, forKey: .spkcacheUpdatePeriod) ?? 300
        chunkLeftContext = try c.decodeIfPresent(Int.self, forKey: .chunkLeftContext) ?? 0
        chunkRightContext = try c.decodeIfPresent(Int.self, forKey: .chunkRightContext) ?? 40
        spkcacheSilFramesPerSpk = try c.decodeIfPresent(Int.self, forKey: .spkcacheSilFramesPerSpk) ?? 1
        predScoreThreshold = try c.decodeIfPresent(Float.self, forKey: .predScoreThreshold) ?? 0.25
        maxIndex = try c.decodeIfPresent(Int.self, forKey: .maxIndex) ?? 99999
        scoresBoostLatest = try c.decodeIfPresent(Float.self, forKey: .scoresBoostLatest) ?? 0.05
        silThreshold = try c.decodeIfPresent(Float.self, forKey: .silThreshold) ?? 0.2
        strongBoostRate = try c.decodeIfPresent(Float.self, forKey: .strongBoostRate) ?? 0.75
        weakBoostRate = try c.decodeIfPresent(Float.self, forKey: .weakBoostRate) ?? 1.5
        minPosScoresRate = try c.decodeIfPresent(Float.self, forKey: .minPosScoresRate) ?? 0.5
        useAosc = try c.decodeIfPresent(Bool.self, forKey: .useAosc) ?? true
        useLearnableSilEmb = try c.decodeIfPresent(Bool.self, forKey: .useLearnableSilEmb) ?? true
        useActivityHead = try c.decodeIfPresent(Bool.self, forKey: .useActivityHead) ?? true
    }

    public var asSortformerModulesConfig: ModulesConfig {
        var cfg = ModulesConfig(
            numSpeakers: numSpeakers,
            fcDModel: fcDModel,
            tfDModel: tfDModel,
            subsamplingFactor: subsamplingFactor
        )
        cfg.chunkLen = chunkLen
        cfg.fifoLen = fifoLen
        cfg.spkcacheLen = spkcacheLen
        cfg.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        cfg.chunkLeftContext = chunkLeftContext
        cfg.chunkRightContext = chunkRightContext
        cfg.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        cfg.predScoreThreshold = predScoreThreshold
        cfg.maxIndex = maxIndex
        cfg.scoresBoostLatest = scoresBoostLatest
        cfg.silThreshold = silThreshold
        cfg.strongBoostRate = strongBoostRate
        cfg.weakBoostRate = weakBoostRate
        cfg.minPosScoresRate = minPosScoresRate
        cfg.useAosc = useAosc
        return cfg
    }
}

public struct NemotronDiarizationMelConfig: Codable, Equatable, Sendable {
    public var featureSize: Int
    public var samplingRate: Int
    public var hopLength: Int
    public var nFft: Int
    public var winLength: Int
    public var preemphasis: Float
    public var padTo: Int

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        hopLength: Int = 160,
        nFft: Int = 512,
        winLength: Int = 400,
        preemphasis: Float = 0.97,
        padTo: Int = 16
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.hopLength = hopLength
        self.nFft = nFft
        self.winLength = winLength
        self.preemphasis = preemphasis
        self.padTo = padTo
    }

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case nFft = "n_fft"
        case winLength = "win_length"
        case preemphasis
        case padTo = "pad_to"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? 128
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 512
        winLength = try c.decodeIfPresent(Int.self, forKey: .winLength) ?? 400
        preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? 0.97
        padTo = try c.decodeIfPresent(Int.self, forKey: .padTo) ?? 16
    }
}

public struct NemotronDiarizationConfig: Codable, Equatable, Sendable {
    public var modelType: String
    public var numSpeakers: Int
    public var dtype: String
    public var outputSubsamplingFactor: Int
    public var encoderConfig: NemotronDiarizationEncoderConfig
    public var modulesConfig: NemotronDiarizationModulesConfig
    public var processorConfig: NemotronDiarizationMelConfig
    public var quantization: BaseConfiguration.Quantization?

    public init(
        modelType: String = "nemotron_diarization",
        numSpeakers: Int = 8,
        dtype: String = "float32",
        outputSubsamplingFactor: Int = 1,
        encoderConfig: NemotronDiarizationEncoderConfig = NemotronDiarizationEncoderConfig(),
        modulesConfig: NemotronDiarizationModulesConfig = NemotronDiarizationModulesConfig(),
        processorConfig: NemotronDiarizationMelConfig = NemotronDiarizationMelConfig(),
        quantization: BaseConfiguration.Quantization? = nil
    ) throws {
        self.modelType = modelType
        self.numSpeakers = numSpeakers
        self.dtype = dtype
        self.outputSubsamplingFactor = outputSubsamplingFactor
        self.encoderConfig = encoderConfig
        self.modulesConfig = modulesConfig
        self.processorConfig = processorConfig
        self.quantization = quantization
        try validate()
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numSpeakers = "num_speakers"
        case dtype
        case outputSubsamplingFactor = "output_subsampling_factor"
        case encoderConfig = "encoder_config"
        case modulesConfig = "modules_config"
        case processorConfig = "processor_config"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_diarization"
        numSpeakers = try c.decodeIfPresent(Int.self, forKey: .numSpeakers) ?? 8
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? "float32"
        outputSubsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .outputSubsamplingFactor) ?? 1
        encoderConfig = try c.decodeIfPresent(NemotronDiarizationEncoderConfig.self, forKey: .encoderConfig)
            ?? NemotronDiarizationEncoderConfig()
        modulesConfig = try c.decodeIfPresent(NemotronDiarizationModulesConfig.self, forKey: .modulesConfig)
            ?? NemotronDiarizationModulesConfig()
        processorConfig = try c.decodeIfPresent(NemotronDiarizationMelConfig.self, forKey: .processorConfig)
            ?? NemotronDiarizationMelConfig()
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        try validate()
    }

    public func validate() throws {
        let enc = encoderConfig
        let mod = modulesConfig

        guard enc.nHeads > 0, enc.dModel % enc.nHeads == 0 else {
            throw NemotronDiarizationConfigError.invalidConfiguration("d_model must be divisible by n_heads")
        }
        let rotaryDim = Int(Float(enc.dModel / enc.nHeads) * enc.rotaryFraction)
        guard rotaryDim >= 2, rotaryDim % 2 == 0, enc.rotaryFraction > 0, enc.rotaryFraction <= 1 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "rotary_fraction must select an even, positive head dimension"
            )
        }
        guard enc.featIn == processorConfig.featureSize else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Encoder and preprocessor feature dimensions must match"
            )
        }
        guard enc.dModel == mod.fcDModel, numSpeakers == mod.numSpeakers else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Encoder and speaker head dimensions must match"
            )
        }
        guard enc.subsamplingFactor == mod.subsamplingFactor else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Encoder and speaker head subsampling factors must match"
            )
        }
        guard mod.chunkLen >= 1, mod.spkcacheUpdatePeriod >= 1, mod.spkcacheLen >= 1 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "chunk_len, spkcache_update_period, and spkcache_len must be positive integers"
            )
        }
        guard mod.fifoLen >= 0, mod.chunkRightContext >= 0, mod.spkcacheSilFramesPerSpk >= 0 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "fifo_len, chunk_right_context, and spkcache_sil_frames_per_spk must be non-negative integers"
            )
        }
        guard mod.chunkLeftContext == 0 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Feature stacking requires chunk_left_context=0"
            )
        }
        guard mod.spkcacheLen >= (1 + mod.spkcacheSilFramesPerSpk) * numSpeakers else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Speaker cache is too small for the number of speakers"
            )
        }
        guard outputSubsamplingFactor >= 1 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "output_subsampling_factor must be a positive integer"
            )
        }
        guard (mod.chunkLen * enc.subsamplingFactor) % outputSubsamplingFactor == 0 else {
            throw NemotronDiarizationConfigError.invalidConfiguration(
                "Output stride must divide the chunk's feature frame count"
            )
        }
    }
}
