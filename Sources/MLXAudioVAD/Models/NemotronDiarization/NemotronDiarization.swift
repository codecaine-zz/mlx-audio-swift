import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFast
import MLXFFT
import MLXNN

public enum NemotronDiarizationStreamingPreset: String, CaseIterable, Sendable {
    case offline = "offline"
    case low = "low"
    case veryLow = "very_low"
    case ultraLow = "ultra_low"

    var parameters: (chunkLen: Int, rightContext: Int, fifoLen: Int, updatePeriod: Int) {
        switch self {
        case .offline: return (340, 40, 40, 300)
        case .low: return (9, 4, 264, 222)
        case .veryLow: return (6, 2, 264, 222)
        case .ultraLow: return (3, 1, 264, 222)
        }
    }
}

public enum NemotronDiarizationError: Error, LocalizedError, Equatable {
    case invalidRepositoryID(String)
    case unknownPreset(String)
    case streamFinished
    case invalidSampleRate(expected: Int, actual: Int)
    case nonMonoChunk

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let repo):
            return "Invalid Nemotron Diarization repository ID: \(repo)"
        case .unknownPreset(let preset):
            return "Unknown streaming preset '\(preset)'; choose from offline, low, very_low, ultra_low"
        case .streamFinished:
            return "This stream is finished; initialize a new state"
        case .invalidSampleRate(let expected, let actual):
            return "feed requires \(expected) Hz audio (got \(actual) Hz); resample before streaming"
        case .nonMonoChunk:
            return "feed expects one-dimensional mono audio"
        }
    }
}

public struct NemotronDiarizationStreamingState: @unchecked Sendable {
    public var spkcache: MLXArray
    public var spkcachePreds: MLXArray
    public var fifo: MLXArray
    public var fifoPreds: MLXArray
    public var spkcacheCompressed: Bool
    public var framesProcessed: Int
    public var samplesReceived: Int
    public var sampleOffset: Int
    public var audioBuffer: MLXArray
    public var finished: Bool

    public init(
        spkcache: MLXArray,
        spkcachePreds: MLXArray,
        fifo: MLXArray,
        fifoPreds: MLXArray,
        spkcacheCompressed: Bool = false,
        framesProcessed: Int = 0,
        samplesReceived: Int = 0,
        sampleOffset: Int = 0,
        audioBuffer: MLXArray = MLXArray.zeros([0], dtype: .float32),
        finished: Bool = false
    ) {
        self.spkcache = spkcache
        self.spkcachePreds = spkcachePreds
        self.fifo = fifo
        self.fifoPreds = fifoPreds
        self.spkcacheCompressed = spkcacheCompressed
        self.framesProcessed = framesProcessed
        self.samplesReceived = samplesReceived
        self.sampleOffset = sampleOffset
        self.audioBuffer = audioBuffer
        self.finished = finished
    }
}

final class NemotronMelFeatures: Module {
    let config: NemotronDiarizationMelConfig
    @ParameterInfo(key: "window") var window: MLXArray
    @ParameterInfo(key: "fb") var fb: MLXArray

    init(_ config: NemotronDiarizationMelConfig) {
        self.config = config
        self._window.wrappedValue = hanningWindow(size: config.winLength, periodic: false)
        // Python mel_filters returns [nMels, nFreqs], so fb has shape [1, nMels, nFreqs].
        let filters = melFilters(
            sampleRate: config.samplingRate,
            nFft: config.nFft,
            nMels: config.featureSize,
            fMin: 0,
            fMax: nil,
            norm: "slaney",
            melScale: .slaney
        ).transposed(1, 0).expandedDimensions(axis: 0)
        self._fb.wrappedValue = filters
    }

    func callAsFunction(
        _ audio: MLXArray,
        start: Int = 0,
        count: Int? = nil,
        sampleOffset: Int = 0,
        totalSamples: Int? = nil
    ) -> MLXArray {
        let cfg = config
        let resolvedTotal = totalSamples ?? audio.dim(0)
        let validFrames = resolvedTotal / cfg.hopLength
        let resolvedCount = count ?? validFrames
        if resolvedCount <= 0 {
            return MLXArray.zeros([1, cfg.featureSize, 0], dtype: .float32)
        }

        let frameIds = MLXArray(Int32(start)..<Int32(start + resolvedCount))
        let positions =
            frameIds.expandedDimensions(axis: 1) * Int32(cfg.hopLength)
            + MLXArray(0..<Int32(cfg.nFft)).expandedDimensions(axis: 0)
            - Int32(cfg.nFft / 2)

        func gather(_ indices: MLXArray) -> MLXArray {
            guard audio.dim(0) > 0 else {
                return MLXArray.zeros(indices.shape, dtype: .float32)
            }
            let local = indices - Int32(sampleOffset)
            let clipped = MLX.clip(local, min: 0, max: max(0, audio.dim(0) - 1))
            let values = audio[clipped]
            let inBounds = (indices .>= 0) & (indices .< Int32(resolvedTotal))
            return MLX.where(inBounds, values, MLXArray(Float(0.0)))
        }

        let currentSamples = gather(positions)
        let previousSamples = gather(positions - 1)
        let inBounds = (positions .>= 0) & (positions .< Int32(resolvedTotal))
        let frames = MLX.where(
            inBounds,
            currentSamples - cfg.preemphasis * previousSamples,
            MLXArray(Float(0.0))
        )

        let pad = cfg.nFft - cfg.winLength
        let paddedWindow = MLX.padded(
            window.asType(.float32),
            widths: [.init((pad / 2, pad - pad / 2))]
        )
        let power = MLX.abs(MLXFFT.rfft(frames * paddedWindow, axis: -1)).square()
        let filterMatrix = fb[0].asType(.float32).transposed(1, 0)
        let logGuard = Float(pow(2.0, -24.0))
        var features = MLX.log(MLX.matmul(power, filterMatrix) + logGuard)
        let validMask = frameIds.expandedDimensions(axis: 1) .< Int32(validFrames)
        features = MLX.where(validMask, features, MLXArray(Float(0.0)))
        return contiguous(features.transposed(1, 0)).expandedDimensions(axis: 0)
    }
}

final class NemotronFeatureStacking: Module {
    let factor: Int
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: NemotronDiarizationEncoderConfig) {
        self.factor = config.subsamplingFactor
        self._proj.wrappedValue = Linear(
            config.featIn * config.subsamplingFactor,
            config.dModel,
            bias: false
        )
    }

    func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var x = features.transposed(0, 2, 1)
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let padFrames = (factor - (t % factor)) % factor
        if padFrames > 0 {
            x = MLX.padded(x, widths: [.init((0, 0)), .init((0, padFrames)), .init((0, 0))])
        }
        let stacked = proj(x.reshaped(b, -1, c * factor))
        let outLengths = (lengths + Int32(factor - 1)) / Int32(factor)
        return (stacked, outLengths)
    }
}

final class NemotronAttention: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "w_qkv") var wQkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let rope: RoPE
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?

    init(_ config: NemotronDiarizationEncoderConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.dModel / config.nHeads
        self.scale = pow(Float(headDim), -0.5)
        self._wQkv.wrappedValue = Linear(config.dModel, 3 * config.dModel, bias: config.qkvBias)
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        self.rope = RoPE(
            dimensions: Int(Float(headDim) * config.rotaryFraction),
            traditional: false,
            base: config.ropeBase
        )
        if config.qkNorm {
            self._qNorm.wrappedValue = LayerNorm(dimensions: headDim)
            self._kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        } else {
            self._qNorm.wrappedValue = nil
            self._kNorm.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, t, d) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = wQkv(x).reshaped(b, t, 3, nHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)

        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        q = rope(q)
        k = rope(k)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .array(mask)
        )
        return outProj(attn.transposed(0, 2, 1, 3).reshaped(b, t, d))
    }
}

final class NemotronFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: NemotronDiarizationEncoderConfig) {
        let hidden = Int(Float(config.dModel) * config.ffExpansion)
        self._linear1.wrappedValue = Linear(config.dModel, hidden, bias: true)
        self._linear2.wrappedValue = Linear(hidden, config.dModel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

final class NemotronTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NemotronAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NemotronFeedForward

    init(_ config: NemotronDiarizationEncoderConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._attn.wrappedValue = NemotronAttention(config)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._ffn.wrappedValue = NemotronFeedForward(config)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), mask: mask)
        return h + ffn(norm2(h))
    }
}

final class NemotronDiarizationEncoder: Module {
    let scale: Float

    @ModuleInfo(key: "pre_encode") var preEncode: NemotronFeatureStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm?
    @ModuleInfo(key: "layers") var layers: [NemotronTransformerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ config: NemotronDiarizationEncoderConfig) {
        self.scale = config.xscaling ? sqrt(Float(config.dModel)) : 1.0
        self._preEncode.wrappedValue = NemotronFeatureStacking(config)
        self._embedNorm.wrappedValue = config.preBlockNorm ? LayerNorm(dimensions: config.dModel) : nil
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            NemotronTransformerBlock(config)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> MLXArray {
        let valid = MLXArray(0..<Int32(x.dim(1))).expandedDimensions(axis: 0)
            .< lengths.asType(.int32).expandedDimensions(axis: 1)
        let mask = valid.expandedDimensions(axes: [1, 2])
        var h = x * scale
        if let embedNorm {
            h = embedNorm(h)
        }
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return finalNorm(h)
    }
}

final class NemotronActivityHead: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    init(tfDModel: Int) {
        self._layers.wrappedValue = [
            LayerNorm(dimensions: tfDModel),
            Linear(tfDModel, 3, bias: true)
        ]
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

final class NemotronSpeakerModules: Module, UnaryLayer {
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var firstHiddenToHidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var singleHiddenToSpks: Linear
    @ModuleInfo(key: "hidden_to_spks") var hiddenToSpks: Linear
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ParameterInfo(key: "learnable_sil_emb") var learnableSilEmb: MLXArray
    @ModuleInfo(key: "activity_head") var activityHead: NemotronActivityHead?

    init(_ config: NemotronDiarizationModulesConfig) {
        self._encoderProj.wrappedValue = Linear(config.fcDModel, config.tfDModel, bias: true)
        self._firstHiddenToHidden.wrappedValue = Linear(config.tfDModel, config.tfDModel, bias: true)
        self._singleHiddenToSpks.wrappedValue = Linear(config.tfDModel, config.numSpeakers, bias: true)
        self._hiddenToSpks.wrappedValue = Linear(2 * config.tfDModel, config.numSpeakers, bias: true)
        self._subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: config.tfDModel,
            outputChannels: config.tfDModel * config.subsamplingFactor,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._learnableSilEmb.wrappedValue = MLXArray.zeros([config.fcDModel], dtype: .float32)
        if config.useActivityHead {
            self._activityHead.wrappedValue = NemotronActivityHead(tfDModel: config.tfDModel)
        } else {
            self._activityHead.wrappedValue = nil
        }
    }

    func forwardSpeakerSigmoids(_ hiddenOut: MLXArray) -> MLXArray {
        var h = relu(hiddenOut)
        h = firstHiddenToHidden(h)
        h = relu(h)
        let spkPreds = singleHiddenToSpks(h)
        return sigmoid(spkPreds)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = encoderProj(input)
        let (b, _, h) = (projected.dim(0), projected.dim(1), projected.dim(2))
        let upsampled = subpixelUpsample(projected).reshaped(b, -1, h)
        return forwardSpeakerSigmoids(upsampled)
    }
}

public final class NemotronDiarizationModel: Module, @unchecked Sendable {
    public static let defaultRepo = "mlx-community/Nemotron-3-Diarization"
    public static let quantized8BitRepo = "mlx-community/Nemotron-3-Diarization-8bit"

    public private(set) var config: NemotronDiarizationConfig

    @ModuleInfo(key: "encoder") var encoder: NemotronDiarizationEncoder
    @ModuleInfo(key: "sortformer_modules") var sortformerModules: NemotronSpeakerModules
    @ModuleInfo(key: "preprocessor") var preprocessor: NemotronMelFeatures

    public init(_ config: NemotronDiarizationConfig) {
        self.config = config
        self._encoder.wrappedValue = NemotronDiarizationEncoder(config.encoderConfig)
        self._sortformerModules.wrappedValue = NemotronSpeakerModules(config.modulesConfig)
        self._preprocessor.wrappedValue = NemotronMelFeatures(config.processorConfig)
    }

    public var modelDtype: DType {
        if let qLinear = encoder.preEncode.proj as? QuantizedLinear {
            return qLinear.scales.dtype
        }
        return encoder.preEncode.proj.weight.dtype
    }

    public var sampleRate: Int {
        config.processorConfig.samplingRate
    }

    /// Single-window forward pass: `(B, mel, T) -> (B, ceil(T/8)*8, speakers)`.
    public func callAsFunction(_ audioSignal: MLXArray, audioSignalLength: MLXArray) -> MLXArray {
        let (x, lengths) = encoder.preEncode(
            audioSignal.asType(modelDtype),
            lengths: audioSignalLength
        )
        let probs = sortformerModules(encoder(x, lengths: lengths))
        let valid = MLXArray(0..<Int32(probs.dim(1))).expandedDimensions(axis: 0)
            .< audioSignalLength.asType(.int32).expandedDimensions(axis: 1)
        return probs * valid.expandedDimensions(axis: 2)
    }

    public func initStreamingState() -> NemotronDiarizationStreamingState {
        NemotronDiarizationStreamingState(
            spkcache: MLXArray.zeros([1, 0, config.encoderConfig.dModel], dtype: modelDtype),
            spkcachePreds: MLXArray.zeros([1, 0, config.numSpeakers], dtype: .float32),
            fifo: MLXArray.zeros([1, 0, config.encoderConfig.dModel], dtype: modelDtype),
            fifoPreds: MLXArray.zeros([1, 0, config.numSpeakers], dtype: .float32)
        )
    }

    public func setStreamingConfig(_ preset: NemotronDiarizationStreamingPreset) {
        let p = preset.parameters
        config.modulesConfig.chunkLen = p.chunkLen
        config.modulesConfig.chunkRightContext = p.rightContext
        config.modulesConfig.fifoLen = p.fifoLen
        config.modulesConfig.spkcacheLen = 264
        config.modulesConfig.spkcacheUpdatePeriod = p.updatePeriod
    }

    public func setStreamingConfig(_ presetName: String) throws {
        guard let preset = NemotronDiarizationStreamingPreset(rawValue: presetName) else {
            throw NemotronDiarizationError.unknownPreset(presetName)
        }
        setStreamingConfig(preset)
    }

    public func streamingStep(
        features: MLXArray,
        state: inout NemotronDiarizationStreamingState,
        centralFrames: Int,
        featureLength: Int? = nil,
        leftContextFrames: Int = 0
    ) -> MLXArray {
        let cfg = config.modulesConfig
        let factor = config.encoderConfig.subsamplingFactor
        let resolvedLength = featureLength ?? features.dim(-1)

        let (chunk, lengths) = encoder.preEncode(
            features.asType(modelDtype),
            lengths: MLXArray([Int32(resolvedLength)])
        )
        let cacheLen = state.spkcache.dim(1)
        let fifoLen = state.fifo.dim(1)
        let combined = MLX.concatenated([state.spkcache, state.fifo, chunk], axis: 1)
        let totalLengths = lengths + Int32(cacheLen + fifoLen)
        var high = sortformerModules(encoder(combined, lengths: totalLengths))

        let validCount = (lengths[0].item(Int32.self) + Int32(cacheLen + fifoLen)) * Int32(factor)
        let valid = MLXArray(0..<Int32(high.dim(1))) .< validCount
        high = high * valid.expandedDimensions(axes: [0, 2])

        // Cache scoring always operates at encoder resolution (80 ms), using mean probabilities.
        let low = high.asType(.float32)
            .reshaped(1, -1, factor, config.numSpeakers)
            .mean(axis: 2)

        let n = (centralFrames + factor - 1) / factor
        let left = leftContextFrames / factor
        let start = cacheLen + fifoLen + left
        let result = high[0, (start * factor)..<(start * factor + centralFrames)].asType(.float32)

        state.fifo = MLX.concatenated([state.fifo, chunk[0..., left..<(left + n), 0...]], axis: 1)
        state.fifoPreds = MLX.concatenated(
            [
                low[0..., cacheLen..<(cacheLen + fifoLen), 0...],
                low[0..., start..<(start + n), 0...]
            ],
            axis: 1
        )

        if state.fifo.dim(1) > cfg.fifoLen {
            let pop = min(
                state.fifo.dim(1),
                max(cfg.spkcacheUpdatePeriod, state.fifo.dim(1) - cfg.fifoLen)
            )
            state.spkcache = MLX.concatenated(
                [state.spkcache, state.fifo[0..., 0..<pop, 0...]],
                axis: 1
            )
            let previous = state.spkcacheCompressed
                ? state.spkcachePreds
                : low[0..., 0..<cacheLen, 0...]
            state.spkcachePreds = MLX.concatenated(
                [previous, state.fifoPreds[0..., 0..<pop, 0...]],
                axis: 1
            )
            state.fifo = state.fifo[0..., pop..., 0...]
            state.fifoPreds = state.fifoPreds[0..., pop..., 0...]

            if state.spkcache.dim(1) > cfg.spkcacheLen {
                let (compressedCache, compressedPreds) = SortformerModel.compressSpeakerCache(
                    embeddings: state.spkcache,
                    probabilities: state.spkcachePreds,
                    silenceEmbedding: sortformerModules.learnableSilEmb.expandedDimensions(axis: 0),
                    configuration: cfg.asSortformerModulesConfig
                )
                state.spkcache = compressedCache
                state.spkcachePreds = compressedPreds
                state.spkcacheCompressed = true
            }
        }

        state.framesProcessed += centralFrames
        eval(result, state.spkcache, state.spkcachePreds, state.fifo, state.fifoPreds)
        return result
    }

    private func makeOutput(
        probs rawProbs: MLXArray,
        offset: Float = 0.0,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0,
        state: NemotronDiarizationStreamingState? = nil
    ) -> DiarizationOutput {
        var probs = rawProbs
        let factor = config.outputSubsamplingFactor
        if factor > 1, probs.dim(0) > 0 {
            let length = probs.dim(0)
            let padLen = (factor - (length % factor)) % factor
            let padded = padLen > 0
                ? MLX.padded(probs, widths: [.init((0, padLen)), .init((0, 0))])
                : probs
            let numOut = padded.dim(0) / factor
            let counts = MLX.minimum(
                MLXArray(Int32(factor)),
                MLXArray(Int32(length)) - MLXArray(0..<Int32(numOut)) * Int32(factor)
            ).asType(.float32)
            probs = padded.reshaped(-1, factor, config.numSpeakers).sum(axis: 1)
                / counts.expandedDimensions(axis: 1)
        }

        let strideSec = Float(config.processorConfig.hopLength) / Float(sampleRate)
        let rawSegments = probs.dim(0) > 0
            ? SortformerModel.predsToSegments(
                probs,
                frameDuration: strideSec * Float(factor),
                threshold: threshold,
                minDuration: minDuration,
                mergeGap: mergeGap
            )
            : []

        let maxEnd = state.map { Float($0.framesProcessed) * strideSec }
        let segments = rawSegments.map { seg in
            var end = seg.end + offset
            if let maxEnd {
                end = min(end, maxEnd)
            }
            return DiarizationSegment(
                start: seg.start + offset,
                end: end,
                speaker: seg.speaker
            )
        }
        let activeSpeakers = Set(segments.map(\.speaker)).count
        return DiarizationOutput(
            segments: segments,
            speakerProbs: probs,
            numSpeakers: activeSpeakers
        )
    }

    public func feed(
        audio: MLXArray,
        state initialState: NemotronDiarizationStreamingState,
        sampleRate: Int = 16000,
        final: Bool = false,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) throws -> (DiarizationOutput, NemotronDiarizationStreamingState) {
        var state = initialState
        if state.finished {
            throw NemotronDiarizationError.streamFinished
        }
        guard sampleRate == self.sampleRate else {
            throw NemotronDiarizationError.invalidSampleRate(expected: self.sampleRate, actual: sampleRate)
        }
        let chunk = audio.asType(.float32)
        guard chunk.ndim == 1 else {
            throw NemotronDiarizationError.nonMonoChunk
        }

        state.audioBuffer = MLX.concatenated([state.audioBuffer, chunk], axis: 0)
        state.samplesReceived += chunk.dim(0)

        let cfg = config.modulesConfig
        let proc = config.processorConfig
        let factor = config.encoderConfig.subsamplingFactor
        let central = cfg.chunkLen * factor
        let right = cfg.chunkRightContext * factor
        let offset = Float(state.framesProcessed * proc.hopLength) / Float(self.sampleRate)
        var outputs: [MLXArray] = []

        while true {
            let available = state.samplesReceived / proc.hopLength - state.framesProcessed
            let needed = (state.framesProcessed + central + right - 1) * proc.hopLength + proc.nFft / 2
            if available <= 0 || (!final && state.samplesReceived < needed) {
                break
            }
            let n = min(central, available)
            let windowFrames: Int
            if final {
                var totalFrames = state.samplesReceived / proc.hopLength + 1
                if proc.padTo > 0 {
                    totalFrames += (proc.padTo - (totalFrames % proc.padTo)) % proc.padTo
                }
                windowFrames = min(central + right, totalFrames - state.framesProcessed)
            } else {
                windowFrames = central + right
            }

            let features = preprocessor(
                state.audioBuffer,
                start: state.framesProcessed,
                count: windowFrames,
                sampleOffset: state.sampleOffset,
                totalSamples: state.samplesReceived
            )
            let chunkProbs = streamingStep(
                features: features,
                state: &state,
                centralFrames: n,
                featureLength: min(windowFrames, available)
            )
            outputs.append(chunkProbs)

            let keepFrom = max(0, state.framesProcessed * proc.hopLength - proc.nFft / 2 - 1)
            let trimCount = max(0, keepFrom - state.sampleOffset)
            state.audioBuffer = state.audioBuffer[trimCount...]
            state.sampleOffset = keepFrom
            eval(state.audioBuffer)
        }

        state.finished = final
        if final {
            state.audioBuffer = MLXArray.zeros([0], dtype: .float32)
        }
        let probs = outputs.isEmpty
            ? MLXArray.zeros([0, config.numSpeakers], dtype: .float32)
            : MLX.concatenated(outputs, axis: 0)

        let output = makeOutput(
            probs: probs,
            offset: offset,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap,
            state: state
        )
        return (output, state)
    }

    private func prepareWaveform(_ audio: MLXArray, sampleRate: Int) throws -> MLXArray {
        var waveform = audio.asType(.float32)
        if waveform.ndim > 1 {
            waveform = MLX.mean(waveform, axis: -1)
        }
        if sampleRate != self.sampleRate, waveform.dim(0) > 0 {
            waveform = try resampleAudio(waveform, from: sampleRate, to: self.sampleRate)
        }
        return waveform
    }

    public func generate(
        audio: MLXArray,
        sampleRate: Int = 16000,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0,
        verbose: Bool = false
    ) throws -> DiarizationOutput {
        let startTime = CFAbsoluteTimeGetCurrent()
        let waveform = try prepareWaveform(audio, sampleRate: sampleRate)
        var state = initStreamingState()

        let step =
            config.modulesConfig.chunkLen
            * config.encoderConfig.subsamplingFactor
            * config.processorConfig.hopLength

        var chunkProbs: [MLXArray] = []
        var idx = 0
        let totalSamples = waveform.dim(0)
        while idx < totalSamples {
            let end = min(idx + step, totalSamples)
            let (out, nextState) = try feed(
                audio: waveform[idx..<end],
                state: state,
                sampleRate: self.sampleRate,
                final: false,
                threshold: threshold,
                minDuration: minDuration,
                mergeGap: mergeGap
            )
            state = nextState
            if let sp = out.speakerProbs, sp.dim(0) > 0 {
                chunkProbs.append(sp)
            }
            idx = end
        }

        let (finalOut, finalState) = try feed(
            audio: MLXArray.zeros([0], dtype: .float32),
            state: state,
            sampleRate: self.sampleRate,
            final: true,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )
        state = finalState
        if let sp = finalOut.speakerProbs, sp.dim(0) > 0 {
            chunkProbs.append(sp)
        }

        let probs = chunkProbs.isEmpty
            ? MLXArray.zeros([0, config.numSpeakers], dtype: .float32)
            : MLX.concatenated(chunkProbs, axis: 0)

        let strideSec =
            Float(config.processorConfig.hopLength * config.outputSubsamplingFactor)
            / Float(self.sampleRate)
        let rawSegments = probs.dim(0) > 0
            ? SortformerModel.predsToSegments(
                probs,
                frameDuration: strideSec,
                threshold: threshold,
                minDuration: minDuration,
                mergeGap: mergeGap
            )
            : []

        let maxEnd = Float(state.framesProcessed * config.processorConfig.hopLength) / Float(self.sampleRate)
        let segments = rawSegments.map { seg in
            DiarizationSegment(
                start: seg.start,
                end: chunkProbs.isEmpty ? seg.end : min(seg.end, maxEnd),
                speaker: seg.speaker
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let result = DiarizationOutput(
            segments: segments,
            speakerProbs: probs,
            numSpeakers: Set(segments.map(\.speaker)).count,
            totalTime: elapsed
        )
        if verbose {
            print(result.text)
        }
        return result
    }

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (key, var value) in weights {
            // Conv1d: PyTorch (O, I, K) -> MLX (O, K, I) when unconverted
            if key.hasSuffix("subpixel_upsample.weight"),
               value.ndim == 3,
               value.dim(2) == 3,
               value.dim(1) != 3 {
                value = value.transposed(0, 2, 1)
            }
            sanitized[key] = value
        }
        return sanitized
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> NemotronDiarizationModel {
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(NemotronDiarizationConfig.self, from: configData)

        let model = NemotronDiarizationModel(config)
        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var allWeights: [String: MLXArray] = [:]
        for file in weightFiles {
            let shard = try loadArrays(url: file)
            for (k, v) in shard {
                allWeights[k] = v
            }
        }

        let sanitized = sanitize(allWeights)
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, _ in
                sanitized["\(path).scales"] != nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }

    public static func fromPretrained(
        _ repoId: String = defaultRepo,
        cache: HubCache = .default
    ) async throws -> NemotronDiarizationModel {
        let localURL = URL(fileURLWithPath: (repoId as NSString).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return try fromModelDirectory(localURL)
        }

        guard let repoID = Repo.ID(rawValue: repoId) else {
            throw NemotronDiarizationError.invalidRepositoryID(repoId)
        }
        let modelURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )
        return try fromModelDirectory(modelURL)
    }
}
