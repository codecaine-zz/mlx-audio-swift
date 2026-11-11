import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore

@inline(__always)
private func pocketCreateAdditiveCausalMask(_ n: Int, offset: Int = 0) -> MLXArray {
    let total = offset + n
    let rinds = MLXArray(stride(from: 0, to: total, by: 1))
    let linds = offset > 0 ? MLXArray(stride(from: offset, to: total, by: 1)) : rinds
    let li = linds.reshaped([n, 1])
    let ri = rinds.reshaped([1, total])
    let mask = li .< ri
    return mask.asType(.float32) * MLXArray(-1e9)
}

public final class PocketLayerScale: Module {
    public var scale: MLXArray

    public init(channels: Int, initValue: Float) {
        self.scale = MLXArray.ones([channels]) * MLXArray(initValue)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * scale }
}

public final class PocketRotaryEmbedding: Module {
    public let maxPeriod: Double
    private var cachedFreqsByDim: [Int: MLXArray] = [:]

    public init(maxPeriod: Double = 10000.0) {
        self.maxPeriod = maxPeriod
        super.init()
    }

    private func freqs4D(headDim d: Int) -> MLXArray {
        if let cached = cachedFreqsByDim[d] {
            return cached
        }
        let half = d / 2
        let indices = MLXArray(stride(from: 0, to: half, by: 1)).asType(.float32)
        let freqScale = -Float(log(maxPeriod)) * 2.0 / Float(d)
        let f = MLX.exp(indices * MLXArray(freqScale)).reshaped([1, 1, 1, half])
        eval(f)
        cachedFreqsByDim[d] = f
        return f
    }

    public func callAsFunction(_ q: MLXArray, _ k: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let qSwapped = swappedAxes(q, 1, 2)
        let kSwapped = swappedAxes(k, 1, 2)
        let d = q.shape[3]
        let qR = MLXFast.RoPE(qSwapped, dimensions: d, traditional: true, base: Float(maxPeriod), scale: 1.0, offset: offset)
        let kR = MLXFast.RoPE(kSwapped, dimensions: d, traditional: true, base: Float(maxPeriod), scale: 1.0, offset: offset)
        return (swappedAxes(qR, 1, 2), swappedAxes(kR, 1, 2))
    }

    /// Applies traditional RoPE to swapped `[B, H, T, D]` tensors with per-batch-item `positions` of shape `[B, 1, T, 1]`.
    public func applySwapped(_ q: MLXArray, _ k: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
        let b = q.shape[0]
        let h = q.shape[1]
        let t = q.shape[2]
        let d = q.shape[3]
        let half = d / 2
        let freq4d = freqs4D(headDim: d)
        let angles = positions * freq4d
        let rotr = MLX.cos(angles)
        let roti = MLX.sin(angles)

        let q2 = q.reshaped([b, h, t, half, 2])
        let k2 = k.reshaped([b, h, t, half, 2])
        let qr = q2[.ellipsis, 0].asType(.float32)
        let qi = q2[.ellipsis, 1].asType(.float32)
        let kr = k2[.ellipsis, 0].asType(.float32)
        let ki = k2[.ellipsis, 1].asType(.float32)

        let qor = (qr * rotr - qi * roti).asType(q.dtype)
        let qoi = (qr * roti + qi * rotr).asType(q.dtype)
        let kor = (kr * rotr - ki * roti).asType(k.dtype)
        let koi = (kr * roti + ki * rotr).asType(k.dtype)

        let qOut = stacked([qor, qoi], axis: -1).reshaped([b, h, t, d])
        let kOut = stacked([kor, koi], axis: -1).reshaped([b, h, t, d])
        return (qOut, kOut)
    }
}

public final class PocketStreamingMultiheadAttention: Module {
    public let embedDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rope: PocketRotaryEmbedding

    @ModuleInfo(key: "in_proj") public var in_proj: Linear
    @ModuleInfo(key: "out_proj") public var out_proj: Linear

    public init(embedDim: Int, numHeads: Int, rope: PocketRotaryEmbedding) {
        precondition(embedDim % numHeads == 0, "embed_dim must be divisible by num_heads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = rope
        self._in_proj = ModuleInfo(wrappedValue: Linear(embedDim, 3 * embedDim, bias: false))
        self._out_proj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim, bias: false))
        super.init()
    }

    public func callAsFunction(
        _ query: MLXArray,
        cache: KVCacheSimple?,
        positions: MLXArray? = nil,
        ropeOffsetVec: MLXArray? = nil,
        maskOverride: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        let b = query.shape[0]
        let t = query.shape[1]

        let projected = in_proj(query).reshaped([b, t, 3, numHeads, headDim])

        var q: MLXArray
        var k: MLXArray
        let v: MLXArray
        if t == 1 {
            q = projected[0..<b, 0..<1, 0, 0..<numHeads, 0..<headDim].reshaped([b, numHeads, 1, headDim])
            k = projected[0..<b, 0..<1, 1, 0..<numHeads, 0..<headDim].reshaped([b, numHeads, 1, headDim])
            v = projected[0..<b, 0..<1, 2, 0..<numHeads, 0..<headDim].reshaped([b, numHeads, 1, headDim])
        } else {
            q = swappedAxes(projected[0..<b, 0..<t, 0, 0..<numHeads, 0..<headDim], 1, 2)
            k = swappedAxes(projected[0..<b, 0..<t, 1, 0..<numHeads, 0..<headDim], 1, 2)
            v = swappedAxes(projected[0..<b, 0..<t, 2, 0..<numHeads, 0..<headDim], 1, 2)
        }

        let offset = cache?.offset ?? 0
        if let ropeOffsetVec {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: true, base: Float(rope.maxPeriod), scale: 1.0, offset: ropeOffsetVec)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: true, base: Float(rope.maxPeriod), scale: 1.0, offset: ropeOffsetVec)
        } else if let positions {
            (q, k) = rope.applySwapped(q, k, positions: positions)
        } else {
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: true, base: Float(rope.maxPeriod), scale: 1.0, offset: offset)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: true, base: Float(rope.maxPeriod), scale: 1.0, offset: offset)
        }

        let kFull: MLXArray
        let vFull: MLXArray
        if let cache {
            (kFull, vFull) = cache.update(keys: k, values: v)
        } else {
            kFull = k
            vFull = v
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let maskOverride {
            maskMode = maskOverride
        } else if t == 1 {
            maskMode = .none
        } else {
            maskMode = .causal
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kFull,
            values: vFull,
            scale: scale,
            mask: maskMode
        )
        let merged = swappedAxes(out, 1, 2).reshaped([b, t, embedDim])
        return out_proj(merged)
    }
}

public final class PocketStreamingTransformerLayer: Module {
    private let _useExactGELU: Bool
    @ModuleInfo(key: "self_attn") public var self_attn: PocketStreamingMultiheadAttention
    @ModuleInfo(key: "norm1") public var norm1: PocketLayerNorm
    @ModuleInfo(key: "norm2") public var norm2: PocketLayerNorm
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear
    @ModuleInfo(key: "layer_scale_1") public var layer_scale_1: PocketLayerScale?
    @ModuleInfo(key: "layer_scale_2") public var layer_scale_2: PocketLayerScale?

    public init(dModel: Int, numHeads: Int, dimFeedforward: Int, rope: PocketRotaryEmbedding, layerScale: Float?, gelu: String = "erf") {
        precondition(["erf", "tanh"].contains(gelu), "Unsupported GELU activation: \(gelu)")
        self._useExactGELU = gelu == "erf"
        self._self_attn = ModuleInfo(wrappedValue: PocketStreamingMultiheadAttention(embedDim: dModel, numHeads: numHeads, rope: rope))
        self._norm1 = ModuleInfo(wrappedValue: PocketLayerNorm(channels: dModel, eps: 1e-5))
        self._norm2 = ModuleInfo(wrappedValue: PocketLayerNorm(channels: dModel, eps: 1e-5))
        self._linear1 = ModuleInfo(wrappedValue: Linear(dModel, dimFeedforward, bias: false))
        self._linear2 = ModuleInfo(wrappedValue: Linear(dimFeedforward, dModel, bias: false))
        if let layerScale {
            self._layer_scale_1 = ModuleInfo(wrappedValue: PocketLayerScale(channels: dModel, initValue: layerScale))
            self._layer_scale_2 = ModuleInfo(wrappedValue: PocketLayerScale(channels: dModel, initValue: layerScale))
        } else {
            self._layer_scale_1 = ModuleInfo(wrappedValue: nil)
            self._layer_scale_2 = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    private func applyScale(_ x: MLXArray, _ scale: PocketLayerScale?) -> MLXArray {
        guard let scale else { return x }
        return scale(x)
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: KVCacheSimple?,
        positions: MLXArray? = nil,
        ropeOffsetVec: MLXArray? = nil,
        maskOverride: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        let attnOut = self_attn(norm1(x), cache: cache, positions: positions, ropeOffsetVec: ropeOffsetVec, maskOverride: maskOverride)
        var h = x + applyScale(attnOut, layer_scale_1)
        let hidden = linear1(norm2(h))
        let ff = linear2(_useExactGELU ? gelu(hidden) : geluApproximate(hidden))
        h = h + applyScale(ff, layer_scale_2)
        return h
    }
}

public final class PocketStreamingTransformer: Module {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let headDim: Int
    public let rope: PocketRotaryEmbedding

    @ModuleInfo(key: "layers") public var layers: [PocketStreamingTransformerLayer]

    public init(dModel: Int, numHeads: Int, numLayers: Int, dimFeedforward: Int, maxPeriod: Double, layerScale: Float? = nil, gelu: String = "erf") {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.headDim = dModel / numHeads
        let rope = PocketRotaryEmbedding(maxPeriod: maxPeriod)
        self.rope = rope
        self._layers = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in
            PocketStreamingTransformerLayer(
                dModel: dModel,
                numHeads: numHeads,
                dimFeedforward: dimFeedforward,
                rope: rope,
                layerScale: layerScale,
                gelu: gelu
            )
        })
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: [KVCacheSimple]?,
        positions: MLXArray? = nil,
        ropeOffsetVec: MLXArray? = nil,
        maskOverride: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        var h = x
        let caches = cache ?? (0..<layers.count).map { _ in KVCacheSimple() }
        for (layer, layerCache) in zip(layers, caches) {
            h = layer(h, cache: layerCache, positions: positions, ropeOffsetVec: ropeOffsetVec, maskOverride: maskOverride)
        }
        return h
    }

    public func makeCache(step: Int = 256) -> [KVCacheSimple] {
        return (0..<layers.count).map { _ in
            let c = KVCacheSimple()
            c.step = max(64, step)
            return c
        }
    }
}
