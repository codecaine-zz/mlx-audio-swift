import Foundation
import MLX
import MLXNN

// MARK: - MimiConv1d (NCL adapter over MLXNN.Conv1d)

public final class MimiConv1d: MLXNN.Conv1d {
    private let inChannels: Int
    private let outChannels: Int
    private let ksize: Int
    private var cachedMatmulWeightID: ObjectIdentifier?
    private var cachedMatmulWeightT: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.ksize = ksize
        super.init(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: ksize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    // NCL -> NLC -> conv1d -> NCL
    public override func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        let y = callNLC(xsNLC)
        return swappedAxes(y, 1, 2)
    }

    // Zero-transpose NLC fast path: [B, T, C_in] -> [B, T_out, C_out]
    public func callNLC(_ xsNLC: MLXArray) -> MLXArray {
        if ksize == 1 && stride == 1 && padding == 0 && groups == 1 {
            let wID = ObjectIdentifier(weight)
            let wT: MLXArray
            if cachedMatmulWeightID == wID, let cached = cachedMatmulWeightT {
                wT = cached
            } else {
                let transposed = swappedAxes(weight.reshaped([outChannels, inChannels]), 0, 1)
                cachedMatmulWeightID = wID
                cachedMatmulWeightT = transposed
                wT = transposed
            }
            var y = matmul(xsNLC, wT)
            if let bias { y = y + bias }
            return y
        }
        return super.callAsFunction(xsNLC)
    }
}

// MARK: - MimiConvTransposed1d (NCL adapter over MLXNN.ConvTransposed1d)

public final class MimiConvTransposed1d: MLXNN.ConvTransposed1d {

    private let ksize: Int
    private let inChannels: Int
    private let outChannels: Int
    private var cachedWeightID: ObjectIdentifier?
    private var cachedExpandedWeight: MLXArray?
    private var cachedExpandedGroups: Int = 1

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.ksize = ksize
        self.inChannels = inChannels
        self.outChannels = outChannels
        super.init(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: ksize,
            stride: stride,
            padding: padding,
            groups: groups,
            bias: bias
        )
    }

    // Expand weight as needed to emulate grouped depthwise transposed conv like the Python version
    private func expandedWeightAndGroups() -> (MLXArray, Int) {
        if groups == inChannels && groups == outChannels {
            let wID = ObjectIdentifier(weight)
            if cachedWeightID == wID, let cached = cachedExpandedWeight {
                return (cached, cachedExpandedGroups)
            }
            var w = weight
            if w.shape.count == 3
                && w.shape[0] == outChannels
                && w.shape[1] == ksize
                && w.shape[2] == 1 {
                // Legacy depthwise layout: (C, k, 1) -> (1, k, C)
                w = swappedAxes(w, 0, 2)
            }
            var eyeW = eye(outChannels)
                .asType(weight.dtype)
                .reshaped([outChannels, 1, outChannels])
            eyeW = repeated(eyeW, count: ksize, axis: 1) // repeat along kernel dim
            let wRep = repeated(w, count: groups, axis: 0)
            let expanded = wRep * eyeW
            eval(expanded)
            cachedWeightID = wID
            cachedExpandedWeight = expanded
            cachedExpandedGroups = 1
            return (expanded, 1)
        } else if groups > 1 {
            fatalError("groups > 1 (non-depthwise) not supported in MimiConvTransposed1d")
        } else {
            return (weight, groups)
        }
    }

    public func callNLC(_ xsNLC: MLXArray) -> MLXArray {
        let (wEff, gEff) = expandedWeightAndGroups()
        var y = convTransposed1d(
            xsNLC,
            wEff,
            stride: stride,
            padding: padding,
            dilation: dilation,
            outputPadding: outputPadding,
            groups: gEff
        )
        if let b = bias { y = y + b }
        return y
    }

    public override func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        let y = callNLC(xsNLC)
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - Normalized wrappers (kept as simple pass-through like Python)

public final class NormConv1d: Module {
    @ModuleInfo public var conv: MimiConv1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, dilation: Int = 1, bias: Bool = true
    ) {
        self._conv = ModuleInfo(wrappedValue: MimiConv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, dilation: dilation, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
    public func callNLC(_ xsNLC: MLXArray) -> MLXArray { conv.callNLC(xsNLC) }
}

public final class NormConvTranspose1d: Module {
    @ModuleInfo public var convtr: MimiConvTransposed1d

    public init(
        inChannels: Int, outChannels: Int, ksize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, bias: Bool = true
    ) {
        self._convtr = ModuleInfo(wrappedValue: MimiConvTransposed1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: padding, groups: groups, bias: bias
        ))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray { convtr(xs) }
    public func callNLC(_ xsNLC: MLXArray) -> MLXArray { convtr.callNLC(xsNLC) }
}

// MARK: - Helpers

@inline(__always)
func getExtraPaddingForConv1dLen(_ len: Int, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let nframes = max(len + paddingTotal - ksize, 0)
    let nf = Double(nframes) / Double(stride) + 1.0
    let idealLen = (Int(ceil(nf)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLen - len)
}

@inline(__always)
func getExtraPaddingForConv1d(xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    getExtraPaddingForConv1dLen(xs.shape[2], ksize: ksize, stride: stride, paddingTotal: paddingTotal)
}

// Unpad along last axis using split (avoids relying on slicing syntax)
@inline(__always)
func unpad1d(_ xs: MLXArray, unpadL: Int, unpadR: Int) -> MLXArray {
    let L = xs.shape[2]
    let parts = split(xs, indices: [unpadL, L - unpadR], axis: 2)
    return parts[1] // middle segment
}

// MARK: - StreamableConv1d

public final class StreamableConv1d: Module {
    private let causal: Bool
    private let padMode: PadMode
    private let ksizeBase: Int
    @ModuleInfo public var conv: NormConv1d

    private var prevXs: MLXArray? = nil
    private var leftPadApplied = false
    private let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool,
        causal: Bool,
        padMode: PadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self.ksizeBase = ksize
        self._conv = ModuleInfo(wrappedValue: NormConv1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, dilation: dilation, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() {
        prevXs = nil
        leftPadApplied = false
    }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        // Effective kernel size with dilation
        let dil = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dil + 1
        let paddingTotal = kEff - conv.conv.stride
        let extra = getExtraPaddingForConv1d(
            xs: xsNCL, ksize: kEff, stride: conv.conv.stride, paddingTotal: paddingTotal
        )
        let z = IntOrPair(0)
        let pad: (Int, Int) = {
            if causal { return (paddingTotal, 0) }
            let pr = paddingTotal / 2
            return (paddingTotal - pr, pr)
        }()
        let (padL, padR) = pad
        let totalRight = padR + extra
        if padL == 0 && totalRight == 0 {
            return conv(xsNCL)
        }
        let widths: [IntOrPair] = [z, z, IntOrPair((padL, totalRight))]
        let xPad = padded(xsNCL, widths: widths, mode: padMode)
        return conv(xPad)
    }

    public func callNLC(_ xsNLC: MLXArray) -> MLXArray {
        let dil = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dil + 1
        let paddingTotal = kEff - conv.conv.stride
        let extra = getExtraPaddingForConv1dLen(
            xsNLC.shape[1], ksize: kEff, stride: conv.conv.stride, paddingTotal: paddingTotal
        )
        let (padL, padR): (Int, Int) = {
            if causal { return (paddingTotal, 0) }
            let pr = paddingTotal / 2
            return (paddingTotal - pr, pr)
        }()
        let totalRight = padR + extra
        if padL == 0 && totalRight == 0 {
            return conv.callNLC(xsNLC)
        }
        let z = IntOrPair(0)
        let widths: [IntOrPair] = [z, IntOrPair((padL, totalRight)), z]
        let xPad = padded(xsNLC, widths: widths, mode: padMode)
        return conv.callNLC(xPad)
    }

    // Streaming step; input/output are NCL
    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = conv.conv.stride
        let dilation = conv.conv.dilation
        let kEff = (ksizeBase - 1) * dilation + 1

        var x = xsNCL
        if !leftPadApplied {
            leftPadApplied = true
            let padTotal = kEff - stride
            x = padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((padTotal, 0))], mode: padMode)
        }

        if let prev = prevXs {
            x = concatenated([prev, x], axis: 2)
        }

        let L = x.shape[2]
        let nframes = max(L + stride - kEff, 0) / stride
        if nframes > 0 {
            let offset = nframes * stride
            // stash tail for next call: x[..., offset:]
            let tailSplit = split(x, indices: [offset], axis: 2)
            prevXs = tailSplit.count > 1 ? tailSplit[1] : nil
            
            let inLen = (nframes - 1) * stride + kEff
            let keep = split(x, indices: [inLen], axis: 2)[0]
            return conv(keep)
        } else {
            prevXs = x
            return MLXArray.zeros([b, outChannels, 0])
        }
    }
}

// MARK: - StreamableConvTranspose1d

public final class StreamableConvTranspose1d: Module {
    private let causal: Bool
    private let ksize: Int
    private let inChannels: Int
    @ModuleInfo public var convtr: NormConvTranspose1d

    private var prevYs: MLXArray? = nil
    private let outChannels: Int
    private var cachedUpsampleWeightID: ObjectIdentifier? = nil
    private var cachedW0: MLXArray? = nil
    private var cachedW1: MLXArray? = nil

    public init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        groups: Int,
        bias: Bool,
        causal: Bool
    ) {
        self.causal = causal
        self.ksize = ksize
        self.inChannels = inChannels
        self._convtr = ModuleInfo(wrappedValue: NormConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels, ksize: ksize,
            stride: stride, padding: 0, groups: groups, bias: bias
        ))
        self.outChannels = outChannels
    }

    public func resetState() { prevYs = nil }

    public func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let stride = convtr.convtr.stride
        let paddingTotal = max(ksize - stride, 0)
        let y = convtr(xsNCL)
        let (unL, unR): (Int, Int) = {
            if causal { return (0, paddingTotal) }
            let r = paddingTotal / 2
            return (paddingTotal - r, r)
        }()
        return unpad1d(y, unpadL: unL, unpadR: unR)
    }

    public func callNLC(_ xsNLC: MLXArray) -> MLXArray {
        let s = convtr.convtr.stride
        if causal
            && s > 1
            && ksize == 2 * s
            && convtr.convtr.groups == 1
            && convtr.convtr.padding == 0
            && convtr.convtr.dilation == 1
            && convtr.convtr.outputPadding == 0
        {
            let b = xsNLC.shape[0]
            let t = xsNLC.shape[1]
            if t == 0 {
                return MLXArray.zeros([b, 0, outChannels], dtype: xsNLC.dtype)
            }
            let w = convtr.convtr.weight
            let wID = ObjectIdentifier(w)
            let w0: MLXArray
            let w1: MLXArray
            if cachedUpsampleWeightID == wID, let c0 = cachedW0, let c1 = cachedW1 {
                w0 = c0
                w1 = c1
            } else {
                let eyeIn = eye(inChannels).asType(w.dtype).reshaped([inChannels, 1, inChannels])
                let imp = convTransposed1d(
                    eyeIn, w, stride: s, padding: 0, dilation: 1, outputPadding: 0, groups: 1
                ) // [inChannels, 2*s, outChannels]
                let newW0 = imp[0..., 0..<s, 0...].reshaped([inChannels, s * outChannels])
                let newW1 = imp[0..., s..<(2 * s), 0...].reshaped([inChannels, s * outChannels])
                eval(newW0, newW1)
                cachedUpsampleWeightID = wID
                cachedW0 = newW0
                cachedW1 = newW1
                w0 = newW0
                w1 = newW1
            }
            let y0 = matmul(xsNLC, w0)
            var out: MLXArray
            if t == 1 {
                out = y0.reshaped([b, s, outChannels])
            } else {
                let y1Part = matmul(xsNLC[0..., 0..<(t - 1), 0...], w1)
                let y1 = padded(
                    y1Part,
                    widths: [IntOrPair(0), IntOrPair((1, 0)), IntOrPair(0)]
                )
                out = (y0 + y1).reshaped([b, t * s, outChannels])
            }
            if let bias = convtr.convtr.bias {
                out = out + bias
            }
            return out
        }

        let paddingTotal = max(ksize - s, 0)
        let y = convtr.callNLC(xsNLC)
        if paddingTotal == 0 { return y }
        let (unL, unR): (Int, Int) = {
            if causal { return (0, paddingTotal) }
            let r = paddingTotal / 2
            return (paddingTotal - r, r)
        }()
        let L = y.shape[1]
        return y[0..., unL ..< max(unL, L - unR), 0...]
    }

    public func step(_ xsNCL: MLXArray) -> MLXArray {
        let b = xsNCL.shape[0]
        let len = xsNCL.shape[2]
        if len == 0 { return MLXArray.zeros([b, outChannels, 0]) }

        let stride = convtr.convtr.stride
        var y = convtr(xsNCL)
        let ot = y.shape[2]

        if var prev = prevYs {
            let pt = prev.shape[2]
            if let b = convtr.convtr.bias { prev = prev - b.reshaped([1, b.shape[0], 1]) }
            // overlap-add
            let head = split(y, indices: [pt], axis: 2)
            let combined = head[0] + prev
            y = concatenated([combined, head[1]], axis: 2)
        }

        let invalid = ksize - stride
        let parts = split(y, indices: [max(ot - invalid, 0)], axis: 2)
        let valid = parts[0]
        prevYs = parts.count > 1 ? parts[1] : nil
        return valid
    }
}

// MARK: - Upsample/Downsample wrappers

public final class ConvDownsample1d: Module {
    @ModuleInfo public var conv: StreamableConv1d

    public init(stride: Int, dim: Int, causal: Bool) {
        self._conv = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, dilation: 1, groups: 1, bias: false,
            causal: causal, padMode: .edge
        ))
    }

    public func resetState() { conv.resetState() }
    public func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
    public func step(_ xs: MLXArray) -> MLXArray { conv.step(xs) }
}

public final class ConvTrUpsample1d: Module {
    @ModuleInfo public var convtr: StreamableConvTranspose1d
    private let strideVal: Int
    private let dimVal: Int
    private let isCausal: Bool
    private var cachedWeightID: ObjectIdentifier?
    private var cachedW0: MLXArray?
    private var cachedW1: MLXArray?

    public init(stride: Int, dim: Int, causal: Bool) {
        self.strideVal = stride
        self.dimVal = dim
        self.isCausal = causal
        self._convtr = ModuleInfo(wrappedValue: StreamableConvTranspose1d(
            inChannels: dim, outChannels: dim, ksize: 2*stride,
            stride: stride, groups: dim, bias: false, causal: causal
        ))
    }

    public func resetState() { convtr.resetState() }

    private func resolveDepthwiseKernels() -> (MLXArray, MLXArray) {
        let w = convtr.convtr.convtr.weight
        let wID = ObjectIdentifier(w)
        if cachedWeightID == wID, let w0 = cachedW0, let w1 = cachedW1 {
            return (w0, w1)
        }
        let impNCL = convtr.convtr(MLXArray.ones([1, dimVal, 1], dtype: w.dtype)) // [1, dim, 2*stride]
        let impNLC = swappedAxes(impNCL, 1, 2) // [1, 2*stride, dim]
        let w0 = impNLC[0..., 0..<strideVal, 0...].reshaped([1, 1, strideVal, dimVal])
        let w1 = impNLC[0..., strideVal..<(2 * strideVal), 0...].reshaped([1, 1, strideVal, dimVal])
        eval(w0, w1)
        cachedWeightID = wID
        cachedW0 = w0
        cachedW1 = w1
        return (w0, w1)
    }

    /// True O(B * T * stride * dim) depthwise causal upsampling directly to NLC `[B, T * stride, dim]` (512x fewer FLOPs than dense `eye(512)` convTransposed1d).
    public func callToNLC(_ xsNCL: MLXArray) -> MLXArray {
        guard isCausal && strideVal > 0 else {
            return swappedAxes(convtr(xsNCL), 1, 2)
        }
        let b = xsNCL.shape[0]
        let t = xsNCL.shape[2]
        if t == 0 {
            return MLXArray.zeros([b, 0, dimVal], dtype: xsNCL.dtype)
        }
        let (w0, w1) = resolveDepthwiseKernels()
        let x = swappedAxes(xsNCL, 1, 2).expandedDimensions(axis: 2) // [B, T, 1, C]
        if t == 1 {
            return (x * w0).reshaped([b, strideVal, dimVal])
        }
        let z = IntOrPair(0)
        let prevX = padded(x[0..., 0..<(t - 1), 0..., 0...], widths: [z, IntOrPair((1, 0)), z, z])
        return (x * w0 + prevX * w1).reshaped([b, t * strideVal, dimVal])
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        if isCausal && strideVal > 0 {
            return swappedAxes(callToNLC(xs), 1, 2)
        }
        return convtr(xs)
    }

    public func step(_ xs: MLXArray) -> MLXArray { convtr.step(xs) }
}
