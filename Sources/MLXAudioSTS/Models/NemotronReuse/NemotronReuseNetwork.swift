// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
// Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.

import Foundation
import MLX
import MLXNN

final class NemotronReuseInstanceNorm: Module, UnaryLayer {
    let eps: Float
    var weight: MLXArray
    var bias: MLXArray

    init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions], dtype: .float32)
        self.bias = MLXArray.zeros([dimensions], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let axes = Array(1..<(x.ndim - 1))
        let mean = MLX.mean(x, axes: axes, keepDims: true)
        let variance = MLX.variance(x, axes: axes, keepDims: true)
        let normalized = (x - mean) * MLX.rsqrt(variance + eps)
        return weight * normalized + bias
    }
}

final class NemotronReusePReLU: Module, UnaryLayer {
    var weight: MLXArray

    init(count: Int, initialValue: Float = 0.25) {
        self.weight = MLXArray.full([count], values: MLXArray(initialValue), type: Float.self)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.maximum(0, x) + weight * MLX.minimum(0, x)
    }
}

private func makeNormalizedStage(_ conv: UnaryLayer, dim: Int) -> [UnaryLayer] {
    [conv, NemotronReuseInstanceNorm(dimensions: dim), NemotronReusePReLU(count: dim)]
}

private func applyStage(_ layers: [UnaryLayer], _ input: MLXArray) -> MLXArray {
    var x = input
    for layer in layers {
        x = layer(x)
    }
    return x
}

final class NemotronReuseDenseBlock: Module, UnaryLayer {
    @ModuleInfo(key: "dense_block") var denseBlock: [[UnaryLayer]]

    init(dim: Int) {
        self._denseBlock.wrappedValue = (0..<4).map { i in
            let dilation = 1 << i
            let conv = Conv2d(
                inputChannels: dim * (i + 1),
                outputChannels: dim,
                kernelSize: .init((3, 3)),
                padding: .init((dilation, 1)),
                dilation: .init((dilation, 1))
            )
            return makeNormalizedStage(conv, dim: dim)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var skip = input
        var x = input
        for layers in denseBlock {
            x = applyStage(layers, skip)
            skip = MLX.concatenated([x, skip], axis: -1)
        }
        return x
    }
}

final class NemotronReuseDenseEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "dense_conv_1") var denseConv1: [UnaryLayer]
    @ModuleInfo(key: "dense_block") var denseBlock: NemotronReuseDenseBlock
    @ModuleInfo(key: "dense_conv_2") var denseConv2: [UnaryLayer]

    init(_ config: NemotronReuseConfig) {
        let dim = config.hidFeature
        self._denseConv1.wrappedValue = makeNormalizedStage(
            Conv2d(inputChannels: config.inputChannel, outputChannels: dim, kernelSize: 1),
            dim: dim
        )
        self._denseBlock.wrappedValue = NemotronReuseDenseBlock(dim: dim)
        self._denseConv2.wrappedValue = makeNormalizedStage(
            Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: .init((1, 3)),
                stride: .init((4, 2))
            ),
            dim: dim
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = denseBlock(applyStage(denseConv1, x))
        return applyStage(denseConv2, hidden)
    }
}

final class NemotronReuseSubpixelConv2d: Module, UnaryLayer {
    let factor: Int
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(dim: Int, factor: Int) {
        self.factor = factor
        self._conv.wrappedValue = Conv2d(
            inputChannels: dim,
            outputChannels: dim * factor,
            kernelSize: .init((1, 3)),
            padding: .init((0, 1))
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = conv(input)
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Upstream channel order is [factor, channels], not [channels, factor].
        let reshaped = x.reshaped(batch, height, width, factor, channels / factor)
        return reshaped.reshaped(batch, height, width * factor, channels / factor)
    }
}

final class NemotronReuseDecoder: Module, UnaryLayer {
    let isPhase: Bool
    @ModuleInfo(key: "dense_block") var denseBlock: NemotronReuseDenseBlock
    @ModuleInfo(key: "up_conv1") var upConv1: [UnaryLayer]
    @ModuleInfo(key: "up_conv2") var upConv2: [UnaryLayer]
    @ModuleInfo(key: "phase_conv_r") var phaseConvR: Conv2d?
    @ModuleInfo(key: "phase_conv_i") var phaseConvI: Conv2d?
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d?

    init(_ config: NemotronReuseConfig, phase: Bool = false) {
        let dim = config.hidFeature
        self.isPhase = phase
        self._denseBlock.wrappedValue = NemotronReuseDenseBlock(dim: dim)
        self._upConv1.wrappedValue = makeNormalizedStage(
            NemotronReuseSubpixelConv2d(dim: dim, factor: 2),
            dim: dim
        )
        self._upConv2.wrappedValue = makeNormalizedStage(
            NemotronReuseSubpixelConv2d(dim: dim, factor: 4),
            dim: dim
        )
        if phase {
            self._phaseConvR.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: config.outputChannel,
                kernelSize: 1
            )
            self._phaseConvI.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: config.outputChannel,
                kernelSize: 1
            )
            self._finalConv.wrappedValue = nil
        } else {
            self._phaseConvR.wrappedValue = nil
            self._phaseConvI.wrappedValue = nil
            self._finalConv.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: config.outputChannel,
                kernelSize: 1
            )
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = applyStage(upConv1, denseBlock(input))
        x = applyStage(upConv2, x.transposed(0, 2, 1, 3)).transposed(0, 2, 1, 3)
        if isPhase, let phaseConvI, let phaseConvR {
            return MLX.atan2(phaseConvI(x), phaseConvR(x))
        }
        return finalConv!(x)
    }
}
