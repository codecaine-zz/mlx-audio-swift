// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
// Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.

import Foundation
import MLX
import MLXFast
import MLXNN

enum NemotronReuseSelectiveScan {
    private static let kernel = MLXFast.metalKernel(
        name: "reuse_selective_scan",
        inputNames: ["x", "dt", "A", "B", "C", "D"],
        outputNames: ["y"],
        source: """
            uint channel = thread_position_in_grid.x;
            uint batch = thread_position_in_grid.y;
            if (channel >= H) return;
            float state[N];
            float a[N];
            for (int s = 0; s < N; ++s) {
                state[s] = 0.0f;
                a[s] = A[channel * N + s];
            }
            for (int t = 0; t < L; ++t) {
                uint index = (batch * L + t) * H + channel;
                uint bc = (batch * L + t) * N;
                float delta = dt[index];
                float u = x[index];
                float value = D[channel] * u;
                for (int s = 0; s < N; ++s) {
                    state[s] = metal::exp(delta * a[s]) * state[s]
                        + delta * B[bc + s] * u;
                    value += state[s] * C[bc + s];
                }
                y[index] = value;
            }
            """
    )

    static func scan(
        _ x: MLXArray,
        dt: MLXArray,
        a: MLXArray,
        b: MLXArray,
        c: MLXArray,
        d: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let channels = x.dim(2)
        let stateSize = a.dim(-1)

        if Device.defaultDevice() == .gpu {
            let outputs = kernel(
                [
                    x.asType(.float32),
                    dt.asType(.float32),
                    a.asType(.float32),
                    b.asType(.float32),
                    c.asType(.float32),
                    d.asType(.float32),
                ],
                template: [("H", channels), ("N", stateSize), ("L", length)],
                grid: (channels, batch, 1),
                threadGroup: (min(256, channels), 1, 1),
                outputShapes: [x.shape],
                outputDTypes: [.float32]
            )
            return outputs[0].asType(x.dtype)
        }

        var state = MLXArray.zeros([batch, channels, stateSize], dtype: .float32)
        let aFloat = a.asType(.float32)
        let bFloat = b.asType(.float32)
        let cFloat = c.asType(.float32)
        let dFloat = d.asType(.float32)
        var output: [MLXArray] = []
        output.reserveCapacity(length)

        for t in 0..<length {
            let delta = dt[0..., t, 0...].asType(.float32).expandedDimensions(axis: -1)
            let u = x[0..., t, 0...].asType(.float32).expandedDimensions(axis: -1)
            let bt = bFloat[0..., t, 0...].expandedDimensions(axis: 1)
            let ct = cFloat[0..., t, 0...].expandedDimensions(axis: 1)
            state = MLX.exp(delta * aFloat) * state + delta * bt * u
            let y = MLX.sum(state * ct, axis: -1) + dFloat * x[0..., t, 0...].asType(.float32)
            output.append(y)
        }

        return MLX.stacked(output, axis: 1).asType(x.dtype)
    }
}

final class NemotronReuseMamba: Module, UnaryLayer {
    let dConv: Int
    let dState: Int
    let dtRank: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: NemotronReuseConfig) {
        let dim = config.hidFeature
        let inner = dim * config.expand
        dConv = config.dConv
        dState = config.dState
        dtRank = Int(ceil(Double(dim) / 16.0))

        self._inProj.wrappedValue = Linear(dim, 2 * inner, bias: false)
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: inner,
            outputChannels: inner,
            kernelSize: config.dConv,
            groups: inner
        )
        self._xProj.wrappedValue = Linear(inner, dtRank + 2 * dState, bias: false)
        self._dtProj.wrappedValue = Linear(dtRank, inner, bias: true)

        let baseA = MLX.log(MLXArray(1...(config.dState)).asType(.float32))
        self._aLog.wrappedValue = MLX.broadcast(baseA, to: [inner, config.dState])
        self._d.wrappedValue = MLXArray.ones([inner], dtype: .float32)
        self._outProj.wrappedValue = Linear(inner, dim, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let parts = MLX.split(inProj(input), parts: 2, axis: -1)
        var x = parts[0]
        let z = parts[1]

        let padded = MLX.padded(
            x,
            widths: [.init((0, 0)), .init((dConv - 1, 0)), .init((0, 0))]
        )
        x = silu(conv1d(padded))

        let proj = xProj(x)
        let dtRaw = proj[0..., 0..., 0..<dtRank]
        let b = proj[0..., 0..., dtRank..<(dtRank + dState)]
        let c = proj[0..., 0..., (dtRank + dState)...]

        let dt = softplus(dtProj(dtRaw).asType(.float32))
        let a = -MLX.exp(aLog.asType(.float32))
        let y = NemotronReuseSelectiveScan.scan(x, dt: dt, a: a, b: b, c: c, d: d)
        return outProj(y * silu(z))
    }
}

final class NemotronReuseBidirectionalMamba: Module, UnaryLayer {
    @ModuleInfo(key: "forward_blocks") var forwardBlocks: NemotronReuseMamba
    @ModuleInfo(key: "backward_blocks") var backwardBlocks: NemotronReuseMamba
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: NemotronReuseConfig) {
        let dim = config.hidFeature
        self._forwardBlocks.wrappedValue = NemotronReuseMamba(config)
        self._backwardBlocks.wrappedValue = NemotronReuseMamba(config)
        self._outputProj.wrappedValue = Linear(2 * dim, dim, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dim)
    }

    private static func reverseTime(_ x: MLXArray) -> MLXArray {
        let len = x.dim(1)
        guard len > 1 else { return x }
        let idx = MLXArray(stride(from: Int32(len - 1), through: 0, by: -1))
        return x[0..., idx, 0...]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let forward = forwardBlocks(x) + x
        let xRev = Self.reverseTime(x)
        let backward = Self.reverseTime(backwardBlocks(xRev) + xRev)
        return norm(outputProj(MLX.concatenated([forward, backward], axis: -1)))
    }
}

final class NemotronReuseTFMambaBlock: Module, UnaryLayer {
    @ModuleInfo(key: "time_mamba") var timeMamba: NemotronReuseBidirectionalMamba
    @ModuleInfo(key: "freq_mamba") var freqMamba: NemotronReuseBidirectionalMamba

    init(_ config: NemotronReuseConfig) {
        self._timeMamba.wrappedValue = NemotronReuseBidirectionalMamba(config)
        self._freqMamba.wrappedValue = NemotronReuseBidirectionalMamba(config)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (batch, time, freq, channels) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3))
        var x = input.transposed(0, 2, 1, 3).reshaped(batch * freq, time, channels)
        x = timeMamba(x) + x
        x = x.reshaped(batch, freq, time, channels).transposed(0, 2, 1, 3)
        x = x.reshaped(batch * time, freq, channels)
        x = freqMamba(x) + x
        return x.reshaped(batch, time, freq, channels)
    }
}
