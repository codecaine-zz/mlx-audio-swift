import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioSTS

@Suite("NemotronReuse")
struct NemotronReuseTests {
    private func tinyConfig() -> NemotronReuseConfig {
        NemotronReuseConfig(
            hidFeature: 4,
            numTfmamba: 1,
            dState: 3,
            expand: 2
        )
    }

    @Test func selectiveScanMatchesReferenceRecurrence() throws {
        let batch = 2, length = 19, channels = 7, states = 3
        let xVals = (0..<(batch * length * channels)).map { Float(($0 % 11) - 5) * 0.13 }
        let dtVals = (0..<(batch * length * channels)).map { Float(($0 % 7) + 1) * 0.03 }
        let aVals = (0..<(channels * states)).map { -Float(($0 % 5) + 1) * 0.25 }
        let bVals = (0..<(batch * length * states)).map { Float(($0 % 9) - 4) * 0.11 }
        let cVals = (0..<(batch * length * states)).map { Float(($0 % 13) - 6) * 0.09 }
        let dVals = (0..<channels).map { Float($0 - 3) * 0.2 }

        let x = MLXArray(xVals).reshaped(batch, length, channels)
        let dt = MLXArray(dtVals).reshaped(batch, length, channels)
        let a = MLXArray(aVals).reshaped(channels, states)
        let b = MLXArray(bVals).reshaped(batch, length, states)
        let c = MLXArray(cVals).reshaped(batch, length, states)
        let d = MLXArray(dVals)

        let cpuActual = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let out = NemotronReuseSelectiveScan.scan(x, dt: dt, a: a, b: b, c: c, d: d)
            eval(out)
            return out
        }
        let gpuActual = Device.withDefaultDevice(.gpu) { () -> MLXArray in
            let out = NemotronReuseSelectiveScan.scan(x, dt: dt, a: a, b: b, c: c, d: d)
            eval(out)
            return out
        }

        #expect(max(abs(gpuActual - cpuActual)).item(Float.self) < 2e-6)
    }

    @Test func subpixelChannelOrderMatchesUpstream() throws {
        let layer = NemotronReuseSubpixelConv2d(dim: 2, factor: 3)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "conv.weight": MLXArray.zeros(layer.conv.weight.shape, dtype: .float32),
                "conv.bias": MLX.arange(6, dtype: .float32)
            ]),
            verify: .all
        )
        let actual = layer(MLXArray.zeros([1, 1, 2, 2], dtype: .float32))
        eval(actual)
        #expect(actual.shape == [1, 1, 6, 2])
        #expect(actual.asArray(Float.self) == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    }

    @Test func sourceLayoutSanitizationAndStrictLoad() throws {
        let config = tinyConfig()
        let model = NemotronReuseModel(config)
        let native = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var source: [String: MLXArray] = [:]
        for (k, v) in native {
            if v.ndim == 4 {
                source[k] = v.transposed(0, 3, 1, 2)
            } else if k.hasSuffix("conv1d.weight"), v.ndim == 3 {
                source[k] = v.transposed(0, 2, 1)
            } else {
                source[k] = v
            }
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-reuse-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let rawJSON = """
        {"model_cfg":{"hid_feature":4,"num_tfmamba":1,"d_state":3,"expand":2},"stft_cfg":{"n_fft":320}}
        """
        try rawJSON.write(to: tmpDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try save(arrays: source, url: tmpDir.appendingPathComponent("model.safetensors"))

        let loaded = try NemotronReuseModel.fromDirectory(tmpDir, strict: true)
        for (key, value) in loaded.parameters().flattened() {
            let expected = native[key]!
            #expect(max(abs(value - expected)).item(Float.self) == 0)
        }

        let sanitized = loaded.sanitize(source)
        let resanitized = loaded.sanitize(sanitized)
        for (key, value) in resanitized {
            #expect(max(abs(value - sanitized[key]!)).item(Float.self) == 0)
        }

        let mag = MLXArray.ones([1, 17, 11], dtype: .float32)
        let phase = MLXArray.zeros([1, 17, 11], dtype: .float32)
        let (outMag, outPhase) = loaded(mag, phase: phase)
        eval(outMag, outPhase)
        #expect(outMag.shape == mag.shape && outPhase.shape == phase.shape)
        #expect(all(isFinite(outMag)).item(Bool.self) && all(isFinite(outPhase)).item(Bool.self))
    }

    @Test func detectionWithoutRepositoryName() {
        let config: [String: Any] = [
            "model_cfg": ["num_tfmamba": 30],
            "stft_cfg": ["n_fft": 320]
        ]
        #expect(STS.inferModelType(fromConfig: config) == "nemotron_reuse")
        var explicit = config
        explicit["model_type"] = "explicit"
        #expect(STS.inferModelType(fromConfig: explicit) == "explicit")
    }

    @Test func chunkOverlapAndMultichannelPreserveAudio() throws {
        let model = NemotronReuseModel(tinyConfig())
        model.chunkEnhanceOverride = { chunk, _ in chunk }
        let samples = (0..<(3501 * 2)).map { sin(Float($0) * 0.017) * 0.1 }
        let audio = MLXArray(samples).reshaped(3501, 2)
        let result = try model.generate(
            audio: audio,
            sampleRate: 8000,
            chunkSeconds: 0.1,
            overlapSeconds: 0.025
        )
        #expect(result.audio.shape == [3501, 2])
        #expect(result.sampleRate == 8000)
        #expect(max(abs(result.audio - audio)).item(Float.self) < 1e-6)
    }

    @Test func negativeZeroImaginaryUsesPositivePiPhase() throws {
        let model = NemotronReuseModel(tinyConfig())
        let real = MLXArray.full([2, 161], values: MLXArray(Float(-1.0)), type: Float.self)
        let imag = MLXArray.full([2, 161], values: MLXArray(Float(-0.0)), type: Float.self)
        let spectrum = real + MLXArray(real: Float(0), imaginary: Float(1)) * imag
        model.stftOverride = { _, _, _, _, _ in spectrum }

        let output = try model.generate(
            audio: MLXArray.zeros([400], dtype: .float32),
            sampleRate: 8000,
            chunkSeconds: nil
        )
        #expect(output.audio.shape == [400])
        #expect(all(isFinite(output.audio)).item(Bool.self))
    }

    @Test func shortSilentAndInvalidInputs() throws {
        let model = NemotronReuseModel(tinyConfig())
        for length in [1, 79, 160, 161] {
            let output = try model.generate(
                audio: MLXArray.zeros([length], dtype: .float32),
                sampleRate: 8000,
                chunkSeconds: nil
            )
            #expect(output.audio.shape == [length])
            #expect(all(isFinite(output.audio)).item(Bool.self))
        }

        #expect(throws: NemotronReuseError.self) {
            _ = try model.generate(audio: MLXArray.zeros([10], dtype: .float32), sampleRate: 7999)
        }
        #expect(throws: NemotronReuseError.self) {
            _ = try model.generate(audio: MLXArray.zeros([10], dtype: .float32), sampleRate: 48001)
        }
        #expect(throws: NemotronReuseError.self) {
            _ = try model.generate(audio: MLXArray.zeros([0], dtype: .float32), sampleRate: 8000)
        }
        #expect(throws: NemotronReuseError.self) {
            _ = try model.generate(audio: MLXArray([Float.nan]), sampleRate: 8000)
        }
        #expect(throws: NemotronReuseError.self) {
            _ = try model.generate(
                audio: MLXArray.zeros([10], dtype: .float32),
                sampleRate: 8000,
                chunkSeconds: 1.0,
                overlapSeconds: 1.0
            )
        }
    }
}
