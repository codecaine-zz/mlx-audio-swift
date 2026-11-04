import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioVAD

@Suite("NemotronDiarization")
struct NemotronDiarizationTests {
    private func tinyConfig() throws -> NemotronDiarizationConfig {
        let enc = NemotronDiarizationEncoderConfig(
            featIn: 16,
            dModel: 16,
            nLayers: 1,
            nHeads: 2,
            subsamplingFactor: 8,
            ffExpansion: 2.0
        )
        let mod = NemotronDiarizationModulesConfig(
            numSpeakers: 4,
            fcDModel: 16,
            tfDModel: 8,
            subsamplingFactor: 8,
            chunkLen: 2,
            fifoLen: 2,
            spkcacheLen: 8,
            spkcacheUpdatePeriod: 2,
            chunkLeftContext: 0,
            chunkRightContext: 1,
            spkcacheSilFramesPerSpk: 1
        )
        let proc = NemotronDiarizationMelConfig(
            featureSize: 16,
            samplingRate: 16000,
            hopLength: 160,
            nFft: 512,
            winLength: 400,
            preemphasis: 0.97,
            padTo: 16
        )
        return try NemotronDiarizationConfig(
            numSpeakers: 4,
            encoderConfig: enc,
            modulesConfig: mod,
            processorConfig: proc
        )
    }

    @Test func configurationValidation() throws {
        let defaultCfg = try JSONDecoder().decode(NemotronDiarizationConfig.self, from: Data("{}".utf8))
        #expect(defaultCfg.numSpeakers == 8)
        #expect(defaultCfg.encoderConfig.nLayers == 31)
        #expect(defaultCfg.modulesConfig.spkcacheLen == 264)

        #expect(throws: NemotronDiarizationConfigError.self) {
            var bad = defaultCfg
            bad.encoderConfig.nHeads = 7
            try bad.validate()
        }
        #expect(throws: NemotronDiarizationConfigError.self) {
            var bad = defaultCfg
            bad.modulesConfig.chunkLeftContext = 1
            try bad.validate()
        }
    }

    @Test func forwardPassAndMasking() throws {
        let config = try tinyConfig()
        let model = NemotronDiarizationModel(config)
        let features = MLXArray.zeros([1, 16, 19], dtype: .float32)
        let lengths = MLXArray([Int32(19)])
        let probs = model(features, audioSignalLength: lengths)
        eval(probs)
        // ceil(19 / 8) * 8 = 24 frames, with frames 19..<24 masked to 0
        #expect(probs.shape == [1, 24, 4])
        #expect(all(isFinite(probs)).item(Bool.self))
        let tailMax = max(abs(probs[0, 19..<24, 0...])).item(Float.self)
        #expect(tailMax == 0)
    }

    @Test func streamingPresetsAndGenerateRoundtrip() throws {
        let config = try tinyConfig()
        let model = NemotronDiarizationModel(config)
        try model.setStreamingConfig("ultra_low")
        #expect(model.config.modulesConfig.chunkLen == 3)
        #expect(model.config.modulesConfig.chunkRightContext == 1)

        // Restore tiny chunk config for fast test execution
        let tiny = try tinyConfig()
        var state = model.initStreamingState()
        let (emptyOut, flushedState) = try model.feed(
            audio: MLXArray.zeros([0], dtype: .float32),
            state: state,
            sampleRate: 16000,
            final: true
        )
        #expect(emptyOut.segments.isEmpty)
        #expect(flushedState.finished)
        #expect(throws: NemotronDiarizationError.self) {
            _ = try model.feed(
                audio: MLXArray.zeros([160], dtype: .float32),
                state: flushedState,
                sampleRate: 16000
            )
        }

        let model2 = NemotronDiarizationModel(tiny)
        let waveform = MLX.sin(MLX.arange(6400, dtype: .float32) * 0.05) * 0.1
        let output = try model2.generate(audio: waveform, sampleRate: 16000, threshold: 0.99)
        #expect(output.speakerProbs?.shape == [40, 4])
        state = model2.initStreamingState()
    }
}
