// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
// Adapted for MLX from NVIDIA-NemotronLabs-RE-USE.

import Foundation

public enum NemotronReuseCompressFactor: Codable, Equatable, Sendable {
    case reluLog1p
    case log1p
    case signedLog1p
    case power(Float)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            switch text {
            case "relu_log1p":
                self = .reluLog1p
            case "log1p":
                self = .log1p
            case "signed_log1p":
                self = .signedLog1p
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Unsupported NemotronReuse compress_factor: \(text)"
                )
            }
        } else {
            let value = try container.decode(Float.self)
            guard value.isFinite, value > 0 else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Numeric compress_factor must be finite and positive"
                )
            }
            self = .power(value)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .reluLog1p:
            try container.encode("relu_log1p")
        case .log1p:
            try container.encode("log1p")
        case .signedLog1p:
            try container.encode("signed_log1p")
        case .power(let value):
            try container.encode(value)
        }
    }
}

public struct NemotronReuseConfig: Codable, Equatable, Sendable {
    public var modelType: String
    public var hidFeature: Int
    public var numTfmamba: Int
    public var dState: Int
    public var dConv: Int
    public var expand: Int
    public var inputChannel: Int
    public var outputChannel: Int
    public var compressFactor: NemotronReuseCompressFactor
    public var nFft: Int
    public var hopSize: Int
    public var winSize: Int
    public var samplingRate: Int

    public init(
        modelType: String = "nemotron_reuse",
        hidFeature: Int = 64,
        numTfmamba: Int = 30,
        dState: Int = 16,
        dConv: Int = 4,
        expand: Int = 4,
        inputChannel: Int = 2,
        outputChannel: Int = 1,
        compressFactor: NemotronReuseCompressFactor = .reluLog1p,
        nFft: Int = 320,
        hopSize: Int = 40,
        winSize: Int = 320,
        samplingRate: Int = 8000
    ) {
        self.modelType = modelType
        self.hidFeature = hidFeature
        self.numTfmamba = numTfmamba
        self.dState = dState
        self.dConv = dConv
        self.expand = expand
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.compressFactor = compressFactor
        self.nFft = nFft
        self.hopSize = hopSize
        self.winSize = winSize
        self.samplingRate = samplingRate
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hidFeature = "hid_feature"
        case numTfmamba = "num_tfmamba"
        case dState = "d_state"
        case dConv = "d_conv"
        case expand
        case inputChannel = "input_channel"
        case outputChannel = "output_channel"
        case compressFactor = "compress_factor"
        case nFft = "n_fft"
        case hopSize = "hop_size"
        case winSize = "win_size"
        case samplingRate = "sampling_rate"
        case modelCfg = "model_cfg"
        case stftCfg = "stft_cfg"
    }

    private struct PartialSection: Decodable {
        var modelType: String?
        var hidFeature: Int?
        var numTfmamba: Int?
        var dState: Int?
        var dConv: Int?
        var expand: Int?
        var inputChannel: Int?
        var outputChannel: Int?
        var compressFactor: NemotronReuseCompressFactor?
        var nFft: Int?
        var hopSize: Int?
        var winSize: Int?
        var samplingRate: Int?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hidFeature = "hid_feature"
            case numTfmamba = "num_tfmamba"
            case dState = "d_state"
            case dConv = "d_conv"
            case expand
            case inputChannel = "input_channel"
            case outputChannel = "output_channel"
            case compressFactor = "compress_factor"
            case nFft = "n_fft"
            case hopSize = "hop_size"
            case winSize = "win_size"
            case samplingRate = "sampling_rate"
        }
    }

    public init(from decoder: Decoder) throws {
        let root = try PartialSection(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let modelCfg = try container.decodeIfPresent(PartialSection.self, forKey: .modelCfg)
        let stftCfg = try container.decodeIfPresent(PartialSection.self, forKey: .stftCfg)

        // Match Python `{**config, **config.get("model_cfg", {}), **config.get("stft_cfg", {})}`
        modelType = stftCfg?.modelType ?? modelCfg?.modelType ?? root.modelType ?? "nemotron_reuse"
        hidFeature = stftCfg?.hidFeature ?? modelCfg?.hidFeature ?? root.hidFeature ?? 64
        numTfmamba = stftCfg?.numTfmamba ?? modelCfg?.numTfmamba ?? root.numTfmamba ?? 30
        dState = stftCfg?.dState ?? modelCfg?.dState ?? root.dState ?? 16
        dConv = stftCfg?.dConv ?? modelCfg?.dConv ?? root.dConv ?? 4
        expand = stftCfg?.expand ?? modelCfg?.expand ?? root.expand ?? 4
        inputChannel = stftCfg?.inputChannel ?? modelCfg?.inputChannel ?? root.inputChannel ?? 2
        outputChannel = stftCfg?.outputChannel ?? modelCfg?.outputChannel ?? root.outputChannel ?? 1
        compressFactor = stftCfg?.compressFactor ?? modelCfg?.compressFactor ?? root.compressFactor ?? .reluLog1p
        nFft = stftCfg?.nFft ?? modelCfg?.nFft ?? root.nFft ?? 320
        hopSize = stftCfg?.hopSize ?? modelCfg?.hopSize ?? root.hopSize ?? 40
        winSize = stftCfg?.winSize ?? modelCfg?.winSize ?? root.winSize ?? 320
        samplingRate = stftCfg?.samplingRate ?? modelCfg?.samplingRate ?? root.samplingRate ?? 8000
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hidFeature, forKey: .hidFeature)
        try container.encode(numTfmamba, forKey: .numTfmamba)
        try container.encode(dState, forKey: .dState)
        try container.encode(dConv, forKey: .dConv)
        try container.encode(expand, forKey: .expand)
        try container.encode(inputChannel, forKey: .inputChannel)
        try container.encode(outputChannel, forKey: .outputChannel)
        try container.encode(compressFactor, forKey: .compressFactor)
        try container.encode(nFft, forKey: .nFft)
        try container.encode(hopSize, forKey: .hopSize)
        try container.encode(winSize, forKey: .winSize)
        try container.encode(samplingRate, forKey: .samplingRate)
    }

    public func stftSizes(sampleRate: Int) -> (nFft: Int, hopSize: Int, winSize: Int) {
        func scale(_ value: Int) -> Int {
            let scaled = (value * sampleRate) / samplingRate
            return scaled + (scaled % 2)
        }
        return (scale(nFft), scale(hopSize), scale(winSize))
    }
}
