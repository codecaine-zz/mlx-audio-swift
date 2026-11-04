import Foundation
import MLX

// PyTorch Sequential indices used by the official Gradium checkpoints.
func pocketTTSSEANetNames(prefix: String, config: PocketTTSSeanetConfig, encoder: Bool) -> [String: String] {
    var names = ["\(prefix).model.0": "\(prefix).init_conv1d"]
    var index = 1
    for layer in config.ratios.indices {
        let destination = "\(prefix).layers.\(layer)"
        if !encoder {
            names["\(prefix).model.\(index + 1)"] = "\(destination).upsample"
            index += 2
        }
        for residual in 0..<config.nResidualLayers {
            for block in 0..<2 {
                names["\(prefix).model.\(index).block.\(2 * block + 1)"] = "\(destination).residuals.\(residual).block.\(block)"
            }
            index += 1
        }
        if encoder {
            names["\(prefix).model.\(index + 1)"] = "\(destination).downsample"
            index += 2
        }
    }
    names["\(prefix).model.\(index + 1)"] = "\(prefix).final_conv1d"
    return names
}

extension PocketTTSModel {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let weights = weights.filter {
            !$0.key.hasSuffix(".learnt_padding") && !["speaker_wavs", "emb", "audio_prompt"].contains($0.key)
        }
        guard weights.keys.contains(where: { $0.contains(".decoder.model.") || $0.contains(".condition_provider.") }) else {
            return weights
        }

        var names = pocketTTSSEANetNames(prefix: "mimi.encoder", config: config.mimi.seanet, encoder: true)
        names.merge(pocketTTSSEANetNames(prefix: "mimi.decoder", config: config.mimi.seanet, encoder: false)) { _, new in new }
        let prefixes = names.sorted { $0.key.count > $1.key.count }
        let targetKeys = Set(parameters().flattened().map(\.0))
        var mapped: [String: MLXArray] = [:]
        for (source, tensor) in weights {
            if [".w_s_t.", ".quantizer.vq.", ".quantizer.logvar_proj."].contains(where: source.contains) { continue }
            var key = source.replacingOccurrences(
                of: "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight", with: "speaker_proj_weight")
            key = key.replacingOccurrences(of: "flow_lm.speaker_proj_weight", with: "speaker_proj_weight")
            key = key.replacingOccurrences(of: "flow_lm.condition_provider.conditioners.transcript_in_segment.", with: "flow_lm.conditioner.")
            for name in ["num_speakers", "duration_delta"] {
                key = key.replacingOccurrences(of: "flow_lm.condition_provider.conditioners.\(name).", with: "flow_lm.\(name).")
            }
            key = key.replacingOccurrences(of: "flow_lm.backbone.", with: "flow_lm.transformer.")
                .replacingOccurrences(of: "flow_lm.flow.", with: "flow_lm.flow_net.")
                .replacingOccurrences(of: "mimi.model.", with: "mimi.")
            for (prefix, destination) in prefixes where key.hasPrefix(prefix + ".") {
                key = destination + key.dropFirst(prefix.count)
                break
            }
            key = key.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
                .replacingOccurrences(
                    of: #"(\.(?:encoder|decoder)_transformer\.transformer\.layers\.\d+)\.(linear[12])\."#,
                    with: "$1.gating.$2.", options: .regularExpression)
            if !targetKeys.contains(key) {
                key = key.replacingOccurrences(
                    of: #"\.(conv|convtr)\.(weight|bias)$"#, with: ".$1.$1.$2", options: .regularExpression)
            }
            var value = tensor
            if value.ndim == 3 && key.hasSuffix(".weight") {
                if key.contains(".convtr.") {
                    value = key.hasPrefix("mimi.upsample.") ? value.transposed(0, 2, 1) : value.transposed(1, 2, 0)
                } else {
                    value = value.transposed(0, 2, 1)
                }
            }
            mapped[key] = value
        }
        return mapped
    }
}

/// Reads Gradium Phonon `.gguf` checkpoints (`model.q8.gguf`, etc.) containing `BF16`, `F16`, `F32`, and `Q8_0` tensors.
func loadPocketTTSGGUF(url: URL) throws -> [String: MLXArray] {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    guard data.count >= 24 else {
        throw NSError(domain: "PocketTTSModel", code: 10, userInfo: [NSLocalizedDescriptionKey: "GGUF file too small: \(url.lastPathComponent)"])
    }

    return try data.withUnsafeBytes { raw -> [String: MLXArray] in
        func readU32(_ offset: inout Int) throws -> UInt32 {
            guard offset + 4 <= raw.count else { throw NSError(domain: "PocketTTSModel", code: 11, userInfo: [NSLocalizedDescriptionKey: "Truncated GGUF header"]) }
            let val = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            offset += 4
            return val
        }
        func readU64(_ offset: inout Int) throws -> UInt64 {
            guard offset + 8 <= raw.count else { throw NSError(domain: "PocketTTSModel", code: 11, userInfo: [NSLocalizedDescriptionKey: "Truncated GGUF header"]) }
            let val = UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
            offset += 8
            return val
        }
        func readString(_ offset: inout Int) throws -> String {
            let len = Int(try readU64(&offset))
            guard len >= 0, offset + len <= raw.count else {
                throw NSError(domain: "PocketTTSModel", code: 12, userInfo: [NSLocalizedDescriptionKey: "Invalid GGUF string length"])
            }
            let slice = UnsafeRawBufferPointer(rebasing: raw[offset ..< offset + len])
            offset += len
            return String(decoding: slice, as: UTF8.self)
        }
        func skipValue(type: UInt32, offset: inout Int) throws {
            switch type {
            case 0, 1, 7: offset += 1 // uint8, int8, bool
            case 2, 3: offset += 2    // uint16, int16
            case 4, 5, 6: offset += 4 // uint32, int32, float32
            case 8: _ = try readString(&offset)
            case 9: // array
                let elemType = try readU32(&offset)
                let count = Int(try readU64(&offset))
                for _ in 0..<count { try skipValue(type: elemType, offset: &offset) }
            case 10, 11, 12: offset += 8 // uint64, int64, float64
            default:
                throw NSError(domain: "PocketTTSModel", code: 13, userInfo: [NSLocalizedDescriptionKey: "Unsupported GGUF metadata type \(type)"])
            }
        }

        var offset = 0
        let magic = try readU32(&offset)
        guard magic == 0x46554747 else { // "GGUF"
            throw NSError(domain: "PocketTTSModel", code: 14, userInfo: [NSLocalizedDescriptionKey: "Invalid GGUF magic in \(url.lastPathComponent)"])
        }
        let version = try readU32(&offset)
        guard version == 2 || version == 3 else {
            throw NSError(domain: "PocketTTSModel", code: 15, userInfo: [NSLocalizedDescriptionKey: "Unsupported GGUF version \(version)"])
        }
        let tensorCount = Int(try readU64(&offset))
        let kvCount = Int(try readU64(&offset))

        for _ in 0..<kvCount {
            _ = try readString(&offset)
            let vtype = try readU32(&offset)
            try skipValue(type: vtype, offset: &offset)
        }

        struct TensorDescriptor {
            let name: String
            let shape: [Int]
            let ggmlType: UInt32
            let relOffset: Int
        }

        var descriptors: [TensorDescriptor] = []
        descriptors.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try readString(&offset)
            let nDims = Int(try readU32(&offset))
            var ggufDims: [Int] = []
            ggufDims.reserveCapacity(nDims)
            for _ in 0..<nDims {
                ggufDims.append(Int(try readU64(&offset)))
            }
            let ggmlType = try readU32(&offset)
            let relOffset = Int(try readU64(&offset))
            // GGUF stores dimensions in reverse order compared to row-major PyTorch/Safetensors
            descriptors.append(TensorDescriptor(name: name, shape: ggufDims.reversed(), ggmlType: ggmlType, relOffset: relOffset))
        }

        let dataStart = (offset + 31) & ~31
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(tensorCount)

        for desc in descriptors {
            let numElements = desc.shape.reduce(1, *)
            let tensorStart = dataStart + desc.relOffset
            switch desc.ggmlType {
            case 30: // GGML_TYPE_BF16
                let byteCount = numElements * 2
                guard tensorStart + byteCount <= raw.count else {
                    throw NSError(domain: "PocketTTSModel", code: 16, userInfo: [NSLocalizedDescriptionKey: "Truncated BF16 tensor \(desc.name)"])
                }
                let slice = data.subdata(in: tensorStart ..< (tensorStart + byteCount))
                result[desc.name] = MLXArray(slice, desc.shape, dtype: .bfloat16)

            case 0: // GGML_TYPE_F32
                let byteCount = numElements * 4
                guard tensorStart + byteCount <= raw.count else {
                    throw NSError(domain: "PocketTTSModel", code: 16, userInfo: [NSLocalizedDescriptionKey: "Truncated F32 tensor \(desc.name)"])
                }
                let slice = data.subdata(in: tensorStart ..< (tensorStart + byteCount))
                result[desc.name] = MLXArray(slice, desc.shape, dtype: .float32)

            case 1: // GGML_TYPE_F16
                let byteCount = numElements * 2
                guard tensorStart + byteCount <= raw.count else {
                    throw NSError(domain: "PocketTTSModel", code: 16, userInfo: [NSLocalizedDescriptionKey: "Truncated F16 tensor \(desc.name)"])
                }
                let slice = data.subdata(in: tensorStart ..< (tensorStart + byteCount))
                result[desc.name] = MLXArray(slice, desc.shape, dtype: .float16)

            case 8: // GGML_TYPE_Q8_0 (32 elements per 34-byte block: Float16 scale + 32 x Int8)
                guard numElements % 32 == 0 else {
                    throw NSError(domain: "PocketTTSModel", code: 17, userInfo: [NSLocalizedDescriptionKey: "Invalid Q8_0 element count for \(desc.name)"])
                }
                let numBlocks = numElements / 32
                let byteCount = numBlocks * 34
                guard tensorStart + byteCount <= raw.count else {
                    throw NSError(domain: "PocketTTSModel", code: 16, userInfo: [NSLocalizedDescriptionKey: "Truncated Q8_0 tensor \(desc.name)"])
                }
                var floats = [Float](repeating: 0, count: numElements)
                guard let basePtr = raw.baseAddress else { continue }
                floats.withUnsafeMutableBufferPointer { dstBuf in
                    guard let dst = dstBuf.baseAddress else { return }
                    for b in 0..<numBlocks {
                        let blockOffset = tensorStart + b * 34
                        let f16Bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: blockOffset, as: UInt16.self))
                        #if !arch(x86_64)
                        let scale = Float(Float16(bitPattern: f16Bits))
                        #else
                        let scale: Float = 0
                        #endif
                        let qsPtr = basePtr.advanced(by: blockOffset + 2).assumingMemoryBound(to: Int8.self)
                        let outBase = dst.advanced(by: b * 32)
                        for i in 0..<32 {
                            outBase[i] = scale * Float(qsPtr[i])
                        }
                    }
                }
                result[desc.name] = MLXArray(floats, desc.shape).asType(.bfloat16)

            default:
                throw NSError(domain: "PocketTTSModel", code: 18, userInfo: [NSLocalizedDescriptionKey: "Unsupported GGUF tensor type \(desc.ggmlType) for \(desc.name)"])
            }
        }
        return result
    }
}
