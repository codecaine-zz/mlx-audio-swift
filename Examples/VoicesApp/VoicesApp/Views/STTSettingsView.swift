import SwiftUI

struct STTSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: STTViewModel
    @State private var modelFilter: String = ""
    @State private var showCachedOnly: Bool = false

    private var filteredPresets: [(name: String, id: String)] {
        let base = showCachedOnly
            ? STTViewModel.presetModels.filter { ModelCacheHelper.isModelCached($0.id) }
            : STTViewModel.presetModels
        if modelFilter.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return base
        }
        return base.filter {
            $0.name.localizedCaseInsensitiveContains(modelFilter) ||
            $0.id.localizedCaseInsensitiveContains(modelFilter)
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                settingsContent
            }
            .navigationTitle("STT Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 600)
        #endif
    }

    #if os(iOS)
    private let sectionSpacing: CGFloat = 12
    private let labelFont: Font = .caption
    private let textFont: Font = .footnote
    private let horizontalPadding: CGFloat = 16
    #else
    private let sectionSpacing: CGFloat = 16
    private let labelFont: Font = .subheadline
    private let textFont: Font = .subheadline
    private let horizontalPadding: CGFloat = 20
    #endif

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Model")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Button(action: {
                        Task {
                            await viewModel.reloadModel()
                        }
                    }) {
                        Text("Load")
                            .font(textFont)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isLoading)
                }
                .padding(.top, 4)

                if viewModel.isLoading {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text(viewModel.generationProgress.isEmpty ? "Loading model…" : viewModel.generationProgress)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)
                } else if let err = viewModel.errorMessage {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.top, 4)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Known Downloadable STT Models (\(filteredPresets.count)):")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Toggle("Cached Only", isOn: $showCachedOnly)
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                            .font(.caption2)
                    }
                    .padding(.top, 6)

                    TextField("Search STT models (e.g. phonon, whisper, parakeet, qwen)...", text: $modelFilter)
                        .font(.caption)
                        .textFieldStyle(.plain)
                        .padding(6)
                        .background(Color.gray.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(filteredPresets, id: \.id) { preset in
                                let isCached = ModelCacheHelper.isModelCached(preset.id)
                                let sizeStr = ModelCacheHelper.cachedModelSizeString(preset.id)
                                Button {
                                    viewModel.modelId = preset.id
                                    Task {
                                        await viewModel.reloadModel()
                                    }
                                } label: {
                                    HStack(spacing: 6) {
                                        Image(systemName: viewModel.modelId == preset.id ? "checkmark.circle.fill" : (isCached ? "internaldrive.fill" : "arrow.down.circle"))
                                            .font(.caption)
                                            .foregroundStyle(viewModel.modelId == preset.id ? .blue : (isCached ? .green : .secondary))
                                        Text(preset.name)
                                            .font(.caption)
                                            .fontWeight(viewModel.modelId == preset.id ? .semibold : .regular)
                                            .foregroundStyle(.primary)
                                        if isCached {
                                            Text(sizeStr.map { "Cached · \($0)" } ?? "Cached")
                                                .font(.caption2)
                                                .foregroundStyle(.green)
                                                .padding(.horizontal, 5)
                                                .padding(.vertical, 1)
                                                .background(Color.green.opacity(0.12))
                                                .clipShape(Capsule())
                                        }
                                        Spacer()
                                        Text(preset.id)
                                            .font(.system(.caption2, design: .monospaced))
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(viewModel.modelId == preset.id ? Color.blue.opacity(0.12) : Color.gray.opacity(0.08))
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                                .buttonStyle(.plain)
                                .disabled(viewModel.isLoading)
                            }
                        }
                    }
                    .frame(maxHeight: 220)
                }
            }
            .padding(.bottom, sectionSpacing)

            // Max Tokens Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Length")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max Tokens")
                        .font(textFont)
                    Spacer()
                    Text("\(viewModel.maxTokens)")
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    range: 512...16384,
                    step: 512
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 512...16384,
                    step: 512
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Temperature Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Temperature")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Temperature")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.temperature))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Chunk Duration Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Chunking")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Chunk Duration")
                        .font(textFont)
                    Spacer()
                    Text("\(Int(viewModel.chunkDuration))s")
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.chunkDuration) },
                        set: { viewModel.chunkDuration = Float($0) }
                    ),
                    range: 30...600,
                    step: 10
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.chunkDuration) },
                        set: { viewModel.chunkDuration = Float($0) }
                    ),
                    in: 30...600,
                    step: 10
                )
                .tint(.blue)
                #endif

                Text("Split long audio into chunks at silence boundaries")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.bottom, sectionSpacing)

            // Streaming Delay Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Live Streaming")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Confirmation Delay")
                        .font(textFont)
                    Spacer()
                    Text(delayLabel(viewModel.streamingDelayMs))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.streamingDelayMs) },
                        set: { viewModel.streamingDelayMs = Int($0) }
                    ),
                    range: 0...5000,
                    step: 100
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.streamingDelayMs) },
                        set: { viewModel.streamingDelayMs = Int($0) }
                    ),
                    in: 0...5000,
                    step: 100
                )
                .tint(.blue)
                #endif

                Text("How long tokens must be stable before confirming. Lower = faster but more corrections.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.bottom, sectionSpacing)

            // Language Section
            VStack(alignment: .leading, spacing: 6) {
                Text("Language")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(["Auto", "English", "Chinese", "Japanese", "Spanish", "French", "German"], id: \.self) { lang in
                            let val = (lang == "Auto") ? "" : lang
                            let isSelected = (lang == "Auto" && viewModel.language.trimmingCharacters(in: .whitespaces).isEmpty) || viewModel.language.caseInsensitiveCompare(lang) == .orderedSame
                            Button(lang) {
                                viewModel.language = val
                            }
                            .font(.caption.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(isSelected ? .white : .primary)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                            .background(isSelected ? Color.blue : Color.gray.opacity(0.15), in: Capsule())
                            .buttonStyle(.plain)
                        }
                    }
                }

                TextField("Auto", text: $viewModel.language)
                    .font(textFont)
                    .textFieldStyle(.plain)
                    .padding(8)
                    .background(Color.gray.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding(.top, 2)

                Text("Leave empty for model default/auto, or enter e.g. English, Chinese, Japanese, Korean")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.bottom, sectionSpacing)

            // Reset button
            Button(action: {
                viewModel.resetSettingsToDefaults()
            }) {
                Text("Reset to Defaults")
                    .font(textFont)
                    .fontWeight(.medium)
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .padding(.top, 16)
            .padding(.bottom, 12)
        }
        .padding(.horizontal, horizontalPadding)
    }
}

private func delayLabel(_ ms: Int) -> String {
    switch ms {
    case 0: return "0ms (instant)"
    case 200: return "200ms (realtime)"
    case 480: return "480ms (agent)"
    case 2400: return "2400ms (subtitle)"
    default: return "\(ms)ms"
    }
}

#Preview {
    STTSettingsView(viewModel: STTViewModel())
}
