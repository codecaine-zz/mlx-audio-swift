import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel
    @FocusState private var focusedField: Field?
    @State private var modelFilter: String = ""
    @State private var showCachedOnly: Bool = false
    @State private var showFavoritesOnly: Bool = false
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var prefs = StudioUserPreferencesManager.shared
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var showDownloadedModelsSheet: Bool = false

    private var filteredPresets: [(name: String, id: String)] {
        _ = modelsManager.cacheRevision
        var base = showCachedOnly
            ? TTSViewModel.presetModels.filter { modelsManager.isModelCached($0.id) }
            : TTSViewModel.presetModels
        if showFavoritesOnly {
            base = base.filter { prefs.isFavoriteTTSModel($0.id) }
        }
        if !modelFilter.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            base = base.filter {
                $0.name.localizedCaseInsensitiveContains(modelFilter) ||
                $0.id.localizedCaseInsensitiveContains(modelFilter)
            }
        }
        return base.sorted { a, b in
            let favA = prefs.isFavoriteTTSModel(a.id)
            let favB = prefs.isFavoriteTTSModel(b.id)
            if favA != favB { return favA && !favB }
            return false
        }
    }

    private enum Field {
        case modelId
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                settingsContent
            }
            .navigationTitle("Settings")
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
        .frame(minWidth: 500, minHeight: 720)
        #endif
        .background(Color(red: 0.08, green: 0.09, blue: 0.13))
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showDownloadedModelsSheet) {
            DownloadedModelsSheetView()
        }
        .alert(
            "Delete Downloaded Model?",
            isPresented: Binding(
                get: { pendingDeleteModel != nil },
                set: { if !$0 { pendingDeleteModel = nil } }
            ),
            presenting: pendingDeleteModel
        ) { target in
            Button("Delete Model (\(target.size))", role: .destructive) {
                modelsManager.deleteModel(target.id, displayName: target.name)
                pendingDeleteModel = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteModel = nil
            }
        } message: { target in
            Text("Remove '\(target.name)' (\(target.id)) from your local HuggingFace cache and free \(target.size) of disk space? You can re-download it at any time.")
        }
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
                HStack {
                    Text("Model")
                        .font(labelFont)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        modelsManager.refresh()
                        showDownloadedModelsSheet = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "internaldrive.fill")
                                .font(.caption2)
                            Text("Manage Downloaded Models (\(modelsManager.completeModelsCount) · \(modelsManager.totalSizeString))")
                                .font(.caption2.weight(.semibold))
                        }
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.blue.opacity(0.12), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }

                if let banner = modelsManager.statusBanner {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.green)
                        Text(banner)
                            .font(.caption)
                            .foregroundStyle(.green)
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.green.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                    .padding(.top, 4)
                }

                HStack(spacing: 6) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .font(textFont)
                        .textFieldStyle(.plain)
#if os(iOS)
                        .textInputAutocapitalization(.never)
#endif
                        .autocorrectionDisabled()
                        .focused($focusedField, equals: Field.modelId)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .onTapGesture {
                            focusedField = .modelId
                        }

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

                    if modelsManager.isModelCached(viewModel.modelId) {
                        let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId) ?? "Cached"
                        Button {
                            let name = TTSViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
                            pendingDeleteModel = (name: name, id: viewModel.modelId, size: activeSize)
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "trash")
                                    .font(.caption)
                                Text("Delete")
                                    .font(.caption.weight(.medium))
                            }
                            .foregroundStyle(.red)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(Color.red.opacity(0.14))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.isLoading)
                        .help("Delete cached weights for \(viewModel.modelId) (\(activeSize))")
                    }
                }
                .padding(.top, 4)

                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 10) {
                        Text("Known Downloadable Models (\(filteredPresets.count)):")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Toggle("★ Favorites (\(prefs.favoriteTTSModelIds.count))", isOn: $showFavoritesOnly)
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                            .font(.caption2)
                        Toggle("Cached Only", isOn: $showCachedOnly)
                            .toggleStyle(.switch)
                            .controlSize(.mini)
                            .font(.caption2)
                    }
                    .padding(.top, 6)

                    TextField("Search models (e.g. fish, phonon, kokoro, qwen)...", text: $modelFilter)
                        .font(.caption)
                        .textFieldStyle(.plain)
                        .padding(6)
                        .background(Color.gray.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(filteredPresets, id: \.id) { preset in
                                let isCached = modelsManager.isModelCached(preset.id)
                                let isFav = prefs.isFavoriteTTSModel(preset.id)
                                let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                HStack(spacing: 6) {
                                    Button {
                                        prefs.toggleFavoriteTTSModel(preset.id)
                                    } label: {
                                        Image(systemName: isFav ? "star.fill" : "star")
                                            .font(.caption)
                                            .foregroundStyle(isFav ? .yellow : .secondary.opacity(0.55))
                                    }
                                    .buttonStyle(.plain)
                                    .help(isFav ? "Remove from favorite TTS models" : "Add to favorite TTS models")

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
                                        .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(viewModel.isLoading)

                                    if isCached {
                                        #if os(macOS)
                                        Button {
                                            ModelCacheHelper.revealModelInFinder(preset.id)
                                        } label: {
                                            Image(systemName: "folder")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                                .padding(4)
                                                .background(Color.gray.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                                        }
                                        .buttonStyle(.plain)
                                        .help("Reveal \(preset.name) in Finder")
                                        #endif

                                        Button {
                                            pendingDeleteModel = (
                                                name: preset.name,
                                                id: preset.id,
                                                size: sizeStr ?? "Cached"
                                            )
                                        } label: {
                                            HStack(spacing: 3) {
                                                Image(systemName: "trash")
                                                    .font(.caption2)
                                                Text("Delete")
                                                    .font(.caption2.weight(.medium))
                                            }
                                            .foregroundStyle(.red)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 3)
                                            .background(Color.red.opacity(0.14), in: Capsule())
                                        }
                                        .buttonStyle(.plain)
                                        .disabled(viewModel.isLoading)
                                        .help("Delete downloaded model \(preset.name) (\(sizeStr ?? "Cached"))")
                                    }
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(viewModel.modelId == preset.id ? Color.blue.opacity(0.12) : Color.gray.opacity(0.08))
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            }
                        }
                    }
                    .frame(maxHeight: 220)

                    // HuggingFace Token & Cache Management
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("HuggingFace Token (optional, for gated models)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Spacer()
                            #if os(macOS)
                            Button("Reveal Cache in Finder") {
                                ModelCacheHelper.revealCacheInFinder()
                            }
                            .font(.caption2)
                            .buttonStyle(.link)
                            #endif
                        }
                        SecureField("hf_...", text: $viewModel.hfToken)
                            .font(.caption)
                            .textFieldStyle(.roundedBorder)
                    }
                    .padding(.top, 6)
                }
            }
            .padding(.bottom, sectionSpacing)

            // Voice Design Section
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Voice Design")
                        .font(labelFont)
                        .foregroundStyle(.secondary)

                    Spacer()

                    #if os(iOS)
                    CompactToggle(label: "", isOn: $viewModel.useVoiceDesign, font: textFont, toggleWidth: 36, toggleHeight: 20, thumbSize: 16)
                    #else
                    Toggle("", isOn: $viewModel.useVoiceDesign)
                        .labelsHidden()
                    #endif
                }

                // Info box about VoiceDesign
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "info.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.blue)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("VoiceDesign Feature")
                                .font(.caption)
                                .fontWeight(.medium)

                            Text("Describe the voice you want in natural language. Only works with VoiceDesign models.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    // Recommended models
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Recommended Models:")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)

                        VStack(alignment: .leading, spacing: 2) {
                            ModelHintRow(modelId: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16") { selected in
                                viewModel.modelId = selected
                                Task { await viewModel.reloadModel() }
                            }
                        }
                    }

                    // Example descriptions
                    if viewModel.useVoiceDesign {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Example Descriptions (Click to Apply):")
                                .font(.caption2)
                                .fontWeight(.medium)
                                .foregroundStyle(.secondary)

                            VStack(alignment: .leading, spacing: 2) {
                                ExampleDescription(text: "A young woman with a bright, energetic voice") {
                                    viewModel.voiceDescription = $0
                                }
                                ExampleDescription(text: "A deep, calm male narrator") {
                                    viewModel.voiceDescription = $0
                                }
                                ExampleDescription(text: "An elderly person speaking slowly") {
                                    viewModel.voiceDescription = $0
                                }
                            }
                        }
                    }
                }
                .padding(10)
                .background(Color.blue.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                // Voice description input (only shown when enabled)
                if viewModel.useVoiceDesign {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Voice Description")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        
                        TextEditor(text: $viewModel.voiceDescription)
                            .font(textFont)
                            .frame(height: 80)
                            .scrollContentBackground(.hidden)
                            .padding(8)
                            .background(Color.gray.opacity(0.15))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(alignment: .topLeading) {
                                if viewModel.voiceDescription.isEmpty {
                                    Text("Describe the voice you want...")
                                        .font(textFont)
                                        .foregroundStyle(.tertiary)
                                        .padding(.horizontal, 12)
                                        .padding(.top, 16)
                                        .allowsHitTesting(false)
                                }
                            }
                    }
                }
            }
            .padding(.bottom, sectionSpacing)

            // Length Section
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
                    range: 100...2000,
                    step: 100
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 100...2000,
                    step: 100
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

            // Top P Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Top P")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Top P")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.topP))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Repetition Penalty Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Repetition Penalty")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Repetition Penalty")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.repetitionPenalty))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.repetitionPenalty) },
                        set: { viewModel.repetitionPenalty = Float($0) }
                    ),
                    range: 1.0...2.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.repetitionPenalty) },
                        set: { viewModel.repetitionPenalty = Float($0) }
                    ),
                    in: 1.0...2.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif

                // Hint for VoiceDesign models
                HStack(spacing: 6) {
                    Image(systemName: "info.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)

                    Text("For Qwen3-TTS VoiceDesign, use minimum 1.1 to reduce repetition")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 4)
                .padding(8)
                .background(Color.orange.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding(.bottom, sectionSpacing)

            // Text Chunking Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Text Chunking")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                #if os(iOS)
                CompactToggle(label: "Chunk text", isOn: $viewModel.enableChunking, font: textFont)
                    .padding(.top, 4)
                #else
                Toggle("Chunk text", isOn: $viewModel.enableChunking)
                    .font(textFont)
                    .padding(.top, 4)
                #endif

                #if os(iOS)
                CompactToggle(label: "Stream audio", isOn: $viewModel.streamingPlayback, font: textFont)
                #else
                Toggle("Stream audio", isOn: $viewModel.streamingPlayback)
                    .font(textFont)
                #endif

                if viewModel.enableChunking {
                    HStack {
                        Text("Max chunk length")
                            .font(textFont)
                        Spacer()
                        Text("\(viewModel.maxChunkLength)")
                            .font(textFont)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)

                    #if os(iOS)
                    CompactSlider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        range: 180...1200,
                        step: 30
                    )
                    #else
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        in: 180...1200,
                        step: 30
                    )
                    .tint(.blue)
                    #endif

                    TextField("Split pattern", text: $viewModel.splitPattern)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

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

// MARK: - Helper Views

struct ModelHintRow: View {
    let modelId: String
    var onSelect: ((String) -> Void)? = nil

    var body: some View {
        Button {
            onSelect?(modelId)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.forward.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue.opacity(0.6))

                Text(modelId)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .buttonStyle(.plain)
        .help("Click to select and load \(modelId)")
    }
}

struct ExampleDescription: View {
    let text: String
    var onSelect: ((String) -> Void)? = nil

    var body: some View {
        Button {
            onSelect?(text)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "quote.bubble.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue.opacity(0.4))

                Text(text)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .italic()
            }
        }
        .buttonStyle(.plain)
        .help("Click to use this VoiceDesign description")
    }
}

#Preview {
    SettingsView(viewModel: TTSViewModel())
}
