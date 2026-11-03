//
//  STSView.swift
//  VoicesApp
//
//  Created by Alpay Calalli on 25.02.26.
//

import SwiftUI
import UniformTypeIdentifiers

struct STSView: View {
    @State private var viewModel = STSViewModel()
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var isFilePickerPresented = false
    @State private var selectedInputURL: URL?
    @State private var isDraggingOver = false
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var showDownloadedModelsSheet = false

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                // Hero Mode Switcher Bento Card
                AppleBentoCard(padding: 20) {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack(spacing: 14) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(
                                        LinearGradient(
                                            colors: [Color.purple.opacity(0.32), Color.blue.opacity(0.24)],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .frame(width: 46, height: 46)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                                            .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
                                    )

                                Image(systemName: "tuningfork")
                                    .font(.system(size: 20, weight: .semibold))
                                    .foregroundStyle(.purple)
                            }

                            VStack(alignment: .leading, spacing: 3) {
                                Text("AI Audio Lab — Stem Separation, Denoise & Language ID")
                                    .font(.system(size: 17, weight: .semibold))
                                Text("Isolate vocals, drums, speech, or instruments with SAM-Audio, clean noisy recordings with MossFormer2/DeepFilterNet, or detect 107–256 spoken languages.")
                                    .font(.system(size: 12.5))
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()

                            if viewModel.isGenerating {
                                Button(role: .destructive) {
                                    viewModel.stop()
                                } label: {
                                    Label("Cancel", systemImage: "xmark.circle.fill")
                                        .font(.system(size: 12, weight: .semibold))
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 7)
                                        .background(Color.red.opacity(0.15), in: Capsule())
                                        .foregroundStyle(.red)
                                }
                                .buttonStyle(.plain)
                            }
                        }

                        // Apple Glass Studio Mode Selector
                        HStack(spacing: 8) {
                            ForEach(AudioStudioMode.allCases) { mode in
                                let isSelected = viewModel.studioMode == mode
                                let icon: String = {
                                    switch mode {
                                    case .separation: return "arrow.triangle.branch"
                                    case .enhancement: return "wand.and.stars"
                                    case .languageId: return "globe"
                                    }
                                }()
                                Button {
                                    withAnimation(.spring(response: 0.25, dampingFraction: 0.82)) {
                                        viewModel.studioMode = mode
                                    }
                                    if ModelCacheHelper.isModelCached(viewModel.modelId) {
                                        Task { await viewModel.loadModel() }
                                    }
                                } label: {
                                    HStack(spacing: 8) {
                                        Image(systemName: icon)
                                            .font(.system(size: 12.5, weight: .semibold))
                                            .foregroundStyle(isSelected ? .white : .secondary)
                                        Text(mode.rawValue)
                                            .font(.system(size: 12.5, weight: isSelected ? .semibold : .medium))
                                            .foregroundStyle(isSelected ? .white : .primary.opacity(0.85))
                                    }
                                    .padding(.vertical, 9)
                                    .frame(maxWidth: .infinity)
                                    .background(
                                        RoundedRectangle(cornerRadius: 11, style: .continuous)
                                            .fill(
                                                isSelected
                                                ? LinearGradient(colors: [Color.blue, Color.indigo], startPoint: .topLeading, endPoint: .bottomTrailing)
                                                : LinearGradient(colors: [Color.white.opacity(0.05), Color.white.opacity(0.03)], startPoint: .top, endPoint: .bottom)
                                            )
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 11, style: .continuous)
                                                    .strokeBorder(Color.white.opacity(isSelected ? 0.24 : 0.08), lineWidth: 0.8)
                                            )
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                // 2-Column Bento Row: Model Engine + Input Audio Drop Zone
                HStack(alignment: .top, spacing: 18) {
                    modelSection
                    inputSection
                }

                if viewModel.studioMode == .separation {
                    settingsSection
                }

                if viewModel.isGenerating {
                    generationProgressView
                }

                separateButton

                if let error = viewModel.errorMessage {
                    errorBanner(error)
                }

                if !viewModel.lidPredictions.isEmpty {
                    lidResultsSection
                }

                if viewModel.audioURL != nil {
                    audioPlayerSection
                }
            }
            .padding(22)
        }
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
        .task {
            if ModelCacheHelper.isModelCached(viewModel.modelId) {
                await viewModel.loadModel()
            }
        }
    }

    private var modeFilteredPresets: [(name: String, id: String)] {
        STSViewModel.presetModels.filter { preset in
            let lower = preset.id.lowercased()
            switch viewModel.studioMode {
            case .separation:
                return lower.contains("sam-audio")
            case .enhancement:
                return lower.contains("mossformer") || lower.contains("deepfilter")
            case .languageId:
                return lower.contains("voxlingua") || lower.contains("mms-lid")
            }
        }
    }

    // MARK: - Model Section

    private var modelSection: some View {
        let _ = modelsManager.cacheRevision
        let isActiveCached = modelsManager.isModelCached(viewModel.modelId)
        let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId)
        return AppleBentoCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Label("Neural Audio Engine", systemImage: "cpu.fill")
                        .font(.system(size: 14, weight: .semibold))
                    if isActiveCached {
                        Text(activeSize.map { "✓ Cached · \($0)" } ?? "✓ Cached")
                            .font(.system(size: 10.5, weight: .semibold))
                            .foregroundStyle(.green)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.14), in: Capsule())

                        Button {
                            let name = STSViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
                            pendingDeleteModel = (name: name, id: viewModel.modelId, size: activeSize ?? "Cached")
                        } label: {
                            HStack(spacing: 3) {
                                Image(systemName: "trash")
                                    .font(.system(size: 9.5, weight: .semibold))
                                Text("Delete")
                                    .font(.system(size: 10, weight: .semibold))
                            }
                            .foregroundStyle(.red)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2.5)
                            .background(Color.red.opacity(0.14), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .help("Delete cached weights for \(viewModel.modelId)")
                    }
                    Spacer()
                    Button {
                        modelsManager.refresh()
                        showDownloadedModelsSheet = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "internaldrive.fill")
                                .font(.system(size: 10))
                            Text("Models")
                                .font(.system(size: 10.5, weight: .semibold))
                        }
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.blue.opacity(0.14), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Manage all downloaded AI models on disk")
                    modelStatusBadge
                }

                // Preset Model Pills
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(modeFilteredPresets, id: \.id) { preset in
                            let cached = modelsManager.isModelCached(preset.id)
                            let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                            let isSelected = viewModel.modelId == preset.id
                            Button {
                                viewModel.modelId = preset.id
                                Task { await viewModel.reloadModel() }
                            } label: {
                                HStack(spacing: 5) {
                                    if cached {
                                        Image(systemName: "checkmark.circle.fill")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.green)
                                    }
                                    Text(preset.name)
                                        .font(.system(size: 11.5, weight: isSelected ? .semibold : .medium))
                                    if let sizeStr, cached {
                                        Text(sizeStr)
                                            .font(.system(size: 9.5, weight: .semibold))
                                            .foregroundStyle(.green.opacity(0.9))
                                    }
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule()
                                        .fill(isSelected ? Color.blue.opacity(0.28) : Color.white.opacity(0.05))
                                        .overlay(
                                            Capsule()
                                                .strokeBorder(isSelected ? Color.blue.opacity(0.8) : Color.white.opacity(0.08), lineWidth: 1)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                            .disabled(viewModel.isLoading || viewModel.isGenerating)
                            .contextMenu {
                                if cached {
                                    #if os(macOS)
                                    Button {
                                        ModelCacheHelper.revealModelInFinder(preset.id)
                                    } label: {
                                        Label("Reveal in Finder", systemImage: "folder")
                                    }
                                    #endif
                                    Button(role: .destructive) {
                                        pendingDeleteModel = (
                                            name: preset.name,
                                            id: preset.id,
                                            size: sizeStr ?? "Cached"
                                        )
                                    } label: {
                                        Label("Delete Downloaded Model (\(sizeStr ?? "Cached"))…", systemImage: "trash")
                                    }
                                }
                            }
                        }
                    }
                }

                // Active Repo & Load Action Bar (without ugly auto-focus text field border)
                HStack(spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "shippingbox")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        TextField("HuggingFace Model ID", text: $viewModel.modelId)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12, design: .monospaced))
                            .disabled(viewModel.isLoading || viewModel.isGenerating)
                    }
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .background(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .fill(Color.black.opacity(0.28))
                            .overlay(
                                RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                    )

                    if viewModel.isLoading {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Button {
                            Task { await viewModel.reloadModel() }
                        } label: {
                            Text(viewModel.isModelLoaded ? "Reload" : "Load Model")
                                .font(.system(size: 12, weight: .semibold))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color.white.opacity(0.09), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.isLoading)
                    }
                }
            }
        }
    }

    private var modelStatusBadge: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(viewModel.isModelLoaded ? Color.green : Color.orange)
                .frame(width: 7, height: 7)
            Text(viewModel.isModelLoaded ? "Ready" : "On Demand")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Input Section

    private var inputSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Source Audio File", systemImage: "waveform.circle.fill")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    if selectedInputURL != nil {
                        Button("Change File") {
                            isFilePickerPresented = true
                        }
                        .font(.system(size: 11.5, weight: .medium))
                        .buttonStyle(.plain)
                        .foregroundStyle(.blue)
                    }
                }

                ZStack {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(
                            isDraggingOver ? Color.blue : Color.white.opacity(0.12),
                            style: StrokeStyle(lineWidth: 1.5, dash: [6])
                        )
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(isDraggingOver
                                    ? Color.blue.opacity(0.08)
                                    : Color.black.opacity(0.22))
                        )
                        .frame(height: 102)

                    VStack(spacing: 6) {
                        if let url = selectedInputURL {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.system(size: 22))
                            Text(url.lastPathComponent)
                                .font(.system(size: 13, weight: .semibold))
                                .lineLimit(1)
                            Text("Ready for processing · Click to replace")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        } else {
                            Image(systemName: "arrow.up.doc.fill")
                                .font(.system(size: 22))
                                .foregroundStyle(.blue.opacity(0.85))
                            Text("Drop WAV, MP3, M4A, or AIFF here, or click to browse")
                                .font(.system(size: 12.5, weight: .medium))
                                .foregroundStyle(.primary.opacity(0.85))
                            Text("100% local processing on Apple Silicon Metal")
                                .font(.system(size: 11))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { isFilePickerPresented = true }
                .onDrop(of: [.fileURL, .audio, .wav, .mp3, .aiff, .mpeg4Audio], isTargeted: $isDraggingOver) { providers in
                    handleDrop(providers: providers)
                }
                .fileImporter(
                    isPresented: $isFilePickerPresented,
                    allowedContentTypes: [.audio, .wav, .mp3, .aiff, .mpeg4Audio],
                    allowsMultipleSelection: false
                ) { result in
                    if case .success(let urls) = result, let url = urls.first {
                        let scoped = url.startAccessingSecurityScopedResource()
                        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                        let dest = FileManager.default.temporaryDirectory
                            .appendingPathComponent("sts_pick_\(UUID().uuidString)_\(url.lastPathComponent)")
                        if (try? FileManager.default.copyItem(at: url, to: dest)) != nil {
                            selectedInputURL = dest
                        } else {
                            selectedInputURL = url
                        }
                    }
                }
            }
        }
    }

    // MARK: - Settings Section

    private var settingsSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Stem Target & Residual Routing", systemImage: "tuningfork")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Picker("Output", selection: $viewModel.useResidual) {
                        Text("Isolate Target Stem").tag(false)
                        Text("Keep Background (Residual)").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 320)
                    .disabled(viewModel.isGenerating)
                }

                // Stem Presets
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 7) {
                        ForEach(STSViewModel.separationPresets, id: \.self) { preset in
                            let isSelected = viewModel.separationDescription == preset
                            Button {
                                viewModel.separationDescription = preset
                            } label: {
                                Text(preset.capitalized)
                                    .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                                    .foregroundStyle(isSelected ? .white : .primary.opacity(0.85))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(
                                        Capsule()
                                            .fill(isSelected ? Color.blue.opacity(0.35) : Color.white.opacity(0.05))
                                            .overlay(
                                                Capsule()
                                                    .strokeBorder(isSelected ? Color.blue : Color.white.opacity(0.08), lineWidth: 1)
                                            )
                                    )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                HStack(spacing: 8) {
                    Image(systemName: "sparkle.magnifyingglass")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                    TextField("Or type any custom sound description (e.g. acoustic guitar, applause, siren)…", text: $viewModel.separationDescription)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12.5))
                        .disabled(viewModel.isGenerating)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.black.opacity(0.26))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                        )
                )
            }
        }
    }

    // MARK: - Action Button

    private var separateButton: some View {
        let actionTitle: String = {
            switch viewModel.studioMode {
            case .separation: return "Separate Audio Stem"
            case .enhancement: return "Enhance Speech & Remove Background Noise"
            case .languageId: return "Identify Spoken Language"
            }
        }()
        let actionIcon: String = {
            switch viewModel.studioMode {
            case .separation: return "arrow.triangle.branch"
            case .enhancement: return "wand.and.stars"
            case .languageId: return "globe"
            }
        }()
        let enabled = selectedInputURL != nil && !viewModel.isGenerating

        return Button {
            guard let url = selectedInputURL else { return }
            viewModel.startSeparation(inputURL: url)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: actionIcon)
                    .font(.system(size: 14, weight: .semibold))
                Text(actionTitle)
                    .font(.system(size: 14.5, weight: .semibold))
            }
            .foregroundStyle(enabled ? .white : .secondary)
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        enabled
                        ? LinearGradient(colors: [Color.blue, Color.indigo], startPoint: .topLeading, endPoint: .bottomTrailing)
                        : LinearGradient(colors: [Color.white.opacity(0.06), Color.white.opacity(0.04)], startPoint: .top, endPoint: .bottom)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(Color.white.opacity(enabled ? 0.24 : 0.08), lineWidth: 0.8)
                    )
                    .shadow(color: enabled ? Color.blue.opacity(0.32) : .clear, radius: 10, y: 4)
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }

    // MARK: - Spoken Language Identification Results

    private var lidResultsSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Label("Detected Spoken Language", systemImage: "globe")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    if let top = viewModel.lidTopLanguage {
                        Text(top)
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 3)
                            .background(Color.blue.opacity(0.14), in: Capsule())
                    }
                }
                ForEach(Array(viewModel.lidPredictions.enumerated()), id: \.offset) { _, pred in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(pred.language)
                                .font(.system(size: 12, weight: .medium))
                            Spacer()
                            Text(String(format: "%.1f%%", pred.confidence * 100))
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                        ProgressView(value: Double(min(max(pred.confidence, 0), 1)))
                            .tint(.blue)
                    }
                }
            }
        }
    }

    private var generationProgressView: some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
            Text(viewModel.generationProgress)
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(14)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: - Audio Player Section

    private var audioPlayerSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Studio Output Compare (A/B)", systemImage: "waveform.badge.checkmark")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Picker("Compare", selection: Binding(
                        get: { viewModel.isPlayingOriginal },
                        set: { viewModel.switchPlaybackSource(original: $0) }
                    )) {
                        Text("Processed Output").tag(false)
                        Text("Original Input").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 220)

                    Menu {
                        ForEach(AudioExportFormat.allCases) { fmt in
                            Button("Save as \(fmt.label)") {
                                viewModel.saveAudioInFormat(fmt)
                            }
                        }
                    } label: {
                        Label("Export Audio", systemImage: "square.and.arrow.down")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                VStack(spacing: 4) {
                    Slider(
                        value: Binding(
                            get: { viewModel.currentTime },
                            set: { viewModel.seek(to: $0) }
                        ),
                        in: 0...max(viewModel.duration, 1)
                    )

                    HStack {
                        Text(formatTime(viewModel.currentTime))
                        Spacer()
                        Text(formatTime(viewModel.duration))
                    }
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                }

                HStack(spacing: 20) {
                    Spacer()
                    Button {
                        viewModel.seek(to: 0)
                    } label: {
                        Image(systemName: "backward.end.fill")
                            .font(.title3)
                    }
                    .buttonStyle(.plain)

                    Button {
                        viewModel.togglePlayPause()
                    } label: {
                        Image(systemName: viewModel.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 42))
                            .foregroundStyle(.blue)
                    }
                    .buttonStyle(.plain)

                    Button {
                        viewModel.stop()
                    } label: {
                        Image(systemName: "stop.fill")
                            .font(.title3)
                    }
                    .buttonStyle(.plain)

                    Spacer()
                }
            }
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.system(size: 12))
            Spacer()
            Button {
                viewModel.errorMessage = nil
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: - Helpers

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        if provider.canLoadObject(ofClass: URL.self) {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                let dest = FileManager.default.temporaryDirectory
                    .appendingPathComponent("sts_drop_\(UUID().uuidString)_\(url.lastPathComponent)")
                if (try? FileManager.default.copyItem(at: url, to: dest)) != nil {
                    DispatchQueue.main.async {
                        selectedInputURL = dest
                    }
                } else {
                    DispatchQueue.main.async {
                        selectedInputURL = url
                    }
                }
            }
            return true
        }
        provider.loadFileRepresentation(forTypeIdentifier: UTType.audio.identifier) { url, _ in
            guard let url else { return }

            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("sts_drop_\(UUID().uuidString)_\(url.lastPathComponent)")
            try? FileManager.default.copyItem(at: url, to: dest)
            DispatchQueue.main.async {
                selectedInputURL = dest
            }
        }
        return true
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
