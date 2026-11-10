//
//  STSView.swift
//  VoicesApp
//
//  Created by Alpay Calalli on 25.02.26.
//

import SwiftUI
import UniformTypeIdentifiers

struct STSView: View {
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @State private var viewModel = STSViewModel()
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var prefs = StudioUserPreferencesManager.shared
    @State private var isFilePickerPresented = false
    @State private var selectedInputURL: URL?
    @State private var isDraggingOver = false
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var showDownloadedModelsSheet = false

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

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
                                            .strokeBorder(activeTheme.pillStroke, lineWidth: 1)
                                    )

                                Image(systemName: "tuningfork")
                                    .font(.system(size: 20, weight: .semibold))
                                    .foregroundStyle(.purple)
                            }

                            VStack(alignment: .leading, spacing: 3) {
                                Text("AI Audio Lab — Stem Separation, Denoise, Diarization & Language ID")
                                    .font(.system(size: 17, weight: .semibold))
                                Text("Isolate vocals or instruments with SAM-Audio, enhance speech with Nemotron RE-USE / MossFormer2 / DeepFilterNet, diarize up to 8 speakers with Nemotron 3 & Sortformer, or detect 107–256 spoken languages.")
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
                                    case .diarization: return "person.2.wave.2.fill"
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
                                                : LinearGradient(colors: [activeTheme.pillFill, activeTheme.recessedFill], startPoint: .top, endPoint: .bottom)
                                            )
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 11, style: .continuous)
                                                    .strokeBorder(isSelected ? Color.white.opacity(0.24) : activeTheme.pillStroke, lineWidth: 0.8)
                                            )
                                    )
                                }
                                .buttonStyle(.plain)
                                .disabled(viewModel.isGenerating || viewModel.isLoading)
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

                if viewModel.studioMode == .diarization && !viewModel.diarizationSegments.isEmpty {
                    diarizationResultsSection
                }

                if viewModel.studioMode == .languageId && !viewModel.lidPredictions.isEmpty {
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
        let base = STSViewModel.presetModels.filter { preset in
            let lower = preset.id.lowercased()
            switch viewModel.studioMode {
            case .separation:
                return lower.contains("sam-audio")
            case .enhancement:
                return lower.contains("mossformer") || lower.contains("deepfilter") || lower.contains("re-use")
            case .diarization:
                return lower.contains("diar") || lower.contains("sortformer") || lower.contains("vad")
            case .languageId:
                return lower.contains("voxlingua") || lower.contains("mms-lid")
            }
        }
        return base.sorted { a, b in
            let favA = prefs.isFavoriteSTSModel(a.id)
            let favB = prefs.isFavoriteSTSModel(b.id)
            if favA != favB { return favA && !favB }
            return false
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
                            let isFav = prefs.isFavoriteSTSModel(preset.id)
                            HStack(spacing: 5) {
                                Button {
                                    prefs.toggleFavoriteSTSModel(preset.id)
                                } label: {
                                    Image(systemName: isFav ? "star.fill" : "star")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundStyle(isFav ? (activeTheme.isLight ? .orange : .yellow) : .secondary.opacity(0.6))
                                }
                                .buttonStyle(.plain)
                                .help(isFav ? "Remove from favorite Audio Lab models" : "Add to favorite Audio Lab models")

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
                                }
                                .buttonStyle(.plain)
                                .disabled(viewModel.isLoading || viewModel.isGenerating)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(
                                Capsule()
                                    .fill(isSelected ? Color.blue.opacity(0.18) : (isFav ? Color.yellow.opacity(0.08) : activeTheme.pillFill))
                                    .overlay(
                                        Capsule()
                                            .strokeBorder(isSelected ? Color.blue.opacity(0.8) : (isFav ? Color.yellow.opacity(0.4) : activeTheme.pillStroke), lineWidth: 1)
                                    )
                            )
                            .contextMenu {
                                Button {
                                    prefs.toggleFavoriteSTSModel(preset.id)
                                } label: {
                                    Label(isFav ? "Remove from ★ Favorites" : "Add to ★ Favorites", systemImage: isFav ? "star.slash" : "star.fill")
                                }
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
                            .fill(activeTheme.recessedFill)
                            .overlay(
                                RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .strokeBorder(activeTheme.recessedStroke, lineWidth: 0.8)
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
                                .background(activeTheme.pillFill, in: Capsule())
                                .overlay(Capsule().strokeBorder(activeTheme.pillStroke, lineWidth: 0.8))
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
                HStack(spacing: 10) {
                    Label("Source Audio File", systemImage: "waveform.circle.fill")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()

                    Button {
                        if let url = viewModel.prepareModeDemoAudio() {
                            selectedInputURL = url
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.fill")
                                .font(.system(size: 10, weight: .bold))
                            Text(viewModel.studioMode == .diarization ? "Load 3-Speaker Demo" : (viewModel.studioMode == .enhancement ? "Load Noisy Speech Demo" : "Load Demo Audio"))
                                .font(.system(size: 11, weight: .semibold))
                        }
                        .foregroundStyle(.cyan)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .background(Color.cyan.opacity(0.14), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Instantly load a mode-matched test recording (multi-speaker conversation for Diarization, noisy speech for Enhancement, or studio sample)")

                    let studioSamples = Voice.bundledStudioSamples()
                    if !studioSamples.isEmpty {
                        Menu {
                            Section("Mode-Matched Test Generators") {
                                Button {
                                    if let url = viewModel.prepareModeDemoAudio() {
                                        selectedInputURL = url
                                    }
                                } label: {
                                    Label("⚡ Auto Mode Demo (\(viewModel.studioMode.rawValue))", systemImage: "bolt.fill")
                                }
                            }
                            Divider()
                            Section("Bundled Studio Voice Samples") {
                                ForEach(studioSamples, id: \.url) { sample in
                                    Button {
                                        selectInputAudio(sample.url)
                                    } label: {
                                        Label("\(sample.name) — \(sample.subtitle)", systemImage: "waveform")
                                    }
                                }
                            }
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "sparkles")
                                    .font(.system(size: 10, weight: .semibold))
                                Text("Samples")
                                    .font(.system(size: 11, weight: .semibold))
                            }
                            .foregroundStyle(.purple)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                            .background(Color.purple.opacity(0.14), in: Capsule())
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                    }

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
                            isDraggingOver ? Color.blue : activeTheme.pillStroke,
                            style: StrokeStyle(lineWidth: 1.5, dash: [6])
                        )
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(isDraggingOver
                                    ? Color.blue.opacity(0.08)
                                    : activeTheme.recessedFill)
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
                                .font(.system(size: 20))
                                .foregroundStyle(.blue.opacity(0.85))
                            Text("Drop any audio or video file (WAV, MP3, FLAC, OGG, OPUS, M4A, MP4, MKV, MOV) here, or click to browse")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.primary.opacity(0.85))
                            Button {
                                if let url = viewModel.prepareModeDemoAudio() {
                                    selectedInputURL = url
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "bolt.fill")
                                        .font(.system(size: 9.5, weight: .bold))
                                    Text("Or Click Here to Load 1-Click Demo Audio")
                                        .font(.system(size: 11, weight: .semibold))
                                }
                                .foregroundStyle(.cyan)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 3.5)
                                .background(Color.cyan.opacity(0.14), in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { isFilePickerPresented = true }
                .onDrop(of: [.fileURL, .item, .audio, .movie, .wav, .mp3, .aiff, .mpeg4Audio], isTargeted: $isDraggingOver) { providers in
                    handleDrop(providers: providers)
                }
                .fileImporter(
                    isPresented: $isFilePickerPresented,
                    allowedContentTypes: [.item, .data, .audio, .movie, .video, .wav, .mp3, .aiff, .mpeg4Audio],
                    allowsMultipleSelection: false
                ) { result in
                    if case .success(let urls) = result, let url = urls.first {
                        let scoped = url.startAccessingSecurityScopedResource()
                        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                        let dest = FileManager.default.temporaryDirectory
                            .appendingPathComponent("sts_pick_\(UUID().uuidString)_\(url.lastPathComponent)")
                        if (try? FileManager.default.copyItem(at: url, to: dest)) != nil {
                            selectInputAudio(dest)
                        } else {
                            selectInputAudio(url)
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
                    .labelsHidden()
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
                                            .fill(isSelected ? Color.blue : activeTheme.pillFill)
                                            .overlay(
                                                Capsule()
                                                    .strokeBorder(isSelected ? Color.blue : activeTheme.pillStroke, lineWidth: 1)
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
                        .fill(activeTheme.recessedFill)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(activeTheme.recessedStroke, lineWidth: 0.8)
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
            case .diarization: return "Run Speaker Diarization & VAD"
            case .languageId: return "Identify Spoken Language"
            }
        }()
        let actionIcon: String = {
            switch viewModel.studioMode {
            case .separation: return "arrow.triangle.branch"
            case .enhancement: return "wand.and.stars"
            case .diarization: return "person.2.wave.2.fill"
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
                        : LinearGradient(colors: [activeTheme.pillFill, activeTheme.recessedFill], startPoint: .top, endPoint: .bottom)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(enabled ? Color.white.opacity(0.24) : activeTheme.pillStroke, lineWidth: 0.8)
                    )
                    .shadow(color: enabled ? Color.blue.opacity(0.32) : .clear, radius: 10, y: 4)
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .keyboardShortcut(.return, modifiers: [.command])
    }

    // MARK: - Speaker Diarization & VAD Results

    private var diarizationResultsSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Label("Speaker Diarization & VAD Timeline", systemImage: "person.2.wave.2.fill")
                        .font(.system(size: 14, weight: .semibold))
                    Spacer()

                    Button {
                        #if os(macOS)
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(viewModel.diarizationRTTM, forType: .string)
                        #endif
                    } label: {
                        Label("Copy RTTM", systemImage: "doc.on.doc")
                            .font(.system(size: 11, weight: .semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3.5)
                            .background(activeTheme.pillFill, in: Capsule())
                            .overlay(Capsule().strokeBorder(activeTheme.pillStroke, lineWidth: 0.8))
                    }
                    .buttonStyle(.plain)
                    .help("Copy NIST RTTM speaker diarization timeline to clipboard")

                    Button {
                        #if os(macOS)
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(viewModel.diarizationJSONString(), forType: .string)
                        #endif
                    } label: {
                        Label("Copy JSON", systemImage: "curlybraces")
                            .font(.system(size: 11, weight: .semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3.5)
                            .background(Color.cyan.opacity(0.14), in: Capsule())
                            .foregroundStyle(.cyan)
                    }
                    .buttonStyle(.plain)
                    .help("Copy structured JSON speaker segments to clipboard for RAD integration")

                    if let url = selectedInputURL ?? viewModel.audioURL {
                        Button {
                            UserDefaults.standard.set(url.path, forKey: "VoicesApp.Transcribe.pendingAudioPath")
                            NotificationCenter.default.post(name: .sendAudioToTranscribe, object: url)
                        } label: {
                            Label("Transcribe →", systemImage: "mic.fill")
                                .font(.system(size: 11, weight: .semibold))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3.5)
                                .background(Color.blue.opacity(0.16), in: Capsule())
                                .foregroundStyle(.blue)
                        }
                        .buttonStyle(.plain)
                        .help("Send this audio recording to Transcribe Studio")
                    }

                    Text("\(viewModel.diarizationSpeakerCount) Speaker\(viewModel.diarizationSpeakerCount == 1 ? "" : "s") · \(viewModel.diarizationSegments.count) Segments")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.indigo)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 3)
                        .background(Color.indigo.opacity(0.14), in: Capsule())
                }

                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(viewModel.diarizationSegments.enumerated()), id: \.offset) { _, seg in
                            let spkColors: [Color] = [.blue, .purple, .orange, .teal, .pink, .green, .indigo]
                            let spkColor = spkColors[abs(seg.speaker.hashValue) % spkColors.count]
                            Button {
                                viewModel.seek(to: Double(seg.start))
                                if !viewModel.isPlaying {
                                    viewModel.togglePlayPause()
                                }
                            } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: "play.circle.fill")
                                        .font(.system(size: 12))
                                        .foregroundStyle(spkColor)
                                    Text("Speaker \(seg.speaker)")
                                        .font(.system(size: 11.5, weight: .semibold))
                                        .foregroundStyle(spkColor)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3)
                                        .background(spkColor.opacity(0.16), in: Capsule())

                                    Text(String(format: "%.2fs → %.2fs", seg.start, seg.end))
                                        .font(.system(size: 12, design: .monospaced))
                                        .foregroundStyle(.primary.opacity(0.9))

                                    Spacer()

                                    Text(String(format: "%.2fs", max(0, seg.end - seg.start)))
                                        .font(.system(size: 11.5, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(activeTheme.elevatedRowFill, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help("Click to jump & play Speaker \(seg.speaker) turn (\(String(format: "%.2fs", seg.start)))")
                        }
                    }
                }
                .frame(maxHeight: 240)
            }
        }
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
        .background(activeTheme.elevatedRowFill, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: - Audio Player Section

    private var audioPlayerSection: some View {
        AppleBentoCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 10) {
                    Label("Studio Output Compare (A/B)", systemImage: "waveform.badge.checkmark")
                        .font(.system(size: 14, weight: .semibold))

                    Text(viewModel.isPlayingOriginal ? "LISTENING: ORIGINAL INPUT" : "LISTENING: PROCESSED OUTPUT")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(viewModel.isPlayingOriginal ? .orange : .green)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            (viewModel.isPlayingOriginal ? Color.orange : Color.green).opacity(0.14),
                            in: Capsule()
                        )

                    Spacer()
                    Picker("Compare", selection: Binding(
                        get: { viewModel.isPlayingOriginal },
                        set: { viewModel.switchPlaybackSource(original: $0) }
                    )) {
                        Text("Processed Output").tag(false)
                        Text("Original Input").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
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

                    if let outURL = viewModel.audioURL {
                        Button {
                            UserDefaults.standard.set(outURL.path, forKey: "VoicesApp.Transcribe.pendingAudioPath")
                            NotificationCenter.default.post(name: .sendAudioToTranscribe, object: outURL)
                        } label: {
                            Label("Transcribe →", systemImage: "waveform.badge.mic")
                                .font(.system(size: 11.5, weight: .semibold))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Send this isolated/denoised audio stem directly to the Transcribe tab")

                        Button {
                            if let cloned = Voice.createInstantClone(
                                from: outURL,
                                transcription: nil,
                                suggestedName: "Audio Lab Vocal Clone"
                            ) {
                                NotificationCenter.default.post(name: .cloneVoiceFromAudio, object: cloned)
                            }
                        } label: {
                            Label("Clone Voice →", systemImage: "person.crop.circle.badge.plus")
                                .font(.system(size: 11.5, weight: .semibold))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Clone this isolated/denoised vocal stem directly into Voice Studio")
                    }
                }

                CompactAudioPlayer(
                    isPlaying: viewModel.isPlaying,
                    currentTime: viewModel.currentTime,
                    duration: viewModel.duration,
                    onPlayPause: { viewModel.togglePlayPause() },
                    onSeek: { viewModel.seek(to: $0) },
                    audioURL: viewModel.isPlayingOriginal ? selectedInputURL : viewModel.audioURL,
                    accentColor: viewModel.isPlayingOriginal ? .orange : activeTheme.activeWordBackground
                )
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(activeTheme.recessedFill)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(activeTheme.recessedStroke, lineWidth: 0.8)
                        )
                )
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

    private func selectInputAudio(_ url: URL) {
        selectedInputURL = url
        viewModel.selectInputAudio(url)
    }

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
                        selectInputAudio(dest)
                    }
                } else {
                    DispatchQueue.main.async {
                        selectInputAudio(url)
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
                selectInputAudio(dest)
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
