import SwiftUI
import UniformTypeIdentifiers

// MARK: - Compact Slider (iOS)

#if os(iOS)
struct CompactToggle: View {
    let label: String
    @Binding var isOn: Bool
    var font: Font = .footnote
    var toggleWidth: CGFloat = 40
    var toggleHeight: CGFloat = 24
    var thumbSize: CGFloat = 20
    var tint: Color = .blue

    var body: some View {
        HStack {
            Text(label)
                .font(font)
            Spacer()
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule()
                    .fill(isOn ? tint : Color.gray.opacity(0.3))
                    .frame(width: toggleWidth, height: toggleHeight)

                Circle()
                    .fill(.white)
                    .shadow(color: .black.opacity(0.15), radius: 1, x: 0, y: 1)
                    .frame(width: thumbSize, height: thumbSize)
                    .padding(2)
            }
            .animation(.easeInOut(duration: 0.15), value: isOn)
            .onTapGesture {
                isOn.toggle()
            }
        }
    }
}

struct CompactSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double? = nil
    var thumbSize: CGFloat = 16
    var trackHeight: CGFloat = 4
    var tint: Color = .blue

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let thumbOffset = (value - range.lowerBound) / (range.upperBound - range.lowerBound) * (width - thumbSize)

            ZStack(alignment: .leading) {
                // Track background
                Capsule()
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: trackHeight)

                // Track fill
                Capsule()
                    .fill(tint)
                    .frame(width: thumbOffset + thumbSize / 2, height: trackHeight)

                // Thumb
                Circle()
                    .fill(.white)
                    .shadow(color: .black.opacity(0.15), radius: 2, x: 0, y: 1)
                    .frame(width: thumbSize, height: thumbSize)
                    .offset(x: thumbOffset)
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { gesture in
                                let newValue = range.lowerBound + (gesture.location.x / width) * (range.upperBound - range.lowerBound)
                                let clampedValue = min(max(newValue, range.lowerBound), range.upperBound)
                                if let step = step {
                                    value = (clampedValue / step).rounded() * step
                                } else {
                                    value = clampedValue
                                }
                            }
                    )
            }
        }
        .frame(height: thumbSize)
    }
}
#endif

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @State private var viewModel = TTSViewModel()
    @State private var textInput: String = {
        if ProcessInfo.processInfo.arguments.contains("--read-along") {
            return """
            Welcome to Codecaine MLX RAD AI Audio Studio on Apple Silicon. As neural speech plays or generates, every word highlights in real time with acoustic waveform alignment and automatic teleprompter scrolling.

            Large books, PDFs, EPUBs, Markdown documents, and transcripts of any size are automatically split into natural paragraph, sentence, and clause chunks so you can read unlimited text with zero memory spikes.

            Click any chunk pill above to synthesize from that section immediately, or click any word in Read-Along mode to jump directly to that moment in the audio timeline.
            """
        }
        return ""
    }()
    @State private var showVoices = ProcessInfo.processInfo.arguments.contains("--open-voices")
    @State private var showSettings = false
    @State private var showDownloadedModelsSheet = false
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var showHistory = false
    @State private var showDocImporter = false
    @State private var showTeleprompter = ProcessInfo.processInfo.arguments.contains("--read-along")
    @State private var previewWordId: Int? = ProcessInfo.processInfo.arguments.contains("--read-along") ? 6 : nil
    @State private var isDropTargeted = false
    @State private var cleanOptions = SpeechCleanOptions()
    @State private var recentlyUsed: [Voice] = Voice.samples
    @State private var customVoices: [Voice] = Voice.loadCustomVoices()

    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .title3
    private let inputFont: Font = .title2

    private var wordCount: Int {
        textChunks.reduce(0) { $0 + $1.wordCount }
    }

    private var estimatedSeconds: Int {
        max(0, Int(ceil(Double(wordCount) / 2.5)))
    }

    private var textChunks: [PhononTextChunk] {
        guard viewModel.enableChunking else {
            let trimmed = textInput.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return [] }
            let words = trimmed.split { $0.isWhitespace || $0.isNewline }.count
            return [PhononTextChunk(id: 0, index: 0, paragraphIndex: 0, text: trimmed, charCount: trimmed.count, wordCount: words)]
        }
        return HierarchicalChunker.splitText(textInput, maxChars: max(180, viewModel.maxChunkLength), mergeShortParagraphs: true)
    }

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? viewModel.readAlongTheme
    }

    private func importDocumentAsync(from url: URL) {
        let fileName = url.lastPathComponent
        viewModel.errorMessage = nil
        viewModel.generationProgress = "Importing \(fileName)…"

        let scoped = url.startAccessingSecurityScopedResource()
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("VoicesAppDocImports", isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let stagedURL = tempDir.appendingPathComponent("\(UUID().uuidString)_\(fileName)")
        do {
            try FileManager.default.linkItem(at: url, to: stagedURL)
        } catch {
            try? FileManager.default.copyItem(at: url, to: stagedURL)
        }
        if scoped {
            url.stopAccessingSecurityScopedResource()
        }
        let targetURL = FileManager.default.fileExists(atPath: stagedURL.path) ? stagedURL : url

        Task.detached(priority: .userInitiated) {
            do {
                let extracted = try SpeechTextCleaner.extractDocumentText(from: targetURL)
                try? FileManager.default.removeItem(at: stagedURL)
                let trimmed = extracted.trimmingCharacters(in: .whitespacesAndNewlines)
                await MainActor.run {
                    self.viewModel.generationProgress = ""
                    if !trimmed.isEmpty {
                        self.viewModel.stop()
                        self.viewModel.audioURL = nil
                        self.viewModel.duration = 0
                        self.viewModel.synthesizedWordTimings = []
                        self.viewModel.activeChunkIndex = 0
                        self.viewModel.synthesizingChunkIndex = 0
                        self.viewModel.synthesizedStartChunkIndex = 0
                        self.viewModel.synthesizedEndChunkIndex = nil
                        self.previewWordId = nil
                        self.textInput = extracted
                        self.viewModel.errorMessage = nil
                        if UserDefaults.standard.bool(forKey: "VoicesApp.VoiceStudio.autoSynthesizeOnImport") {
                            UserDefaults.standard.removeObject(forKey: "VoicesApp.VoiceStudio.autoSynthesizeOnImport")
                            self.showTeleprompter = true
                            Task { @MainActor in
                                for _ in 0..<60 {
                                    if self.viewModel.isModelLoaded { break }
                                    try? await Task.sleep(for: .milliseconds(250))
                                }
                                if self.viewModel.isModelLoaded {
                                    self.viewModel.startSynthesis(
                                        text: self.textInput,
                                        voice: self.viewModel.selectedVoice,
                                        startChunkIndex: 0
                                    )
                                }
                            }
                        }
                    } else {
                        self.viewModel.errorMessage = "No readable text found in \(fileName)."
                    }
                }
            } catch {
                try? FileManager.default.removeItem(at: stagedURL)
                await MainActor.run {
                    self.viewModel.generationProgress = ""
                    self.viewModel.errorMessage = "Failed to import \(fileName): \(error.localizedDescription)"
                }
            }
        }
    }

    var body: some View {
        let theme = activeTheme
        let chunks = textChunks
        let totalWords = chunks.reduce(0) { $0 + $1.wordCount }
        let timings: [WordTiming] = {
            guard !textInput.isEmpty && (showTeleprompter || viewModel.isGenerating || viewModel.isPlaying || viewModel.audioURL != nil) else {
                return []
            }
            if viewModel.synthesizedWordTimings.count == totalWords,
               let firstPlayable = viewModel.synthesizedWordTimings.first(where: { $0.end > $0.start }),
               firstPlayable.end > firstPlayable.start {
                return viewModel.synthesizedWordTimings
            }
            return WordTimingEngine.computeWordTimings(
                text: textInput,
                audioURL: viewModel.audioURL,
                duration: viewModel.duration > 0.05 ? viewModel.duration : Double(max(estimatedSeconds, 2)),
                maxChunkChars: max(180, viewModel.maxChunkLength),
                enableChunking: viewModel.enableChunking,
                startChunkIndex: viewModel.synthesizedStartChunkIndex,
                endChunkIndex: viewModel.synthesizedEndChunkIndex,
                paragraphPauseMs: viewModel.paragraphPauseMs
            )
        }()
        let activeWordId: Int? = {
            if viewModel.isSpeaking || (viewModel.isPlaying && viewModel.duration > 0 && viewModel.currentTime > 0) {
                return WordTimingEngine.activeWordIndex(
                    in: timings,
                    at: viewModel.currentTime,
                    offsetMs: viewModel.wordSyncOffsetMs
                )
            }
            if viewModel.isGenerating {
                if viewModel.currentTime > 0 {
                    return WordTimingEngine.activeWordIndex(
                        in: timings,
                        at: viewModel.currentTime,
                        offsetMs: viewModel.wordSyncOffsetMs
                    )
                }
                return timings.first(where: { $0.chunkIndex == viewModel.activeChunkIndex })?.id ?? timings.first?.id
            }
            if showTeleprompter {
                if let pid = previewWordId, pid >= 0 && pid < timings.count {
                    return pid
                }
                return timings.first(where: { $0.chunkIndex == viewModel.activeChunkIndex })?.id ?? timings.first?.id
            }
            return nil
        }()
        let activeChunkIdx: Int = {
            if let wid = activeWordId {
                if wid >= 0 && wid < timings.count && timings[wid].id == wid {
                    return timings[wid].chunkIndex
                }
                if let match = timings.first(where: { $0.id == wid }) {
                    return match.chunkIndex
                }
            }
            return viewModel.activeChunkIndex
        }()

        VStack(spacing: 10) {
            // 1. Apple Sequoia Studio Ribbon (Engine Status Pill + Grouped Glass Action Bar)
            HStack(spacing: 12) {
                // Engine & Cache Status Capsule
                HStack(spacing: 8) {
                    Circle()
                        .fill(viewModel.isModelLoaded ? Color.green : (viewModel.isLoading ? Color.orange : Color.gray))
                        .frame(width: 8, height: 8)
                        .shadow(color: (viewModel.isModelLoaded ? Color.green : Color.orange).opacity(0.6), radius: 4)

                    Text(viewModel.isModelLoaded ? "Engine Ready" : (viewModel.isLoading ? "Loading Model…" : "Engine Idle"))
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary.opacity(0.92))

                    if modelsManager.isModelCached(viewModel.modelId) {
                        let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId)
                        HStack(spacing: 3) {
                            Image(systemName: "checkmark.seal.fill")
                                .font(.system(size: 10))
                            Text(activeSize.map { "Cached · \($0)" } ?? "Cached")
                                .font(.system(size: 10.5, weight: .semibold))
                        }
                        .foregroundStyle(.green)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background(Color.green.opacity(0.14), in: Capsule())
                        .contextMenu {
                            Button {
                                modelsManager.refresh()
                                showDownloadedModelsSheet = true
                            } label: {
                                Label("Manage Downloaded Models…", systemImage: "internaldrive.fill")
                            }
                            Button(role: .destructive) {
                                let name = TTSViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
                                pendingDeleteModel = (name: name, id: viewModel.modelId, size: activeSize ?? "Cached")
                            } label: {
                                Label("Delete Cached Model (\(activeSize ?? "Cached"))…", systemImage: "trash")
                            }
                        }
                    }

                    if viewModel.isModelLoaded {
                        Divider()
                            .frame(height: 12)
                        Button {
                            viewModel.unloadModelAndPurgeVRAM()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "memorychip")
                                    .font(.system(size: 10.5, weight: .semibold))
                                Text("Purge VRAM")
                                    .font(.system(size: 11, weight: .medium))
                            }
                            .foregroundStyle(.orange)
                        }
                        .buttonStyle(.plain)
                        .help("Unload model & free Apple Silicon Metal memory immediately (keep_alive: 0)")
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(
                    Capsule()
                        .fill(Color.white.opacity(0.045))
                        .overlay(
                            Capsule()
                                .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                        )
                )
                .fixedSize()

                Spacer(minLength: 10)

                // Segmented Edit Script vs Read-Along Karaoke Switcher
                HStack(spacing: 2) {
                    Button {
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                            showTeleprompter = false
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "square.and.pencil")
                                .font(.system(size: 11))
                            Text("Edit Script")
                                .font(.system(size: 11.5, weight: !showTeleprompter ? .semibold : .medium))
                        }
                        .foregroundStyle(!showTeleprompter ? .white : .secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5.5)
                        .background(!showTeleprompter ? Color.white.opacity(0.16) : Color.clear, in: Capsule())
                    }
                    .buttonStyle(.plain)

                    Button {
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                            if textInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                textInput = """
                                Welcome to Codecaine Voice Studio on Apple Silicon MLX. As neural speech plays or generates, every word highlights in real time with acoustic waveform alignment and automatic teleprompter scrolling.

                                Large books, PDFs, EPUBs, Markdown documents, and transcripts of any size are automatically split into natural paragraph, sentence, and clause chunks so you can read unlimited text with zero memory spikes.

                                Click any chunk pill above to synthesize from that section immediately, or click any word in Read-Along mode to jump directly to that moment in the audio timeline.
                                """
                            }
                            showTeleprompter = true
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "text.word.spacing")
                                .font(.system(size: 11))
                            Text("Read-Along Karaoke")
                                .font(.system(size: 11.5, weight: showTeleprompter ? .semibold : .medium))
                        }
                        .foregroundStyle(showTeleprompter ? .white : .secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5.5)
                        .background(showTeleprompter ? Color.blue.opacity(0.65) : Color.clear, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(3)
                .background(
                    Capsule()
                        .fill(Color.black.opacity(0.28))
                        .overlay(
                            Capsule()
                                .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                        )
                )
                .fixedSize()

                // Unified Apple Glass Action Capsule Group
                HStack(spacing: 2) {
                    // Import Document (.pdf, .epub, .docx, .rtf, .md, .html, .srt, .ipynb, .csv, .json, .png/.jpg OCR)
                    Button {
                        showDocImporter = true
                    } label: {
                        Label("Import", systemImage: "doc.text.magnifyingglass")
                            .font(.system(size: 12, weight: .medium))
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Import PDF, EPUB, DOCX, ODT, Jupyter .ipynb, CSV/TSV, JSON, SRT/VTT, Markdown, HTML, or Image OCR")

                    Divider().frame(height: 14).opacity(0.35)

                    // Smart Speech Cleanup Filters
                    Menu {
                        Button {
                            textInput = SpeechTextCleaner.clean(textInput, options: cleanOptions)
                        } label: {
                            Label("Clean Text for Speech Now", systemImage: "wand.and.stars")
                        }
                        Divider()
                        Toggle("Strip Markdown (#, **, tables)", isOn: $cleanOptions.stripMarkdown)
                        Toggle("Strip HTML Tags & Entities", isOn: $cleanOptions.stripHtml)
                        Toggle("Strip Code Blocks (```)", isOn: $cleanOptions.stripCodeBlocks)
                        Toggle("Simplify URLs to Domain", isOn: $cleanOptions.simplifyUrls)
                        Toggle("Strip SRT/VTT/LRC Timestamps", isOn: $cleanOptions.stripTimestamps)
                        Toggle("Strip ASCII Divider Lines", isOn: $cleanOptions.stripDividers)
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "wand.and.stars")
                                .font(.system(size: 11.5))
                            Text("Clean")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Divider().frame(height: 14).opacity(0.35)

                    // Templates & Calibration Scripts menu
                    Menu {
                        Section("Studio & Calibration Scripts") {
                            ForEach(Voice.promptTemplates) { tpl in
                                Button {
                                    textInput = tpl.text
                                } label: {
                                    Label("\(tpl.title) — \(tpl.subtitle)", systemImage: tpl.icon)
                                }
                            }
                        }
                        Divider()
                        Button(role: .destructive) {
                            textInput = ""
                        } label: {
                            Label("Clear Editor", systemImage: "trash")
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "text.book.closed")
                                .font(.system(size: 11.5))
                            Text("Scripts")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Divider().frame(height: 14).opacity(0.35)

                    // Direct Silent Audio, Read-Along MP4 Video, Subtitles & Batch Export menu
                    Menu {
                        Section("Direct Silent Audio Export (No Playback Needed)") {
                            ForEach(AudioExportFormat.allCases) { fmt in
                                Button {
                                    viewModel.saveAudioInFormat(
                                        fmt,
                                        fallbackText: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: activeChunkIdx,
                                        scopeOverride: viewModel.synthesisScope
                                    )
                                } label: {
                                    Label("Export Audio as \(fmt.label) (Silent)", systemImage: "waveform.badge.plus")
                                }
                            }
                        }
                        Divider()
                        Section("Instant 2s Active Chunk (§\(activeChunkIdx + 1)) Silent Export") {
                            ForEach([AudioExportFormat.wav, .mp3, .m4a]) { fmt in
                                Button {
                                    viewModel.quickExportActiveChunk(
                                        text: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: activeChunkIdx,
                                        format: fmt,
                                        scope: .activeChunk
                                    )
                                } label: {
                                    Label("⚡ 2s Export Chunk §\(activeChunkIdx + 1) as \(fmt.rawValue.uppercased()) (Silent)", systemImage: "bolt.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Synchronized Read-Along Karaoke Video (.MP4)") {
                            ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                Button {
                                    viewModel.exportReadAlongVideo(
                                        text: textInput,
                                        theme: viewModel.readAlongTheme,
                                        aspect: aspect
                                    )
                                } label: {
                                    Label("Export Read-Along Video — \(aspect.label)", systemImage: "film.stack")
                                }
                            }
                        }
                        Divider()
                        Section("Karaoke & Read-Along Subtitles") {
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "srt")
                            } label: {
                                Label("Export Subtitles (.SRT)", systemImage: "captions.bubble")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "vtt")
                            } label: {
                                Label("Export WebVTT (.VTT)", systemImage: "captions.bubble.fill")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "ass")
                            } label: {
                                Label("Export Karaoke Subtitles (.ASS)", systemImage: "music.mic")
                            }
                        }
                        Divider()
                        Section("Batch Line-by-Line Synthesis") {
                            Button {
                                viewModel.batchExportLines(text: textInput, voice: viewModel.selectedVoice)
                            } label: {
                                Label("Batch Export Every Line as WAV…", systemImage: "square.stack.3d.down.right")
                            }
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 11.5))
                            Text("Export")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Divider().frame(height: 14).opacity(0.35)

                    // Takes History button
                    Button {
                        showHistory = true
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "clock.arrow.circlepath")
                                .font(.system(size: 11.5))
                            Text("Takes")
                                .font(.system(size: 12, weight: .medium))
                            if !viewModel.history.isEmpty {
                                Text("\(viewModel.history.count)")
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Color.blue, in: Capsule())
                            }
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 3)
                .background(
                    Capsule()
                        .fill(Color.white.opacity(0.045))
                        .overlay(
                            Capsule()
                                .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                        )
                )
                .fixedSize()
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)

            // 1B. Codecaine Prosody, Cinema Theme & Hierarchical Chunk Navigator Ribbon
            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    // Cinema Theme Picker
                    Menu {
                        Section("Studio & Read-Along Cinema Theme") {
                            ForEach(ReadAlongTheme.allCases) { item in
                                Button {
                                    withAnimation(.easeInOut(duration: 0.22)) {
                                        rawStudioTheme = item.rawValue
                                        viewModel.readAlongTheme = item
                                    }
                                } label: {
                                    if item == theme {
                                        Label(item.label, systemImage: "checkmark")
                                    } else {
                                        Label(item.label, systemImage: item.icon)
                                    }
                                }
                            }
                        }
                        Divider()
                        Toggle("Auto-Open Read-Along When Playing", isOn: $viewModel.autoReadAlong)
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: theme.icon)
                                .font(.system(size: 10.5))
                                .foregroundStyle(theme.activeWordBackground)
                            Text(theme.label)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.primary)
                            Image(systemName: "chevron.down")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            Capsule()
                                .fill(theme.activeWordBackground.opacity(0.16))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(theme.activeWordBackground.opacity(0.38), lineWidth: 0.8)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Speed Control Menu
                    Menu {
                        Section("Playback & Synthesis Speed") {
                            ForEach([Float(0.5), 0.75, 0.9, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0], id: \.self) { rate in
                                let label = String(format: "%.2gx Speed", rate)
                                Button {
                                    viewModel.playbackRate = rate
                                } label: {
                                    if abs(viewModel.playbackRate - rate) < 0.01 {
                                        Label(label, systemImage: "checkmark")
                                    } else {
                                        Text(label)
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "gauge.with.dots.needle.67percent")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.blue)
                            Text(String(format: "Speed %.2gx", viewModel.playbackRate))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.white.opacity(0.05), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Pitch Semitone Control Menu
                    Menu {
                        Section("Acoustic Pitch Shift (Semitones)") {
                            ForEach([Float(-6), -4, -2, -1, 0, 1, 2, 4, 6], id: \.self) { st in
                                let label = st == 0 ? "0 st (Natural Pitch)" : String(format: "%+.0f st", st)
                                Button {
                                    viewModel.pitchSemitones = st
                                } label: {
                                    if abs(viewModel.pitchSemitones - st) < 0.05 {
                                        Label(label, systemImage: "checkmark")
                                    } else {
                                        Text(label)
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "tuningfork")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.purple)
                            Text(viewModel.pitchSemitones == 0 ? "Pitch 0st" : String(format: "Pitch %+.0fst", viewModel.pitchSemitones))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.white.opacity(0.05), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Paragraph Pause Control Menu
                    Menu {
                        Section("Inter-Chunk / Paragraph Pause") {
                            ForEach([0, 150, 280, 350, 450, 750, 1200, 2000], id: \.self) { ms in
                                let label = "\(ms) ms Pause"
                                Button {
                                    viewModel.paragraphPauseMs = ms
                                } label: {
                                    if viewModel.paragraphPauseMs == ms {
                                        Label(label, systemImage: "checkmark")
                                    } else {
                                        Text(label)
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "pause.rectangle")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.indigo)
                            Text("Pause \(viewModel.paragraphPauseMs)ms")
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.white.opacity(0.05), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Word Sync Offset Control Menu
                    Menu {
                        Section("Karaoke Word Highlight Sync Offset") {
                            ForEach([-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0], id: \.self) { ms in
                                let label = ms == 0 ? "0 ms (Auto RMS Sync)" : String(format: "%+.0f ms", ms)
                                Button {
                                    viewModel.wordSyncOffsetMs = ms
                                } label: {
                                    if abs(viewModel.wordSyncOffsetMs - ms) < 1.0 {
                                        Label(label, systemImage: "checkmark")
                                    } else {
                                        Text(label)
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "metronome")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.teal)
                            Text(String(format: "Sync %+.0fms", viewModel.wordSyncOffsetMs))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.white.opacity(0.05), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Max Chunk Characters Control Menu
                    Menu {
                        Section("Unlimited Hierarchical Chunk Size (Paragraph → Sentence → Clause)") {
                            ForEach([200, 350, 520, 750, 1000, 1400], id: \.self) { maxC in
                                let label = maxC == 520 ? "520 chars / chunk (Codecaine Default)" : "\(maxC) chars / chunk"
                                Button {
                                    viewModel.maxChunkLength = maxC
                                } label: {
                                    if viewModel.maxChunkLength == maxC {
                                        Label(label, systemImage: "checkmark")
                                    } else {
                                        Text(label)
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "rectangle.split.3x1")
                                .font(.system(size: 10.5))
                                .foregroundStyle(.orange)
                            Text("Chunk \(viewModel.maxChunkLength)c")
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.white.opacity(0.05), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Codecaine File Manager 2-Second Synthesis Scope Picker
                    Menu {
                        Section("EPUB & Document Synthesis Speed / Scope") {
                            ForEach(SynthesisScope.allCases) { scope in
                                Button {
                                    viewModel.synthesisScope = scope
                                } label: {
                                    if viewModel.synthesisScope == scope {
                                        Label(scope.label, systemImage: "checkmark")
                                    } else {
                                        Text(scope.label)
                                    }
                                }
                            }
                        }
                        Divider()
                        Button {
                            viewModel.quickExportActiveChunk(
                                text: textInput,
                                voice: viewModel.selectedVoice,
                                chunkIndex: activeChunkIdx,
                                format: .wav
                            )
                        } label: {
                            Label("⚡ Instant 2s Synthesize & Export Active Chunk (§\(activeChunkIdx + 1))…", systemImage: "bolt.fill")
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.yellow)
                            Text(viewModel.synthesisScope.shortBadge)
                                .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.yellow.opacity(0.12), in: Capsule())
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("Codecaine File Manager 2-second synthesis & instant WAV export mode")

                    Spacer()

                    // Live Hierarchical Chunk Navigator Strip (§1 · §2 · §3...)
                    if !chunks.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            LazyHStack(spacing: 5) {
                                Text("CHUNKS (\(chunks.count))")
                                    .font(.system(size: 9.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(.secondary)

                                ForEach(chunks) { chunk in
                                    let isCurrent = chunk.index == activeChunkIdx
                                    Button {
                                        viewModel.activeChunkIndex = chunk.index
                                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                                            showTeleprompter = true
                                        }
                                        if let firstWord = timings.first(where: { $0.chunkIndex == chunk.index }) {
                                            previewWordId = firstWord.id
                                            if viewModel.audioURL != nil && firstWord.end > firstWord.start {
                                                viewModel.seek(to: max(0, firstWord.start + viewModel.wordSyncOffsetMs / 1000.0))
                                            }
                                        }
                                        #if os(macOS)
                                        if !viewModel.isGenerating && viewModel.isModelLoaded && NSEvent.modifierFlags.contains(.option) {
                                            viewModel.startSynthesis(
                                                text: textInput,
                                                voice: viewModel.selectedVoice,
                                                startChunkIndex: chunk.index
                                            )
                                        }
                                        #endif
                                    } label: {
                                        HStack(spacing: 4) {
                                            Text("§\(chunk.index + 1)")
                                                .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                            Text("\(chunk.wordCount)w")
                                                .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                                                .opacity(0.8)
                                        }
                                        .foregroundStyle(isCurrent ? .white : .primary.opacity(0.85))
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3.5)
                                        .background(
                                            Capsule()
                                                .fill(isCurrent ? Color.blue.opacity(0.65) : Color.white.opacity(0.06))
                                                .overlay(
                                                    Capsule()
                                                        .strokeBorder(isCurrent ? Color.blue : Color.white.opacity(0.10), lineWidth: 0.8)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                    .help("Chunk §\(chunk.index + 1) (\(chunk.wordCount) words, \(chunk.charCount) chars) — Click to jump, Option-Click to synthesize from §\(chunk.index + 1)")
                                }
                            }
                        }
                        .frame(maxWidth: 340, maxHeight: 28)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.horizontal, 20)

            // 2. Inset Apple Editorial Script Canvas Card
            ZStack(alignment: .bottomTrailing) {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(theme.canvasBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: isDropTargeted
                                        ? [Color.blue, Color.cyan]
                                        : theme.canvasBorderColors,
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: isDropTargeted ? 2 : 1.2
                            )
                    )
                    .shadow(color: .black.opacity(theme.colorScheme == .light ? 0.12 : 0.28), radius: 14, y: 6)
                    .animation(.easeInOut(duration: 0.22), value: rawStudioTheme)

                if showTeleprompter && !timings.isEmpty {
                    KaraokeTeleprompterView(
                        timings: timings,
                        chunks: chunks,
                        activeWordId: activeWordId,
                        activeChunkIndex: activeChunkIdx,
                        theme: theme,
                        onSelectWord: { timing in
                            previewWordId = timing.id
                            viewModel.activeChunkIndex = timing.chunkIndex
                            if viewModel.audioURL != nil && timing.end > timing.start {
                                viewModel.seek(to: max(0, timing.start + viewModel.wordSyncOffsetMs / 1000.0))
                            }
                        },
                        onSelectChunk: { chunkIdx in
                            let safeIdx = min(max(0, chunkIdx), max(0, chunks.count - 1))
                            viewModel.activeChunkIndex = safeIdx
                            if let firstWord = timings.first(where: { $0.chunkIndex == safeIdx }) {
                                previewWordId = firstWord.id
                                if viewModel.audioURL != nil && firstWord.end > firstWord.start {
                                    viewModel.seek(to: max(0, firstWord.start + viewModel.wordSyncOffsetMs / 1000.0))
                                }
                            }
                        },
                        onSynthesizeFromChunk: { chunkIdx in
                            viewModel.activeChunkIndex = chunkIdx
                            viewModel.startSynthesis(
                                text: textInput,
                                voice: viewModel.selectedVoice,
                                startChunkIndex: chunkIdx
                            )
                        },
                        onQuickExportChunk: { chunkIdx in
                            viewModel.quickExportActiveChunk(
                                text: textInput,
                                voice: viewModel.selectedVoice,
                                chunkIndex: chunkIdx,
                                format: .wav
                            )
                        }
                    )
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 52)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if textInput.count > 20_000 && chunks.count > 16 {
                    // Large Book / EPUB Sectioned Editor Mode (prevents NSTextView from stalling on 500k+ char books)
                    let chunksPerSection = 12
                    let totalSections = max(1, (chunks.count + chunksPerSection - 1) / chunksPerSection)
                    let currentSection = min(totalSections - 1, max(0, activeChunkIdx / chunksPerSection))
                    let sectionStartIdx = currentSection * chunksPerSection
                    let sectionEndIdx = min(chunks.count - 1, sectionStartIdx + chunksPerSection - 1)
                    let sectionChunks = Array(chunks[sectionStartIdx...sectionEndIdx])
                    let sectionWords = sectionChunks.reduce(0) { $0 + $1.wordCount }
                    let sectionBinding = Binding<String>(
                        get: {
                            sectionChunks.map(\.text).joined(separator: "\n\n")
                        },
                        set: { newSectionText in
                            var updatedAll = chunks.map(\.text)
                            let replacementParts = [newSectionText.trimmingCharacters(in: .whitespacesAndNewlines)]
                            updatedAll.replaceSubrange(sectionStartIdx...sectionEndIdx, with: replacementParts)
                            textInput = updatedAll.joined(separator: "\n\n")
                        }
                    )

                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 10) {
                            HStack(spacing: 6) {
                                Image(systemName: "book.pages.fill")
                                    .font(.system(size: 11.5, weight: .bold))
                                    .foregroundStyle(theme.activeWordBackground)
                                Text("LARGE BOOK / EPUB STUDIO")
                                    .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(theme.activeWordBackground)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(theme.activeWordBackground.opacity(0.16), in: Capsule())

                            Text("Section \(currentSection + 1) of \(totalSections) · Chunks §\(sectionStartIdx + 1)–§\(sectionEndIdx + 1) of §\(chunks.count) (\(sectionWords)w in section · \(totalWords)w total)")
                                .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)

                            Spacer()

                            Button {
                                viewModel.activeChunkIndex = max(0, (currentSection - 1) * chunksPerSection)
                            } label: {
                                Label("Prev Section", systemImage: "chevron.left")
                                    .font(.system(size: 11, weight: .semibold))
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 4)
                                    .background(Color.white.opacity(0.07), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(currentSection <= 0)

                            Button {
                                viewModel.activeChunkIndex = min(chunks.count - 1, (currentSection + 1) * chunksPerSection)
                            } label: {
                                Label("Next Section", systemImage: "chevron.right")
                                    .font(.system(size: 11, weight: .semibold))
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 4)
                                    .background(Color.white.opacity(0.07), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(currentSection >= totalSections - 1)

                            Button {
                                viewModel.activeChunkIndex = sectionStartIdx
                                if viewModel.autoReadAlong {
                                    withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                                        showTeleprompter = true
                                    }
                                }
                                viewModel.startSynthesis(
                                    text: textInput,
                                    voice: viewModel.selectedVoice,
                                    startChunkIndex: sectionStartIdx
                                )
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "play.fill")
                                        .font(.system(size: 9.5))
                                    Text("Synthesize §\(sectionStartIdx + 1) (2s)")
                                        .font(.system(size: 11, weight: .semibold))
                                }
                                .foregroundStyle(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4.5)
                                .background(theme.activeWordBackground, in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(viewModel.isGenerating || !viewModel.isModelLoaded)

                            Button {
                                viewModel.quickExportActiveChunk(
                                    text: textInput,
                                    voice: viewModel.selectedVoice,
                                    chunkIndex: sectionStartIdx,
                                    format: .wav
                                )
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "bolt.fill")
                                        .font(.system(size: 9.5))
                                    Text("2s Export §\(sectionStartIdx + 1)")
                                        .font(.system(size: 11, weight: .semibold))
                                }
                                .foregroundStyle(.yellow)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4.5)
                                .background(Color.yellow.opacity(0.16), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(viewModel.isGenerating || !viewModel.isModelLoaded)
                        }
                        .padding(.horizontal, 18)
                        .padding(.top, 14)

                        TextEditor(text: sectionBinding)
                            .font(.system(size: 18, weight: .regular))
                            .foregroundStyle(theme.editorTextColor)
                            .tint(theme.activeWordBackground)
                            .lineSpacing(5)
                            .scrollContentBackground(.hidden)
                            .background(.clear)
                            .disabled(viewModel.isGenerating)
                            .padding(.horizontal, 18)
                            .padding(.bottom, 44)
                    }
                } else {
                    TextEditor(text: $textInput)
                        .font(.system(size: 19, weight: .regular))
                        .foregroundStyle(theme.editorTextColor)
                        .tint(theme.activeWordBackground)
                        .lineSpacing(5)
                        .scrollContentBackground(.hidden)
                        .background(.clear)
                        .disabled(viewModel.isGenerating)
                        .padding(.horizontal, 18)
                        .padding(.top, 16)
                        .padding(.bottom, 44)
                        .overlay(alignment: .topLeading) {
                            if textInput.isEmpty {
                                VStack(alignment: .leading, spacing: 22) {
                                    Text("Write or paste your script, or choose a studio starter below…")
                                        .font(.system(size: 19, weight: .regular))
                                        .foregroundStyle(theme.upcomingWordForeground)
                                        .allowsHitTesting(false)

                                    // Apple Quick-Start Bento Cards inside the empty canvas
                                    VStack(alignment: .leading, spacing: 10) {
                                        Text("QUICK START STUDIO SCRIPTS · \(theme.label.uppercased()) THEME")
                                            .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                            .foregroundStyle(theme.activeWordBackground)

                                        HStack(spacing: 12) {
                                            ForEach(Voice.promptTemplates.prefix(3)) { tpl in
                                                Button {
                                                    withAnimation(.easeOut(duration: 0.15)) {
                                                        textInput = tpl.text
                                                    }
                                                } label: {
                                                    VStack(alignment: .leading, spacing: 6) {
                                                        HStack(spacing: 6) {
                                                            Image(systemName: tpl.icon)
                                                                .font(.system(size: 12, weight: .semibold))
                                                                .foregroundStyle(theme.activeWordBackground)
                                                            Text(tpl.title)
                                                                .font(.system(size: 12.5, weight: .semibold))
                                                                .foregroundStyle(theme.editorTextColor)
                                                                .lineLimit(1)
                                                        }
                                                        Text(tpl.subtitle)
                                                            .font(.system(size: 11))
                                                            .foregroundStyle(.secondary)
                                                            .lineLimit(1)
                                                        Text(tpl.text)
                                                            .font(.system(size: 11))
                                                            .foregroundStyle(theme.upcomingWordForeground)
                                                            .lineLimit(2)
                                                            .multilineTextAlignment(.leading)
                                                    }
                                                    .padding(12)
                                                    .frame(maxWidth: .infinity, alignment: .leading)
                                                    .background(
                                                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                                                            .fill(theme.cardBackground)
                                                            .overlay(
                                                                RoundedRectangle(cornerRadius: 13, style: .continuous)
                                                                    .strokeBorder(theme.activeWordBackground.opacity(0.24), lineWidth: 0.8)
                                                            )
                                                    )
                                                }
                                                .buttonStyle(.plain)
                                            }

                                            // Import Document Card
                                            Button {
                                                showDocImporter = true
                                            } label: {
                                                VStack(alignment: .leading, spacing: 6) {
                                                    HStack(spacing: 6) {
                                                        Image(systemName: "doc.badge.plus")
                                                            .font(.system(size: 12, weight: .semibold))
                                                            .foregroundStyle(theme.activeWordBackground)
                                                        Text("Import Document")
                                                            .font(.system(size: 12.5, weight: .semibold))
                                                            .foregroundStyle(theme.editorTextColor)
                                                    }
                                                    Text("PDF · Markdown · SRT · HTML")
                                                        .font(.system(size: 11))
                                                        .foregroundStyle(.secondary)
                                                    Text("Extract and clean any document for natural speech synthesis.")
                                                        .font(.system(size: 11))
                                                        .foregroundStyle(theme.upcomingWordForeground)
                                                        .lineLimit(2)
                                                        .multilineTextAlignment(.leading)
                                                }
                                                .padding(12)
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                                .background(
                                                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                                                        .fill(theme.activeWordBackground.opacity(0.11))
                                                        .overlay(
                                                            RoundedRectangle(cornerRadius: 13, style: .continuous)
                                                                .strokeBorder(theme.activeWordBackground.opacity(0.32), lineWidth: 0.8)
                                                        )
                                                )
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                    .padding(.top, 12)
                                }
                                .padding(.horizontal, 22)
                                .padding(.top, 20)
                            }
                        }
                }

                // Floating Bottom Bar inside Canvas: Director Tags on Left + Telemetry Pill on Right
                HStack(spacing: 12) {
                    // Expressive & Multi-Speaker Performance Director Tags
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 5) {
                            Text("DIRECTOR TAGS")
                                .font(.system(size: 9.5, weight: .bold, design: .rounded))
                                .foregroundStyle(.secondary)
                                .padding(.trailing, 2)

                            ForEach(Voice.expressiveTags) { item in
                                Button {
                                    if textInput.isEmpty || textInput.hasSuffix(" ") || textInput.hasSuffix("\n") {
                                        textInput += "\(item.tag) "
                                    } else {
                                        textInput += " \(item.tag) "
                                    }
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: item.icon)
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundStyle(.purple.opacity(0.9))
                                        Text(item.label)
                                            .font(.system(size: 11, weight: .medium))
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(
                                        Capsule()
                                            .fill(Color.purple.opacity(0.13))
                                            .overlay(
                                                Capsule()
                                                    .strokeBorder(Color.purple.opacity(0.25), lineWidth: 0.8)
                                            )
                                    )
                                }
                                .buttonStyle(.plain)
                                .help("Insert \(item.tag)")
                            }
                        }
                    }

                    Spacer(minLength: 8)

                    HStack(spacing: 8) {
                        if !textInput.isEmpty {
                            Button {
                                textInput = ""
                            } label: {
                                Label("Clear", systemImage: "xmark.circle.fill")
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)

                            Divider().frame(height: 10)
                        }

                        Text("\(wordCount) words · \(textInput.count) chars · \(max(chunks.count, 1)) chunks · ~\(estimatedSeconds)s est.")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.09), lineWidth: 0.8)
                    )
                    .fixedSize()
                }
                .padding(14)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
                guard let first = providers.first else { return false }
                _ = first.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in
                        importDocumentAsync(from: url)
                    }
                }
                return true
            }

            // 3. Unified Apple Glass Bottom Studio Dock
            VStack(spacing: 12) {
                // Status / Error Banner inside Dock
                if !viewModel.generationProgress.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.generationProgress)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal, 6)
                }

                if let error = viewModel.errorMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                        Text(error)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.red)
                        Spacer()
                        Button {
                            viewModel.errorMessage = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 13))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Dismiss error")
                    }
                    .padding(.horizontal, 6)
                }

                // Upper Dock Row: Dedicated Full-Width Built-In Voices Palette + Category Menu + Voice Browser Trigger
                if !viewModel.builtInVoices.isEmpty || !customVoices.isEmpty {
                    let allQuickVoices = customVoices + viewModel.builtInVoices
                    HStack(spacing: 8) {
                        // Categorized All-Voices Dropdown Menu
                        Menu {
                            let conversational = viewModel.builtInVoices.filter { $0.category == "Conversational" }
                            let narrators = viewModel.builtInVoices.filter { $0.category == "Narrator" }
                            let characters = viewModel.builtInVoices.filter { $0.category == "Character" }
                            let clones = viewModel.builtInVoices.filter { $0.category == "Reference Clones" }
                            let others = viewModel.builtInVoices.filter {
                                !["Conversational", "Narrator", "Character", "Reference Clones"].contains($0.category)
                            }

                            if !customVoices.isEmpty {
                                Section("Custom Cloned Voices (\(customVoices.count))") {
                                    ForEach(customVoices) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            if !conversational.isEmpty {
                                Section("Conversational (\(conversational.count))") {
                                    ForEach(conversational) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            if !narrators.isEmpty {
                                Section("Narrator (\(narrators.count))") {
                                    ForEach(narrators) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            if !characters.isEmpty {
                                Section("Character (\(characters.count))") {
                                    ForEach(characters) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            if !clones.isEmpty {
                                Section("Reference Clones (\(clones.count))") {
                                    ForEach(clones) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            if !others.isEmpty {
                                Section("Model Built-In (\(others.count))") {
                                    ForEach(others) { voice in
                                        Button {
                                            viewModel.selectedVoice = voice
                                        } label: {
                                            Text("\(voice.name) — \(voice.description)")
                                        }
                                    }
                                }
                            }
                            Divider()
                            Button {
                                showVoices = true
                            } label: {
                                Label("Open Full Voice Browser & Cloner...", systemImage: "square.grid.2x2")
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "person.2.wave.2.fill")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(.blue)
                                Text("BUILT-IN VOICES (\(viewModel.builtInVoices.count))")
                                    .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(.primary)
                                Image(systemName: "chevron.down")
                                    .font(.system(size: 8.5, weight: .bold))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5.5)
                            .background(
                                Capsule()
                                    .fill(Color.blue.opacity(0.14))
                                    .overlay(
                                        Capsule()
                                            .strokeBorder(Color.blue.opacity(0.32), lineWidth: 0.8)
                                    )
                            )
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .help("Browse all \(viewModel.builtInVoices.count) built-in voices grouped by Conversational, Narrator, Character, and Clones")

                        // Full-width horizontal voice strip
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(allQuickVoices) { voice in
                                    let isSelected = viewModel.selectedVoice?.name == voice.name && viewModel.selectedVoice?.voiceId == voice.voiceId
                                    Button {
                                        withAnimation(.spring(response: 0.24, dampingFraction: 0.8)) {
                                            viewModel.selectedVoice = voice
                                        }
                                    } label: {
                                        HStack(spacing: 6) {
                                            VoiceAvatar(color: voice.color, size: 15)
                                            Text(voice.name)
                                                .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                                                .foregroundStyle(isSelected ? .white : .primary.opacity(0.88))
                                        }
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(
                                            Capsule()
                                                .fill(isSelected ? Color.blue.opacity(0.34) : Color.white.opacity(0.05))
                                                .overlay(
                                                    Capsule()
                                                        .strokeBorder(isSelected ? Color.blue.opacity(0.88) : Color.white.opacity(0.09), lineWidth: 1)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                    .help("\(voice.name) — \(voice.description)")
                                }
                            }
                        }

                        // 1-Click Visual Voice Browser Grid Button
                        Button {
                            showVoices = true
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "square.grid.2x2")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("Browse All")
                                    .font(.system(size: 11.5, weight: .semibold))
                            }
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(
                                Capsule()
                                    .fill(Color.white.opacity(0.05))
                                    .overlay(
                                        Capsule()
                                            .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                                    )
                            )
                        }
                        .buttonStyle(.plain)
                        .fixedSize()
                        .help("Open visual Voice Browser grid and zero-shot voice cloner")
                    }
                }

                // Middle Dock Row: Apple Podcasts-style Transport Bar (when audio is generated)
                if viewModel.audioURL != nil {
                    HStack(spacing: 12) {
                        CompactAudioPlayer(
                            isPlaying: viewModel.isPlaying,
                            currentTime: viewModel.currentTime,
                            duration: viewModel.duration,
                            onPlayPause: { viewModel.togglePlayPause() },
                            onSeek: { viewModel.seek(to: $0) }
                        )

                        // Speed selector pill
                        Menu {
                            ForEach([Float(0.5), 0.75, 1.0, 1.25, 1.5, 2.0], id: \.self) { rate in
                                Button {
                                    viewModel.playbackRate = rate
                                } label: {
                                    if abs(viewModel.playbackRate - rate) < 0.01 {
                                        Label(String(format: "%.2gx", rate), systemImage: "checkmark")
                                    } else {
                                        Text(String(format: "%.2gx", rate))
                                    }
                                }
                            }
                        } label: {
                            Text(String(format: "%.2gx", viewModel.playbackRate))
                                .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 9)
                                .padding(.vertical, 5)
                                .background(Color.blue.opacity(0.18), in: Capsule())
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()

                        if viewModel.tokensPerSecond > 0 {
                            Text(String(format: "%.1f tok/s", viewModel.tokensPerSecond))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }

                        // Multi-format Download menu
                        Menu {
                            Section("Save Audio Format") {
                                ForEach(AudioExportFormat.allCases) { fmt in
                                    Button {
                                        viewModel.saveAudioInFormat(fmt)
                                    } label: {
                                        Label("Save as \(fmt.label)", systemImage: "waveform")
                                    }
                                }
                            }
                            Divider()
                            Section("Save Read-Along Karaoke Video (.MP4)") {
                                ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                    Button("Save Read-Along MP4 — \(aspect.label)") {
                                        viewModel.exportReadAlongVideo(
                                            text: textInput,
                                            theme: viewModel.readAlongTheme,
                                            aspect: aspect
                                        )
                                    }
                                }
                            }
                            Divider()
                            Section("Save Synced Subtitles") {
                                Button("Save Subtitles (.SRT)") {
                                    viewModel.exportSubtitles(text: textInput, format: "srt")
                                }
                                Button("Save WebVTT (.VTT)") {
                                    viewModel.exportSubtitles(text: textInput, format: "vtt")
                                }
                                Button("Save Karaoke (.ASS)") {
                                    viewModel.exportSubtitles(text: textInput, format: "ass")
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "arrow.down.circle.fill")
                                    .font(.system(size: 15))
                                Text("Save Audio")
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .background(Color.blue.opacity(0.14), in: Capsule())
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .help("Save audio in WAV, M4A, AIFF, FLAC, MP3, Read-Along MP4 Video, or SRT/VTT/ASS subtitles")
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.white.opacity(0.04))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                    )
                }

                Divider()
                    .opacity(0.25)

                // Lower Dock Row: Voice Browser Card + Model Engine Selector + Primary Generate Button
                HStack(spacing: 10) {
                    // Active Voice Selector Card
                    Button(action: { showVoices = true }) {
                        HStack(spacing: 10) {
                            if viewModel.useVoiceDesign && !viewModel.voiceDescription.isEmpty {
                                Image(systemName: "wand.and.stars")
                                    .font(.system(size: 15, weight: .semibold))
                                    .foregroundStyle(.purple)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text("Voice Design Active")
                                        .font(.system(size: 13.5, weight: .semibold))
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                    Text(viewModel.voiceDescription)
                                        .font(.system(size: 10.5))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            } else if let voice = viewModel.selectedVoice {
                                VoiceAvatar(color: voice.color, size: 28)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(voice.name)
                                        .font(.system(size: 13.5, weight: .semibold))
                                        .foregroundStyle(.primary)
                                        .lineLimit(1)
                                    Text(voice.description.isEmpty ? (voice.isCustom ? "Custom Cloned Voice" : "Built-In Studio Voice") : voice.description)
                                        .font(.system(size: 10.5))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            } else {
                                Image(systemName: "person.wave.2.fill")
                                    .font(.system(size: 15))
                                    .foregroundStyle(.blue)
                                Text("Select Voice")
                                    .font(.system(size: 13.5, weight: .semibold))
                            }
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundStyle(.secondary)
                                .padding(.leading, 4)
                        }
                        .padding(.horizontal, 14)
                        .frame(height: 44)
                        .background(
                            Capsule()
                                .fill(Color.white.opacity(0.06))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.8)
                                )
                        )
                    }
                    .buttonStyle(.plain)
                    .help("Browse all built-in voices or clone a custom voice from audio/microphone")

                    // Quick Model Engine Picker Capsule
                    Menu {
                        Section("Select MLX TTS Engine (\(TTSViewModel.presetModels.count) Models)") {
                            ForEach(TTSViewModel.presetModels, id: \.id) { preset in
                                let cached = modelsManager.isModelCached(preset.id)
                                let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                let labelText = cached
                                    ? "✓ \(preset.name) [Cached\(sizeStr.map { " · \($0)" } ?? "")]"
                                    : "\(preset.name) (\(preset.id))"
                                Button {
                                    viewModel.modelId = preset.id
                                    Task {
                                        await viewModel.reloadModel()
                                    }
                                } label: {
                                    if viewModel.modelId == preset.id {
                                        Label(labelText, systemImage: "checkmark")
                                    } else {
                                        Text(labelText)
                                    }
                                }
                            }
                        }
                        Divider()
                        if modelsManager.isModelCached(viewModel.modelId) {
                            let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId) ?? "Cached"
                            let activeName = TTSViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
                            Button(role: .destructive) {
                                pendingDeleteModel = (name: activeName, id: viewModel.modelId, size: activeSize)
                            } label: {
                                Label("Delete Cached \(activeName) (\(activeSize))…", systemImage: "trash")
                            }
                        }
                        Button {
                            modelsManager.refresh()
                            showDownloadedModelsSheet = true
                        } label: {
                            Label("Manage Downloaded Models (\(modelsManager.completeModelsCount) · \(modelsManager.totalSizeString))…", systemImage: "internaldrive.fill")
                        }
                        Button {
                            showSettings = true
                        } label: {
                            Label("Model & Generation Settings…", systemImage: "slider.horizontal.3")
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "cpu.fill")
                                .font(.system(size: 12.5))
                                .foregroundStyle(.blue)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(viewModel.modelId.split(separator: "/").last.map(String.init) ?? viewModel.modelId)
                                    .font(.system(size: 12.5, weight: .semibold))
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                Text("MLX Neural Engine")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                            }
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 14)
                        .frame(height: 44)
                        .background(
                            Capsule()
                                .fill(Color.white.opacity(0.06))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.8)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    // Inspector / Settings Button
                    Button(action: { showSettings = true }) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 44, height: 44)
                            .background(
                                Circle()
                                    .fill(Color.white.opacity(0.06))
                                    .overlay(
                                        Circle()
                                            .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.8)
                                    )
                            )
                    }
                    .buttonStyle(.plain)
                    .help("Configure temperature, top-p, max tokens, HuggingFace token & cache")

                    Spacer()

                    // Direct Silent Export Menu (Export without pressing Generate Speech & without playing sound)
                    Menu {
                        Section("Direct Silent Audio Export (No Playback)") {
                            ForEach(AudioExportFormat.allCases) { fmt in
                                Button {
                                    viewModel.saveAudioInFormat(
                                        fmt,
                                        fallbackText: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: activeChunkIdx,
                                        scopeOverride: viewModel.synthesisScope
                                    )
                                } label: {
                                    Label("Export as \(fmt.label) (Silent)", systemImage: "arrow.down.doc.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Instant 2s Active Chunk (§\(activeChunkIdx + 1)) Silent Export") {
                            ForEach([AudioExportFormat.wav, .mp3, .m4a]) { fmt in
                                Button {
                                    viewModel.quickExportActiveChunk(
                                        text: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: activeChunkIdx,
                                        format: fmt,
                                        scope: .activeChunk
                                    )
                                } label: {
                                    Label("⚡ 2s Export Chunk §\(activeChunkIdx + 1) as \(fmt.rawValue.uppercased())", systemImage: "bolt.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Export Full Book / Document Batch (Silent)") {
                            ForEach([AudioExportFormat.wav, .mp3, .m4a]) { fmt in
                                Button {
                                    viewModel.quickExportActiveChunk(
                                        text: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: 0,
                                        format: fmt,
                                        scope: .fullDocument
                                    )
                                } label: {
                                    Label("📚 Export Full Document as \(fmt.rawValue.uppercased()) (Silent)", systemImage: "books.vertical.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Synchronized Read-Along Karaoke Video (.MP4)") {
                            ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                Button {
                                    viewModel.exportReadAlongVideo(
                                        text: textInput,
                                        theme: viewModel.readAlongTheme,
                                        aspect: aspect
                                    )
                                } label: {
                                    Label("Export Read-Along MP4 — \(aspect.label) (Silent)", systemImage: "film.stack")
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "bolt.horizontal.circle.fill")
                                .font(.system(size: 13.5, weight: .semibold))
                                .foregroundStyle(.yellow)
                            Text("Silent Export")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.primary)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 9.5, weight: .bold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 14)
                        .frame(height: 44)
                        .background(
                            Capsule()
                                .fill(Color.white.opacity(0.06))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.yellow.opacity(0.32), lineWidth: 0.9)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .disabled(textInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isGenerating)
                    .help("Export audio or Read-Along MP4 directly without pressing Generate Speech — synthesizes silently with no sound played")

                    // Primary Generate / Stop Button
                    if viewModel.isGenerating {
                        Button(action: {
                            viewModel.stop()
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "stop.fill")
                                    .font(.system(size: 12, weight: .bold))
                                Text("Stop Synthesis")
                                    .font(.system(size: 14, weight: .semibold))
                            }
                            .foregroundStyle(.white)
                            .frame(height: 44)
                            .padding(.horizontal, 22)
                            .background(
                                LinearGradient(
                                    colors: [Color.red, Color.pink],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                in: Capsule()
                            )
                            .shadow(color: Color.red.opacity(0.35), radius: 8, y: 3)
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button(action: {
                            if viewModel.autoReadAlong {
                                withAnimation(.spring(response: 0.25, dampingFraction: 0.82)) {
                                    showTeleprompter = true
                                }
                            }
                            viewModel.startSynthesis(
                                text: textInput,
                                voice: viewModel.selectedVoice,
                                startChunkIndex: viewModel.activeChunkIndex
                            )
                            if let voice = viewModel.selectedVoice {
                                recentlyUsed.removeAll { $0.id == voice.id }
                                recentlyUsed.insert(voice, at: 0)
                            }
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "waveform.badge.plus")
                                    .font(.system(size: 14, weight: .semibold))
                                Text("Generate Speech")
                                    .font(.system(size: 14, weight: .semibold))
                                Text("⌘↩")
                                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.white.opacity(0.16), in: Capsule())
                            }
                            .foregroundStyle(canGenerate ? .white : .secondary)
                            .frame(height: 44)
                            .padding(.horizontal, 20)
                            .background(
                                LinearGradient(
                                    colors: canGenerate
                                        ? [Color.blue, Color.indigo]
                                        : [Color.white.opacity(0.08), Color.white.opacity(0.05)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                in: Capsule()
                            )
                            .overlay(
                                Capsule()
                                    .strokeBorder(Color.white.opacity(canGenerate ? 0.25 : 0.08), lineWidth: 0.8)
                            )
                            .shadow(color: canGenerate ? Color.blue.opacity(0.35) : .clear, radius: 10, y: 4)
                        }
                        .buttonStyle(.plain)
                        .disabled(!canGenerate)
                        .keyboardShortcut(.return, modifiers: [.command])
                    }
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(theme.dockBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: theme.canvasBorderColors,
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(theme.colorScheme == .light ? 0.12 : 0.32), radius: 16, y: 6)
            )
            .padding(.horizontal, 20)
            .padding(.bottom, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: rawStudioTheme) { _, newRaw in
            if let updated = ReadAlongTheme(rawValue: newRaw), viewModel.readAlongTheme != updated {
                viewModel.readAlongTheme = updated
            }
        }
        .fileImporter(
            isPresented: $showDocImporter,
            allowedContentTypes: [.epub, .pdf, .plainText, .rtf, .html, .json, .image, .data, .item],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                importDocumentAsync(from: url)
            }
        }
        .sheet(isPresented: $showVoices) {
            VoicesView(
                modelId: viewModel.modelId,
                builtInVoices: viewModel.builtInVoices,
                activeVoice: viewModel.selectedVoice,
                recentlyUsed: $recentlyUsed,
                customVoices: $customVoices
            ) { voice in
                viewModel.selectedVoice = voice
                showVoices = false
            }
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
                .presentationDetents([.large])
                .presentationDragIndicator(.visible)
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
        .sheet(isPresented: $showHistory) {
            TakesHistorySheet(viewModel: viewModel) { selectedTake in
                textInput = selectedTake.text
                if FileManager.default.fileExists(atPath: selectedTake.audioURL.path) {
                    viewModel.playTake(selectedTake)
                }
            }
        }
        .onChange(of: customVoices) { _, newValue in
            Voice.saveCustomVoices(newValue)
        }
        .onReceive(NotificationCenter.default.publisher(for: .sendTranscriptToVoiceStudio)) { note in
            let incoming = (note.object as? String)
                ?? UserDefaults.standard.string(forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                ?? ""
            let cleaned = incoming.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty {
                UserDefaults.standard.removeObject(forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                textInput = cleaned
                viewModel.synthesizedWordTimings = []
                viewModel.activeChunkIndex = 0
                viewModel.synthesizingChunkIndex = 0
                viewModel.synthesizedStartChunkIndex = 0
                viewModel.synthesizedEndChunkIndex = nil
                previewWordId = nil
                showTeleprompter = false
            }
        }
        .onAppear {
            if let docPath = UserDefaults.standard.string(forKey: "VoicesApp.VoiceStudio.autoImportDocPath"),
               !docPath.isEmpty,
               FileManager.default.fileExists(atPath: docPath) {
                UserDefaults.standard.removeObject(forKey: "VoicesApp.VoiceStudio.autoImportDocPath")
                importDocumentAsync(from: URL(fileURLWithPath: docPath))
            } else if let pending = UserDefaults.standard.string(forKey: "VoicesApp.VoiceStudio.pendingTranscript")?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !pending.isEmpty {
                UserDefaults.standard.removeObject(forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                textInput = pending
                viewModel.synthesizedWordTimings = []
                viewModel.activeChunkIndex = 0
                viewModel.synthesizingChunkIndex = 0
                viewModel.synthesizedStartChunkIndex = 0
                viewModel.synthesizedEndChunkIndex = nil
                previewWordId = nil
                showTeleprompter = false
            }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                viewModel.pause()
                viewModel.stop()
            default:
                break
            }
        }
        .task {
            LocalhostServerManager.shared.startIfNeeded()
            await viewModel.loadModel()
        }
    }

    private var canGenerate: Bool {
        !textInput.isEmpty && !viewModel.isGenerating && viewModel.isModelLoaded
    }
}

// MARK: - Waveform-Synced Karaoke Read-Along Teleprompter View (Codecaine File Manager Style)

struct KaraokeFlowLayout: Layout {
    var spacing: CGFloat = 6
    var lineSpacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? 760
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var totalWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX > 0 && currentX + size.width > maxWidth {
                currentX = 0
                currentY += lineHeight + lineSpacing
                lineHeight = 0
            }
            totalWidth = max(totalWidth, currentX + size.width)
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: max(totalWidth, min(maxWidth, 200)), height: currentY + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX = bounds.minX
        var currentY = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX > bounds.minX && currentX + size.width > bounds.maxX {
                currentX = bounds.minX
                currentY += lineHeight + lineSpacing
                lineHeight = 0
            }
            subview.place(
                at: CGPoint(x: currentX, y: currentY),
                anchor: .topLeading,
                proposal: ProposedViewSize(size)
            )
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

struct KaraokeTeleprompterView: View {
    let timings: [WordTiming]
    let chunks: [PhononTextChunk]
    let activeWordId: Int?
    let activeChunkIndex: Int
    let theme: ReadAlongTheme
    let onSelectWord: (WordTiming) -> Void
    let onSelectChunk: (Int) -> Void
    let onSynthesizeFromChunk: (Int) -> Void
    let onQuickExportChunk: (Int) -> Void

    @State private var stageMode: Bool = true

    private func snippet(for chunk: PhononTextChunk?) -> String {
        guard let raw = chunk?.text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return ""
        }
        return raw.count > 105 ? String(raw.prefix(105)) + "…" : raw
    }

    var body: some View {
        let wordsByChunk = Dictionary(grouping: timings, by: \.chunkIndex)
        let chunksByIndex = Dictionary(uniqueKeysWithValues: chunks.map { ($0.index, $0) })
        let totalChunks = max(chunks.count, 1)
        let safeActiveChunk = min(max(0, activeChunkIndex), max(0, totalChunks - 1))
        let useFocusedStage = stageMode && chunks.count > 1

        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    // Header Banner
                    HStack(spacing: 8) {
                        HStack(spacing: 6) {
                            Image(systemName: "waveform.and.magnifyingglass")
                                .font(.system(size: 11.5, weight: .bold))
                                .foregroundStyle(theme.activeWordBackground)
                            Text("LIVE READ-ALONG KARAOKE")
                                .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                .foregroundStyle(theme.activeWordBackground)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(theme.activeWordBackground.opacity(0.16), in: Capsule())

                        if chunks.count > 1 {
                            // Focused Stage vs All Chunks Toggle
                            HStack(spacing: 2) {
                                Button {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        stageMode = true
                                    }
                                } label: {
                                    Text("🎯 Stage (§\(safeActiveChunk + 1)/\(totalChunks))")
                                        .font(.system(size: 10.5, weight: stageMode ? .bold : .medium, design: .rounded))
                                        .foregroundStyle(stageMode ? .white : .secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3.5)
                                        .background(stageMode ? theme.activeWordBackground : Color.clear, in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Button {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        stageMode = false
                                    }
                                } label: {
                                    Text("📜 All (§1–§\(totalChunks))")
                                        .font(.system(size: 10.5, weight: !stageMode ? .bold : .medium, design: .rounded))
                                        .foregroundStyle(!stageMode ? .white : .secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3.5)
                                        .background(!stageMode ? theme.activeWordBackground : Color.clear, in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(2)
                            .background(Color.white.opacity(0.06), in: Capsule())

                            Button {
                                onSelectChunk(max(0, safeActiveChunk - 1))
                            } label: {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 10.5, weight: .bold))
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 4.5)
                                    .background(Color.white.opacity(0.07), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(safeActiveChunk <= 0)
                            .help("Previous Chunk (§\(max(1, safeActiveChunk)))")

                            Button {
                                onSelectChunk(min(totalChunks - 1, safeActiveChunk + 1))
                            } label: {
                                Image(systemName: "chevron.right")
                                    .font(.system(size: 10.5, weight: .bold))
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 4.5)
                                    .background(Color.white.opacity(0.07), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(safeActiveChunk >= totalChunks - 1)
                            .help("Next Chunk (§\(min(totalChunks, safeActiveChunk + 2)))")
                        } else {
                            Text("Click any word to highlight & seek audio")
                                .font(.system(size: 11.5))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }

                        Spacer()

                        Button {
                            onQuickExportChunk(safeActiveChunk)
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "bolt.fill")
                                    .font(.system(size: 9.5))
                                Text("2s Export §\(safeActiveChunk + 1)")
                                    .font(.system(size: 10.5, weight: .semibold))
                            }
                            .foregroundStyle(.yellow)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                            .background(Color.yellow.opacity(0.15), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .help("Synthesize & export Chunk §\(safeActiveChunk + 1) as WAV in ~2 seconds")

                        if let activeId = activeWordId,
                           let current = (activeId >= 0 && activeId < timings.count && timings[activeId].id == activeId)
                            ? timings[activeId]
                            : timings.first(where: { $0.id == activeId }) {
                            Text(String(format: "Word #%d · %.2fs–%.2fs", current.id + 1, current.start, current.end))
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                .foregroundStyle(theme.activeWordBackground)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(Color.white.opacity(0.06), in: Capsule())
                        }
                    }

                    if useFocusedStage {
                        // Codecaine File Manager Focused Stage Mode: keeps the active spoken chunk & highlighted word centered on screen
                        if safeActiveChunk > 0, let prevMeta = chunksByIndex[safeActiveChunk - 1] {
                            Button {
                                onSelectChunk(safeActiveChunk - 1)
                            } label: {
                                HStack(spacing: 8) {
                                    Text("↑ Prev (§\(safeActiveChunk))")
                                        .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                        .foregroundStyle(theme.activeWordBackground)
                                    Text(snippet(for: prevMeta))
                                        .font(.system(size: 11.5))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Spacer()
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .fill(Color.white.opacity(0.04))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                                .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }

                        let activeWords = wordsByChunk[safeActiveChunk] ?? []
                        let activeMeta = chunksByIndex[safeActiveChunk]
                        let localWordIndex = activeWords.firstIndex(where: { $0.id == activeWordId }).map { $0 + 1 } ?? 1
                        let localWordTotal = max(activeWords.count, 1)
                        let chunkProgress = Double(localWordIndex) / Double(localWordTotal)

                        VStack(alignment: .leading, spacing: 12) {
                            // Stage Chunk Header + Progress
                            HStack(spacing: 8) {
                                Text("ACTIVE STAGE · CHUNK §\(safeActiveChunk + 1) OF \(totalChunks)")
                                    .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                    .foregroundStyle(theme.activeWordForeground)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 3.5)
                                    .background(theme.activeWordBackground, in: Capsule())

                                if let meta = activeMeta {
                                    Text("Word \(localWordIndex)/\(localWordTotal) · \(meta.charCount) chars · Paragraph #\(meta.paragraphIndex + 1)")
                                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }

                                Spacer()

                                Button {
                                    onSynthesizeFromChunk(safeActiveChunk)
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "play.fill")
                                            .font(.system(size: 9))
                                        Text("Read from §\(safeActiveChunk + 1)")
                                            .font(.system(size: 10.5, weight: .semibold))
                                    }
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(theme.activeWordBackground, in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }

                            // Progress bar inside active chunk stage
                            GeometryReader { geo in
                                ZStack(alignment: .leading) {
                                    Capsule()
                                        .fill(Color.white.opacity(0.08))
                                    Capsule()
                                        .fill(theme.activeWordBackground)
                                        .frame(width: max(8, geo.size.width * min(1.0, max(0.02, chunkProgress))))
                                }
                            }
                            .frame(height: 4)

                            // Word-by-Word Flowing Karaoke Tokens (always visible in stage center)
                            KaraokeFlowLayout(spacing: 8, lineSpacing: 10) {
                                ForEach(activeWords) { item in
                                    let isActive = item.id == activeWordId
                                    let isSpoken = (activeWordId != nil) && (item.id < activeWordId!)
                                    KaraokeWordTokenView(
                                        item: item,
                                        isActive: isActive,
                                        isSpoken: isSpoken,
                                        theme: theme,
                                        onSelect: { onSelectWord(item) }
                                    )
                                    .id("word-\(item.id)")
                                }
                            }
                        }
                        .padding(16)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(theme.tokenSpokenFill)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                                        .strokeBorder(theme.activeWordBackground.opacity(0.65), lineWidth: 1.4)
                                )
                        )
                        .id("chunk-\(safeActiveChunk)")

                        if safeActiveChunk + 1 < totalChunks, let nextMeta = chunksByIndex[safeActiveChunk + 1] {
                            Button {
                                onSelectChunk(safeActiveChunk + 1)
                            } label: {
                                HStack(spacing: 8) {
                                    Text("↓ Next (§\(safeActiveChunk + 2))")
                                        .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                        .foregroundStyle(theme.activeWordBackground)
                                    Text(snippet(for: nextMeta))
                                        .font(.system(size: 11.5))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Spacer()
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .fill(Color.white.opacity(0.04))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                                .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    } else {
                        // All-Chunks View with distinct string IDs ("chunk-\(cIdx)" vs "word-\(item.id)") so ScrollViewReader never collides
                        let chunkIndices = chunks.isEmpty ? [0] : chunks.map(\.index)
                        ForEach(chunkIndices, id: \.self) { cIdx in
                            let chunkWords = wordsByChunk[cIdx] ?? []
                            let isChunkActive = cIdx == safeActiveChunk
                            let chunkMeta = chunksByIndex[cIdx]

                            if !chunkWords.isEmpty {
                                VStack(alignment: .leading, spacing: 10) {
                                    // Chunk Section Ribbon
                                    HStack(spacing: 8) {
                                        Text("CHUNK §\(cIdx + 1)")
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .foregroundStyle(isChunkActive ? theme.activeWordForeground : .secondary)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 3)
                                            .background(
                                                isChunkActive ? theme.activeWordBackground : Color.white.opacity(0.07),
                                                in: Capsule()
                                            )

                                        if let meta = chunkMeta {
                                            Text("\(meta.wordCount) words · \(meta.charCount) chars · Paragraph #\(meta.paragraphIndex + 1)")
                                                .font(.system(size: 11, design: .monospaced))
                                                .foregroundStyle(.secondary)
                                        }

                                        Spacer()

                                        Button {
                                            onQuickExportChunk(cIdx)
                                        } label: {
                                            HStack(spacing: 3) {
                                                Image(systemName: "bolt.fill")
                                                    .font(.system(size: 8.5))
                                                Text("2s Export")
                                                    .font(.system(size: 10, weight: .semibold))
                                            }
                                            .foregroundStyle(.yellow)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 3)
                                            .background(Color.yellow.opacity(0.14), in: Capsule())
                                        }
                                        .buttonStyle(.plain)

                                        Button {
                                            onSynthesizeFromChunk(cIdx)
                                        } label: {
                                            HStack(spacing: 4) {
                                                Image(systemName: "play.fill")
                                                    .font(.system(size: 9))
                                                Text("Read from §\(cIdx + 1)")
                                                    .font(.system(size: 10.5, weight: .semibold))
                                            }
                                            .foregroundStyle(.blue)
                                            .padding(.horizontal, 9)
                                            .padding(.vertical, 3.5)
                                            .background(Color.blue.opacity(0.14), in: Capsule())
                                        }
                                        .buttonStyle(.plain)
                                    }

                                    // Word-by-Word Flowing Karaoke Tokens
                                    KaraokeFlowLayout(spacing: 7, lineSpacing: 8) {
                                        ForEach(chunkWords) { item in
                                            let isActive = item.id == activeWordId
                                            let isSpoken = (activeWordId != nil) && (item.id < activeWordId!)
                                            KaraokeWordTokenView(
                                                item: item,
                                                isActive: isActive,
                                                isSpoken: isSpoken,
                                                theme: theme,
                                                onSelect: { onSelectWord(item) }
                                            )
                                            .id("word-\(item.id)")
                                        }
                                    }
                                }
                                .padding(14)
                                .background(
                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                        .fill(isChunkActive ? theme.tokenSpokenFill : theme.cardBackground)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                                .strokeBorder(
                                                    isChunkActive
                                                        ? theme.activeWordBackground.opacity(0.55)
                                                        : Color.primary.opacity(0.08),
                                                    lineWidth: 1
                                                )
                                        )
                                )
                                .id("chunk-\(cIdx)")
                            }
                        }
                    }
                }
            }
            .onChange(of: activeWordId) { _, newId in
                guard let newId else { return }
                withAnimation(.easeInOut(duration: 0.20)) {
                    proxy.scrollTo("word-\(newId)", anchor: .center)
                }
            }
        }
    }
}

struct KaraokeWordTokenView: View {
    let item: WordTiming
    let isActive: Bool
    let isSpoken: Bool
    let theme: ReadAlongTheme
    let onSelect: () -> Void

    private var textColor: Color {
        if isActive { return theme.activeWordForeground }
        return isSpoken ? theme.spokenWordForeground : theme.upcomingWordForeground
    }

    private var fillColor: Color {
        if isActive { return theme.activeWordBackground }
        return isSpoken ? theme.tokenSpokenFill : theme.tokenRestingFill
    }

    var body: some View {
        Button(action: onSelect) {
            Text(item.word)
                .font(.system(size: 16, weight: isActive ? .bold : .medium))
                .foregroundStyle(textColor)
                .fixedSize(horizontal: true, vertical: true)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(fillColor)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(
                                    isActive ? Color.white.opacity(0.45) : Color.primary.opacity(0.08),
                                    lineWidth: isActive ? 1.2 : 0.6
                                )
                        )
                )
                .shadow(
                    color: isActive ? theme.activeWordBackground.opacity(0.55) : .clear,
                    radius: 8,
                    y: 2
                )
                .scaleEffect(isActive ? 1.04 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Takes History Sheet

struct TakesHistorySheet: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel
    let onRestoreText: (GeneratedTake) -> Void

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if viewModel.history.isEmpty {
                    VStack(spacing: 12) {
                        Spacer()
                        Image(systemName: "waveform.badge.magnifyingglass")
                            .font(.system(size: 44))
                            .foregroundStyle(.tertiary)
                        Text("No generated takes in this session yet")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        Text("Every speech generation is automatically recorded here so you can compare models, voices, and speeds.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                        Spacer()
                    }
                } else {
                    List {
                        ForEach(viewModel.history) { take in
                            let isThisPlaying = viewModel.isPlaying && viewModel.audioURL == take.audioURL
                            HStack(alignment: .top, spacing: 12) {
                                Button {
                                    if isThisPlaying {
                                        viewModel.pause()
                                    } else {
                                        viewModel.playTake(take)
                                    }
                                } label: {
                                    Image(systemName: isThisPlaying ? "pause.circle.fill" : "play.circle.fill")
                                        .font(.system(size: 28))
                                        .foregroundStyle(.blue)
                                }
                                .buttonStyle(.plain)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(take.text)
                                        .font(.subheadline)
                                        .lineLimit(2)
                                    HStack(spacing: 8) {
                                        Text(take.voiceName)
                                            .font(.caption2.weight(.semibold))
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.blue.opacity(0.15))
                                            .clipShape(Capsule())
                                        Text(take.shortModelName)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                        Text(String(format: "%.1fs · %.1f tok/s", take.duration, take.tokensPerSecond))
                                            .font(.caption2.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                    }
                                }

                                Spacer()

                                Button("Use Text") {
                                    onRestoreText(take)
                                    dismiss()
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)

                                Menu {
                                    ForEach(AudioExportFormat.allCases) { fmt in
                                        Button("Export \(fmt.label)") {
                                            viewModel.saveAudioInFormat(fmt, from: take.audioURL, baseName: "\(take.voiceName)_take")
                                        }
                                    }
                                    Divider()
                                    Button(role: .destructive) {
                                        viewModel.deleteTake(take)
                                    } label: {
                                        Label("Delete Take", systemImage: "trash")
                                    }
                                } label: {
                                    Image(systemName: "square.and.arrow.down")
                                }
                                .menuStyle(.borderlessButton)
                                .fixedSize()
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .navigationTitle("Generated Takes (\(viewModel.history.count))")
            .toolbar {
                if !viewModel.history.isEmpty {
                    ToolbarItem(placement: .automatic) {
                        Button("Clear All", role: .destructive) {
                            viewModel.clearHistory()
                        }
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 540, minHeight: 420)
        #endif
    }
}

// MARK: - Voice Selector Button

struct VoiceSelectorButton: View {
    let selectedVoice: Voice?
    let isLoading: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                if let voice = selectedVoice {
                    VoiceAvatar(color: voice.color, size: 44)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(voice.name)
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundStyle(.primary)

                        Text(voice.description.isEmpty ? voice.language : voice.description)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    ZStack {
                        Circle()
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: 44, height: 44)

                        Image(systemName: "waveform")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Select a voice")
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundStyle(.primary)

                        Text("Tap to browse voices")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(Color.gray.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
    }
}

// MARK: - Status View

struct StatusView: View {
    let message: String
    let tokensPerSecond: Double

    var body: some View {
        HStack(spacing: 12) {
            ProgressView()
                .scaleEffect(0.8)

            VStack(alignment: .leading, spacing: 2) {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                if tokensPerSecond > 0 {
                    Text(String(format: "%.1f tokens/sec", tokensPerSecond))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Error View

struct ErrorView: View {
    let message: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.primary)

            Spacer()
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Compact Audio Player

struct CompactAudioPlayer: View {
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSeek: (TimeInterval) -> Void

    private let playButtonSize: CGFloat = 44
    private let spacing: CGFloat = 12
    private let padding: CGFloat = 12

    var body: some View {
        HStack(spacing: spacing) {
            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: playButtonSize))
                    .foregroundStyle(.blue)
            }
            .buttonStyle(.plain)

            VStack(spacing: 2) {
                // Progress bar
                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { currentTime },
                        set: { onSeek($0) }
                    ),
                    range: 0...max(duration, 0.01)
                )
                #else
                Slider(
                    value: Binding(
                        get: { currentTime },
                        set: { onSeek($0) }
                    ),
                    in: 0...max(duration, 0.01)
                )
                .tint(.blue)
                #endif

                // Time labels
                HStack {
                    Text(formatTime(currentTime))
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text(formatTime(duration))
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(padding)
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Bottom Action Bar

struct BottomActionBar: View {
    let isGenerating: Bool
    let isModelLoaded: Bool
    let canGenerate: Bool
    let onGenerate: () -> Void

    var body: some View {
        Button(action: onGenerate) {
            HStack(spacing: 8) {
                if isGenerating {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "waveform")
                }

                Text(buttonTitle)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(canGenerate ? Color.blue : Color.gray)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
        .disabled(!canGenerate)
        .padding()
    }

    private var buttonTitle: String {
        if !isModelLoaded {
            return "Loading Model..."
        } else if isGenerating {
            return "Generating..."
        } else {
            return "Generate Speech"
        }
    }
}

#Preview("CompactAudioPlayer") {
    CompactAudioPlayer(isPlaying: true, currentTime: 1.0, duration: 5.0, onPlayPause: {}, onSeek: { _ in })
}

#Preview("ContentView") {
    ContentView()
}
