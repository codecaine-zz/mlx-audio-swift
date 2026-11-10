import SwiftUI
import AVFoundation
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
    @State private var textInput: String = Voice.deadOrbitNovelText
    @State private var showVoices = ProcessInfo.processInfo.arguments.contains("--open-voices")
    @State private var showSettings = ProcessInfo.processInfo.arguments.contains("--open-settings")
    @State private var showDownloadedModelsSheet = ProcessInfo.processInfo.arguments.contains("--open-models")
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var showHistory = ProcessInfo.processInfo.arguments.contains("--open-history")
    @State private var showDocImporter = false
    @State private var showTeleprompter: Bool = {
        if ProcessInfo.processInfo.arguments.contains("--read-along") { return true }
        return UserDefaults.standard.bool(forKey: "VoicesApp.VoiceStudio.showTeleprompter")
    }()
    @State private var previewWordId: Int? = ProcessInfo.processInfo.arguments.contains("--read-along") ? 6 : nil
    @State private var isDropTargeted = false
    @State private var cleanOptions: SpeechCleanOptions = SpeechCleanOptions.loadFromDefaults()
    @State private var recentlyUsed: [Voice] = Voice.samples
    @State private var customVoices: [Voice] = Voice.loadCustomVoices()
    @State private var copiedScript = false
    @State private var isAuditioningVoice = false
    @State private var auditionPlayer: AVAudioPlayer? = nil
    @State private var sessionManager = StudioDocumentSessionManager.shared
    @State private var lexicon = SpeechPronunciationLexicon.shared
    @State private var castingManager = CharacterCastingManager.shared
    @State private var masteringManager = StudioMasteringManager.shared
    @State private var miniReader = StudioMiniReaderCoordinator.shared
    @State private var prefs = StudioUserPreferencesManager.shared
    @State private var showSaveProfilePopover = false
    @State private var newProfileName: String = ""
    @State private var showPronunciationSheet = ProcessInfo.processInfo.arguments.contains("--open-pronunciation")
    @State private var showCastingSheet = ProcessInfo.processInfo.arguments.contains("--open-casting")
    @State private var showM4BExporterSheet = ProcessInfo.processInfo.arguments.contains("--open-m4b")
    @State private var showMasteringRackSheet = ProcessInfo.processInfo.arguments.contains("--open-mastering")
    @State private var showChunkSearchPopover = ProcessInfo.processInfo.arguments.contains("--open-chunk-search")
    @State private var chunkSearchQuery: String = ""
    @State private var chunkSearchChaptersOnly: Bool = false

    private func syncMiniReaderState() {
        let chunks = textChunks
        let safeIdx = min(max(0, viewModel.activeChunkIndex), max(0, chunks.count - 1))
        let snippet = safeIdx < chunks.count ? chunks[safeIdx].text : textInput
        miniReader.sync(
            title: sessionManager.currentDocumentTitle,
            chunkIndex: safeIdx,
            totalChunks: max(1, chunks.count),
            snippet: snippet,
            isPlaying: viewModel.isPlaying,
            isGenerating: viewModel.isGenerating,
            voiceName: viewModel.selectedVoice?.name ?? "Zoey",
            rate: viewModel.playbackRate,
            currentTime: viewModel.currentTime,
            duration: viewModel.duration
        )
    }

    private func handleMiniReaderTransport(_ action: String) {
        let chunks = textChunks
        switch action {
        case "togglePlayPause":
            if viewModel.audioURL != nil {
                viewModel.togglePlayPause()
            } else if viewModel.isModelLoaded {
                viewModel.startSynthesis(text: textInput, voice: viewModel.selectedVoice, startChunkIndex: viewModel.activeChunkIndex)
            }
        case "prevChunk":
            let prev = max(0, viewModel.activeChunkIndex - 1)
            viewModel.activeChunkIndex = prev
            if viewModel.isModelLoaded {
                viewModel.startSynthesis(text: textInput, voice: viewModel.selectedVoice, startChunkIndex: prev)
            }
        case "nextChunk":
            let next = min(max(0, chunks.count - 1), viewModel.activeChunkIndex + 1)
            viewModel.activeChunkIndex = next
            if viewModel.isModelLoaded {
                viewModel.startSynthesis(text: textInput, voice: viewModel.selectedVoice, startChunkIndex: next)
            }
        case "rate_1.0":
            viewModel.playbackRate = 1.0
        case "rate_1.25":
            viewModel.playbackRate = 1.25
        case "rate_1.5":
            viewModel.playbackRate = 1.5
        case "rate_2.0":
            viewModel.playbackRate = 2.0
        case "openMasteringRack":
            showMasteringRackSheet = true
        case "retakeActiveChunk":
            viewModel.retakeAndSpliceChunk(fullText: textInput, chunkIndex: viewModel.activeChunkIndex, voice: viewModel.selectedVoice)
        case "applyMasterBus":
            viewModel.applyMasterBusInPlaceToCurrentAudio()
        case "preset_acx":
            masteringManager.applyPreset(.acxAudiobook)
            viewModel.applyMasterBusInPlaceToCurrentAudio()
        case "preset_podcast":
            masteringManager.applyPreset(.podcastBroadcast)
            viewModel.applyMasterBusInPlaceToCurrentAudio()
        case "preset_youtube":
            masteringManager.applyPreset(.youtubeStreaming)
            viewModel.applyMasterBusInPlaceToCurrentAudio()
        case "preset_cinema":
            masteringManager.applyPreset(.warmCinema)
            viewModel.applyMasterBusInPlaceToCurrentAudio()
        case "preset_bypass":
            masteringManager.applyPreset(.bypass)
        case "clearAllText":
            clearAllStudioText()
        case "saveCurrentProfile":
            let vName = viewModel.selectedVoice?.name.components(separatedBy: " ").first ?? "Studio"
            newProfileName = "\(vName) \(String(format: "%.2gx", viewModel.playbackRate)) (\(masteringManager.preset.shortBadge))"
            showSaveProfilePopover = true
        default:
            if action.hasPrefix("applyProfile_") {
                let idStr = String(action.dropFirst("applyProfile_".count))
                if let uuid = UUID(uuidString: idStr),
                   let profile = prefs.userProfiles.first(where: { $0.id == uuid }) {
                    viewModel.applyStudioProfile(profile)
                    viewModel.applyMasterBusInPlaceToCurrentAudio()
                }
            }
        }
        syncMiniReaderState()
    }

    private func pasteAndReadFromClipboard() {
        #if os(macOS)
        guard let raw = NSPasteboard.general.string(forType: .string)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            viewModel.errorMessage = "Clipboard does not contain any readable text."
            return
        }
        let cleaned = SpeechTextCleaner.clean(raw, options: cleanOptions)
        let finalScript = cleaned.isEmpty ? raw : cleaned
        viewModel.stop()
        viewModel.synthesizedWordTimings = []
        viewModel.activeChunkIndex = 0
        viewModel.synthesizingChunkIndex = 0
        viewModel.synthesizedStartChunkIndex = 0
        viewModel.synthesizedEndChunkIndex = nil
        previewWordId = nil
        textInput = finalScript
        sessionManager.saveSessionScript(finalScript, chunkIndex: 0, title: "Clipboard Script")
        if viewModel.autoReadAlong {
            withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                showTeleprompter = true
            }
        }
        if viewModel.isModelLoaded {
            viewModel.startSynthesis(
                text: finalScript,
                voice: viewModel.selectedVoice,
                startChunkIndex: 0
            )
        }
        #endif
    }

    private func clearAllStudioText() {
        viewModel.stop()
        withAnimation(.easeOut(duration: 0.16)) {
            textInput = ""
            viewModel.synthesizedWordTimings = []
            viewModel.activeChunkIndex = 0
            viewModel.synthesizingChunkIndex = 0
            viewModel.synthesizedStartChunkIndex = 0
            viewModel.synthesizedEndChunkIndex = nil
            previewWordId = nil
            showTeleprompter = false
            viewModel.errorMessage = nil
            sessionManager.saveSessionScript("", chunkIndex: 0, title: "Studio Script")
        }
        syncMiniReaderState()
    }

    private func toggleVoiceAudition() {
        viewModel.toggleVoiceAudition(viewModel.selectedVoice)
    }

    private func isChapterHeadingChunk(_ chunk: PhononTextChunk) -> Bool {
        let lines = chunk.text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        for line in lines.prefix(3) {
            if HierarchicalChunker.isStandaloneBookHeading(line) {
                return true
            }
        }
        return false
    }

    @ViewBuilder
    private func chunkSearchPopoverView(
        chunks: [PhononTextChunk],
        timings: [WordTiming],
        activeChunkIdx: Int
    ) -> some View {
        let queryTrimmed = chunkSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let chapterChunksCount = chunks.filter { isChapterHeadingChunk($0) }.count
        let filteredChunks = chunks.filter { c in
            let matchesChapter = !chunkSearchChaptersOnly || isChapterHeadingChunk(c)
            let matchesQuery = queryTrimmed.isEmpty || c.text.localizedCaseInsensitiveContains(queryTrimmed)
            return matchesChapter && matchesQuery
        }

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search across \(chunks.count) chunks (e.g. Chapter, Cabin)…", text: $chunkSearchQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12.5))
                if !chunkSearchQuery.isEmpty {
                    Button {
                        chunkSearchQuery = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))

            HStack(spacing: 8) {
                Button {
                    chunkSearchChaptersOnly.toggle()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: chunkSearchChaptersOnly ? "book.closed.fill" : "book.closed")
                            .font(.system(size: 10.5, weight: .bold))
                        Text("Chapters Only (\(chapterChunksCount))")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(chunkSearchChaptersOnly ? .white : .blue)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(
                        chunkSearchChaptersOnly ? Color.blue : Color.blue.opacity(0.14),
                        in: Capsule()
                    )
                }
                .buttonStyle(.plain)
                .help("Filter navigator to detected book chapters & headings only")

                Spacer()

                Text("\(filteredChunks.count) of \(chunks.count) chunks matching")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(filteredChunks.prefix(50)) { c in
                        HStack(spacing: 8) {
                            Text("§\(c.index + 1)")
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 3)
                                .background(c.index == activeChunkIdx ? Color.blue : Color.secondary.opacity(0.35), in: Capsule())

                            Text(c.text.replacingOccurrences(of: "\n", with: " "))
                                .font(.system(size: 11.5))
                                .lineLimit(2)
                                .foregroundStyle(.primary)

                            Spacer()

                            Button("Jump") {
                                viewModel.activeChunkIndex = c.index
                                if let firstWord = timings.first(where: { $0.chunkIndex == c.index }) {
                                    previewWordId = firstWord.id
                                }
                                showTeleprompter = true
                                showChunkSearchPopover = false
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)

                            Button {
                                viewModel.activeChunkIndex = c.index
                                if let firstWord = timings.first(where: { $0.chunkIndex == c.index }) {
                                    previewWordId = firstWord.id
                                }
                                showTeleprompter = true
                                showChunkSearchPopover = false
                                if viewModel.isModelLoaded {
                                    viewModel.startSynthesis(
                                        text: textInput,
                                        voice: viewModel.selectedVoice,
                                        startChunkIndex: c.index
                                    )
                                }
                            } label: {
                                Image(systemName: "play.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.mini)
                            .help("Speak immediately from Chunk §\(c.index + 1)")
                        }
                        .padding(6)
                        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            .frame(height: 240)
        }
        .padding(12)
        .frame(width: 440)
    }

    @ViewBuilder
    private var exportCompletionBannerRow: some View {
        if let exportedURL = viewModel.lastExportedFileURL,
           FileManager.default.fileExists(atPath: exportedURL.path) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.green)

                Text("Saved \(viewModel.lastExportedSummary.isEmpty ? exportedURL.lastPathComponent : viewModel.lastExportedSummary)")
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Spacer()

                #if os(macOS)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([exportedURL])
                } label: {
                    Label("Reveal in Finder", systemImage: "folder.fill")
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3.5)
                        .background(Color.green.opacity(0.15), in: Capsule())
                }
                .buttonStyle(.plain)

                Button {
                    NSWorkspace.shared.open(exportedURL)
                } label: {
                    Label("Open File", systemImage: "play.rectangle.fill")
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3.5)
                        .background(Color.blue.opacity(0.15), in: Capsule())
                }
                .buttonStyle(.plain)
                #endif

                Button {
                    viewModel.lastExportedFileURL = nil
                    viewModel.lastExportedSummary = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Dismiss export notice")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(Color.green.opacity(0.08))
                    .overlay(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .strokeBorder(Color.green.opacity(0.28), lineWidth: 0.8)
                    )
            )
        }
    }

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

    private func importDocumentsAsync(from urls: [URL]) {
        guard !urls.isEmpty else { return }
        if urls.count == 1, let single = urls.first {
            importDocumentAsync(from: single)
            return
        }
        viewModel.errorMessage = nil
        viewModel.generationProgress = "Importing \(urls.count) files…"

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("VoicesAppDocImports", isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        var stagedItems: [(original: URL, staged: URL, name: String)] = []
        for url in urls {
            let fileName = url.lastPathComponent
            let scoped = url.startAccessingSecurityScopedResource()
            let stagedURL = tempDir.appendingPathComponent("\(UUID().uuidString)_\(fileName)")
            do {
                try FileManager.default.linkItem(at: url, to: stagedURL)
            } catch {
                try? FileManager.default.copyItem(at: url, to: stagedURL)
            }
            if scoped {
                url.stopAccessingSecurityScopedResource()
            }
            let target = FileManager.default.fileExists(atPath: stagedURL.path) ? stagedURL : url
            stagedItems.append((url, target, fileName))
        }

        Task.detached(priority: .userInitiated) {
            var sections: [String] = []
            for item in stagedItems {
                if let extracted = try? SpeechTextCleaner.extractDocumentText(from: item.staged) {
                    let trimmed = extracted.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        sections.append("[\(item.name)]\n\(trimmed)")
                    }
                }
                try? FileManager.default.removeItem(at: item.staged)
            }
            let combined = sections.joined(separator: "\n\n")
            let batchTitle = "\(stagedItems.first?.name ?? "Batch") + \(stagedItems.count - 1) more"
            await MainActor.run {
                self.viewModel.generationProgress = ""
                if !combined.isEmpty {
                    self.viewModel.stop()
                    self.viewModel.audioURL = nil
                    self.viewModel.duration = 0
                    self.viewModel.synthesizedWordTimings = []
                    self.viewModel.activeChunkIndex = 0
                    self.viewModel.synthesizingChunkIndex = 0
                    self.viewModel.synthesizedStartChunkIndex = 0
                    self.viewModel.synthesizedEndChunkIndex = nil
                    self.previewWordId = nil
                    self.textInput = combined
                    self.viewModel.errorMessage = nil
                    self.sessionManager.recordRecentDocument(
                        title: batchTitle,
                        sourcePath: stagedItems.first?.original.path ?? "",
                        text: combined,
                        chunkIndex: 0
                    )
                    self.sessionManager.saveSessionScript(combined, chunkIndex: 0, title: batchTitle)
                } else {
                    self.viewModel.errorMessage = "No readable text found in selected files."
                }
            }
        }
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
                        self.viewModel.completedChunkIndex = -1
                        self.viewModel.synthesisScope = .fullDocument
                        self.previewWordId = nil
                        self.textInput = extracted
                        self.viewModel.errorMessage = nil
                        self.sessionManager.recordRecentDocument(
                            title: fileName,
                            sourcePath: url.path,
                            text: extracted,
                            chunkIndex: 0
                        )
                        self.sessionManager.saveSessionScript(extracted, chunkIndex: 0, title: fileName)
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
            } catch let nsErr as NSError {
                try? FileManager.default.removeItem(at: stagedURL)
                await MainActor.run {
                    self.viewModel.generationProgress = ""
                    if nsErr.code == 415 && UniversalFileFormatCatalog.isAudioOrVideo(url: url) {
                        NotificationCenter.default.post(
                            name: Notification.Name("VoicesApp.OpenAudioInTranscribe"),
                            object: url
                        )
                    } else {
                        self.viewModel.errorMessage = "Failed to import \(fileName): \(nsErr.localizedDescription)"
                    }
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
            let matchesLastSynth = viewModel.lastSynthesizedText == textInput.trimmingCharacters(in: .whitespacesAndNewlines)
            if matchesLastSynth,
               viewModel.synthesizedWordTimings.count == totalWords,
               let firstPlayable = viewModel.synthesizedWordTimings.first(where: { $0.end > $0.start }),
               firstPlayable.end > firstPlayable.start {
                return viewModel.synthesizedWordTimings
            }
            return WordTimingEngine.computeWordTimings(
                text: textInput,
                audioURL: matchesLastSynth ? viewModel.audioURL : nil,
                duration: (matchesLastSynth && viewModel.duration > 0.05) ? viewModel.duration : Double(max(estimatedSeconds, 2)),
                maxChunkChars: max(180, viewModel.maxChunkLength),
                enableChunking: viewModel.enableChunking,
                startChunkIndex: matchesLastSynth ? viewModel.synthesizedStartChunkIndex : 0,
                endChunkIndex: matchesLastSynth ? viewModel.synthesizedEndChunkIndex : nil,
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
            HStack(spacing: 8) {
                // Engine & Cache Status Capsule
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isModelLoaded ? Color.green : (viewModel.isLoading ? Color.orange : Color.gray))
                        .frame(width: 8, height: 8)
                        .shadow(color: (viewModel.isModelLoaded ? Color.green : Color.orange).opacity(0.6), radius: 4)

                    Text(viewModel.isModelLoaded ? "Engine Ready" : (viewModel.isLoading ? "Loading Model…" : "Engine Idle"))
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(.primary.opacity(0.92))

                    if modelsManager.isModelCached(viewModel.modelId) {
                        let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId)
                        HStack(spacing: 3) {
                            Image(systemName: "checkmark.seal.fill")
                                .font(.system(size: 9.5))
                            Text(activeSize.map { "Cached · \($0)" } ?? "Cached")
                                .font(.system(size: 10, weight: .semibold))
                        }
                        .foregroundStyle(.green)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
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
                            HStack(spacing: 3.5) {
                                Image(systemName: "memorychip")
                                    .font(.system(size: 10, weight: .semibold))
                                Text("Purge")
                                    .font(.system(size: 10.5, weight: .medium))
                            }
                            .foregroundStyle(.orange)
                        }
                        .buttonStyle(.plain)
                        .help("Unload model & free Apple Silicon Metal memory immediately (keep_alive: 0)")
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(theme.pillFill)
                        .overlay(
                            Capsule()
                                .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                        )
                )
                .fixedSize()

                Spacer(minLength: 4)

                // Segmented Edit Script vs Read-Along Karaoke Switcher
                HStack(spacing: 2) {
                    Button {
                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                            showTeleprompter = false
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "square.and.pencil")
                                .font(.system(size: 10.5))
                            Text("Edit Script")
                                .font(.system(size: 11, weight: !showTeleprompter ? .semibold : .medium))
                        }
                        .foregroundStyle(!showTeleprompter ? (theme.isLight ? Color.white : Color.white) : .secondary)
                        .padding(.horizontal, 8.5)
                        .padding(.vertical, 5)
                        .background(
                            !showTeleprompter
                                ? (theme.isLight ? Color.black.opacity(0.78) : Color.white.opacity(0.18))
                                : Color.clear,
                            in: Capsule()
                        )
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
                        HStack(spacing: 4) {
                            Image(systemName: "text.word.spacing")
                                .font(.system(size: 10.5))
                            Text("Read-Along")
                                .font(.system(size: 11, weight: showTeleprompter ? .semibold : .medium))
                        }
                        .foregroundStyle(showTeleprompter ? theme.activeWordForeground : .secondary)
                        .padding(.horizontal, 8.5)
                        .padding(.vertical, 5)
                        .background(showTeleprompter ? theme.activeWordBackground : Color.clear, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(2.5)
                .background(
                    Capsule()
                        .fill(theme.recessedFill)
                        .overlay(
                            Capsule()
                                .strokeBorder(theme.recessedStroke, lineWidth: 0.8)
                        )
                )
                .fixedSize()

                // Unified Apple Glass Action Capsule Group
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 1) {
                        // Import Any File Format (180+ extensions across E-Books, PDF, Office, iWork, Subtitles, Markup, Data, Code, Archives & OCR)
                        Button {
                            showDocImporter = true
                        } label: {
                            Label("Import", systemImage: "doc.text.magnifyingglass")
                                .font(.system(size: 11, weight: .medium))
                                .padding(.horizontal, 6.5)
                                .padding(.vertical, 5)
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .help("Import any file format (\(UniversalFileFormatCatalog.supportedFormatsSummary))")

                        Divider().frame(height: 14).opacity(0.35)

                        // 1-Click Paste & Read Clipboard (⇧⌘V)
                        Button {
                            pasteAndReadFromClipboard()
                        } label: {
                            HStack(spacing: 3.5) {
                                Image(systemName: "doc.on.clipboard.fill")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.cyan)
                                Text("Paste")
                                    .font(.system(size: 11, weight: .medium))
                            }
                            .padding(.horizontal, 6.5)
                            .padding(.vertical, 5)
                            .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .help("Clean & immediately read whatever text is on your macOS clipboard (⇧⌘V)")

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
                            HStack(spacing: 4) {
                                Image(systemName: "wand.and.stars")
                                    .font(.system(size: 11))
                                Text("Clean")
                                    .font(.system(size: 11, weight: .medium))
                            }
                            .padding(.horizontal, 7)
                            .padding(.vertical, 5)
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()

                        Divider().frame(height: 14).opacity(0.35)

                        // 1-Click Clear All Text Button (⇧⌘⌫)
                        Button {
                            clearAllStudioText()
                        } label: {
                            HStack(spacing: 3.5) {
                                Image(systemName: "trash.fill")
                                    .font(.system(size: 10))
                                    .foregroundStyle(textInput.isEmpty ? Color.secondary.opacity(0.5) : Color.red)
                                Text("Clear Text")
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(textInput.isEmpty ? .secondary : .primary)
                            }
                            .padding(.horizontal, 6.5)
                            .padding(.vertical, 5)
                            .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .disabled(textInput.isEmpty && !showTeleprompter)
                        .keyboardShortcut(.delete, modifiers: [.command, .shift])
                        .help("Clear all text and reset editor to blank canvas (⇧⌘⌫)")

                    Divider().frame(height: 14).opacity(0.35)

                    // Reading Bookmarks & Recent Documents Menu
                    Menu {
                        let currentChunkText = (activeChunkIdx < chunks.count) ? chunks[activeChunkIdx].text : textInput
                        let isBookmarked = sessionManager.isChunkBookmarked(activeChunkIdx)
                        Button {
                            sessionManager.toggleBookmark(chunkIndex: activeChunkIdx, chunkText: currentChunkText)
                        } label: {
                            Label(
                                isBookmarked ? "Remove Bookmark for Chunk §\(activeChunkIdx + 1)" : "Bookmark Current Chunk (§\(activeChunkIdx + 1))",
                                systemImage: isBookmarked ? "bookmark.slash.fill" : "bookmark.fill"
                            )
                        }
                        if !sessionManager.bookmarks.isEmpty {
                            Divider()
                            Section("Saved Reading Bookmarks (\(sessionManager.bookmarks.count))") {
                                ForEach(sessionManager.bookmarks) { bm in
                                    Button {
                                        let target = min(max(0, bm.chunkIndex), max(0, chunks.count - 1))
                                        viewModel.activeChunkIndex = target
                                        if let firstWord = timings.first(where: { $0.chunkIndex == target }) {
                                            previewWordId = firstWord.id
                                        }
                                        withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                                            showTeleprompter = true
                                        }
                                    } label: {
                                        Label("§\(bm.chunkIndex + 1) — \(bm.previewSnippet)", systemImage: "bookmark.fill")
                                    }
                                }
                            }
                        }
                        if !sessionManager.recentDocuments.isEmpty {
                            Divider()
                            Section("Recent Imported Books & Documents") {
                                ForEach(sessionManager.recentDocuments) { doc in
                                    Button {
                                        if let restored = sessionManager.loadRecentDocumentText(doc) {
                                            textInput = restored
                                            viewModel.activeChunkIndex = doc.lastChunkIndex
                                        }
                                    } label: {
                                        Label("\(doc.title) (§\(doc.lastChunkIndex + 1))", systemImage: "clock")
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 3.5) {
                            Image(systemName: sessionManager.isChunkBookmarked(activeChunkIdx) ? "bookmark.fill" : "bookmark")
                                .font(.system(size: 10.5))
                                .foregroundStyle(sessionManager.isChunkBookmarked(activeChunkIdx) ? .orange : .primary)
                            Text("Bookmarks")
                                .font(.system(size: 11, weight: .medium))
                            if !sessionManager.bookmarks.isEmpty {
                                Text("\(sessionManager.bookmarks.count)")
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(Color.orange, in: Capsule())
                            }
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 5)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("Save reading bookmarks (§Chunk) or reopen recently imported books and documents")

                    Divider().frame(height: 14).opacity(0.35)

                    // Pronunciation Lexicon Button
                    Button {
                        showPronunciationSheet = true
                    } label: {
                        HStack(spacing: 3.5) {
                            Image(systemName: "character.phonetic")
                                .font(.system(size: 10.5))
                                .foregroundStyle(lexicon.isEnabled ? .mint : .secondary)
                            Text("Lexicon")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 5)
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Customize phonetic pronunciation rules for acronyms, technical terms & character names")

                    Divider().frame(height: 14).opacity(0.35)

                    // Multi-Voice Character Casting Table Button
                    Button {
                        showCastingSheet = true
                    } label: {
                        HStack(spacing: 3.5) {
                            Image(systemName: "person.2.wave.2.fill")
                                .font(.system(size: 10.5))
                                .foregroundStyle(castingManager.isEnabled ? .purple : .secondary)
                            Text("Cast")
                                .font(.system(size: 11, weight: .medium))
                            if castingManager.isEnabled {
                                Text("\(castingManager.cast.filter(\.isEnabled).count)")
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(Color.purple, in: Capsule())
                            }
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 5)
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Interactive Multi-Voice Character Casting Table — assign distinct voices & pitch profiles to book characters and dialogue")
                    .sheet(isPresented: $showCastingSheet) {
                        CharacterCastingSheet(scriptText: $textInput)
                    }
                    .sheet(isPresented: $showM4BExporterSheet) {
                        M4BAudiobookExportSheet(viewModel: viewModel, scriptText: textInput, defaultTitle: sessionManager.currentDocumentTitle)
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .openCharacterCastingSheet)) { _ in
                        showCastingSheet = true
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .openM4BAudiobookSheet)) { _ in
                        showM4BExporterSheet = true
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .miniReaderQuickSpeak)) { note in
                        guard let text = note.object as? String, !text.isEmpty else { return }
                        textInput = text
                        viewModel.activeChunkIndex = 0
                        if viewModel.isModelLoaded {
                            viewModel.startSynthesis(text: text, voice: viewModel.selectedVoice, startChunkIndex: 0)
                        }
                    }
                    .onReceive(NotificationCenter.default.publisher(for: .miniReaderTransportAction)) { note in
                        if let action = note.object as? String {
                            handleMiniReaderTransport(action)
                        }
                    }
                    .onReceive(NotificationCenter.default.publisher(for: Notification.Name("VoicesApp.ImportDocumentURL"))) { note in
                        if let url = note.object as? URL {
                            importDocumentAsync(from: url)
                        } else if let urls = note.object as? [URL] {
                            importDocumentsAsync(from: urls)
                        }
                    }

                    Divider().frame(height: 14).opacity(0.35)

                    // Studio Master Bus DSP Rack & ACX / LUFS Broadcast QC Button
                    Button {
                        showMasteringRackSheet = true
                    } label: {
                        HStack(spacing: 3.5) {
                            Image(systemName: "slider.vertical.3")
                                .font(.system(size: 10.5))
                                .foregroundStyle(masteringManager.preset != .bypass ? .orange : .secondary)
                            Text("Master")
                                .font(.system(size: 11, weight: .medium))
                            Text(masteringManager.preset.shortBadge)
                                .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(masteringManager.preset != .bypass ? .white : .secondary)
                                .padding(.horizontal, 4.5)
                                .padding(.vertical, 1)
                                .background(
                                    masteringManager.preset != .bypass ? Color.orange : Color.primary.opacity(0.08),
                                    in: Capsule()
                                )
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 5)
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Studio Master Bus DSP Rack — ACX / Audible -20 dB RMS compliance, -16 LUFS Podcast mastering, tube warmth, de-esser & ambience beds")
                    .sheet(isPresented: $showMasteringRackSheet) {
                        MasteringRackSheet(viewModel: viewModel)
                    }

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
                            clearAllStudioText()
                        } label: {
                            Label("Clear All Text (⇧⌘⌫)", systemImage: "trash")
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "text.book.closed")
                                .font(.system(size: 10.5))
                            Text("Scripts")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .padding(.horizontal, 6.5)
                        .padding(.vertical, 5)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Divider().frame(height: 14).opacity(0.35)

                    // Direct Silent Audio, Read-Along MP4 Video, Subtitles & Batch Export menu
                    Menu {
                        Section("Chapterized .M4B Audiobook (Chapters + Cover Art)") {
                            Button {
                                showM4BExporterSheet = true
                            } label: {
                                Label("📖 Configure & Export Chapterized .M4B Audiobook…", systemImage: "books.vertical.circle.fill")
                            }
                            Button {
                                viewModel.exportChapterizedM4B(
                                    text: textInput,
                                    bookTitle: sessionManager.currentDocumentTitle,
                                    author: "Codecaine MLX Audio Studio",
                                    voice: viewModel.selectedVoice
                                )
                            } label: {
                                Label("⚡ Quick Export .M4B Audiobook (\(max(1, chunks.count)) Chapters)", systemImage: "bolt.badge.clock.fill")
                            }
                        }
                        Divider()
                        Section("Full Book / All \(max(1, chunks.count)) Chapters Audio Export") {
                            ForEach(AudioExportFormat.allCases) { fmt in
                                Button {
                                    viewModel.saveAudioInFormat(
                                        fmt,
                                        fallbackText: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: 0,
                                        scopeOverride: .fullDocument
                                    )
                                } label: {
                                    Label("Export Full Audio as \(fmt.label) (All \(max(1, chunks.count)) Chapters)", systemImage: "waveform.badge.plus")
                                }
                            }
                        }
                        Divider()
                        Section("Quick Active Section (§\(activeChunkIdx + 1)) Export") {
                            ForEach([AudioExportFormat.wav, .m4b, .mp3, .m4a]) { fmt in
                                Button {
                                    viewModel.quickExportActiveChunk(
                                        text: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: activeChunkIdx,
                                        format: fmt,
                                        scope: .activeChunk
                                    )
                                } label: {
                                    Label("⚡ 1.5s Export Active Chunk §\(activeChunkIdx + 1) as \(fmt.rawValue.uppercased())", systemImage: "bolt.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Synchronized Read-Along Karaoke Video (All Chapters in 1 .MP4)") {
                            ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                Button {
                                    viewModel.exportReadAlongVideo(
                                        text: textInput,
                                        theme: viewModel.readAlongTheme,
                                        aspect: aspect,
                                        scope: .fullDocument
                                    )
                                } label: {
                                    Label("🎬 Export Read-Along MP4 — \(aspect.label) (All Chapters · \(max(1, chunks.count)) Sections in 1 MP4)", systemImage: "film.stack")
                                }
                            }
                            Menu("Partial / Section Clip Preview Only (Not All Chapters)…") {
                                if viewModel.audioURL != nil {
                                    ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                        Button {
                                            viewModel.exportCurrentSyncedAudioAsMP4(
                                                text: textInput,
                                                theme: viewModel.readAlongTheme,
                                                aspect: aspect
                                            )
                                        } label: {
                                            Label("⚡ Export Already-Played Clip Only — \(aspect.label)", systemImage: "bolt.badge.film.fill")
                                        }
                                    }
                                    Divider()
                                }
                                ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                    Button {
                                        viewModel.exportReadAlongVideo(
                                            text: textInput,
                                            theme: viewModel.readAlongTheme,
                                            aspect: aspect,
                                            scope: .fastSection
                                        )
                                    } label: {
                                        Label("⚡ Preview Section Clip Only — \(aspect.label) (§\(activeChunkIdx + 1) · 4 Chunks)", systemImage: "bolt.fill")
                                    }
                                }
                            }
                        }
                        Divider()
                        Section("Karaoke, Subtitles & Transcripts") {
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
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "lrc")
                            } label: {
                                Label("Export Synchronized Lyrics (.LRC)", systemImage: "music.note.list")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "ttml")
                            } label: {
                                Label("Export Broadcast Timed Text (.TTML)", systemImage: "chevron.left.forwardslash.chevron.right")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "sbv")
                            } label: {
                                Label("Export YouTube Captions (.SBV)", systemImage: "play.rectangle")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "json")
                            } label: {
                                Label("Export Word Timings (.JSON)", systemImage: "curlybraces")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "md")
                            } label: {
                                Label("Export Cleaned Document (.MD)", systemImage: "doc.richtext")
                            }
                            Button {
                                viewModel.exportSubtitles(text: textInput, format: "txt")
                            } label: {
                                Label("Export Cleaned Plain Text (.TXT)", systemImage: "doc.plaintext")
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
                        HStack(spacing: 4) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 10.5))
                            Text("Export")
                                .font(.system(size: 11, weight: .medium))
                        }
                        .padding(.horizontal, 6.5)
                        .padding(.vertical, 5)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Divider().frame(height: 14).opacity(0.35)

                    // Takes History button
                    Button {
                        showHistory = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "clock.arrow.circlepath")
                                .font(.system(size: 10.5))
                            Text("Takes")
                                .font(.system(size: 11, weight: .medium))
                            if !viewModel.history.isEmpty {
                                Text("\(viewModel.history.count)")
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 4.5)
                                    .padding(.vertical, 1)
                                    .background(Color.blue, in: Capsule())
                            }
                        }
                        .padding(.horizontal, 6.5)
                        .padding(.vertical, 5)
                    }
                    .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 3)
                .background(
                    Capsule()
                        .fill(theme.pillFill)
                        .overlay(
                            Capsule()
                                .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                        )
                )
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)

            // 1B. Codecaine Prosody, Cinema Theme & Hierarchical Chunk Navigator Ribbon
            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
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
                                .background(theme.pillFill, in: Capsule())
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
                                .background(theme.pillFill, in: Capsule())
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
                                .background(theme.pillFill, in: Capsule())
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
                                .background(theme.pillFill, in: Capsule())
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
                                .background(theme.pillFill, in: Capsule())
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()

                            // Codecaine File Manager 2-Second Synthesis Scope Picker
                            Menu {
                                Section("Universal Book, Office & Document Synthesis Speed / Scope") {
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
                                        .foregroundStyle(theme.isLight ? .orange : .yellow)
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

                            // Saved Studio Profiles / Presets Quick Menu
                            Menu {
                                Section("★ Saved Studio Profiles (\(prefs.userProfiles.count))") {
                                    ForEach(prefs.userProfiles) { profile in
                                        Button {
                                            viewModel.applyStudioProfile(profile)
                                            viewModel.applyMasterBusInPlaceToCurrentAudio()
                                        } label: {
                                            Label("\(profile.name) — \(profile.subtitle)", systemImage: profile.icon)
                                        }
                                    }
                                }
                                Divider()
                                Button {
                                    let vName = viewModel.selectedVoice?.name.components(separatedBy: " ").first ?? "Studio"
                                    newProfileName = "\(vName) \(String(format: "%.2gx", viewModel.playbackRate)) (\(masteringManager.preset.shortBadge))"
                                    showSaveProfilePopover = true
                                } label: {
                                    Label("★ Save Current Setup as Favorite Profile…", systemImage: "star.circle.fill")
                                }

                                let customProfiles = prefs.userProfiles.filter { !$0.isBuiltIn }
                                if !customProfiles.isEmpty {
                                    Menu("Delete Custom Profile") {
                                        ForEach(customProfiles) { profile in
                                            Button(role: .destructive) {
                                                prefs.deleteProfile(id: profile.id)
                                            } label: {
                                                Label("Delete '\(profile.name)'", systemImage: "trash")
                                            }
                                        }
                                    }
                                }

                                Button {
                                    prefs.resetProfilesToDefaults()
                                } label: {
                                    Label("Restore Default Starter Profiles", systemImage: "arrow.counterclockwise")
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "star.circle.fill")
                                        .font(.system(size: 10.5))
                                        .foregroundStyle(theme.isLight ? .orange : .yellow)
                                    Text("Presets (\(prefs.userProfiles.count))")
                                        .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                                }
                                .padding(.horizontal, 9)
                                .padding(.vertical, 5)
                                .background(Color.yellow.opacity(0.14), in: Capsule())
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.yellow.opacity(0.38), lineWidth: 0.8)
                                )
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()
                            .help("1-click Studio Profiles — save or recall your favorite Model + Voice + Speed + Pitch + Mastering + Theme combinations")
                            .popover(isPresented: $showSaveProfilePopover, arrowEdge: .bottom) {
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "star.circle.fill")
                                            .foregroundStyle(.yellow)
                                        Text("Save Current Studio Setup as Profile")
                                            .font(.system(size: 13, weight: .bold))
                                    }
                                    Text("Saves \(viewModel.selectedVoice?.name ?? "Freya") · \(String(format: "%.2gx", viewModel.playbackRate)) · \(masteringManager.preset.shortBadge) · \(viewModel.readAlongTheme.label)")
                                        .font(.system(size: 11))
                                        .foregroundStyle(.secondary)
                                    TextField("Profile name (e.g. Late Night Sci-Fi Narrator)", text: $newProfileName)
                                        .textFieldStyle(.roundedBorder)
                                        .font(.system(size: 12))
                                    HStack {
                                        Spacer()
                                        Button("Cancel") {
                                            showSaveProfilePopover = false
                                        }
                                        .buttonStyle(.bordered)
                                        .controlSize(.small)
                                        Button("Save Profile") {
                                            prefs.saveCustomProfile(
                                                name: newProfileName,
                                                modelId: viewModel.modelId,
                                                voice: viewModel.selectedVoice,
                                                playbackRate: viewModel.playbackRate,
                                                pitchSemitones: viewModel.pitchSemitones,
                                                paragraphPauseMs: viewModel.paragraphPauseMs,
                                                maxChunkLength: viewModel.maxChunkLength,
                                                theme: viewModel.readAlongTheme,
                                                masteringPreset: masteringManager.preset
                                            )
                                            showSaveProfilePopover = false
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .controlSize(.small)
                                    }
                                }
                                .padding(14)
                                .frame(width: 340)
                            }
                        }
                    }
                    .fixedSize(horizontal: false, vertical: true)

                    Spacer(minLength: 6)

                    // Live Hierarchical Chunk Navigator Strip (§1 · §2 · §3...) + Surgical Retake & In-Book Search Popover
                    if !chunks.isEmpty {
                        Button {
                            viewModel.retakeAndSpliceChunk(
                                fullText: textInput,
                                chunkIndex: activeChunkIdx,
                                voice: viewModel.selectedVoice
                            )
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.triangle.2.circlepath.circle.fill")
                                    .font(.system(size: 10.5, weight: .bold))
                                    .foregroundStyle(.orange)
                                Text("Retake §\(activeChunkIdx + 1)")
                                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4.5)
                            .background(Color.orange.opacity(0.14), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .fixedSize()
                        .disabled(viewModel.isGenerating)
                        .help("Surgical Retake & Splice — re-synthesize only §\(activeChunkIdx + 1) and splice it into the master WAV with 12ms crossfades")

                        Button {
                            showChunkSearchPopover.toggle()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "magnifyingglass")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.cyan)
                                Text("Find §")
                                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4.5)
                            .background(theme.pillFill, in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .fixedSize()
                        .help("Search chapters, character names, or keywords across all \(chunks.count) chunks")
                        .popover(isPresented: $showChunkSearchPopover, arrowEdge: .bottom) {
                            chunkSearchPopoverView(chunks: chunks, timings: timings, activeChunkIdx: activeChunkIdx)
                        }

                        ScrollView(.horizontal, showsIndicators: false) {
                            LazyHStack(spacing: 5) {
                                Text("CHUNKS (\(chunks.count))")
                                    .font(.system(size: 9.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(.secondary)

                                ForEach(chunks) { chunk in
                                    let isCurrent = chunk.index == activeChunkIdx
                                    let isBookmarked = sessionManager.isChunkBookmarked(chunk.index)
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
                                            if isBookmarked {
                                                Image(systemName: "bookmark.fill")
                                                    .font(.system(size: 8.5))
                                                    .foregroundStyle(.orange)
                                            }
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
                                                .fill(isCurrent ? Color.blue : theme.pillFill)
                                                .overlay(
                                                    Capsule()
                                                        .strokeBorder(isCurrent ? Color.blue : (isBookmarked ? Color.orange.opacity(0.6) : theme.pillStroke), lineWidth: 0.8)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                    .help("Chunk §\(chunk.index + 1) (\(chunk.wordCount) words, \(chunk.charCount) chars) — Click to jump, Option-Click to synthesize from §\(chunk.index + 1)")
                                }
                            }
                        }
                        .frame(maxWidth: 235, maxHeight: 28)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.horizontal, 14)

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
                        },
                        onClearAllText: {
                            clearAllStudioText()
                        }
                    )
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 52)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if textInput.count > 12_000 && chunks.count > 10 {
                    // Universal Book & Multi-Format Sectioned Editor Mode (prevents NSTextView from stalling on 500k+ char books/docs)
                    let chunksPerSection = 12
                    let totalSections = max(1, (chunks.count + chunksPerSection - 1) / chunksPerSection)
                    let currentSection = min(totalSections - 1, max(0, activeChunkIdx / chunksPerSection))
                    let sectionStartIdx = currentSection * chunksPerSection
                    let sectionEndIdx = min(chunks.count - 1, sectionStartIdx + chunksPerSection - 1)
                    let sectionChunks = Array(chunks[sectionStartIdx...sectionEndIdx])
                    let sectionWords = sectionChunks.reduce(0) { $0 + $1.wordCount }
                    let docCategory = UniversalFileFormatCatalog.category(
                        forExtension: URL(fileURLWithPath: sessionManager.currentDocumentTitle).pathExtension
                    )
                    let docBadgeLabel = UniversalFileFormatCatalog.badgeLabel(
                        forDocumentTitle: sessionManager.currentDocumentTitle
                    )
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
                                Image(systemName: docCategory.icon)
                                    .font(.system(size: 11.5, weight: .bold))
                                    .foregroundStyle(theme.activeWordBackground)
                                Text(docBadgeLabel)
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
                                clearAllStudioText()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "trash.fill")
                                        .font(.system(size: 9.5))
                                    Text("Clear All")
                                        .font(.system(size: 11, weight: .semibold))
                                }
                                .foregroundStyle(.red)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4.5)
                                .background(Color.red.opacity(0.14), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .help("Clear entire loaded document and return to empty script editor (⇧⌘⌫)")

                            Button {
                                viewModel.activeChunkIndex = max(0, (currentSection - 1) * chunksPerSection)
                            } label: {
                                Label("Prev Section", systemImage: "chevron.left")
                                    .font(.system(size: 11, weight: .semibold))
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 4)
                                    .background(theme.pillFill, in: Capsule())
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
                                    .background(theme.pillFill, in: Capsule())
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
                                .foregroundStyle(theme.activeWordForeground)
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
                                .foregroundStyle(theme.isLight ? .orange : .yellow)
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

                                            // Import Any File Format Card
                                            Button {
                                                showDocImporter = true
                                            } label: {
                                                VStack(alignment: .leading, spacing: 6) {
                                                    HStack(spacing: 6) {
                                                        Image(systemName: "doc.badge.plus")
                                                            .font(.system(size: 12, weight: .semibold))
                                                            .foregroundStyle(theme.activeWordBackground)
                                                        Text("Import Any Format")
                                                            .font(.system(size: 12.5, weight: .semibold))
                                                            .foregroundStyle(theme.editorTextColor)
                                                    }
                                                    Text("EPUB · MOBI · PDF · DOCX · PPTX · XLSX · SRT · OCR")
                                                        .font(.system(size: 11))
                                                        .foregroundStyle(.secondary)
                                                        .lineLimit(1)
                                                    Text("Open 180+ file formats: E-Books, Office, iWork, LaTeX, Code, Archives & Images.")
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
                                #if os(macOS)
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(textInput, forType: .string)
                                #endif
                                copiedScript = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    copiedScript = false
                                }
                            } label: {
                                Label(
                                    copiedScript ? "Copied" : "Copy",
                                    systemImage: copiedScript ? "checkmark.circle.fill" : "doc.on.doc"
                                )
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(copiedScript ? .green : .secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Copy entire script to clipboard")

                            Divider().frame(height: 10)

                            Button {
                                clearAllStudioText()
                            } label: {
                                Label("Clear All", systemImage: "trash.fill")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(.red.opacity(0.9))
                            }
                            .buttonStyle(.plain)
                            .help("Clear all text in editor (⇧⌘⌫)")

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
                            .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                    )
                    .fixedSize()
                }
                .padding(14)
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onDrop(of: [.fileURL, .item], isTargeted: $isDropTargeted) { providers in
                guard !providers.isEmpty else { return false }
                let group = DispatchGroup()
                let lock = NSLock()
                var droppedURLs: [URL] = []
                for provider in providers {
                    group.enter()
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        if let url {
                            lock.lock()
                            droppedURLs.append(url)
                            lock.unlock()
                        }
                        group.leave()
                    }
                }
                group.notify(queue: .main) {
                    if !droppedURLs.isEmpty {
                        importDocumentsAsync(from: droppedURLs)
                    }
                }
                return true
            }

            // 3. Unified Apple Glass Bottom Studio Dock
            VStack(spacing: 12) {
                // Status / Error Banner inside Dock
                if !viewModel.generationProgress.isEmpty {
                    VStack(spacing: 6) {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text(viewModel.generationProgress)
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.primary.opacity(0.90))
                            Spacer()
                            if viewModel.isGenerating && viewModel.totalChunkCount > 1 {
                                let completedFrac = Double(min(viewModel.totalChunkCount, viewModel.synthesizingChunkIndex + 1)) / Double(max(1, viewModel.totalChunkCount))
                                Text("Chapter §\(min(viewModel.totalChunkCount, viewModel.synthesizingChunkIndex + 1)) of \(viewModel.totalChunkCount) · \(Int(round(completedFrac * 100)))%")
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundStyle(theme.activeWordBackground)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 2.5)
                                    .background(theme.activeWordBackground.opacity(0.14), in: Capsule())
                            }
                            if viewModel.isGenerating {
                                Button {
                                    viewModel.stop()
                                } label: {
                                    Label("Cancel", systemImage: "xmark.circle.fill")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(.red)
                                }
                                .buttonStyle(.plain)
                                .help("Cancel active synthesis or export")
                            }
                        }
                        if viewModel.isGenerating && viewModel.totalChunkCount > 1 {
                            let completedFrac = Double(min(viewModel.totalChunkCount, viewModel.synthesizingChunkIndex + 1)) / Double(max(1, viewModel.totalChunkCount))
                            GeometryReader { geo in
                                ZStack(alignment: .leading) {
                                    Capsule()
                                        .fill(theme.pillFill)
                                    Capsule()
                                        .fill(theme.activeWordBackground)
                                        .frame(width: max(12, geo.size.width * min(1.0, max(0.04, completedFrac))))
                                }
                            }
                            .frame(height: 5)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(theme.recessedFill)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .strokeBorder(theme.recessedStroke, lineWidth: 0.7)
                            )
                    )
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

                exportCompletionBannerRow

                // Upper Dock Row: Dedicated Full-Width Built-In Voices Palette + Category Menu + Voice Browser Trigger
                upperDockVoicesPaletteRow

                // Middle Dock Row: Apple Podcasts-style Transport Bar (when audio is generated)
                if viewModel.audioURL != nil {
                    HStack(spacing: 12) {
                        CompactAudioPlayer(
                            isPlaying: viewModel.isPlaying,
                            currentTime: viewModel.currentTime,
                            duration: viewModel.duration,
                            onPlayPause: { viewModel.togglePlayPause() },
                            onSeek: { viewModel.seek(to: $0) },
                            audioURL: viewModel.audioURL,
                            accentColor: theme.activeWordBackground
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
                            Section("Save Full Book / All \(max(1, chunks.count)) Chapters Audio") {
                                ForEach(AudioExportFormat.allCases) { fmt in
                                    Button {
                                        viewModel.saveAudioInFormat(
                                            fmt,
                                            fallbackText: textInput,
                                            voice: viewModel.selectedVoice,
                                            chunkIndex: 0,
                                            scopeOverride: .fullDocument
                                        )
                                    } label: {
                                        Label("Save Full Audio as \(fmt.label)", systemImage: "waveform")
                                    }
                                }
                            }
                            Divider()
                            Section("Save Chapterized .M4B Audiobook") {
                                Button {
                                    showM4BExporterSheet = true
                                } label: {
                                    Label("📖 Configure & Export Chapterized .M4B Audiobook…", systemImage: "books.vertical.circle.fill")
                                }
                            }
                            Divider()
                            Section("Save Read-Along Karaoke Video (All Chapters in 1 .MP4)") {
                                ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                    Button("🎬 Save Read-Along MP4 — \(aspect.label) (All Chapters · \(max(1, chunks.count)) Sections in 1 MP4)") {
                                        viewModel.exportReadAlongVideo(
                                            text: textInput,
                                            theme: viewModel.readAlongTheme,
                                            aspect: aspect,
                                            scope: .fullDocument
                                        )
                                    }
                                }
                                Menu("Partial / Played Clip Preview Only (Not All Chapters)…") {
                                    ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                        Button("⚡ Save Current Played Audio Clip Only — \(aspect.label)") {
                                            viewModel.exportCurrentSyncedAudioAsMP4(
                                                text: textInput,
                                                theme: viewModel.readAlongTheme,
                                                aspect: aspect
                                            )
                                        }
                                    }
                                }
                            }
                            Divider()
                            Section("Save Synced Subtitles & Transcripts") {
                                Button("Save Subtitles (.SRT)") {
                                    viewModel.exportSubtitles(text: textInput, format: "srt")
                                }
                                Button("Save WebVTT (.VTT)") {
                                    viewModel.exportSubtitles(text: textInput, format: "vtt")
                                }
                                Button("Save Karaoke (.ASS)") {
                                    viewModel.exportSubtitles(text: textInput, format: "ass")
                                }
                                Button("Save Synchronized Lyrics (.LRC)") {
                                    viewModel.exportSubtitles(text: textInput, format: "lrc")
                                }
                                Button("Save Timed Text (.TTML)") {
                                    viewModel.exportSubtitles(text: textInput, format: "ttml")
                                }
                                Button("Save YouTube Captions (.SBV)") {
                                    viewModel.exportSubtitles(text: textInput, format: "sbv")
                                }
                                Button("Save Word Timings (.JSON)") {
                                    viewModel.exportSubtitles(text: textInput, format: "json")
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
                            .fill(theme.recessedFill)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(theme.recessedStroke, lineWidth: 0.8)
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
                                .fill(theme.pillFill)
                                .overlay(
                                    Capsule()
                                        .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                )
                        )
                    }
                    .buttonStyle(.plain)
                    .help("Browse all built-in voices or clone a custom voice from audio/microphone")

                    // 1-Click Favorite Active Voice Star Button
                    if let activeVoice = viewModel.selectedVoice {
                        let isFavVoice = prefs.isFavoriteVoice(activeVoice)
                        Button {
                            prefs.toggleFavoriteVoice(activeVoice)
                        } label: {
                            Image(systemName: isFavVoice ? "star.fill" : "star")
                                .font(.system(size: 13.5, weight: .semibold))
                                .foregroundStyle(isFavVoice ? (theme.isLight ? .orange : .yellow) : .secondary)
                                .frame(width: 38, height: 44)
                                .background(
                                    Circle()
                                        .fill(isFavVoice ? Color.yellow.opacity(0.14) : theme.pillFill)
                                        .overlay(
                                            Circle()
                                                .strokeBorder(isFavVoice ? Color.yellow.opacity(0.45) : theme.pillStroke, lineWidth: 0.8)
                                        )
                                )
                        }
                        .buttonStyle(.plain)
                        .help(isFavVoice ? "Remove '\(activeVoice.name)' from favorite voices" : "Star '\(activeVoice.name)' as a favorite voice")
                    }

                    // 1-Click Active Voice Audition Button (Synthesizes/plays genuine neural sample for the selected voice)
                    if let activeVoice = viewModel.selectedVoice {
                        let isAuditioningThis = viewModel.isAuditioningVoice && viewModel.auditioningVoiceKey == activeVoice.voiceKey
                        Button {
                            viewModel.toggleVoiceAudition(activeVoice)
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: isAuditioningThis ? "stop.fill" : "speaker.wave.2.fill")
                                    .font(.system(size: 12.5, weight: .semibold))
                                    .foregroundStyle(isAuditioningThis ? .orange : .blue)
                                Text(isAuditioningThis ? "Stop" : "Sample")
                                    .font(.system(size: 11.5, weight: .semibold))
                                    .foregroundStyle(.primary)
                            }
                            .padding(.horizontal, 12)
                            .frame(height: 44)
                            .background(
                                Capsule()
                                    .fill(isAuditioningThis ? Color.orange.opacity(0.16) : theme.pillFill)
                                    .overlay(
                                        Capsule()
                                            .strokeBorder(
                                                isAuditioningThis ? Color.orange.opacity(0.45) : theme.pillStroke,
                                                lineWidth: 0.8
                                            )
                                    )
                            )
                        }
                        .buttonStyle(.plain)
                        .help("Audition '\(activeVoice.name)' with a real neural voice sample")
                    }

                    // Quick Model Engine Picker Capsule (Favorites + Categorized + Cached Pinned)
                    HStack(spacing: 4) {
                        Menu {
                            let favPresets = TTSViewModel.presetModels.filter { prefs.isFavoriteTTSModel($0.id) }
                            if !favPresets.isEmpty {
                                Section("★ Favorite TTS Engines (\(favPresets.count))") {
                                    ForEach(favPresets, id: \.id) { preset in
                                        let cached = modelsManager.isModelCached(preset.id)
                                        let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                        let labelText = "★ \(preset.name)\(cached ? " · \(sizeStr ?? "Cached")" : "")"
                                        Button {
                                            viewModel.modelId = preset.id
                                            Task { await viewModel.reloadModel() }
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
                            }

                            let cachedPresets = TTSViewModel.presetModels.filter { modelsManager.isModelCached($0.id) }
                            if !cachedPresets.isEmpty {
                                Section("✓ Downloaded & Ready on Disk (\(cachedPresets.count))") {
                                    ForEach(cachedPresets, id: \.id) { preset in
                                        let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                        let labelText = "✓ \(preset.name)\(sizeStr.map { " · \($0)" } ?? "")"
                                        Button {
                                            viewModel.modelId = preset.id
                                            Task { await viewModel.reloadModel() }
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
                            }

                            Section("Browse \(TTSViewModel.presetModels.count) MLX TTS Engines by Category") {
                                ForEach(TTSViewModel.categorizedPresetModels, id: \.category) { group in
                                    Menu {
                                        ForEach(group.models, id: \.id) { preset in
                                            let cached = modelsManager.isModelCached(preset.id)
                                            let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                            let isFav = prefs.isFavoriteTTSModel(preset.id)
                                            let prefix = isFav ? "★ " : ""
                                            let labelText = cached
                                                ? "\(prefix)✓ \(preset.name) [Cached\(sizeStr.map { " · \($0)" } ?? "")]"
                                                : "\(prefix)\(preset.name)"
                                            Button {
                                                viewModel.modelId = preset.id
                                                Task { await viewModel.reloadModel() }
                                            } label: {
                                                if viewModel.modelId == preset.id {
                                                    Label(labelText, systemImage: "checkmark")
                                                } else {
                                                    Text(labelText)
                                                }
                                            }
                                        }
                                    } label: {
                                        Label("\(group.category) (\(group.models.count))", systemImage: group.icon)
                                    }
                                }
                            }
                            Divider()
                            Button {
                                prefs.toggleFavoriteTTSModel(viewModel.modelId)
                            } label: {
                                Label(
                                    prefs.isFavoriteTTSModel(viewModel.modelId) ? "Remove Current Engine from ★ Favorites" : "Add Current Engine to ★ Favorites",
                                    systemImage: prefs.isFavoriteTTSModel(viewModel.modelId) ? "star.slash.fill" : "star.fill"
                                )
                            }
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
                                    .fill(theme.pillFill)
                                    .overlay(
                                        Capsule()
                                            .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                    )
                            )
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()

                        Button {
                            prefs.toggleFavoriteTTSModel(viewModel.modelId)
                        } label: {
                            let isFav = prefs.isFavoriteTTSModel(viewModel.modelId)
                            Image(systemName: isFav ? "star.fill" : "star")
                                .font(.system(size: 13.5, weight: .semibold))
                                .foregroundStyle(isFav ? (theme.isLight ? .orange : .yellow) : .secondary)
                                .frame(width: 38, height: 44)
                                .background(
                                    Circle()
                                        .fill(isFav ? Color.yellow.opacity(0.14) : theme.pillFill)
                                        .overlay(
                                            Circle()
                                                .strokeBorder(isFav ? Color.yellow.opacity(0.45) : theme.pillStroke, lineWidth: 0.8)
                                        )
                                )
                        }
                        .buttonStyle(.plain)
                        .help(prefs.isFavoriteTTSModel(viewModel.modelId) ? "Remove active TTS engine from favorites" : "Star active TTS engine as favorite")
                    }

                    // Inspector / Settings Button
                    Button(action: { showSettings = true }) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 44, height: 44)
                            .background(
                                Circle()
                                    .fill(theme.pillFill)
                                    .overlay(
                                        Circle()
                                            .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                    )
                            )
                    }
                    .buttonStyle(.plain)
                    .help("Configure temperature, top-p, max tokens, HuggingFace token & cache")

                    // 1-Click RAD Developer Code & API Snippet Menu
                    Menu {
                        let port = LocalhostServerManager.shared.primaryPort
                        let voiceLabel = viewModel.selectedVoice?.name ?? "Freya"
                        Section("Copy Live RAD Snippet (\(viewModel.modelId.split(separator: "/").last.map(String.init) ?? viewModel.modelId) · \(voiceLabel))") {
                            Button {
                                #if os(macOS)
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(viewModel.radCodeSnippet(language: "curl", text: textInput, port: port), forType: .string)
                                #endif
                            } label: {
                                Label("Copy cURL Command (POST /v1/audio/speech)", systemImage: "terminal.fill")
                            }

                            Button {
                                #if os(macOS)
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(viewModel.radCodeSnippet(language: "python", text: textInput, port: port), forType: .string)
                                #endif
                            } label: {
                                Label("Copy Python Snippet (OpenAI SDK Drop-In)", systemImage: "chevron.left.forwardslash.chevron.right")
                            }

                            Button {
                                #if os(macOS)
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(viewModel.radCodeSnippet(language: "typescript", text: textInput, port: port), forType: .string)
                                #endif
                            } label: {
                                Label("Copy TypeScript / Node.js Snippet (fetch)", systemImage: "curlybraces")
                            }

                            Button {
                                #if os(macOS)
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(viewModel.radCodeSnippet(language: "swift", text: textInput, port: port), forType: .string)
                                #endif
                            } label: {
                                Label("Copy Swift async/await Snippet (URLSession)", systemImage: "swift")
                            }
                        }
                        Divider()
                        Button {
                            #if os(macOS)
                            if let url = URL(string: "http://127.0.0.1:\(port)/playground") {
                                NSWorkspace.shared.open(url)
                            }
                            #endif
                        } label: {
                            Label("Open Interactive Web Playground (:8080/playground)…", systemImage: "safari.fill")
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "chevron.left.forwardslash.chevron.right")
                                .font(.system(size: 11.5, weight: .bold))
                                .foregroundStyle(.cyan)
                            Text("RAD API")
                                .font(.system(size: 11.5, weight: .semibold))
                                .foregroundStyle(.primary)
                        }
                        .padding(.horizontal, 12)
                        .frame(height: 44)
                        .background(
                            Capsule()
                                .fill(Color.cyan.opacity(0.10))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.cyan.opacity(0.32), lineWidth: 0.8)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("1-click copy ready-to-run cURL, Python, TypeScript, or Swift code for the active model, voice, speed & script")

                    Spacer()

                    // Direct Silent Export Menu (Export without pressing Generate Speech & without playing sound)
                    Menu {
                        Section("Full Book / All \(max(1, chunks.count)) Chapters Silent Export") {
                            Button {
                                showM4BExporterSheet = true
                            } label: {
                                Label("📖 Configure & Export Chapterized .M4B Audiobook…", systemImage: "books.vertical.circle.fill")
                            }
                            ForEach(AudioExportFormat.allCases) { fmt in
                                Button {
                                    viewModel.saveAudioInFormat(
                                        fmt,
                                        fallbackText: textInput,
                                        voice: viewModel.selectedVoice,
                                        chunkIndex: 0,
                                        scopeOverride: .fullDocument
                                    )
                                } label: {
                                    Label("Export Full Document as \(fmt.label) (All \(max(1, chunks.count)) Chapters)", systemImage: "arrow.down.doc.fill")
                                }
                            }
                        }
                        Divider()
                        Section("Synchronized Read-Along Karaoke Video (All Chapters in 1 .MP4)") {
                            ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                Button {
                                    viewModel.exportReadAlongVideo(
                                        text: textInput,
                                        theme: viewModel.readAlongTheme,
                                        aspect: aspect,
                                        scope: .fullDocument
                                    )
                                } label: {
                                    Label("🎬 Export Read-Along MP4 — \(aspect.label) (All Chapters · \(max(1, chunks.count)) Sections in 1 MP4)", systemImage: "film.stack")
                                }
                            }
                        }
                        Divider()
                        Section("Quick Single-Section Preview (§\(activeChunkIdx + 1) Only — Not All Chapters)") {
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
                                    Label("⚡ 1.5s Export Active Chunk §\(activeChunkIdx + 1) as \(fmt.rawValue.uppercased())", systemImage: "bolt.fill")
                                }
                            }
                            Menu("Partial Clip MP4 Preview (§\(activeChunkIdx + 1) Only)…") {
                                if viewModel.audioURL != nil {
                                    ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                        Button {
                                            viewModel.exportCurrentSyncedAudioAsMP4(
                                                text: textInput,
                                                theme: viewModel.readAlongTheme,
                                                aspect: aspect
                                            )
                                        } label: {
                                            Label("⚡ Export Already-Played Clip Only — \(aspect.label)", systemImage: "bolt.badge.film.fill")
                                        }
                                    }
                                    Divider()
                                }
                                ForEach(ReadAlongVideoAspect.allCases) { aspect in
                                    Button {
                                        viewModel.exportReadAlongVideo(
                                            text: textInput,
                                            theme: viewModel.readAlongTheme,
                                            aspect: aspect,
                                            scope: .fastSection
                                        )
                                    } label: {
                                        Label("⚡ Preview Section §\(activeChunkIdx + 1) MP4 Only — \(aspect.label)", systemImage: "bolt.fill")
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "bolt.horizontal.circle.fill")
                                .font(.system(size: 13.5, weight: .semibold))
                                .foregroundStyle(theme.isLight ? .orange : .yellow)
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
                                .fill(theme.pillFill)
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.yellow.opacity(0.32), lineWidth: 0.9)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .disabled(textInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && viewModel.audioURL == nil)
                    .help("Instant export (<1s when audio is generated, ~1.5–2s silent synthesis when not yet generated)")

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
                                        : [theme.pillFill, theme.recessedFill],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                in: Capsule()
                            )
                            .overlay(
                                Capsule()
                                    .strokeBorder(canGenerate ? Color.white.opacity(0.25) : theme.pillStroke, lineWidth: 0.8)
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
            allowedContentTypes: [.item, .content, .data, .folder],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result, !urls.isEmpty {
                importDocumentsAsync(from: urls)
            }
        }
        .sheet(isPresented: $showVoices) {
            VoicesView(
                modelId: viewModel.modelId,
                builtInVoices: viewModel.builtInVoices,
                activeVoice: viewModel.selectedVoice,
                auditioningVoiceKey: viewModel.isAuditioningVoice ? viewModel.auditioningVoiceKey : nil,
                recentlyUsed: $recentlyUsed,
                customVoices: $customVoices,
                onPreviewVoice: { voice in
                    viewModel.toggleVoiceAudition(voice)
                },
                onPreviewVoiceWithPhrase: { voice, phrase in
                    viewModel.toggleVoiceAudition(voice, customPhrase: phrase)
                }
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
        .sheet(isPresented: $showPronunciationSheet) {
            PronunciationLexiconSheet()
        }
        .onChange(of: customVoices) { _, newValue in
            Voice.saveCustomVoices(newValue)
        }
        .onChange(of: showTeleprompter) { _, newVal in
            UserDefaults.standard.set(newVal, forKey: "VoicesApp.VoiceStudio.showTeleprompter")
        }
        .onChange(of: cleanOptions) { _, newOpts in
            newOpts.saveToDefaults()
        }
        .onChange(of: textInput) { _, newText in
            sessionManager.saveSessionScript(newText, chunkIndex: viewModel.activeChunkIndex)
            syncMiniReaderState()
        }
        .onChange(of: viewModel.activeChunkIndex) { _, newChunk in
            if previewWordId != nil && !viewModel.isPlaying {
                previewWordId = timings.first(where: { $0.chunkIndex == newChunk })?.id
            }
            sessionManager.saveActiveChunkIndex(newChunk)
            syncMiniReaderState()
        }
        .onReceive(NotificationCenter.default.publisher(for: .pasteAndReadClipboard)) { _ in
            pasteAndReadFromClipboard()
        }
        .onReceive(NotificationCenter.default.publisher(for: .cloneVoiceFromAudio)) { note in
            let refreshed = Voice.loadCustomVoices()
            customVoices = refreshed
            if let clonedVoice = (note.object as? Voice) ?? refreshed.first {
                viewModel.selectedVoice = clonedVoice
                viewModel.generationProgress = "Cloned voice '\(clonedVoice.name)' ready in Voice Studio!"
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(3.0))
                    if self.viewModel.generationProgress.hasPrefix("Cloned voice") {
                        self.viewModel.generationProgress = ""
                    }
                }
            }
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
            if ProcessInfo.processInfo.arguments.contains("--demo-player"),
               let sampleURL = Voice.bundledStudioSamples().first?.url {
                viewModel.audioURL = sampleURL
                viewModel.duration = 9.4
                viewModel.currentTime = 3.8
                viewModel.tokensPerSecond = 84.6
                let demoMP4 = URL(fileURLWithPath: ("~/Desktop/read_along_paper_9x16.mp4" as NSString).expandingTildeInPath)
                if FileManager.default.fileExists(atPath: demoMP4.path) {
                    viewModel.lastExportedFileURL = demoMP4
                    viewModel.lastExportedSummary = "read_along_paper_9x16.mp4 • 9:16 Read-Along MP4 + SRT/VTT (All Chapters)"
                } else {
                    viewModel.lastExportedFileURL = sampleURL
                    viewModel.lastExportedSummary = "Deadlocked_2_Audiobook.m4b • 25 Chapters + Cover Art"
                }
            }
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
            } else if !ProcessInfo.processInfo.arguments.contains("--read-along"),
                      let restored = sessionManager.restoreSessionScript() {
                textInput = restored.text
                viewModel.activeChunkIndex = restored.chunkIndex
            }
            if textInput.contains("\\'") || textInput.contains("\\\"") {
                textInput = textInput
                    .replacingOccurrences(of: "\\'", with: "'")
                    .replacingOccurrences(of: "\\\"", with: "\"")
            }
            syncMiniReaderState()
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
            if ProcessInfo.processInfo.arguments.contains("--demo-player") {
                try? await Task.sleep(nanoseconds: 150_000_000)
                let demoURL = Voice.bundledStudioSamples().first?.url ?? URL(fileURLWithPath: "/tmp/demo_studio_waveform.wav")
                viewModel.audioURL = demoURL
                viewModel.duration = 12.6
                viewModel.currentTime = 4.8
                viewModel.tokensPerSecond = 84.6
            }
            if ProcessInfo.processInfo.arguments.contains("--auto-speak"), !textInput.isEmpty {
                showTeleprompter = true
                viewModel.startSynthesis(
                    text: textInput,
                    voice: viewModel.selectedVoice,
                    startChunkIndex: 0
                )
            }
        }
    }

    @ViewBuilder
    private var upperDockVoicesPaletteRow: some View {
        if !viewModel.builtInVoices.isEmpty || !customVoices.isEmpty {
            let allQuickVoices = (customVoices + viewModel.builtInVoices).sorted { a, b in
                let favA = prefs.isFavoriteVoice(a)
                let favB = prefs.isFavoriteVoice(b)
                if favA != favB { return favA && !favB }
                return false
            }
            HStack(spacing: 8) {
                Menu {
                    let favVoices = (customVoices + viewModel.builtInVoices).filter { prefs.isFavoriteVoice($0) }
                    let recentVoices = prefs.recentVoiceKeys.compactMap { key in
                        (customVoices + viewModel.builtInVoices).first(where: { $0.voiceKey == key })
                    }
                    let conversational = viewModel.builtInVoices.filter { $0.category == "Conversational" }
                    let narrators = viewModel.builtInVoices.filter { $0.category == "Narrator" }
                    let characters = viewModel.builtInVoices.filter { $0.category == "Character" }
                    let clones = viewModel.builtInVoices.filter { $0.category == "Reference Clones" }
                    let others = viewModel.builtInVoices.filter {
                        !["Conversational", "Narrator", "Character", "Reference Clones"].contains($0.category)
                    }

                    if !favVoices.isEmpty {
                        Section("★ Favorite Voices (\(favVoices.count))") {
                            ForEach(favVoices) { voice in
                                Button {
                                    viewModel.selectedVoice = voice
                                } label: {
                                    if viewModel.selectedVoice?.voiceKey == voice.voiceKey {
                                        Label("★ \(voice.name) — \(voice.description)", systemImage: "checkmark")
                                    } else {
                                        Text("★ \(voice.name) — \(voice.description)")
                                    }
                                }
                            }
                        }
                        Divider()
                    }
                    if !recentVoices.isEmpty {
                        Section("Recently Used Voices (\(recentVoices.count))") {
                            ForEach(recentVoices.prefix(6)) { voice in
                                Button {
                                    viewModel.selectedVoice = voice
                                } label: {
                                    Text("\(voice.name) — \(voice.description)")
                                }
                            }
                        }
                        Divider()
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

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(allQuickVoices) { voice in
                            quickVoicePillButton(for: voice)
                        }
                    }
                }

                Button {
                    showVoices = true
                } label: {
                    let theme = viewModel.readAlongTheme
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
                            .fill(theme.pillFill)
                            .overlay(
                                Capsule()
                                    .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                            )
                    )
                }
                .buttonStyle(.plain)
                .fixedSize()
                .help("Open visual Voice Browser grid and zero-shot voice cloner")
            }
        }
    }

    @ViewBuilder
    private func quickVoicePillButton(for voice: Voice) -> some View {
        let theme = viewModel.readAlongTheme
        let isSelected = viewModel.selectedVoice?.name == voice.name && viewModel.selectedVoice?.voiceId == voice.voiceId
        let isFav = prefs.isFavoriteVoice(voice)
        let fillStyle: Color = isSelected ? Color.blue : (isFav ? Color.yellow.opacity(0.08) : theme.pillFill)
        let borderStyle: Color = isSelected ? Color.blue : (isFav ? Color.yellow.opacity(0.38) : theme.pillStroke)
        Button {
            withAnimation(.spring(response: 0.24, dampingFraction: 0.8)) {
                viewModel.selectedVoice = voice
            }
        } label: {
            HStack(spacing: 5) {
                if isFav {
                    Image(systemName: "star.fill")
                        .font(.system(size: 9.5, weight: .bold))
                        .foregroundStyle(isSelected ? .yellow : (theme.isLight ? .orange : .yellow))
                }
                VoiceAvatar(color: voice.color, size: 15)
                Text(voice.name)
                    .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                    .foregroundStyle(isSelected ? .white : .primary.opacity(0.88))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(fillStyle)
                    .overlay(
                        Capsule()
                            .strokeBorder(borderStyle, lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
        .help("\(voice.name) — \(voice.description) (Right-click to ★ Favorite)")
        .contextMenu {
            Button {
                prefs.toggleFavoriteVoice(voice)
            } label: {
                Label(
                    isFav ? "Remove '\(voice.name)' from ★ Favorites" : "Add '\(voice.name)' to ★ Favorites",
                    systemImage: isFav ? "star.slash.fill" : "star.fill"
                )
            }
            Button {
                viewModel.selectedVoice = voice
            } label: {
                Label("Select '\(voice.name)'", systemImage: "checkmark.circle")
            }
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
    var onClearAllText: (() -> Void)? = nil

    @AppStorage("VoicesApp.Teleprompter.stageMode") private var stageMode: Bool = true
    @AppStorage("VoicesApp.Teleprompter.fontSize") private var teleprompterFontSize: Double = 16.0

    private func snippet(for chunk: PhononTextChunk?) -> String {
        guard let raw = chunk?.text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return ""
        }
        return raw.count > 105 ? String(raw.prefix(105)) + "…" : raw
    }

    private func chunkMeta(at index: Int) -> PhononTextChunk? {
        guard index >= 0 else { return nil }
        if index < chunks.count && chunks[index].index == index {
            return chunks[index]
        }
        return chunks.first(where: { $0.index == index })
    }

    var body: some View {
        let totalChunks = max(chunks.count, 1)
        let safeActiveChunk = min(max(0, activeChunkIndex), max(0, totalChunks - 1))
        let useFocusedStage = stageMode && chunks.count > 1

        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    // Header Banner
                    ScrollView(.horizontal, showsIndicators: false) {
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
                                        .foregroundStyle(stageMode ? theme.activeWordForeground : .secondary)
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
                                        .foregroundStyle(!stageMode ? theme.activeWordForeground : .secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3.5)
                                        .background(!stageMode ? theme.activeWordBackground : Color.clear, in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(2)
                            .background(theme.pillFill, in: Capsule())

                            Button {
                                onSelectChunk(max(0, safeActiveChunk - 1))
                            } label: {
                                Image(systemName: "chevron.left")
                                    .font(.system(size: 10.5, weight: .bold))
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 4.5)
                                    .background(theme.pillFill, in: Capsule())
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
                                    .background(theme.pillFill, in: Capsule())
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

                        if totalChunks > 1 {
                            let bookPercent = Int(round(Double(safeActiveChunk + 1) / Double(totalChunks) * 100.0))
                            Text("📖 \(bookPercent)%")
                                .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(theme.pillFill, in: Capsule())
                                .help("Book / document section progress (§\(safeActiveChunk + 1) of \(totalChunks))")
                        }

                        // Teleprompter Typography Zoom (A- / A+)
                        HStack(spacing: 4) {
                            Button {
                                teleprompterFontSize = max(13.0, teleprompterFontSize - 1.5)
                            } label: {
                                Text("A−")
                                    .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(.primary.opacity(0.85))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                            }
                            .buttonStyle(.plain)
                            .disabled(teleprompterFontSize <= 13.0)
                            .help("Decrease teleprompter word size")

                            Text("\(Int(teleprompterFontSize))pt")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.secondary)

                            Button {
                                teleprompterFontSize = min(26.0, teleprompterFontSize + 1.5)
                            } label: {
                                Text("A+")
                                    .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(.primary.opacity(0.85))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                            }
                            .buttonStyle(.plain)
                            .disabled(teleprompterFontSize >= 26.0)
                            .help("Increase teleprompter word size")
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1.5)
                        .background(
                            Capsule()
                                .fill(theme.pillFill)
                                .overlay(
                                    Capsule()
                                        .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                )
                        )

                        if let onClearAllText {
                            Button {
                                onClearAllText()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "trash.fill")
                                        .font(.system(size: 9.5))
                                    Text("Clear All")
                                        .font(.system(size: 10.5, weight: .semibold))
                                }
                                .foregroundStyle(.red)
                                .padding(.horizontal, 9)
                                .padding(.vertical, 4)
                                .background(Color.red.opacity(0.14), in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .help("Clear all text and return to empty script editor (⇧⌘⌫)")
                        }

                        Button {
                            onQuickExportChunk(safeActiveChunk)
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "bolt.fill")
                                    .font(.system(size: 9.5))
                                Text("2s Export §\(safeActiveChunk + 1)")
                                    .font(.system(size: 10.5, weight: .semibold))
                            }
                            .foregroundStyle(theme.isLight ? .orange : .yellow)
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
                                .background(theme.pillFill, in: Capsule())
                        }
                    }
                    }

                    if useFocusedStage {
                        // Codecaine File Manager Focused Stage Mode: keeps the active spoken chunk & highlighted word centered on screen
                        if safeActiveChunk > 0, let prevMeta = chunkMeta(at: safeActiveChunk - 1) {
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
                                        .fill(theme.elevatedRowFill)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                                .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }

                        let activeWords = timings.filter { $0.chunkIndex == safeActiveChunk }
                        let activeMeta = chunkMeta(at: safeActiveChunk)
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
                                    .foregroundStyle(theme.activeWordForeground)
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
                                        .fill(theme.pillStroke)
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
                                        fontSize: CGFloat(teleprompterFontSize),
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

                        if safeActiveChunk + 1 < totalChunks, let nextMeta = chunkMeta(at: safeActiveChunk + 1) {
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
                                        .fill(theme.elevatedRowFill)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                                .strokeBorder(theme.pillStroke, lineWidth: 0.8)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    } else {
                        // All-Chunks View with distinct string IDs ("chunk-\(cIdx)" vs "word-\(item.id)") so ScrollViewReader never collides
                        let wordsByChunk = Dictionary(grouping: timings, by: \.chunkIndex)
                        let chunkIndices = chunks.isEmpty ? [0] : chunks.map(\.index)
                        ForEach(chunkIndices, id: \.self) { cIdx in
                            let chunkWords = wordsByChunk[cIdx] ?? []
                            let isChunkActive = cIdx == safeActiveChunk
                            let chunkMeta = chunkMeta(at: cIdx)

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
                                                isChunkActive ? theme.activeWordBackground : theme.pillFill,
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
                                            .foregroundStyle(theme.isLight ? .orange : .yellow)
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
                                                fontSize: CGFloat(teleprompterFontSize),
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
    var fontSize: CGFloat = 16
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
                .font(.system(size: fontSize, weight: isActive ? .bold : .medium))
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

// MARK: - Compact Audio Player (Acoustic Waveform Scrubber + Skip Transport)

struct CompactAudioPlayer: View {
    let isPlaying: Bool
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onPlayPause: () -> Void
    let onSeek: (TimeInterval) -> Void
    var audioURL: URL? = nil
    var accentColor: Color = .blue

    private let barCount: Int = 44

    private var envelope: [CGFloat] {
        WaveformEnvelopeCache.envelope(for: audioURL, barCount: barCount).map { CGFloat($0) }
    }

    private var progressFraction: Double {
        guard duration > 0.01 else { return 0 }
        return min(1.0, max(0.0, currentTime / duration))
    }

    var body: some View {
        HStack(spacing: 10) {
            // Skip Back 5s
            Button {
                onSeek(max(0, currentTime - 5))
            } label: {
                Image(systemName: "gobackward.5")
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 26, height: 26)
                    .background(Color.primary.opacity(0.06), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Skip back 5 seconds")

            // Play/Pause button
            Button(action: onPlayPause) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(accentColor)
                    .shadow(color: accentColor.opacity(0.3), radius: 5, y: 1)
            }
            .buttonStyle(.plain)

            // Skip Forward 5s
            Button {
                onSeek(min(max(duration, 0), currentTime + 5))
            } label: {
                Image(systemName: "goforward.5")
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 26, height: 26)
                    .background(Color.primary.opacity(0.06), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Skip forward 5 seconds")

            // Interactive Acoustic Waveform Scrubber
            GeometryReader { geo in
                let dynamicCount = max(40, min(92, Int(geo.size.width / 6.5)))
                let bars = WaveformEnvelopeCache.envelope(for: audioURL, barCount: dynamicCount).map { CGFloat($0) }
                let totalBars = max(bars.count, 1)
                let playedIndex = Int(round(progressFraction * Double(totalBars)))

                HStack(alignment: .center, spacing: 2.5) {
                    ForEach(0..<totalBars, id: \.self) { idx in
                        let amp = bars[idx]
                        let isPlayed = idx <= playedIndex && duration > 0.01
                        RoundedRectangle(cornerRadius: 2, style: .continuous)
                            .fill(
                                isPlayed
                                    ? accentColor
                                    : Color.primary.opacity(0.20)
                            )
                            .frame(maxWidth: .infinity)
                            .frame(height: max(4, geo.size.height * amp))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let ratio = min(1.0, max(0.0, Double(value.location.x / max(1, geo.size.width))))
                            onSeek(ratio * max(duration, 0.01))
                        }
                )
            }
            .frame(height: 28)

            // Time Readout Pill
            Text("\(formatTime(currentTime)) / \(formatTime(duration))")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .fixedSize()

            #if os(macOS)
            if let url = audioURL, FileManager.default.fileExists(atPath: url.path) {
                // Draggable WAV File Pill (Drag directly into Finder, Logic Pro, Final Cut, iMessage, Slack)
                HStack(spacing: 4) {
                    Image(systemName: "waveform.path")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(accentColor)
                    Text("Drag WAV")
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary.opacity(0.88))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(accentColor.opacity(0.14))
                        .overlay(
                            Capsule()
                                .strokeBorder(accentColor.opacity(0.35), lineWidth: 0.8)
                        )
                )
                .onDrag {
                    NSItemProvider(object: url as NSURL)
                }
                .help("Drag this audio file directly into Finder, Logic Pro, Final Cut Pro, or any app")

                // 1-Click Reveal in Finder Button
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                } label: {
                    Image(systemName: "folder.fill")
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 24, height: 24)
                        .background(Color.primary.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Reveal audio file in macOS Finder")
            }
            #endif
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let safe = max(0, time)
        let minutes = Int(safe) / 60
        let seconds = Int(safe) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

// MARK: - Pronunciation Lexicon Sheet (Custom Phonetic Replacements for Acronyms & Proper Nouns)

struct PronunciationLexiconSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var lexicon = SpeechPronunciationLexicon.shared
    @State private var newTerm: String = ""
    @State private var newPhonetic: String = ""
    @State private var newCaseSensitive: Bool = false
    @State private var sampleTestInput: String = "Testing MLX neural speech on Apple Silicon VRAM with BF16 and arXiv EPUB books."

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Custom Pronunciation Lexicon")
                            .font(.headline)
                        Text("Phonetic rules are applied automatically during speech synthesis while preserving the original spelling in your script and Read-Along Karaoke.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("Enable Lexicon", isOn: $lexicon.isEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }
                .padding(12)
                .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))

                // Add New Rule Bar
                HStack(spacing: 8) {
                    TextField("Word or Acronym (e.g. MLX, Callahan)", text: $newTerm)
                        .textFieldStyle(.roundedBorder)
                    Image(systemName: "arrow.right")
                        .foregroundStyle(.secondary)
                    TextField("Spoken Phonetic (e.g. M L X, Cal-uh-han)", text: $newPhonetic)
                        .textFieldStyle(.roundedBorder)
                    Toggle("Aa", isOn: $newCaseSensitive)
                        .toggleStyle(.button)
                        .help("Case-sensitive match (useful for all-caps acronyms like MLX or SQL)")
                    Button {
                        lexicon.addRule(term: newTerm, phonetic: newPhonetic, caseSensitive: newCaseSensitive)
                        newTerm = ""
                        newPhonetic = ""
                    } label: {
                        Label("Add Rule", systemImage: "plus.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(newTerm.trimmingCharacters(in: .whitespaces).isEmpty || newPhonetic.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                // Live Preview Box
                VStack(alignment: .leading, spacing: 6) {
                    Text("LIVE PHONETIC PREVIEW")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(.secondary)
                    TextField("Type a sentence to preview phonetic expansion…", text: $sampleTestInput)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                    HStack(spacing: 6) {
                        Image(systemName: "speaker.wave.2.bubble.fill")
                            .foregroundStyle(.mint)
                        Text(lexicon.apply(to: sampleTestInput))
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.mint)
                    }
                }
                .padding(10)
                .background(Color.mint.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                List {
                    ForEach($lexicon.rules) { $rule in
                        HStack(spacing: 12) {
                            Toggle("", isOn: $rule.isEnabled)
                                .labelsHidden()
                            Text(rule.term)
                                .font(.system(size: 13, weight: .bold, design: .monospaced))
                                .frame(minWidth: 110, alignment: .leading)
                            Image(systemName: "arrow.right")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("Phonetic", text: $rule.phonetic)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12.5))
                            if rule.caseSensitive {
                                Text("Aa")
                                    .font(.caption2.bold())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.blue.opacity(0.18), in: Capsule())
                            }
                            Button(role: .destructive) {
                                lexicon.removeRule(id: rule.id)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .padding(16)
            .navigationTitle("Pronunciation Lexicon (\(lexicon.rules.filter(\.isEnabled).count) Active)")
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Button("Reset Defaults") {
                        lexicon.resetDefaults()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 620, minHeight: 480)
        #endif
    }
}

// MARK: - Interactive Multi-Voice Character Casting Table Sheet ("Cast & Dialogue" Studio)

struct CharacterCastingSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var scriptText: String
    @State private var manager = CharacterCastingManager.shared
    @State private var newName: String = ""
    @State private var newVoice: String = "Freya"
    @State private var newPitch: Float = 0.0
    @State private var scanBanner: String? = nil

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 14) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Multi-Voice Character Casting Table")
                            .font(.headline)
                        Text("Assign distinct studio voices and acoustic pitch profiles to each character (`Name:` prefixes, `<|speaker:N|>` tags, or quoted novel dialogue). Synthesis switches voices per chunk automatically.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("Enable Casting", isOn: $manager.isEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }
                .padding(12)
                .background(Color.purple.opacity(0.09), in: RoundedRectangle(cornerRadius: 12))

                HStack(spacing: 10) {
                    Button {
                        let added = manager.scanScriptForCharacters(scriptText)
                        scanBanner = added > 0
                            ? "Discovered & added \(added) character(s) from the active script!"
                            : "All detected script speakers are already in the cast table."
                    } label: {
                        Label("Auto-Scan Script for Characters", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)

                    Toggle("Auto-Cast Quoted Dialogue (“…”)", isOn: $manager.castQuotedDialogue)
                        .font(.caption.weight(.semibold))

                    Spacer()

                    if let scanBanner {
                        Text(scanBanner)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.purple)
                    }
                }

                // Add Custom Character Row
                HStack(spacing: 8) {
                    TextField("Character Name (e.g. Captain Miller, Alice)", text: $newName)
                        .textFieldStyle(.roundedBorder)
                    Picker("Voice", selection: $newVoice) {
                        ForEach(CharacterCastingManager.studioVoiceOptions, id: \.self) { v in
                            Text(v).tag(v)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 140)
                    Picker("Pitch", selection: $newPitch) {
                        ForEach([Float(-3.0), -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0], id: \.self) { p in
                            Text(p == 0 ? "0 st" : String(format: "%+.1f st", p)).tag(p)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 110)
                    Button {
                        manager.addCharacter(name: newName, voiceId: newVoice, pitch: newPitch)
                        newName = ""
                    } label: {
                        Label("Add Character", systemImage: "person.badge.plus")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                List {
                    ForEach($manager.cast) { $entry in
                        HStack(spacing: 12) {
                            Toggle("", isOn: $entry.isEnabled)
                                .labelsHidden()
                            VStack(alignment: .leading, spacing: 2) {
                                TextField("Character", text: $entry.characterName)
                                    .font(.system(size: 13, weight: .bold))
                                    .textFieldStyle(.plain)
                                Text(entry.speakerTag)
                                    .font(.system(size: 10.5, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            .frame(minWidth: 150, alignment: .leading)

                            Picker("Voice", selection: $entry.voiceId) {
                                ForEach(CharacterCastingManager.studioVoiceOptions, id: \.self) { v in
                                    Text(v).tag(v)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 140)

                            Picker("Pitch", selection: $entry.pitchSemitones) {
                                ForEach([Float(-3.0), -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], id: \.self) { p in
                                    Text(p == 0 ? "0 st" : String(format: "%+.1f st", p)).tag(p)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 115)

                            Button {
                                if !scriptText.hasSuffix("\n") && !scriptText.isEmpty {
                                    scriptText += "\n"
                                }
                                scriptText += "\(entry.characterName): "
                            } label: {
                                Label("Insert Cue", systemImage: "text.insert")
                                    .font(.caption)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Append '\(entry.characterName): ' dialogue line to the active script")

                            Button(role: .destructive) {
                                manager.removeCharacter(id: entry.id)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .padding(16)
            .navigationTitle("Character Casting (\(manager.cast.filter(\.isEnabled).count) Voices Active)")
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Button("Reset Defaults") {
                        manager.resetDefaults()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(width: 740, height: 520)
        #endif
    }
}

// MARK: - Chapterized .M4B Audiobook Exporter Sheet (Chapters + Cover Art)

struct M4BAudiobookExportSheet: View {
    @Environment(\.dismiss) private var dismiss
    let viewModel: TTSViewModel
    let scriptText: String
    let defaultTitle: String

    @AppStorage("VoicesApp.studioTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @State private var bookTitle: String = ""
    @State private var authorName: String = "Codecaine MLX Audio Studio"
    @State private var customCoverURL: URL? = nil
    @State private var showCoverPicker: Bool = false

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? viewModel.readAlongTheme
    }

    private var previewChunks: [PhononTextChunk] {
        viewModel.chunkTextWithMetadata(scriptText)
    }

    var body: some View {
        let chunks = previewChunks
        let isAlreadyComplete = viewModel.hasCompleteAudio(for: scriptText, voice: viewModel.selectedVoice, requiredScope: .fullDocument)
        let estSec = (isAlreadyComplete && viewModel.duration > 0) ? viewModel.duration : max(2.0, Double(scriptText.count) / 14.0)
        let chapters = M4BAudiobookExporter.buildChapters(
            chunks: chunks,
            timings: isAlreadyComplete ? viewModel.synthesizedWordTimings : [],
            totalDurationSec: estSec
        )

        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [Color.indigo, Color.purple, Color.blue],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 78, height: 78)
                        VStack(spacing: 4) {
                            Image(systemName: "books.vertical.fill")
                                .font(.system(size: 24, weight: .bold))
                                .foregroundStyle(.white)
                            Text(".M4B")
                                .font(.system(size: 10, weight: .heavy, design: .rounded))
                                .foregroundStyle(.white.opacity(0.9))
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Text("Chapterized .M4B Audiobook Exporter")
                                .font(.headline)
                            Text(isAlreadyComplete ? "✓ All \(chapters.count) Chapters Ready" : "Full Book · All \(chapters.count) Chapters")
                                .font(.system(size: 10.5, weight: .bold, design: .rounded))
                                .foregroundStyle(isAlreadyComplete ? .green : .indigo)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2.5)
                                .background((isAlreadyComplete ? Color.green : Color.indigo).opacity(0.14), in: Capsule())
                        }
                        Text("Packages your entire book or script into an Apple Books & podcast player compatible .M4B audiobook with embedded chapter markers (\(chapters.count) chapters) and 600×600 studio cover artwork.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(12)
                .background(activeTheme.elevatedRowFill, in: RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(activeTheme.pillStroke, lineWidth: 0.8)
                )

                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Audiobook Title")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        TextField("Audiobook Title", text: $bookTitle)
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Author / Series")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        TextField("Author", text: $authorName)
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Cover Artwork")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        HStack(spacing: 6) {
                            Button {
                                showCoverPicker = true
                            } label: {
                                Label(customCoverURL?.lastPathComponent ?? "Auto-Generated Studio Cover", systemImage: "photo.artframe")
                                    .lineLimit(1)
                            }
                            if customCoverURL != nil {
                                Button {
                                    customCoverURL = nil
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                Text("EMBEDDED CHAPTER TABLE (\(chapters.count) CHAPTERS)")
                    .font(.system(size: 10.5, weight: .bold, design: .rounded))
                    .foregroundStyle(.secondary)

                List(chapters) { ch in
                    HStack {
                        Text("#\(ch.id + 1)")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(.indigo)
                            .frame(width: 36, alignment: .leading)
                        Text(ch.title)
                            .font(.system(size: 12.5, weight: .medium))
                            .lineLimit(1)
                        Spacer()
                        Text(String(format: "%d:%02d – %d:%02d", Int(ch.startSec) / 60, Int(ch.startSec) % 60, Int(ch.endSec) / 60, Int(ch.endSec) % 60))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))

                HStack {
                    Spacer()
                    Button("Cancel") {
                        dismiss()
                    }
                    Button {
                        let resolvedTitle = bookTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? defaultTitle
                            : bookTitle
                        dismiss()
                        viewModel.exportChapterizedM4B(
                            text: scriptText,
                            bookTitle: resolvedTitle,
                            author: authorName,
                            voice: viewModel.selectedVoice,
                            customCoverURL: customCoverURL
                        )
                    } label: {
                        Label("Export Chapterized .M4B Audiobook…", systemImage: "square.and.arrow.down.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.indigo)
                }
            }
            .padding(16)
            .navigationTitle("Export .M4B Audiobook")
            .onAppear {
                if bookTitle.isEmpty {
                    bookTitle = defaultTitle
                }
            }
            .fileImporter(
                isPresented: $showCoverPicker,
                allowedContentTypes: [.image, .item, .data],
                allowsMultipleSelection: false
            ) { res in
                if case .success(let urls) = res, let first = urls.first {
                    customCoverURL = first
                }
            }
        }
        #if os(macOS)
        .frame(width: 700, height: 520)
        #endif
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

// MARK: - Studio Master Bus DSP Rack & ACX / Broadcast Loudness QC Sheet

struct MasteringRackSheet: View {
    @Environment(\.dismiss) private var dismiss
    var viewModel: TTSViewModel
    @State private var manager = StudioMasteringManager.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack {
                Image(systemName: "slider.vertical.3")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Studio Master Bus DSP Rack & Broadcast Loudness QC")
                        .font(.system(size: 16, weight: .bold))
                    Text("ACX / Audible (-20 dB RMS) & Podcast (-16 LUFS) compliance, tube warmth, dynamic de-esser & auto-ducked ambience beds")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            Divider()

            // Preset Strip
            VStack(alignment: .leading, spacing: 6) {
                Text("Mastering Target Preset")
                    .font(.system(size: 11.5, weight: .bold))
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    ForEach(MasteringPreset.allCases) { p in
                        let isSel = manager.preset == p
                        Button {
                            manager.applyPreset(p)
                        } label: {
                            VStack(alignment: .leading, spacing: 3) {
                                HStack(spacing: 5) {
                                    Circle()
                                        .fill(isSel ? Color.orange : Color.secondary.opacity(0.35))
                                        .frame(width: 7, height: 7)
                                    Text(p.shortBadge)
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                }
                                Text(p.title)
                                    .font(.system(size: 10.5, weight: .medium))
                                    .lineLimit(1)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                isSel ? Color.orange.opacity(0.18) : Color.primary.opacity(0.05),
                                in: RoundedRectangle(cornerRadius: 9)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 9)
                                    .stroke(isSel ? Color.orange.opacity(0.75) : Color.clear, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }

                Text(manager.preset.subtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            // DSP Parameter Grid
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Dynamics & Loudness Chain")
                        .font(.system(size: 12, weight: .bold))

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("Target Loudness")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%.1f LUFS", manager.targetLUFS))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(.orange)
                        }
                        Slider(value: $manager.targetLUFS, in: -24.0...(-12.0), step: 0.5)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("True-Peak Brickwall Ceiling")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%.1f dBTP", manager.truePeakCeilingDB))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(.cyan)
                        }
                        Slider(value: $manager.truePeakCeilingDB, in: -4.0...(-0.5), step: 0.1)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("Optical Vocal Leveler (Ratio)")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%.1f : 1", manager.compressionRatio))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                        }
                        Slider(value: $manager.compressionRatio, in: 1.0...4.0, step: 0.1)
                    }

                    Toggle("ACX 0.65s Head / 0.85s Tail Room-Tone Floor (-72 dBFS)", isOn: $manager.acxHeadTailRoomTone)
                        .font(.system(size: 11.5, weight: .medium))
                }
                .padding(12)
                .frame(maxWidth: .infinity)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))

                VStack(alignment: .leading, spacing: 10) {
                    Text("Tone, Sibilance & Acoustic Bed")
                        .font(.system(size: 12, weight: .bold))

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("Analog Tube Warmth (Harmonics)")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%.0f%%", manager.warmthDrive * 100))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(.pink)
                        }
                        Slider(value: $manager.warmthDrive, in: 0.0...0.8, step: 0.02)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("Presence & Air High-Shelf (4.2 kHz)")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%+.1f dB", manager.presenceBoostDB))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                        }
                        Slider(value: $manager.presenceBoostDB, in: -3.0...5.0, step: 0.2)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text("Dynamic Sibilance De-Esser")
                                .font(.system(size: 11.5, weight: .medium))
                            Spacer()
                            Text(String(format: "%.0f%%", manager.deEsserAmount * 100))
                                .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                                .foregroundStyle(.teal)
                        }
                        Slider(value: $manager.deEsserAmount, in: 0.0...1.0, step: 0.05)
                    }

                    HStack {
                        Text("Ambience Bed:")
                            .font(.system(size: 11.5, weight: .medium))
                        Picker("", selection: $manager.ambienceBed) {
                            ForEach(BackgroundAmbienceBed.allCases) { bed in
                                Text(bed.title).tag(bed)
                            }
                        }
                        .labelsHidden()
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            }

            // Broadcast / ACX Compliance Telemetry Card
            if let rep = manager.lastReport {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Broadcast & ACX Loudness Compliance Telemetry", systemImage: "waveform.badge.magnifyingglass")
                            .font(.system(size: 12, weight: .bold))
                        Spacer()
                        HStack(spacing: 6) {
                            Text(rep.acxCompliant ? "ACX AUDIBLE: PASS" : "ACX AUDIBLE: ADJUST")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 2.5)
                                .background(rep.acxCompliant ? Color.green : Color.orange, in: Capsule())

                            Text(rep.podcastCompliant ? "PODCAST -16 LUFS: PASS" : "PODCAST: INFO")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 2.5)
                                .background(rep.podcastCompliant ? Color.blue : Color.secondary, in: Capsule())
                        }
                    }

                    HStack(spacing: 10) {
                        loudnessMetricPill(title: "Integrated", value: String(format: "%.1f LUFS", rep.integratedLUFS), ok: true)
                        loudnessMetricPill(title: "RMS Level (ACX -23..-18)", value: String(format: "%.1f dBFS", rep.rmsDBFS), ok: rep.acxRMSCompliant)
                        loudnessMetricPill(title: "True Peak (<= -3.0 dBTP)", value: String(format: "%.1f dBTP", rep.truePeakDBTP), ok: rep.acxPeakCompliant)
                        loudnessMetricPill(title: "Noise Floor (<= -60 dB)", value: String(format: "%.1f dBFS", rep.noiseFloorDBFS), ok: rep.acxNoiseFloorCompliant)
                        loudnessMetricPill(title: "Crest Factor", value: String(format: "%.1f dB", rep.crestFactorDB), ok: true)
                    }
                }
                .padding(12)
                .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
            }

            // Footer Actions
            HStack {
                Text(manager.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    let target = viewModel.audioURL ?? Voice.bundledStudioSamples().first?.url
                    if let url = target, let rep = StudioMasteringManager.analyzeWAVFile(at: url) {
                        manager.lastReport = rep
                        manager.statusMessage = "Analyzed \(url.lastPathComponent) (\(String(format: "%.1f", rep.integratedLUFS)) LUFS)"
                    }
                } label: {
                    Label("Analyze Loudness QC", systemImage: "chart.bar.doc.horizontal")
                }
                .buttonStyle(.bordered)

                Button {
                    viewModel.applyMasterBusInPlaceToCurrentAudio()
                } label: {
                    Label("Apply Master Bus to Current Audio", systemImage: "wand.and.stars")
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
            }
        }
        .padding(18)
        #if os(macOS)
        .frame(width: 820, height: 520)
        #endif
        .onAppear {
            if ProcessInfo.processInfo.arguments.contains("--demo-acx-master") {
                manager.applyPreset(.acxAudiobook)
                viewModel.applyMasterBusInPlaceToCurrentAudio()
            } else if manager.lastReport == nil {
                let target = viewModel.audioURL ?? Voice.bundledStudioSamples().first?.url
                if let url = target {
                    manager.lastReport = StudioMasteringManager.analyzeWAVFile(at: url)
                }
            }
        }
    }

    private func loudnessMetricPill(title: String, value: String, ok: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 9.5, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12.5, weight: .bold, design: .monospaced))
                .foregroundStyle(ok ? Color.primary : Color.orange)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
    }
}

#Preview("CompactAudioPlayer") {
    CompactAudioPlayer(isPlaying: true, currentTime: 1.0, duration: 5.0, onPlayPause: {}, onSeek: { _ in })
}

#Preview("ContentView") {
    ContentView()
}
