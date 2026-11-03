import SwiftUI
import UniformTypeIdentifiers
import Foundation

struct STTView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = STTViewModel()
    @State private var modelsManager = DownloadedModelsManager.shared
    @State private var showFileImporter = false
    @State private var showSettings = false
    @State private var showDownloadedModelsSheet = false
    @State private var pendingDeleteModel: (name: String, id: String, size: String)? = nil
    @State private var isAudioHovering = false
    @State private var isDropTargeted = false
    @State private var transcriptSearch = ""
    @State private var copiedRecently = false
    @State private var autoSyncToVoiceStudio = false

    #if os(iOS)
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .callout
    private let bodyFont: Font = .body
    #else
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .subheadline
    private let bodyFont: Font = .title3
    #endif

    var body: some View {
        let transcriptSegments = TaggedTranscriptParser.parse(viewModel.transcriptionText)
        let activeTranscriptSegmentID = viewModel.isPlaying
            ? TaggedTranscriptParser.activeSegmentID(in: transcriptSegments, at: viewModel.currentTime)
            : nil

        VStack(spacing: 14) {
            // Top Contextual Ribbon
            HStack(spacing: 12) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(viewModel.isModelLoaded ? Color.green : (viewModel.isLoading ? Color.orange : Color.gray))
                        .frame(width: 8, height: 8)
                        .shadow(color: (viewModel.isModelLoaded ? Color.green : Color.orange).opacity(0.6), radius: 4)

                    Text(viewModel.isModelLoaded ? "ASR Engine Ready" : (viewModel.isLoading ? "Loading ASR Model…" : "ASR Idle"))
                        .font(.system(size: 12, weight: .semibold))

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
                                let name = STTViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
                                pendingDeleteModel = (name: name, id: viewModel.modelId, size: activeSize ?? "Cached")
                            } label: {
                                Label("Delete Cached Model (\(activeSize ?? "Cached"))…", systemImage: "trash")
                            }
                        }
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

                if viewModel.hasEmbeddedLyrics, let source = viewModel.embeddedLyricsSource {
                    HStack(spacing: 6) {
                        Image(systemName: "music.note.list")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.mint)
                        Text(viewModel.usingEmbeddedLyrics ? "Real Song Lyrics (\(source))" : "Embedded Lyrics Available (\(source))")
                            .font(.system(size: 11.5, weight: .semibold))
                            .foregroundStyle(.mint)
                    }
                    .padding(.horizontal, 11)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(Color.mint.opacity(0.14))
                            .overlay(
                                Capsule()
                                    .strokeBorder(Color.mint.opacity(0.35), lineWidth: 0.8)
                            )
                    )
                    .help("This song file contains readable lyrics (\(source)). Using the real song lyrics instead of AI transcription.")
                }

                Spacer()

                if !viewModel.transcriptionText.isEmpty {
                    // Search Transcript Filter
                    HStack(spacing: 6) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        TextField("Filter transcript…", text: $transcriptSearch)
                            .font(.system(size: 11.5))
                            .textFieldStyle(.plain)
                            .frame(width: 140)
                        if !transcriptSearch.isEmpty {
                            Button {
                                transcriptSearch = ""
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.white.opacity(0.05), in: Capsule())

                    // Send Transcript to Voice Studio
                    let cleanTranscript = STTViewModel.cleanTranscriptForVoiceStudio(viewModel.transcriptionText)
                    let cleanWordCount = cleanTranscript.split { $0.isWhitespace || $0.isNewline }.count
                    Button {
                        if viewModel.isGenerating {
                            autoSyncToVoiceStudio = true
                        }
                        let payload = cleanTranscript.isEmpty ? viewModel.transcriptionText : cleanTranscript
                        UserDefaults.standard.set(payload, forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                        NotificationCenter.default.post(
                            name: .sendTranscriptToVoiceStudio,
                            object: payload
                        )
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "waveform.and.person.filled")
                                .font(.system(size: 11))
                            Text(
                                viewModel.isGenerating
                                    ? "Send & Live-Sync to Voice Studio (\(cleanWordCount)w)"
                                    : "Send to Voice Studio (\(cleanWordCount)w)"
                            )
                                .font(.system(size: 11.5, weight: .semibold))
                        }
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.14), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Send the entire transcribed text (\(cleanWordCount) words) to Voice Studio for editing, Read-Along Karaoke, or re-synthesis")
                }

                if viewModel.tokensPerSecond > 0 {
                    HStack(spacing: 10) {
                        Label(String(format: "%.1f tok/s", viewModel.tokensPerSecond), systemImage: "speedometer")
                        Label(String(format: "%.1f GB Peak", viewModel.peakMemory), systemImage: "memorychip")
                    }
                    .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.white.opacity(0.04), in: Capsule())
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 14)

            // Inset Apple Transcript Canvas Card
            ZStack {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Color.white.opacity(0.03))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        Color.white.opacity(0.13),
                                        Color.white.opacity(0.04)
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(0.25), radius: 14, y: 6)

                ScrollViewReader { proxy in
                    ScrollView {
                        if viewModel.transcriptionText.isEmpty && !viewModel.isGenerating && !viewModel.isRecording {
                            VStack(spacing: 22) {
                                Spacer(minLength: 48)
                                ZStack {
                                    Circle()
                                        .fill(
                                            LinearGradient(
                                                colors: [Color.red.opacity(0.22), Color.blue.opacity(0.18)],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            )
                                        )
                                        .frame(width: 72, height: 72)
                                        .overlay(
                                            Circle()
                                                .strokeBorder(Color.white.opacity(0.14), lineWidth: 1)
                                        )

                                    Image(systemName: "waveform.badge.mic")
                                        .font(.system(size: 30, weight: .semibold))
                                        .foregroundStyle(.white)
                                }

                                VStack(spacing: 6) {
                                    Text("Neural Speech-to-Text & Subtitle Studio")
                                        .font(.system(size: 19, weight: .semibold))
                                    Text("Transcribe audio files or live microphone input with Qwen3-ASR, Whisper Large v3 Turbo, Parakeet, or SenseVoice.")
                                        .font(.system(size: 13))
                                        .foregroundStyle(.secondary)
                                        .multilineTextAlignment(.center)
                                        .frame(maxWidth: 480)
                                }

                                HStack(spacing: 14) {
                                    Button {
                                        showFileImporter = true
                                    } label: {
                                        HStack(spacing: 8) {
                                            Image(systemName: "doc.badge.plus")
                                                .font(.system(size: 14, weight: .semibold))
                                                .foregroundStyle(.blue)
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Import Audio File")
                                                    .font(.system(size: 13, weight: .semibold))
                                                Text("WAV, MP3, M4A, AIFF")
                                                    .font(.system(size: 11))
                                                    .foregroundStyle(.secondary)
                                            }
                                        }
                                        .padding(.horizontal, 18)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                                .fill(Color.white.opacity(0.05))
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.8)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)

                                    Button {
                                        Task { await viewModel.startRecording() }
                                    } label: {
                                        HStack(spacing: 8) {
                                            Image(systemName: "mic.circle.fill")
                                                .font(.system(size: 18, weight: .semibold))
                                                .foregroundStyle(.red)
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Record Live Microphone")
                                                    .font(.system(size: 13, weight: .semibold))
                                                Text("Real-time streaming ASR")
                                                    .font(.system(size: 11))
                                                    .foregroundStyle(.secondary)
                                            }
                                        }
                                        .padding(.horizontal, 18)
                                        .padding(.vertical, 12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                                .fill(Color.red.opacity(0.10))
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                                        .strokeBorder(Color.red.opacity(0.25), lineWidth: 0.8)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(!viewModel.canRecord)
                                }
                                Spacer(minLength: 48)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else if viewModel.transcriptionText.isEmpty && viewModel.isRecording {
                            RecordingTranscriptPlaceholder(supportsRealtime: viewModel.supportsRealtimeRecording)
                                .padding(24)
                        } else {
                            let filteredSegments = transcriptSearch.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                ? transcriptSegments
                                : transcriptSegments.filter {
                                    $0.text.localizedCaseInsensitiveContains(transcriptSearch) ||
                                    $0.speaker.localizedCaseInsensitiveContains(transcriptSearch)
                                }
                            TranscriptOutputView(
                                text: viewModel.transcriptionText,
                                searchFilter: transcriptSearch,
                                font: bodyFont,
                                segments: filteredSegments,
                                activeSegmentID: activeTranscriptSegmentID,
                                onSelectSegment: { seconds in
                                    if viewModel.selectedAudioURL != nil {
                                        viewModel.seek(to: seconds)
                                    }
                                }
                            )
                            .padding(18)

                            Color.clear
                                .frame(height: 1)
                                .id("bottom")
                        }
                    }
                    .onChange(of: viewModel.transcriptionText) {
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                        if autoSyncToVoiceStudio && !viewModel.transcriptionText.isEmpty {
                            let clean = STTViewModel.cleanTranscriptForVoiceStudio(viewModel.transcriptionText)
                            let payload = clean.isEmpty ? viewModel.transcriptionText : clean
                            UserDefaults.standard.set(payload, forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                            NotificationCenter.default.post(
                                name: .sendTranscriptToVoiceStudio,
                                object: payload
                            )
                        }
                    }
                    .onChange(of: viewModel.isGenerating) { _, isGen in
                        if !isGen && autoSyncToVoiceStudio {
                            autoSyncToVoiceStudio = false
                            let clean = STTViewModel.cleanTranscriptForVoiceStudio(viewModel.transcriptionText)
                            let payload = clean.isEmpty ? viewModel.transcriptionText : clean
                            if !payload.isEmpty {
                                UserDefaults.standard.set(payload, forKey: "VoicesApp.VoiceStudio.pendingTranscript")
                                NotificationCenter.default.post(
                                    name: .sendTranscriptToVoiceStudio,
                                    object: payload
                                )
                            }
                        }
                    }
                    .onChange(of: activeTranscriptSegmentID) { _, segmentID in
                        guard viewModel.isPlaying, let segmentID else { return }
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(segmentID, anchor: .center)
                        }
                    }
                }
            }
            .padding(.horizontal, 20)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onDrop(of: [.audio, .fileURL], isTargeted: $isDropTargeted) { providers in
                guard let first = providers.first else { return false }
                _ = first.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in
                        viewModel.selectAudioFile(url)
                    }
                }
                return true
            }

            // Unified Apple Glass Bottom Dock
            VStack(spacing: 12) {
                if viewModel.isRecording {
                    RecordingIndicator(
                        duration: viewModel.recordingDuration,
                        audioLevel: viewModel.audioLevel,
                        supportsRealtime: viewModel.supportsRealtimeRecording,
                        isRealtime: viewModel.usesRealtimeRecording
                    )
                }

                if viewModel.selectedAudioURL != nil && !viewModel.isRecording {
                    VStack(spacing: 4) {
                        if let fileName = viewModel.audioFileName {
                            HStack(spacing: 6) {
                                Image(systemName: "waveform.circle.fill")
                                    .foregroundStyle(.blue)
                                Text(fileName)
                                    .font(.system(size: 12, weight: .semibold))
                                    .lineLimit(1)
                                Spacer()
                                Button(action: { viewModel.removeAudioFile() }) {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.system(size: 16, weight: .semibold))
                                        .foregroundStyle(isAudioHovering ? Color.red : Color.secondary)
                                }
                                .buttonStyle(.plain)
                                .disabled(viewModel.isGenerating)
                            }
                            .padding(.horizontal, 12)
                            .padding(.top, 6)
                        }

                        CompactAudioPlayer(
                            isPlaying: viewModel.isPlaying,
                            currentTime: viewModel.currentTime,
                            duration: viewModel.duration,
                            onPlayPause: { viewModel.togglePlayPause() },
                            onSeek: { viewModel.seek(to: $0) }
                        )
                    }
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.white.opacity(0.04))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                    )
                    .onHover { isAudioHovering = $0 }
                }

                if !viewModel.generationProgress.isEmpty && !viewModel.isRecording {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.generationProgress)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
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
                    }
                }

                // Bottom Control Deck
                HStack(spacing: 10) {
                    if !viewModel.isRecording {
                        Button(action: { showFileImporter = true }) {
                            HStack(spacing: 7) {
                                Image(systemName: "doc.badge.plus")
                                    .font(.system(size: 13, weight: .semibold))
                                Text("Import Audio")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .frame(height: 44)
                            .padding(.horizontal, 16)
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
                        .disabled(viewModel.isGenerating)
                    }

                    // Record Button (Apple Voice Memos crimson capsule)
                    Button(action: {
                        if viewModel.isRecording {
                            viewModel.stopRecording()
                        } else {
                            Task { await viewModel.startRecording() }
                        }
                    }) {
                        HStack(spacing: 7) {
                            Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                                .font(.system(size: 13.5, weight: .semibold))
                            Text(viewModel.isRecording ? "Stop Recording" : "Record Mic")
                                .font(.system(size: 13, weight: .semibold))
                        }
                        .foregroundStyle(viewModel.isRecording ? .white : .red)
                        .frame(height: 44)
                        .padding(.horizontal, 16)
                        .background(
                            Capsule()
                                .fill(viewModel.isRecording ? Color.red : Color.red.opacity(0.14))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.red.opacity(0.32), lineWidth: 0.8)
                                )
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(!viewModel.canRecord)

                    if !viewModel.isRecording {
                        // One-Click Phonon 2.5 EN / Multilingual Quick Toggle Pill
                        HStack(spacing: 3) {
                            let isPhononEn = viewModel.modelId == "FermionResearch/Phonon-2.5-English"
                            let isPhononMulti = viewModel.modelId == "FermionResearch/Phonon-2.5"
                            Button {
                                guard viewModel.modelId != "FermionResearch/Phonon-2.5-English" else { return }
                                viewModel.modelId = "FermionResearch/Phonon-2.5-English"
                                Task { await viewModel.reloadModel() }
                            } label: {
                                Text("Phonon 2.5 EN")
                                    .font(.system(size: 11.5, weight: .semibold))
                                    .foregroundStyle(isPhononEn ? .white : .secondary)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 8)
                                    .background(
                                        Capsule()
                                            .fill(isPhononEn ? Color.blue : Color.clear)
                                    )
                            }
                            .buttonStyle(.plain)
                            .help("Switch to Phonon 2.5 English (FermionResearch/Phonon-2.5-English)")

                            Button {
                                guard viewModel.modelId != "FermionResearch/Phonon-2.5" else { return }
                                viewModel.modelId = "FermionResearch/Phonon-2.5"
                                Task { await viewModel.reloadModel() }
                            } label: {
                                Text("Multilingual")
                                    .font(.system(size: 11.5, weight: .semibold))
                                    .foregroundStyle(isPhononMulti ? .white : .secondary)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 8)
                                    .background(
                                        Capsule()
                                            .fill(isPhononMulti ? Color.blue : Color.clear)
                                    )
                            }
                            .buttonStyle(.plain)
                            .help("Switch to Phonon 2.5 Multilingual (FermionResearch/Phonon-2.5)")
                        }
                        .padding(3)
                        .frame(height: 44)
                        .background(
                            Capsule()
                                .fill(Color.white.opacity(0.06))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 0.8)
                                )
                        )

                        // Quick STT Model Picker
                        Menu {
                            Section("Select MLX ASR Model (\(STTViewModel.presetModels.count) Models)") {
                                ForEach(STTViewModel.presetModels, id: \.id) { preset in
                                    let cached = modelsManager.isModelCached(preset.id)
                                    let sizeStr = modelsManager.cachedModelSizeString(preset.id)
                                    let labelText = cached
                                        ? "✓ \(preset.name) [Cached\(sizeStr.map { " · \($0)" } ?? "")]"
                                        : "\(preset.name) (\(preset.id))"
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
                            if modelsManager.isModelCached(viewModel.modelId) {
                                let activeSize = modelsManager.cachedModelSizeString(viewModel.modelId) ?? "Cached"
                                let activeName = STTViewModel.presetModels.first(where: { $0.id == viewModel.modelId })?.name ?? viewModel.modelId
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
                                Label("STT Settings…", systemImage: "slider.horizontal.3")
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "cpu.fill")
                                    .font(.system(size: 12.5))
                                    .foregroundStyle(.blue)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(viewModel.modelId.split(separator: "/").last.map(String.init) ?? viewModel.modelId)
                                        .font(.system(size: 12.5, weight: .semibold))
                                        .lineLimit(1)
                                    Text("ASR Neural Engine")
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

                        Button(action: { showSettings = true }) {
                            Image(systemName: "slider.horizontal.3")
                                .font(.system(size: 14, weight: .semibold))
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

                        if !viewModel.transcriptionText.isEmpty {
                            Button(action: {
                                viewModel.copyTranscription()
                                copiedRecently = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    copiedRecently = false
                                }
                            }) {
                                Image(systemName: copiedRecently ? "checkmark" : "doc.on.doc")
                                    .font(.system(size: 13.5, weight: .semibold))
                                    .foregroundStyle(copiedRecently ? .green : .primary)
                                    .frame(width: 44, height: 44)
                                    .background(Circle().fill(Color.white.opacity(0.06)))
                            }
                            .buttonStyle(.plain)
                            .help("Copy transcription to clipboard")

                            Menu {
                                Button("Export Plain Text (.TXT)") { viewModel.exportTranscription(format: "txt") }
                                Button("Export Subtitles (.SRT)") { viewModel.exportTranscription(format: "srt") }
                                Button("Export WebVTT (.VTT)") { viewModel.exportTranscription(format: "vtt") }
                                Button("Export Karaoke (.ASS)") { viewModel.exportTranscription(format: "ass") }
                                Button("Export JSON with Timings (.JSON)") { viewModel.exportTranscription(format: "json") }
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: "square.and.arrow.down")
                                    Text("Export")
                                }
                                .font(.system(size: 12.5, weight: .semibold))
                                .foregroundStyle(.blue)
                                .frame(height: 44)
                                .padding(.horizontal, 14)
                                .background(Color.blue.opacity(0.15), in: Capsule())
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()

                            if !viewModel.isGenerating {
                                Button(action: { viewModel.clearTranscription() }) {
                                    Image(systemName: "trash")
                                        .font(.system(size: 13.5))
                                        .foregroundStyle(.red)
                                        .frame(width: 44, height: 44)
                                        .background(Circle().fill(Color.red.opacity(0.12)))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    } else {
                        Button(action: { viewModel.cancelRecording() }) {
                            Text("Cancel")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.secondary)
                                .frame(height: 44)
                                .padding(.horizontal, 16)
                                .background(Capsule().fill(Color.white.opacity(0.06)))
                        }
                        .buttonStyle(.plain)
                    }

                    Spacer()

                    if !viewModel.isRecording {
                        if viewModel.isGenerating {
                            Button(action: { viewModel.stop() }) {
                                HStack(spacing: 7) {
                                    Image(systemName: "stop.fill")
                                    Text("Stop")
                                }
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.white)
                                .frame(height: 44)
                                .padding(.horizontal, 20)
                                .background(Color.red, in: Capsule())
                            }
                            .buttonStyle(.plain)
                        } else {
                            if viewModel.hasEmbeddedLyrics {
                                // Optional AI override button if user explicitly wants AI ASR instead of real song lyrics
                                Button(action: { viewModel.startTranscription(forceAI: true) }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "cpu")
                                            .font(.system(size: 12, weight: .semibold))
                                        Text("AI Transcribe")
                                            .font(.system(size: 12.5, weight: .semibold))
                                    }
                                    .foregroundStyle(viewModel.isModelLoaded ? .secondary : .tertiary)
                                    .frame(height: 44)
                                    .padding(.horizontal, 14)
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
                                .fixedSize()
                                .disabled(!viewModel.isModelLoaded || viewModel.isGenerating)
                                .help("Run AI ASR instead of using the song file's embedded lyrics")
                            }

                            Button(action: { viewModel.startTranscription(forceAI: false) }) {
                                HStack(spacing: 8) {
                                    Image(systemName: viewModel.hasEmbeddedLyrics ? "music.note.list" : "waveform.badge.mic")
                                        .font(.system(size: 14, weight: .semibold))
                                    Text(viewModel.hasEmbeddedLyrics ? "Use Song Lyrics" : "Transcribe Audio")
                                        .font(.system(size: 14, weight: .semibold))
                                        .lineLimit(1)
                                }
                                .foregroundStyle(canTranscribe ? .white : .secondary)
                                .frame(height: 44)
                                .padding(.horizontal, 20)
                                .background(
                                    LinearGradient(
                                        colors: canTranscribe
                                            ? (viewModel.hasEmbeddedLyrics ? [Color.teal, Color.blue] : [Color.blue, Color.indigo])
                                            : [Color.white.opacity(0.08), Color.white.opacity(0.05)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    in: Capsule()
                                )
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.white.opacity(canTranscribe ? 0.24 : 0.08), lineWidth: 0.8)
                                )
                            }
                            .buttonStyle(.plain)
                            .fixedSize()
                            .disabled(!canTranscribe)
                        }
                    }
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [Color.white.opacity(0.16), Color.white.opacity(0.05)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(0.32), radius: 16, y: 6)
            )
            .padding(.horizontal, 20)
            .padding(.bottom, 16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio, .aiff, .plainText],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    viewModel.selectAudioFile(url)
                }
            case .failure(let error):
                viewModel.errorMessage = "File import failed: \(error.localizedDescription)"
            }
        }
        .sheet(isPresented: $showSettings) {
            STTSettingsView(viewModel: viewModel)
                #if os(iOS)
                .presentationDetents([.large])
                .presentationDragIndicator(.visible)
                #endif
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
            await viewModel.loadModel()
        }
    }

    private var canTranscribe: Bool {
        viewModel.selectedAudioURL != nil && !viewModel.isGenerating && (viewModel.hasEmbeddedLyrics || viewModel.isModelLoaded)
    }
}

// MARK: - Transcript Output

private struct TranscriptOutputView: View {
    let text: String
    var searchFilter: String = ""
    let font: Font
    let segments: [TaggedTranscriptSegment]
    let activeSegmentID: TaggedTranscriptSegment.ID?
    var onSelectSegment: ((TimeInterval) -> Void)? = nil

    var body: some View {
        if segments.isEmpty {
            let displayedText: String = {
                let q = searchFilter.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !q.isEmpty else { return text }
                let matchingLines = text
                    .components(separatedBy: .newlines)
                    .filter { $0.localizedCaseInsensitiveContains(q) }
                return matchingLines.isEmpty ? text : matchingLines.joined(separator: "\n")
            }()
            Text(displayedText)
                .font(font)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        } else {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(segments) { segment in
                    TaggedTranscriptSegmentRow(
                        segment: segment,
                        font: font,
                        isActive: segment.id == activeSegmentID,
                        onSelect: { onSelectSegment?(segment.startSeconds) }
                    )
                    .id(segment.id)
                }
            }
            .padding()
            .textSelection(.enabled)
        }
    }
}

private struct TaggedTranscriptSegmentRow: View {
    let segment: TaggedTranscriptSegment
    let font: Font
    let isActive: Bool
    var onSelect: (() -> Void)? = nil

    var body: some View {
        Button {
            onSelect?()
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Capsule()
                    .fill(speakerColor)
                    .frame(width: 4)
                    .padding(.vertical, 2)
                    .opacity(isActive ? 1 : 0)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Text(segment.speaker)
                            .font(.caption.monospaced().weight(.semibold))
                            .foregroundStyle(speakerColor)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(speakerColor.opacity(isActive ? 0.22 : 0.14))
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

                        Text(segment.formattedTimeRange)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(isActive ? .primary : .secondary)

                        Spacer(minLength: 0)

                        Image(systemName: "play.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Text(segment.text)
                        .font(font)
                        .foregroundStyle(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(isActive ? speakerColor.opacity(0.12) : Color.primary.opacity(0.035))
            )
            .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(speakerColor.opacity(isActive ? 0.6 : 0.14), lineWidth: isActive ? 1.5 : 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Click to seek audio to \(segment.formattedTimeRange)")
        .animation(.easeOut(duration: 0.16), value: isActive)
    }

    private var speakerColor: Color {
        let palette: [Color] = [
            .blue,
            .green,
            .orange,
            .purple,
            .pink,
            .teal,
        ]
        let number = Int(segment.speaker.drop { !$0.isNumber }) ?? 1
        return palette[(max(number, 1) - 1) % palette.count]
    }
}

private struct TaggedTranscriptSegment: Identifiable {
    let id: Int
    let speaker: String
    let startTime: String
    let startSeconds: TimeInterval
    let endTime: String?
    let endSeconds: TimeInterval?
    let text: String

    var formattedTimeRange: String {
        if let endTime {
            return "\(Self.formatTimestamp(startTime)) - \(Self.formatTimestamp(endTime))"
        }
        return Self.formatTimestamp(startTime)
    }

    private static func formatTimestamp(_ rawValue: String) -> String {
        guard let totalSeconds = timestampValue(rawValue) else { return rawValue }

        let minutes = Int(totalSeconds) / 60
        let seconds = totalSeconds - Double(minutes * 60)
        return String(
            format: "%d:%05.2f",
            locale: Locale(identifier: "en_US_POSIX"),
            minutes,
            seconds
        )
    }

    static func timestampValue(_ text: String) -> Double? {
        Double(text.replacingOccurrences(of: ",", with: "."))
    }
}

private enum TaggedTranscriptParser {
    private static let startTagRegex = try! NSRegularExpression(
        pattern: #"\[(\d+(?:[\.,]\d+)?)\]\[(S\d{1,3})\]"#,
        options: [.caseInsensitive]
    )
    private static let trailingTimestampRegex = try! NSRegularExpression(
        pattern: #"\[(\d+(?:[\.,]\d+)?)\]\s*$"#,
        options: []
    )

    static func parse(_ text: String) -> [TaggedTranscriptSegment] {
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = startTagRegex.matches(in: text, range: fullRange)
        guard !matches.isEmpty else { return [] }

        return matches.enumerated().compactMap { index, match in
            guard
                let matchRange = Range(match.range, in: text),
                let startRange = Range(match.range(at: 1), in: text),
                let speakerRange = Range(match.range(at: 2), in: text),
                let startSeconds = TaggedTranscriptSegment.timestampValue(String(text[startRange]))
            else {
                return nil
            }

            let bodyEnd: String.Index
            if index + 1 < matches.count,
               let nextRange = Range(matches[index + 1].range, in: text) {
                bodyEnd = nextRange.lowerBound
            } else {
                bodyEnd = text.endIndex
            }

            var segmentText = String(text[matchRange.upperBound..<bodyEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let endTime = stripTrailingTimestamp(from: &segmentText)
            let endSeconds = endTime.flatMap(TaggedTranscriptSegment.timestampValue)

            guard !segmentText.isEmpty else { return nil }

            return TaggedTranscriptSegment(
                id: index,
                speaker: String(text[speakerRange]).uppercased(),
                startTime: String(text[startRange]),
                startSeconds: startSeconds,
                endTime: endTime,
                endSeconds: endSeconds,
                text: segmentText
            )
        }
    }

    static func activeSegmentID(in segments: [TaggedTranscriptSegment], at time: TimeInterval) -> TaggedTranscriptSegment.ID? {
        for (index, segment) in segments.enumerated() {
            let nextStart = index + 1 < segments.count ? segments[index + 1].startSeconds : nil
            let end = segment.endSeconds ?? nextStart

            if let end {
                if time >= segment.startSeconds && time < max(end, segment.startSeconds) {
                    return segment.id
                }
            } else if time >= segment.startSeconds {
                return segment.id
            }
        }

        return nil
    }

    private static func stripTrailingTimestamp(from text: inout String) -> String? {
        var lastTimestamp: String?

        while true {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard
                let match = trailingTimestampRegex.firstMatch(in: text, range: range),
                let timestampRange = Range(match.range(at: 1), in: text),
                let fullMatchRange = Range(match.range, in: text)
            else {
                return lastTimestamp
            }

            lastTimestamp = String(text[timestampRange])
            text.removeSubrange(fullMatchRange)
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
}

// MARK: - Recording Indicator

private struct RecordingTranscriptPlaceholder: View {
    let supportsRealtime: Bool

    var body: some View {
        VStack(spacing: 8) {
            Spacer(minLength: 80)

            Image(systemName: supportsRealtime ? "text.bubble.fill" : "waveform.circle.fill")
                .font(.system(size: 40, weight: .medium))
                .foregroundStyle(.tertiary)

            Text(supportsRealtime ? "Listening..." : "Recording...")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(supportsRealtime ? "Realtime prediction will appear here" : "Starting realtime preview")
                .font(.caption)
                .foregroundStyle(.tertiary)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

private struct RecordingIndicator: View {
    let duration: TimeInterval
    let audioLevel: Float
    let supportsRealtime: Bool
    let isRealtime: Bool

    private let barCount = 28

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.red.opacity(0.14))
                    .frame(width: 34, height: 34)

                Image(systemName: "mic.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.red)
                    .symbolEffect(.pulse, options: .repeating, value: duration)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(isRealtime ? "Live" : "Recording")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text(supportsRealtime ? "Realtime preview" : "Starting...")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(minWidth: 88, alignment: .leading)

            TimelineView(.animation) { timeline in
                let phase = timeline.date.timeIntervalSinceReferenceDate * 5.8
                let level = max(0, min(CGFloat(audioLevel), 1))

                HStack(alignment: .center, spacing: 3) {
                    ForEach(0..<barCount, id: \.self) { index in
                        let position = CGFloat(index) / CGFloat(max(barCount - 1, 1))
                        let ripple = abs(sin(phase + Double(index) * 0.47))
                        let envelope = 0.35 + 0.65 * sin(position * .pi)
                        let idleLift = 0.10 + 0.10 * CGFloat(ripple)
                        let reactiveLift = level * CGFloat(0.35 + 0.65 * ripple) * envelope
                        let height = 8 + (idleLift + reactiveLift) * 34

                        Capsule()
                            .fill(barGradient(at: position))
                            .frame(width: 4, height: height)
                            .shadow(color: Color.red.opacity(0.22 * (idleLift + reactiveLift)), radius: 6)
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .animation(.easeOut(duration: 0.12), value: audioLevel)
            }

            Spacer()

            Text(formatDuration(duration))
                .font(.caption.monospacedDigit().weight(.medium))
                .foregroundStyle(.secondary)
                .frame(minWidth: 42, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.red.opacity(0.07))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.red.opacity(0.16), lineWidth: 1)
        }
    }

    private func barGradient(at position: CGFloat) -> LinearGradient {
        let leading = Color(red: 1.0, green: 0.23, blue: 0.30)
        let center = Color(red: 1.0, green: 0.42, blue: 0.58)
        let trailing = Color(red: 0.72, green: 0.33, blue: 1.0)
        let color = position < 0.5
            ? leading.mix(with: center, by: position * 2)
            : center.mix(with: trailing, by: (position - 0.5) * 2)

        return LinearGradient(
            colors: [color.opacity(0.65), color, color.opacity(0.75)],
            startPoint: .bottom,
            endPoint: .top
        )
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

private extension Color {
    func mix(with other: Color, by amount: CGFloat) -> Color {
        #if os(iOS)
        let first = UIColor(self)
        let second = UIColor(other)
        #else
        let first = NSColor(self)
        let second = NSColor(other)
        #endif

        var red1: CGFloat = 0
        var green1: CGFloat = 0
        var blue1: CGFloat = 0
        var alpha1: CGFloat = 0
        var red2: CGFloat = 0
        var green2: CGFloat = 0
        var blue2: CGFloat = 0
        var alpha2: CGFloat = 0

        first.getRed(&red1, green: &green1, blue: &blue1, alpha: &alpha1)
        second.getRed(&red2, green: &green2, blue: &blue2, alpha: &alpha2)

        let clampedAmount = max(0, min(amount, 1))
        return Color(
            red: red1 + (red2 - red1) * clampedAmount,
            green: green1 + (green2 - green1) * clampedAmount,
            blue: blue1 + (blue2 - blue1) * clampedAmount,
            opacity: alpha1 + (alpha2 - alpha1) * clampedAmount
        )
    }
}

#Preview {
    STTView()
}
