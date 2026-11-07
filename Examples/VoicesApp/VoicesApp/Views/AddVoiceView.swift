import SwiftUI
import UniformTypeIdentifiers
import AVFoundation
import MLX
import MLXAudioCore

struct AddVoiceView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var voiceName = ""
    @State private var voiceDescription = ""
    @State private var selectedLanguage = "English"
    @State private var selectedColor: Color = .blue

    // Voice cloning
    @State private var audioFileURL: URL?
    @State private var transcription = "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors."
    @State private var showFilePicker = false
    @State private var audioDuration: TimeInterval = 0
    @State private var isPlayingPreview = false
    @State private var audioPlayer: AVAudioPlayer?
    @State private var recorder = AudioRecorderManager()
    @State private var biometrics: VoiceBiometricsProfile? = nil

    private let calibrationScripts: [(title: String, text: String)] = [
        (
            "1. The Rainbow Passage (Phonetic Balance)",
            "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon."
        ),
        (
            "2. Dramatic Storyteller (Expressive Range)",
            "Wait — did you hear that? Beneath the ancient stone library, a golden clockwork engine began to hum! \"Quickly!\" whispered Elena, her voice trembling with wonder. \"Before the midnight bell strikes twelve, open the starlight vault!\""
        ),
        (
            "3. Technical & AI Pitch (Clear Articulation)",
            "Welcome to Codecaine Studio. Today we are synthesizing twenty-four kilohertz neural speech directly on local hardware, with zero cloud latency, automatic word alignment, and instant multi-format audio export."
        ),
    ]

    let languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese"]
    let colorOptions: [Color] = [
        .blue, .purple, .pink, .red, .orange, .yellow, .green, .teal, .cyan, .indigo
    ]

    var onSave: ((Voice) -> Void)?

    private var isVoiceCloning: Bool {
        audioFileURL != nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Voice name", text: $voiceName)
                    TextField("Description (optional)", text: $voiceDescription)
                } header: {
                    Text("Basic Info")
                }

                // Voice Cloning Section
                Section {
                    // Guided Calibration Script selector (from Codecaine File Manager)
                    Menu {
                        ForEach(calibrationScripts, id: \.title) { script in
                            Button(script.title) {
                                transcription = script.text
                            }
                        }
                    } label: {
                        HStack {
                            Image(systemName: "text.quote")
                                .foregroundStyle(.purple)
                            Text("Load Guided Calibration Script...")
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    TextField("Calibration / Reference Script (read this aloud)", text: $transcription, axis: .vertical)
                        .lineLimit(3...6)

                    // Live Microphone Recorder button
                    Button {
                        if recorder.isRecording {
                            if let audioArray = recorder.stopRecording() {
                                let samples = audioArray.asArray(Float.self)
                                let tempURL = FileManager.default.temporaryDirectory
                                    .appendingPathComponent("recorded_voice_\(UUID().uuidString).wav")
                                if let writer = try? StreamingWAVWriter(url: tempURL, sampleRate: 16_000) {
                                    try? writer.writeChunk(samples)
                                    let finalURL = writer.finalize()
                                    audioFileURL = finalURL
                                    audioDuration = Double(samples.count) / 16_000.0
                                    let bio = VoiceBiometricsAnalyzer.analyzeAudio(url: finalURL)
                                    biometrics = bio
                                    if voiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let bio {
                                        voiceName = "My \(bio.voiceRegister) Clone"
                                    }
                                    if voiceDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let bio {
                                        voiceDescription = "\(bio.timbreLabel) · \(Int(bio.medianF0Hz)) Hz"
                                    }
                                }
                            }
                        } else {
                            Task {
                                try? await recorder.startRecording()
                            }
                        }
                    } label: {
                        HStack {
                            Image(systemName: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                                .font(.title3)
                                .foregroundStyle(recorder.isRecording ? .red : .pink)
                            Text(recorder.isRecording
                                 ? String(format: "Recording Live... %.1fs (Click to Stop)", recorder.recordingDuration)
                                 : "Record Voice Sample from Microphone")
                                .fontWeight(.medium)
                                .foregroundStyle(recorder.isRecording ? .red : .primary)
                            Spacer()
                        }
                    }
                    .buttonStyle(.plain)

                    // Audio file picker (un-nested from Play/Clear buttons)
                    HStack {
                        Button(action: { showFilePicker = true }) {
                            HStack {
                                Image(systemName: audioFileURL != nil ? "waveform" : "plus.circle")
                                    .foregroundStyle(audioFileURL != nil ? .green : .blue)
                                if let url = audioFileURL {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(url.lastPathComponent)
                                            .foregroundStyle(.primary)
                                            .lineLimit(1)
                                        Text(formatDuration(audioDuration))
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                } else {
                                    Text("Or select existing audio / video file (.wav / .mp3 / .flac / .ogg / .m4a / .mp4 / .mov)")
                                        .foregroundStyle(.primary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        if audioFileURL != nil {
                            Button(action: { playPreview() }) {
                                Image(systemName: isPlayingPreview ? "stop.circle.fill" : "play.circle.fill")
                                    .font(.title2)
                                    .foregroundStyle(.blue)
                            }
                            .buttonStyle(.plain)
                            .help(isPlayingPreview ? "Stop preview" : "Play recorded/selected voice sample")

                            Button(action: { clearAudio() }) {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.title2)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Clear audio sample")
                        }
                    }

                    // Acoustic Voice Biometrics Profile Card (from Codecaine File Manager)
                    if let bio = biometrics {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Label("Acoustic Voice Biometrics Profile", systemImage: "waveform.path.ecg")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(.cyan)
                                Spacer()
                                Text("Matched Base: \(bio.recommendedBaseVoice)")
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 2)
                                    .background(Color.cyan.opacity(0.16), in: Capsule())
                            }

                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("F₀ PITCH")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundStyle(.secondary)
                                    Text(String(format: "%.0f Hz", bio.medianF0Hz))
                                        .font(.caption.weight(.bold).monospacedDigit())
                                }
                                Divider().frame(height: 22)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("REGISTER")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundStyle(.secondary)
                                    Text(bio.voiceRegister)
                                        .font(.caption.weight(.semibold))
                                }
                                Divider().frame(height: 22)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("TIMBRE")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundStyle(.secondary)
                                    Text(bio.timbreLabel)
                                        .font(.caption.weight(.semibold))
                                }
                                Divider().frame(height: 22)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("SNR QUALITY")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundStyle(.secondary)
                                    Text(String(format: "%.1f dB", bio.snrDb))
                                        .font(.caption.weight(.bold).monospacedDigit())
                                        .foregroundStyle(.green)
                                }
                            }
                        }
                        .padding(.vertical, 6)
                    }
                } header: {
                    Text("Zero-Shot Voice Cloning & Calibration Recorder")
                } footer: {
                    if audioFileURL != nil && transcription.isEmpty {
                        Text("Enter the exact text spoken in the audio file for best results.")
                            .foregroundStyle(.orange)
                    } else if audioFileURL == nil {
                        Text("Pick a calibration script above and record 5–15 seconds on your mic, or upload an existing audio sample to clone any voice.")
                    }
                }

                Section {
                    Picker("Language", selection: $selectedLanguage) {
                        ForEach(languages, id: \.self) { language in
                            Text(language).tag(language)
                        }
                    }
                } header: {
                    Text("Language")
                }

                Section {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 5), spacing: 12) {
                        ForEach(colorOptions, id: \.self) { color in
                            ColorButton(
                                color: color,
                                isSelected: selectedColor == color
                            ) {
                                selectedColor = color
                            }
                        }
                    }
                    .padding(.vertical, 8)
                } header: {
                    Text("Color")
                }

                Section {
                    HStack {
                        Spacer()
                        ZStack {
                            VoiceAvatar(color: selectedColor.opacity(0.5), size: 80)
                            if isVoiceCloning {
                                Image(systemName: "waveform")
                                    .font(.title2)
                                    .foregroundStyle(.white)
                            }
                        }
                        Spacer()
                    }
                    .padding(.vertical, 20)

                    HStack {
                        Spacer()
                        VStack(spacing: 4) {
                            Text(voiceName.isEmpty ? "Voice Name" : voiceName)
                                .font(.headline)
                            Text(voiceDescription.isEmpty ? (isVoiceCloning ? "Cloned Voice" : selectedLanguage) : voiceDescription)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            if isVoiceCloning {
                                Label("Voice Clone", systemImage: "waveform")
                                    .font(.caption)
                                    .foregroundStyle(.blue)
                            }
                        }
                        Spacer()
                    }
                } header: {
                    Text("Preview")
                }
            }
            .navigationTitle("Add Voice")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        saveVoice()
                    }
                    .disabled(voiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || audioFileURL == nil || transcription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [.item, .data, .audio, .movie, .video, .wav, .mp3, .aiff],
                allowsMultipleSelection: false
            ) { result in
                handleFileSelection(result)
            }
            .onDisappear {
                audioPlayer?.stop()
                if recorder.isRecording {
                    _ = recorder.stopRecording()
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 540, minHeight: 620)
        #endif
    }

    private func saveVoice() {
        // Copy audio file to app's documents directory if provided
        var savedAudioURL: URL? = nil
        if let sourceURL = audioFileURL {
            savedAudioURL = copyAudioToDocuments(sourceURL) ?? sourceURL
        }

        let trimmedName = voiceName.trimmingCharacters(in: .whitespacesAndNewlines)
        let newVoice = Voice(
            name: trimmedName,
            voiceId: biometrics?.recommendedBaseVoice ?? "1001",
            description: voiceDescription.isEmpty ? (isVoiceCloning ? "Cloned Voice" : "") : voiceDescription,
            language: selectedLanguage,
            color: selectedColor.opacity(0.3),
            isCustom: true,
            audioFileURL: savedAudioURL,
            transcription: transcription.isEmpty ? nil : transcription
        )
        onSave?(newVoice)
        dismiss()
    }

    private func copyAudioToDocuments(_ sourceURL: URL) -> URL? {
        let fileManager = FileManager.default
        guard let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }

        let voicesDir = documentsDir.appendingPathComponent("ClonedVoices", isDirectory: true)
        try? fileManager.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let destURL = voicesDir.appendingPathComponent("\(UUID().uuidString)_\(sourceURL.lastPathComponent)")

        do {
            let scoped = sourceURL.startAccessingSecurityScopedResource()
            defer { if scoped { sourceURL.stopAccessingSecurityScopedResource() } }

            if (try? AVAudioPlayer(contentsOf: sourceURL)) != nil {
                try fileManager.copyItem(at: sourceURL, to: destURL)
                return destURL
            } else if let (sr, audioData) = try? loadAudioArray(from: sourceURL), sr > 0 {
                let wavDest = voicesDir.appendingPathComponent("\(UUID().uuidString)_\(sourceURL.deletingPathExtension().lastPathComponent).wav")
                let samples = audioData.asArray(Float.self)
                try AudioUtils.writeWavFile(samples: samples, sampleRate: Double(sr), fileURL: wavDest)
                return wavDest
            } else {
                try fileManager.copyItem(at: sourceURL, to: destURL)
                return destURL
            }
        } catch {
            print("Failed to copy audio file: \(error)")
            return nil
        }
    }

    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }

            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }

            let localCopy = copyAudioToDocuments(url) ?? url
            audioFileURL = localCopy
            loadAudioDuration(from: localCopy)
            let bio = VoiceBiometricsAnalyzer.analyzeAudio(url: localCopy)
            biometrics = bio
            if voiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                voiceName = url.deletingPathExtension().lastPathComponent
            }
            if voiceDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let bio {
                voiceDescription = "\(bio.timbreLabel) · \(Int(bio.medianF0Hz)) Hz"
            }

        case .failure(let error):
            print("File selection error: \(error)")
        }
    }

    private func loadAudioDuration(from url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            audioDuration = player.duration
        } catch {
            print("Failed to load audio: \(error)")
            audioDuration = 0
        }
    }

    private func playPreview() {
        if isPlayingPreview {
            audioPlayer?.stop()
            isPlayingPreview = false
            return
        }

        guard let url = audioFileURL else { return }

        let scoped = url.startAccessingSecurityScopedResource()

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingPreview = true

            // Stop after playback
            DispatchQueue.main.asyncAfter(deadline: .now() + max(0.5, audioDuration) + 0.1) {
                self.isPlayingPreview = false
                if scoped {
                    url.stopAccessingSecurityScopedResource()
                }
            }
        } catch {
            print("Failed to play audio: \(error)")
            if scoped {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }

    private func clearAudio() {
        audioPlayer?.stop()
        audioFileURL = nil
        biometrics = nil
        transcription = ""
        audioDuration = 0
        isPlayingPreview = false
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

struct ColorButton: View {
    let color: Color
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(color.gradient)
                    .frame(width: 44, height: 44)

                if isSelected {
                    Circle()
                        .strokeBorder(Color.primary, lineWidth: 3)
                        .frame(width: 44, height: 44)

                    Image(systemName: "checkmark")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    AddVoiceView()
}
