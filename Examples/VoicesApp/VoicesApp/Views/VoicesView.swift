import SwiftUI
import AVFoundation

struct VoicesView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @State private var searchText = ""
    @State private var selectedCategory: String = "All"
    @State private var showAddVoice = false
    @State private var selectedVoice: Voice?
    @State private var previewingVoiceId: UUID? = nil
    @State private var previewPlayer: AVAudioPlayer? = nil
    @State private var prefs = StudioUserPreferencesManager.shared

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

    var modelId: String = ""
    var builtInVoices: [Voice] = []
    var activeVoice: Voice? = nil
    @Binding var recentlyUsed: [Voice]
    @Binding var customVoices: [Voice]
    var onVoiceSelected: ((Voice) -> Void)?

    private var shortModelName: String {
        modelId.split(separator: "/").last.map(String.init) ?? modelId
    }

    private var favoriteCount: Int {
        (builtInVoices + customVoices).filter { prefs.isFavoriteVoice($0) }.count
    }

    private var recentCount: Int {
        let recentSet = Set(prefs.recentVoiceKeys)
        let matched = (builtInVoices + customVoices).filter { recentSet.contains($0.voiceKey) }.count
        return max(matched, recentlyUsed.count)
    }

    private var availableCategories: [(name: String, count: Int)] {
        let conversationalCount = builtInVoices.filter { $0.category == "Conversational" }.count
        let narratorCount = builtInVoices.filter { $0.category == "Narrator" }.count
        let characterCount = builtInVoices.filter { $0.category == "Character" }.count
        let cloneCount = builtInVoices.filter { $0.category == "Reference Clones" }.count
        let studioCount = builtInVoices.filter { $0.category == "Studio Voices" }.count
        var items: [(String, Int)] = [
            ("All", builtInVoices.count + customVoices.count),
            ("★ Favorites", favoriteCount)
        ]
        if recentCount > 0 { items.append(("Recent", recentCount)) }
        if conversationalCount > 0 { items.append(("Conversational", conversationalCount)) }
        if narratorCount > 0 { items.append(("Narrator", narratorCount)) }
        if characterCount > 0 { items.append(("Character", characterCount)) }
        if cloneCount > 0 { items.append(("Reference Clones", cloneCount)) }
        if studioCount > 0 { items.append(("Studio Voices", studioCount)) }
        if !customVoices.isEmpty { items.append(("Custom", customVoices.count)) }
        return items
    }

    var filteredBuiltInVoices: [Voice] {
        let base = builtInVoices.filter { voice in
            let matchesCategory: Bool
            switch selectedCategory {
            case "All":
                matchesCategory = true
            case "★ Favorites":
                matchesCategory = prefs.isFavoriteVoice(voice)
            case "Recent":
                matchesCategory = prefs.recentVoiceKeys.contains(voice.voiceKey) ||
                    recentlyUsed.contains(where: { $0.voiceKey == voice.voiceKey })
            default:
                matchesCategory = (voice.category == selectedCategory)
            }
            let matchesSearch = searchText.isEmpty ||
                voice.name.localizedCaseInsensitiveContains(searchText) ||
                voice.description.localizedCaseInsensitiveContains(searchText) ||
                voice.category.localizedCaseInsensitiveContains(searchText)
            return matchesCategory && matchesSearch
        }
        return base.sorted { a, b in
            let favA = prefs.isFavoriteVoice(a)
            let favB = prefs.isFavoriteVoice(b)
            if favA != favB { return favA && !favB }
            return false
        }
    }

    var filteredCustomVoices: [Voice] {
        customVoices.filter { voice in
            let matchesCategory: Bool
            switch selectedCategory {
            case "All", "Custom":
                matchesCategory = true
            case "★ Favorites":
                matchesCategory = prefs.isFavoriteVoice(voice)
            case "Recent":
                matchesCategory = prefs.recentVoiceKeys.contains(voice.voiceKey) ||
                    recentlyUsed.contains(where: { $0.voiceKey == voice.voiceKey })
            default:
                matchesCategory = false
            }
            let matchesSearch = searchText.isEmpty ||
                voice.name.localizedCaseInsensitiveContains(searchText) ||
                voice.description.localizedCaseInsensitiveContains(searchText)
            return matchesCategory && matchesSearch
        }
    }

    private let gridColumns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10)
    ]

    var body: some View {
        VStack(spacing: 0) {
            // Top Apple Studio Sheet Header
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color.blue.opacity(0.28), Color.indigo.opacity(0.20)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 38, height: 38)
                    Image(systemName: "person.2.wave.2.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.blue)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text("Studio Voice Browser")
                            .font(.system(size: 16, weight: .bold))
                        Text("\(builtInVoices.count) Built-In Voices")
                            .font(.system(size: 10.5, weight: .bold))
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2.5)
                            .background(Color.blue.opacity(0.15), in: Capsule())
                    }
                    Text("Built-in voices & zero-shot reference clones for \(shortModelName.isEmpty ? "Fish S2 Pro" : shortModelName)")
                        .font(.system(size: 11.5))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    showAddVoice = true
                } label: {
                    Label("Clone New Voice", systemImage: "plus.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(
                            LinearGradient(colors: [Color.blue, Color.indigo], startPoint: .topLeading, endPoint: .bottomTrailing),
                            in: Capsule()
                        )
                }
                .buttonStyle(.plain)

                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(.ultraThinMaterial)

            Divider().opacity(0.25)

            // Search + Category Filter Ribbon
            VStack(spacing: 10) {
                SearchBar(text: $searchText, totalCount: builtInVoices.count + customVoices.count)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(availableCategories, id: \.name) { item in
                            let isSelected = selectedCategory == item.name
                            Button {
                                withAnimation(.spring(response: 0.24, dampingFraction: 0.82)) {
                                    selectedCategory = item.name
                                }
                            } label: {
                                HStack(spacing: 5) {
                                    Text(item.name)
                                        .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                                    Text("\(item.count)")
                                        .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 1.5)
                                        .background(
                                            isSelected ? Color.white.opacity(0.22) : Color.white.opacity(0.08),
                                            in: Capsule()
                                        )
                                }
                                .foregroundStyle(isSelected ? .white : .primary.opacity(0.85))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(
                                    Capsule()
                                        .fill(isSelected ? Color.blue : activeTheme.pillFill)
                                        .overlay(
                                            Capsule()
                                                .strokeBorder(isSelected ? Color.blue : activeTheme.pillStroke, lineWidth: 0.9)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            Divider().opacity(0.2)

            // 2-Column Bento Grid of Built-In & Custom Voices
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if (selectedCategory != "Custom" && filteredBuiltInVoices.isEmpty) &&
                       filteredCustomVoices.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: selectedCategory == "★ Favorites" ? "star.slash" : "magnifyingglass")
                                .font(.system(size: 36))
                                .foregroundStyle(.tertiary)
                            Text(selectedCategory == "★ Favorites" && searchText.isEmpty
                                 ? "No favorite voices starred yet — click the ★ button on any voice card to pin it here."
                                 : "No voices matching \"\(searchText)\"")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(.secondary)
                            Button("Show All Voices") {
                                searchText = ""
                                selectedCategory = "All"
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 48)
                    }

                    if selectedCategory != "Custom" && !filteredBuiltInVoices.isEmpty {
                        LazyVGrid(columns: gridColumns, spacing: 10) {
                            ForEach(filteredBuiltInVoices) { voice in
                                let isActive = (activeVoice?.name == voice.name && activeVoice?.voiceId == voice.voiceId)
                                StudioVoiceBentoCard(
                                    voice: voice,
                                    isSelected: isActive,
                                    isFavorite: prefs.isFavoriteVoice(voice),
                                    isPreviewing: previewingVoiceId == voice.id,
                                    onToggleFavorite: { prefs.toggleFavoriteVoice(voice) },
                                    onPreview: { togglePreview(for: voice) },
                                    onTap: {
                                        previewPlayer?.stop()
                                        previewingVoiceId = nil
                                        selectedVoice = voice
                                        onVoiceSelected?(voice)
                                    }
                                )
                            }
                        }
                    }

                    if (selectedCategory == "All" || selectedCategory == "Custom" || selectedCategory == "★ Favorites" || selectedCategory == "Recent") && !filteredCustomVoices.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("CUSTOM & RECORDED VOICES (\(filteredCustomVoices.count))")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundStyle(.secondary)

                            LazyVGrid(columns: gridColumns, spacing: 10) {
                                ForEach(filteredCustomVoices) { voice in
                                    let isActive = (activeVoice?.name == voice.name && activeVoice?.voiceId == voice.voiceId)
                                    StudioVoiceBentoCard(
                                        voice: voice,
                                        isSelected: isActive,
                                        isFavorite: prefs.isFavoriteVoice(voice),
                                        isPreviewing: previewingVoiceId == voice.id,
                                        showDelete: true,
                                        onToggleFavorite: { prefs.toggleFavoriteVoice(voice) },
                                        onPreview: { togglePreview(for: voice) },
                                        onDelete: {
                                            customVoices.removeAll { $0.id == voice.id }
                                        },
                                        onTap: {
                                            previewPlayer?.stop()
                                            previewingVoiceId = nil
                                            selectedVoice = voice
                                            onVoiceSelected?(voice)
                                        }
                                    )
                                }
                            }
                        }
                    }
                }
                .padding(20)
            }
        }
        #if os(macOS)
        .frame(minWidth: 780, minHeight: 580)
        #endif
        .background(activeTheme.canvasBackground)
        .preferredColorScheme(activeTheme.colorScheme)
        .onDisappear {
            previewPlayer?.stop()
            previewingVoiceId = nil
        }
        .sheet(isPresented: $showAddVoice) {
            AddVoiceView { newVoice in
                customVoices.append(newVoice)
            }
        }
    }

    private func togglePreview(for voice: Voice) {
        if previewingVoiceId == voice.id {
            previewPlayer?.stop()
            previewingVoiceId = nil
            return
        }
        previewPlayer?.stop()
        guard let url = Voice.previewAudioURL(for: voice) else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        if let player = try? AVAudioPlayer(contentsOf: url) {
            previewPlayer = player
            previewingVoiceId = voice.id
            player.play()
            let duration = max(0.5, min(player.duration, 10.0))
            let targetId = voice.id
            DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
                if self.previewingVoiceId == targetId {
                    self.previewPlayer?.stop()
                    self.previewingVoiceId = nil
                }
            }
        }
    }
}

struct StudioVoiceBentoCard: View {
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    let voice: Voice
    var isSelected: Bool = false
    var isFavorite: Bool = false
    var isPreviewing: Bool = false
    var showDelete: Bool = false
    var onToggleFavorite: (() -> Void)? = nil
    var onPreview: (() -> Void)? = nil
    var onDelete: (() -> Void)? = nil
    var onTap: () -> Void

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

    private var categoryColor: Color {
        switch voice.category {
        case "Conversational": return .blue
        case "Narrator": return .purple
        case "Character": return .orange
        case "Reference Clones": return .green
        default: return .teal
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            Button(action: onTap) {
                HStack(spacing: 12) {
                    VoiceAvatar(color: voice.color, size: 36)

                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Text(voice.name)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.primary)
                                .lineLimit(1)

                            if isSelected {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.blue)
                            }

                            Text(voice.category)
                                .font(.system(size: 9.5, weight: .bold))
                                .foregroundStyle(categoryColor)
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(categoryColor.opacity(0.14), in: Capsule())
                        }

                        Text(voice.description)
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Spacer(minLength: 4)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if let onToggleFavorite {
                Button(action: onToggleFavorite) {
                    Image(systemName: isFavorite ? "star.fill" : "star")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(isFavorite ? (activeTheme.isLight ? .orange : .yellow) : .secondary.opacity(0.65))
                }
                .buttonStyle(.plain)
                .help(isFavorite ? "Remove from favorite voices" : "Add to favorite voices")
            }

            if let onPreview, Voice.previewAudioURL(for: voice) != nil {
                Button(action: onPreview) {
                    Image(systemName: isPreviewing ? "stop.circle.fill" : "speaker.wave.2.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(isPreviewing ? .orange : .blue.opacity(0.85))
                }
                .buttonStyle(.plain)
                .help(isPreviewing ? "Stop voice preview" : "Audition voice sample")
            }

            if showDelete, let onDelete {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 12))
                        .foregroundStyle(.red.opacity(0.8))
                }
                .buttonStyle(.plain)
                .help("Delete custom voice")
            } else {
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary.opacity(0.6))
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .fill(isSelected ? Color.blue.opacity(0.14) : (isFavorite ? Color.yellow.opacity(0.05) : activeTheme.elevatedRowFill))
                .overlay(
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .strokeBorder(isSelected ? Color.blue.opacity(0.75) : (isFavorite ? Color.yellow.opacity(0.35) : activeTheme.pillStroke), lineWidth: isSelected ? 1.2 : 0.8)
                )
        )
    }
}

// MARK: - Search Bar

struct SearchBar: View {
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @Binding var text: String
    var totalCount: Int = 41

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)

            TextField("Search \(totalCount) voices by name, style, or category (Conversational, Narrator, Character)...", text: $text)
                .font(.system(size: 12.5))
                .textFieldStyle(.plain)

            if !text.isEmpty {
                Button(action: { text = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
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

#Preview {
    VoicesView(
        modelId: "mlx-community/fish-audio-s2-pro-bf16",
        builtInVoices: Voice.builtInVoices(for: "mlx-community/fish-audio-s2-pro-bf16"),
        recentlyUsed: .constant(Voice.samples),
        customVoices: .constant(Voice.customVoices)
    )
}
