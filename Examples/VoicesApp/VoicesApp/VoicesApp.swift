import SwiftUI
import Combine
#if os(macOS)
import AppKit
#endif

// MARK: - Studio Futuristic Icon Loader & Showcase

enum StudioBrandIconLoader {
    #if os(macOS)
    @MainActor
    static let nsImage: NSImage? = {
        if let url = Bundle.main.url(forResource: "AppIcon", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            return img
        }
        if let url = Bundle.main.url(forResource: "AppIcon", withExtension: "icns"),
           let img = NSImage(contentsOf: url) {
            return img
        }
        if let img = NSImage(named: "AppIcon") {
            return img
        }
        return nil
    }()

    @MainActor
    static func applyDockIcon() {
        if let icon = nsImage {
            NSApplication.shared.applicationIconImage = icon
        }
    }
    #endif
}

struct FuturisticStudioEmblemView: View {
    let size: CGFloat
    let accentColor: Color

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.24, style: .continuous)
                .fill(
                    RadialGradient(
                        colors: [
                            accentColor.opacity(0.45),
                            Color.purple.opacity(0.28),
                            Color.black.opacity(0.85)
                        ],
                        center: .center,
                        startRadius: 2,
                        endRadius: size * 0.7
                    )
                )
                .frame(width: size, height: size)
                .shadow(color: accentColor.opacity(0.45), radius: size * 0.22, y: 2)

            #if os(macOS)
            if let nsImg = StudioBrandIconLoader.nsImage {
                Image(nsImage: nsImg)
                    .resizable()
                    .interpolation(.high)
                    .antialiased(true)
                    .scaledToFit()
                    .frame(width: size, height: size)
                    .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        Color.cyan.opacity(0.75),
                                        accentColor.opacity(0.55),
                                        Color.purple.opacity(0.65)
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: max(1.0, size * 0.032)
                            )
                    )
            } else {
                Image(systemName: "waveform.badge.magnifyingglass")
                    .font(.system(size: size * 0.46, weight: .bold))
                    .foregroundStyle(.white)
            }
            #else
            Image(systemName: "waveform")
                .font(.system(size: size * 0.46, weight: .bold))
                .foregroundStyle(.white)
            #endif
        }
    }
}

struct FuturisticIconShowcaseSheet: View {
    let theme: ReadAlongTheme
    @Binding var isPresented: Bool

    var body: some View {
        VStack(spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text("FUTURISTIC STUDIO EMBLEM")
                            .font(.system(size: 10, weight: .heavy, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Capsule().fill(Color.cyan.opacity(0.18)))
                            .foregroundStyle(Color.cyan)

                        Text("1024×1024 RETINA MASTER · .ICNS + .XCASSETS")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Text("Codecaine MLX RAD AI Audio Studio — Neural Core Icon")
                        .font(.system(size: 18, weight: .bold))
                }
                Spacer()
                Button {
                    isPresented = false
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "xmark.circle.fill")
                        Text("Close")
                    }
                    .font(.system(size: 12, weight: .semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Capsule().fill(Color.primary.opacity(0.10)))
                }
                .buttonStyle(.plain)
            }

            HStack(alignment: .center, spacing: 24) {
                // Hero 280x280 preview with ambient plasma halo
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color.cyan.opacity(0.32),
                                    Color.purple.opacity(0.24),
                                    Color.clear
                                ],
                                center: .center,
                                startRadius: 12,
                                endRadius: 155
                            )
                        )
                        .frame(width: 300, height: 300)

                    FuturisticStudioEmblemView(size: 268, accentColor: theme.activeWordBackground)
                }

                // Futuristic architectural callouts + multi-scale Dock preview strip
                VStack(alignment: .leading, spacing: 12) {
                    iconSpecRow(
                        icon: "mic.and.signal.meter.fill",
                        color: .cyan,
                        title: "Holographic Studio Condenser Core",
                        detail: "Precision acoustic mesh capsule suspended inside an iridescent optical crystal gyroscope ring."
                    )
                    iconSpecRow(
                        icon: "cpu.fill",
                        color: .orange,
                        title: "Apple Silicon MLX Neural Die Base",
                        detail: "Exposed gold & cyan neural fabric traces representing unified memory zero-shot TTS, ASR, and STS."
                    )
                    iconSpecRow(
                        icon: "waveform.path.ecg",
                        color: .purple,
                        title: "3D Spectral Frequency Breakout Ribbons",
                        detail: "Luminous cyan, ultraviolet, and plasma-magenta harmonic waves breaking beyond the titanium squircle."
                    )
                    iconSpecRow(
                        icon: "dial.high.fill",
                        color: .green,
                        title: "Laser-Etched Radial dB & Frequency Telemetry",
                        detail: "Micro-machined brushed obsidian bezel with precision studio calibration tick marks."
                    )

                    Divider()
                        .opacity(0.4)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("MACOS DOCK & ASSET CATALOG SCALES")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundStyle(.secondary)

                        HStack(alignment: .bottom, spacing: 16) {
                            ForEach([64, 48, 32, 24], id: \.self) { pt in
                                VStack(spacing: 4) {
                                    FuturisticStudioEmblemView(size: CGFloat(pt), accentColor: theme.activeWordBackground)
                                    Text("\(pt)pt")
                                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(22)
        .frame(width: 760)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(red: 0.06, green: 0.08, blue: 0.13).opacity(0.96))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.cyan.opacity(0.55), Color.purple.opacity(0.45)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.2
                        )
                )
        )
        .preferredColorScheme(.dark)
    }

    private func iconSpecRow(icon: String, color: Color, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(color)
                .frame(width: 26, height: 26)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(color.opacity(0.15))
                )

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12.5, weight: .bold))
                    .foregroundStyle(.white)
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.74))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

@main
struct VoicesApp: App {
    @State private var selectedTab: Int = {
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--tab"), idx + 1 < args.count, let t = Int(args[idx + 1]) {
            return t
        }
        return 0
    }()

    init() {
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--theme"), idx + 1 < args.count {
            let requested = args[idx + 1].lowercased()
            if ReadAlongTheme(rawValue: requested) != nil {
                UserDefaults.standard.set(requested, forKey: "VoicesApp.TTSViewModel.readAlongTheme")
            }
        }
        Task { @MainActor in
            #if os(macOS)
            StudioBrandIconLoader.applyDockIcon()
            #endif
            LocalhostServerManager.shared.startIfNeeded()
        }
    }

    var body: some Scene {
        WindowGroup {
            AppleStudioShellView(selectedTab: $selectedTab)
                #if os(macOS)
                .frame(minWidth: 960, minHeight: 720)
                .onAppear {
                    DispatchQueue.main.async {
                        StudioBrandIconLoader.applyDockIcon()
                        NSApplication.shared.activate(ignoringOtherApps: true)
                        for window in NSApplication.shared.windows {
                            window.title = "Codecaine MLX RAD AI Audio Studio"
                            if window.frame.width < 1340 || window.frame.height < 840 {
                                var frame = window.frame
                                frame.size.width = max(frame.size.width, 1380)
                                frame.size.height = max(frame.size.height, 880)
                                window.setFrame(frame, display: true, animate: false)
                                window.center()
                            }
                            window.makeKeyAndOrderFront(nil)
                        }
                        let args = ProcessInfo.processInfo.arguments
                        if let snapIdx = args.firstIndex(of: "--snapshot-out"), snapIdx + 1 < args.count {
                            let outPath = args[snapIdx + 1]
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
                                if let window = NSApplication.shared.windows.first(where: { $0.frame.width > 400 }),
                                   let contentView = window.contentView {
                                    if window.frame.width < 1380 || window.frame.height < 880 {
                                        window.setContentSize(NSSize(width: 1400, height: 900))
                                    }
                                    contentView.layoutSubtreeIfNeeded()
                                    let bounds = contentView.bounds
                                    if let rep = contentView.bitmapImageRepForCachingDisplay(in: bounds) {
                                        contentView.cacheDisplay(in: bounds, to: rep)
                                        if let pngData = rep.representation(using: .png, properties: [:]) {
                                            try? pngData.write(to: URL(fileURLWithPath: outPath))
                                        }
                                    }
                                }
                                NSApplication.shared.terminate(nil)
                            }
                        }
                    }
                }
                #endif
        }
        #if os(macOS)
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1380, height: 880)
        .commands {
            CommandMenu("Studio") {
                Button("Voice Studio (TTS & Cloning)") {
                    selectedTab = 0
                }
                .keyboardShortcut("1", modifiers: .command)

                Button("Transcribe (Speech to Text)") {
                    selectedTab = 1
                }
                .keyboardShortcut("2", modifiers: .command)

                Button("Audio Lab (Stems, Denoise & LID)") {
                    selectedTab = 2
                }
                .keyboardShortcut("3", modifiers: .command)

                Button("Localhost API Server") {
                    selectedTab = 3
                }
                .keyboardShortcut("4", modifiers: .command)

                Button("Packages & Downloaded Models") {
                    selectedTab = 4
                }
                .keyboardShortcut("5", modifiers: .command)

                Divider()

                Menu("Studio & Read-Along Theme") {
                    ForEach(ReadAlongTheme.allCases) { item in
                        Button(item.label) {
                            UserDefaults.standard.set(item.rawValue, forKey: "VoicesApp.TTSViewModel.readAlongTheme")
                        }
                    }
                }

                Divider()

                Button("Purge Metal GPU VRAM Cache") {
                    HardwareGuardInspector.purgeGPUCache()
                    LocalhostServerManager.shared.unloadAllModelsAndPurgeVRAM()
                }
                .keyboardShortcut("k", modifiers: [.command, .shift])
            }
        }
        #endif
    }
}

// MARK: - Apple Sequoia / Logic Pro Unified Studio Shell

struct StudioWorkspaceTab: Identifiable {
    let id: Int
    let title: String
    let subtitle: String
    let icon: String
    var shortcutKey: KeyEquivalent {
        KeyEquivalent(Character("\(id + 1)"))
    }
}

struct AppleStudioShellView: View {
    @Binding var selectedTab: Int
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    @State private var server = LocalhostServerManager.shared
    @State private var hardware = HardwareGuardInspector.inspect()
    @State private var visitedTabs: Set<Int> = []
    @State private var showIconShowcase: Bool = ProcessInfo.processInfo.arguments.contains("--show-icon-showcase")
    @Namespace private var tabNamespace
    private let telemetryTimer = Timer.publish(every: 4.0, on: .main, in: .common).autoconnect()

    private var activeTheme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

    private let tabs: [StudioWorkspaceTab] = [
        StudioWorkspaceTab(id: 0, title: "Voice Studio", subtitle: "TTS & Cloning", icon: "waveform"),
        StudioWorkspaceTab(id: 1, title: "Transcribe", subtitle: "Speech to Text", icon: "mic.fill"),
        StudioWorkspaceTab(id: 2, title: "Audio Lab", subtitle: "Stems & Denoise", icon: "tuningfork"),
        StudioWorkspaceTab(id: 3, title: "Localhost API", subtitle: "Ollama & LM Studio", icon: "server.rack"),
        StudioWorkspaceTab(id: 4, title: "Packages", subtitle: "Models & Tools", icon: "shippingbox.fill")
    ]

    var body: some View {
        let theme = activeTheme
        ZStack {
            // Dynamic Studio & Reader Theme ambient background
            LinearGradient(
                colors: theme.shellGradientColors,
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            .animation(.easeInOut(duration: 0.22), value: rawStudioTheme)

            VStack(spacing: 0) {
                // Unified Apple Glass Header Bar
                HStack(spacing: 12) {
                    // Brand identity (clickable to inspect the full-resolution Futuristic Studio Icon)
                    Button {
                        withAnimation(.spring(response: 0.28, dampingFraction: 0.84)) {
                            showIconShowcase.toggle()
                        }
                    } label: {
                        HStack(spacing: 10) {
                            FuturisticStudioEmblemView(
                                size: 34,
                                accentColor: theme.activeWordBackground
                            )

                            VStack(alignment: .leading, spacing: 1) {
                                Text("Codecaine MLX RAD AI Audio Studio")
                                    .font(.system(size: 12.5, weight: .bold))
                                    .foregroundStyle(.primary)
                                Text("Apple Silicon MLX · MIT Open Source")
                                    .font(.system(size: 10, weight: .semibold))
                                    .foregroundStyle(theme.activeWordBackground)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Click to inspect the 1024×1024 Futuristic Neural Audio Studio Icon")
                    .popover(isPresented: $showIconShowcase, arrowEdge: .bottom) {
                        FuturisticIconShowcaseSheet(theme: theme, isPresented: $showIconShowcase)
                    }
                    .fixedSize()

                    Spacer(minLength: 8)

                    // Center: Apple Segmented Glass Pill Switcher
                    HStack(spacing: 4) {
                        ForEach(tabs) { tab in
                            let isSelected = selectedTab == tab.id
                            Button {
                                withAnimation(.spring(response: 0.28, dampingFraction: 0.82)) {
                                    selectedTab = tab.id
                                }
                            } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: tab.icon)
                                        .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                                        .foregroundStyle(isSelected ? theme.activeWordForeground : .secondary)

                                    Text(tab.title)
                                        .font(.system(size: 12, weight: isSelected ? .semibold : .medium))
                                        .foregroundStyle(isSelected ? theme.activeWordForeground : .primary.opacity(0.82))

                                    Text("⌘\(tab.id + 1)")
                                        .font(.system(size: 9.5, weight: .bold, design: .rounded))
                                        .foregroundStyle(isSelected ? theme.activeWordForeground.opacity(0.82) : .secondary.opacity(0.65))
                                        .padding(.horizontal, 4.5)
                                        .padding(.vertical, 1)
                                        .background(
                                            Capsule()
                                                .fill(isSelected ? theme.activeWordForeground.opacity(0.15) : Color.primary.opacity(0.06))
                                        )
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6.5)
                                .background {
                                    if isSelected {
                                        Capsule()
                                            .fill(
                                                LinearGradient(
                                                    colors: [
                                                        theme.activeWordBackground.opacity(0.92),
                                                        theme.activeWordBackground.opacity(0.74)
                                                    ],
                                                    startPoint: .top,
                                                    endPoint: .bottom
                                                )
                                            )
                                            .overlay(
                                                Capsule()
                                                    .strokeBorder(Color.white.opacity(0.26), lineWidth: 0.8)
                                            )
                                            .shadow(color: theme.activeWordBackground.opacity(0.32), radius: 5, y: 2)
                                            .matchedGeometryEffect(id: "ActiveStudioTab", in: tabNamespace)
                                    }
                                }
                                .contentShape(Capsule())
                            }
                            .buttonStyle(.plain)
                            .keyboardShortcut(tab.shortcutKey, modifiers: .command)
                            .help("\(tab.title) — \(tab.subtitle) (⌘\(tab.id + 1))")
                        }
                    }
                    .padding(4)
                    .background(
                        Capsule()
                            .fill(theme.colorScheme == .light ? Color.black.opacity(0.08) : Color.black.opacity(0.32))
                            .overlay(
                                Capsule()
                                    .strokeBorder(Color.primary.opacity(0.10), lineWidth: 0.8)
                            )
                    )
                    .fixedSize()

                    Spacer(minLength: 8)

                    // Global Studio Theme Quick Switcher Pill
                    Menu {
                        Section("Studio & Read-Along Theme") {
                            ForEach(ReadAlongTheme.allCases) { item in
                                Button {
                                    withAnimation(.easeInOut(duration: 0.22)) {
                                        rawStudioTheme = item.rawValue
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
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: theme.icon)
                                .font(.system(size: 10.5, weight: .semibold))
                                .foregroundStyle(theme.activeWordBackground)
                            Text(theme.label)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.primary)
                            Image(systemName: "chevron.down")
                                .font(.system(size: 8, weight: .bold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule()
                                .fill(theme.activeWordBackground.opacity(0.15))
                                .overlay(
                                    Capsule()
                                        .strokeBorder(theme.activeWordBackground.opacity(0.35), lineWidth: 0.8)
                                )
                        )
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("Switch Studio & Read-Along Theme")

                    // Right: Live Localhost Drop-In & Hardware Guard Status Pill
                    Button {
                        withAnimation(.spring(response: 0.28, dampingFraction: 0.82)) {
                            selectedTab = 3
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(server.isRunning ? Color.green : Color.orange)
                                .frame(width: 7, height: 7)
                                .shadow(color: (server.isRunning ? Color.green : Color.orange).opacity(0.6), radius: 4)

                            Text(
                                server.isRunning
                                    ? ":\(server.primaryPort) + \(max(1, server.activePorts.count)) APIs"
                                    : "Server Idle"
                            )
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(.primary.opacity(0.9))

                            Divider()
                                .frame(height: 12)

                            HStack(spacing: 4) {
                                Image(systemName: "memorychip")
                                    .font(.system(size: 9.5, weight: .semibold))
                                    .foregroundStyle(theme.activeWordBackground)
                                Text(
                                    hardware.activeMetalMB >= 10
                                        ? String(format: "%.0f MB / %.0f GB", hardware.activeMetalMB, hardware.safeBudgetGB)
                                        : String(format: "%.0f GB Guard", hardware.safeBudgetGB)
                                )
                                .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                        .background(
                            Capsule()
                                .fill(theme.cardBackground)
                                .overlay(
                                    Capsule()
                                        .strokeBorder(Color.primary.opacity(0.12), lineWidth: 0.8)
                                )
                        )
                    }
                    .buttonStyle(.plain)
                    .fixedSize()
                    .help("Standalone Localhost API active on ports 8080, 11434 (Ollama), and 1234 (LM Studio) · Click to inspect VRAM & endpoints (⌘4)")
                }
                .padding(.horizontal, 16)
                .padding(.top, 28)
                .padding(.bottom, 10)
                .background(.ultraThinMaterial)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(Color.primary.opacity(0.10))
                        .frame(height: 1)
                }

                // Active Workspace Content (keep views mounted so models, audio, and transcripts persist across tab switches)
                ZStack {
                    if visitedTabs.contains(0) || selectedTab == 0 {
                        ContentView()
                            .opacity(selectedTab == 0 ? 1 : 0)
                            .allowsHitTesting(selectedTab == 0)
                            .accessibilityHidden(selectedTab != 0)
                    }

                    if visitedTabs.contains(1) || selectedTab == 1 {
                        STTView()
                            .opacity(selectedTab == 1 ? 1 : 0)
                            .allowsHitTesting(selectedTab == 1)
                            .accessibilityHidden(selectedTab != 1)
                    }

                    if visitedTabs.contains(2) || selectedTab == 2 {
                        STSView()
                            .opacity(selectedTab == 2 ? 1 : 0)
                            .allowsHitTesting(selectedTab == 2)
                            .accessibilityHidden(selectedTab != 2)
                    }

                    if visitedTabs.contains(3) || selectedTab == 3 {
                        LocalhostServerView()
                            .opacity(selectedTab == 3 ? 1 : 0)
                            .allowsHitTesting(selectedTab == 3)
                            .accessibilityHidden(selectedTab != 3)
                    }

                    if visitedTabs.contains(4) || selectedTab == 4 {
                        PackagesInstallerView()
                            .opacity(selectedTab == 4 ? 1 : 0)
                            .allowsHitTesting(selectedTab == 4)
                            .accessibilityHidden(selectedTab != 4)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .ignoresSafeArea(.container, edges: .top)
        }
        .navigationTitle("Codecaine MLX RAD AI Audio Studio")
        .preferredColorScheme(theme.colorScheme)
        .onChange(of: selectedTab) { _, newTab in
            visitedTabs.insert(newTab)
            hardware = HardwareGuardInspector.inspect()
        }
        .onReceive(telemetryTimer) { _ in
            hardware = HardwareGuardInspector.inspect()
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelCacheDidChange)) { _ in
            hardware = HardwareGuardInspector.inspect()
        }
        .onReceive(NotificationCenter.default.publisher(for: .sendTranscriptToVoiceStudio)) { _ in
            visitedTabs.insert(0)
            withAnimation(.spring(response: 0.28, dampingFraction: 0.82)) {
                selectedTab = 0
            }
        }
    }
}

// MARK: - Reusable Apple Bento Card Container

struct AppleBentoCard<Content: View>: View {
    @AppStorage("VoicesApp.TTSViewModel.readAlongTheme") private var rawStudioTheme: String = ReadAlongTheme.midnight.rawValue
    var padding: CGFloat = 18
    @ViewBuilder var content: () -> Content

    private var theme: ReadAlongTheme {
        ReadAlongTheme(rawValue: rawStudioTheme) ?? .midnight
    }

    var body: some View {
        content()
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(theme.canvasBackground.opacity(theme.colorScheme == .light ? 0.88 : 0.72))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: theme.canvasBorderColors,
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(theme.colorScheme == .light ? 0.10 : 0.24), radius: 12, y: 6)
            )
    }
}

// MARK: - Standalone Localhost API & Hardware Guard View

struct LocalhostServerView: View {
    @State private var server = LocalhostServerManager.shared
    @State private var hardware = HardwareGuardInspector.inspect()
    @State private var copiedBanner: String?
    @State private var portString: String = "8080"

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                // Hero Banner Card
                AppleBentoCard(padding: 20) {
                    HStack(alignment: .center, spacing: 16) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [Color.green.opacity(0.28), Color.blue.opacity(0.22)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 48, height: 48)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                        .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
                                )

                            Image(systemName: "server.rack")
                                .font(.system(size: 22, weight: .semibold))
                                .foregroundStyle(.green)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text("Standalone Localhost AI Audio Server")
                                    .font(.system(size: 18, weight: .semibold))
                                Text(server.isRunning ? "ONLINE" : "STOPPED")
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                                    .foregroundStyle(server.isRunning ? .green : .orange)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background((server.isRunning ? Color.green : Color.orange).opacity(0.16), in: Capsule())
                            }
                            Text("Native Apple Silicon MLX server with automatic drop-in emulation for Ollama (:11434) and LM Studio (:1234) — no external daemon required.")
                                .font(.system(size: 12.5))
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        HStack(spacing: 8) {
                            Button {
                                Task { await runSelfTest() }
                            } label: {
                                Label("Run Loopback Test", systemImage: "bolt.horizontal.fill")
                                    .font(.system(size: 12, weight: .semibold))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(
                                        LinearGradient(colors: [.blue, .indigo], startPoint: .top, endPoint: .bottom),
                                        in: Capsule()
                                    )
                                    .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)

                            Button {
                                if let p = UInt16(portString), p > 0 {
                                    server.primaryPort = p
                                }
                                server.startServer()
                                showBanner("Restarted listeners on \(server.statusMessage)")
                            } label: {
                                Label(server.isRunning ? "Restart" : "Start", systemImage: "arrow.clockwise")
                                    .font(.system(size: 12, weight: .medium))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(Color.white.opacity(0.08), in: Capsule())
                            }
                            .buttonStyle(.plain)

                            if server.isRunning {
                                Button {
                                    server.stopServer()
                                } label: {
                                    Text("Stop")
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundStyle(.red)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 7)
                                        .background(Color.red.opacity(0.12), in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                if let banner = copiedBanner {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text(banner)
                            .font(.system(size: 12.5, weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.green.opacity(0.14), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(Color.green.opacity(0.3), lineWidth: 1)
                    )
                }

                // Bento Row: Hardware Guard + Multi-Port Listeners
                HStack(alignment: .top, spacing: 18) {
                    // 1. Hardware RAM Safety Guard Bento Card
                    AppleBentoCard {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Label("Apple Silicon 70% RAM Guard", systemImage: "memorychip.fill")
                                    .font(.system(size: 14, weight: .semibold))
                                    .foregroundStyle(.primary)
                                Spacer()
                                Button {
                                    HardwareGuardInspector.purgeGPUCache()
                                    server.unloadAllModelsAndPurgeVRAM()
                                    hardware = HardwareGuardInspector.inspect()
                                    showBanner("Metal GPU VRAM cache purged (keep_alive: 0)")
                                } label: {
                                    Label("Purge VRAM", systemImage: "sparkles")
                                        .font(.system(size: 11.5, weight: .semibold))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.orange.opacity(0.16), in: Capsule())
                                        .foregroundStyle(.orange)
                                }
                                .buttonStyle(.plain)
                            }

                            // 4-Metric Grid
                            HStack(spacing: 12) {
                                HardwareMetricTile(
                                    label: "CHIP",
                                    value: hardware.chipName.replacingOccurrences(of: "Apple ", with: ""),
                                    sub: "\(hardware.cpuCores) CPU Cores",
                                    tint: .primary
                                )
                                HardwareMetricTile(
                                    label: "UNIFIED RAM",
                                    value: String(format: "%.0f GB", hardware.totalRamGB),
                                    sub: "Physical Memory",
                                    tint: .primary
                                )
                                HardwareMetricTile(
                                    label: "70% SAFE CEILING",
                                    value: String(format: "%.1f GB", hardware.safeBudgetGB),
                                    sub: "Guard Limit",
                                    tint: .green
                                )
                                HardwareMetricTile(
                                    label: "METAL VRAM",
                                    value: String(format: "%.0f MB", hardware.activeMetalMB),
                                    sub: String(format: "Peak %.0f MB", hardware.peakMetalMB),
                                    tint: .blue
                                )
                            }

                            // Apple System Settings-style Memory Bar
                            VStack(alignment: .leading, spacing: 6) {
                                GeometryReader { geo in
                                    let activeGB = hardware.activeMetalMB / 1024.0
                                    let safeRatio = min(max(hardware.safeBudgetGB / max(hardware.totalRamGB, 1), 0.1), 1.0)
                                    let usedRatio = min(max(activeGB / max(hardware.totalRamGB, 1), 0.015), safeRatio)
                                    ZStack(alignment: .leading) {
                                        Capsule()
                                            .fill(Color.white.opacity(0.07))
                                        Capsule()
                                            .fill(Color.green.opacity(0.18))
                                            .frame(width: geo.size.width * safeRatio)
                                        Capsule()
                                            .fill(
                                                LinearGradient(
                                                    colors: [.blue, .cyan],
                                                    startPoint: .leading,
                                                    endPoint: .trailing
                                                )
                                            )
                                            .frame(width: geo.size.width * usedRatio)
                                    }
                                }
                                .frame(height: 8)

                                Text("\(hardware.tierLabel) — Prevents unified memory swapping so macOS stays responsive.")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    // 2. Multi-Port Emulation Bento Card
                    AppleBentoCard {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Label("Drop-In Localhost Listeners", systemImage: "antenna.radiowaves.left.and.right")
                                    .font(.system(size: 14, weight: .semibold))
                                Spacer()
                                Text("\(server.activePorts.count) Active Ports")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundStyle(.blue)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Color.blue.opacity(0.14), in: Capsule())
                            }

                            // Active port badges
                            HStack(spacing: 8) {
                                PortStatusBadge(
                                    port: Int(server.primaryPort),
                                    label: "VoicesApp API",
                                    active: server.activePorts.contains(server.primaryPort)
                                )
                                PortStatusBadge(
                                    port: 11434,
                                    label: "Ollama Drop-In",
                                    active: server.activePorts.contains(11434)
                                )
                                PortStatusBadge(
                                    port: 1234,
                                    label: "LM Studio Drop-In",
                                    active: server.activePorts.contains(1234)
                                )
                            }

                            HStack(spacing: 14) {
                                HStack(spacing: 6) {
                                    Text("Primary Port:")
                                        .font(.system(size: 11.5))
                                        .foregroundStyle(.secondary)
                                    TextField("8080", text: $portString)
                                        .textFieldStyle(.plain)
                                        .font(.system(size: 11.5, weight: .semibold, design: .monospaced))
                                        .padding(.horizontal, 7)
                                        .padding(.vertical, 3)
                                        .frame(width: 58)
                                        .background(Color.black.opacity(0.28), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                                .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                                        )
                                }

                                Toggle("Emulate Ollama (:11434)", isOn: $server.enableOllamaFallbackPort)
                                    .font(.system(size: 12))
                                    .toggleStyle(.checkbox)
                                    .onChange(of: server.enableOllamaFallbackPort) { _, _ in
                                        server.startServer()
                                    }

                                Toggle("Emulate LM Studio (:1234)", isOn: $server.enableLMStudioFallbackPort)
                                    .font(.system(size: 12))
                                    .toggleStyle(.checkbox)
                                    .onChange(of: server.enableLMStudioFallbackPort) { _, _ in
                                        server.startServer()
                                    }
                            }

                            Text("Automatically binds :11434 and :1234 whenever Ollama or LM Studio aren't running, so Codecaine File Manager and local scripts work seamlessly.")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                // 3. Endpoints & Quick Copy Bento Card
                AppleBentoCard {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Label("OpenAI, Ollama & LM Studio Compatible Endpoints", systemImage: "terminal.fill")
                                .font(.system(size: 14, weight: .semibold))
                            Spacer()
                            HStack(spacing: 8) {
                                Button {
                                    let port = server.activePorts.contains(11434) ? 11434 : Int(server.primaryPort)
                                    copyToClipboard("""
                                    curl -s http://127.0.0.1:\(port)/v1/audio/speech \\
                                      -H "Content-Type: application/json" \\
                                      -d '{"model":"phonon","input":"Hello from native Apple Silicon MLX!","voice":"Zoey"}' \\
                                      --output speech.wav
                                    """)
                                    showBanner("Copied Ollama /v1/audio/speech cURL command")
                                } label: {
                                    Label("Copy Ollama TTS cURL", systemImage: "doc.on.doc")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.white.opacity(0.07), in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Button {
                                    let port = server.activePorts.contains(1234) ? 1234 : Int(server.primaryPort)
                                    copyToClipboard("""
                                    curl -s http://127.0.0.1:\(port)/v1/audio/speech \\
                                      -H "Content-Type: application/json" \\
                                      -d '{"model":"mlx-community/fish-audio-s2-pro-8bit","input":"Testing Fish S2 Pro on localhost.","voice":"Default"}' \\
                                      --output fish_speech.wav
                                    """)
                                    showBanner("Copied LM Studio /v1/audio/speech cURL command")
                                } label: {
                                    Label("Copy Fish S2 cURL", systemImage: "doc.on.doc")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.white.opacity(0.07), in: Capsule())
                                }
                                .buttonStyle(.plain)

                                Button {
                                    let port = Int(server.primaryPort)
                                    let samplePath = Bundle.main.url(forResource: "Lily", withExtension: "wav")?.path ?? "/tmp/hello.wav"
                                    copyToClipboard("""
                                    curl -s http://127.0.0.1:\(port)/v1/audio/transcriptions \\
                                      -H "Content-Type: application/json" \\
                                      -d '{"file":"\(samplePath)","model":"mlx-community/Qwen3-ASR-0.6B-4bit","response_format":"srt"}'
                                    """)
                                    showBanner("Copied /v1/audio/transcriptions cURL command")
                                } label: {
                                    Label("Copy Whisper STT cURL", systemImage: "doc.on.doc")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.white.opacity(0.07), in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                        }

                        VStack(spacing: 8) {
                            EndpointRow(method: "POST", path: "/v1/audio/speech", desc: "Synthesizes WAV speech with 38 Phonon voices (Zoey, etc.) & 54 MLX TTS models")
                            EndpointRow(method: "POST", path: "/v1/audio/transcriptions", desc: "Transcribes local or base64 audio to JSON, TXT, SRT, or VTT")
                            EndpointRow(method: "GET", path: "/v1/models  ·  /api/tags  ·  /api/ps", desc: "Enumerates all 81 MLX Audio models, local cache state, and active Metal VRAM")
                            EndpointRow(method: "POST", path: "/v1/audio/clean  ·  /v1/audio/subtitles", desc: "Codecaine 6-filter speech cleaner & RMS-aligned SRT/VTT/ASS karaoke subtitle generator")
                        }
                    }
                }

                // 4. Live Request Telemetry Bento Card
                AppleBentoCard {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Live HTTP Telemetry (\(server.requestLogs.count))", systemImage: "waveform.path.ecg")
                                .font(.system(size: 14, weight: .semibold))
                            Spacer()
                            if !server.requestLogs.isEmpty {
                                Button {
                                    server.requestLogs.removeAll()
                                } label: {
                                    Text("Clear Log")
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundStyle(.secondary)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.white.opacity(0.06), in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                        }

                        if server.requestLogs.isEmpty {
                            HStack(spacing: 10) {
                                Image(systemName: "network")
                                    .foregroundStyle(.secondary)
                                Text("\(server.statusMessage). Click 'Run Loopback Test' above to verify.")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 8)
                        } else {
                            VStack(spacing: 6) {
                                ForEach(server.requestLogs.prefix(12)) { entry in
                                    let statusColor: Color = entry.status >= 400 ? .red : .green
                                    HStack(spacing: 10) {
                                        Text(":\(entry.port)")
                                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                            .foregroundStyle(.blue)
                                            .padding(.horizontal, 7)
                                            .padding(.vertical, 2)
                                            .background(Color.blue.opacity(0.14), in: Capsule())

                                        Text("HTTP \(entry.status)")
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .foregroundStyle(statusColor)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(statusColor.opacity(0.14), in: Capsule())

                                        Text(entry.method)
                                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                                            .foregroundStyle(entry.method == "POST" ? .orange : .green)
                                            .frame(width: 36, alignment: .leading)

                                        Text(entry.path)
                                            .font(.system(size: 12, weight: .medium, design: .monospaced))

                                        Spacer()

                                        Text(entry.summary)
                                            .font(.system(size: 11.5))
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)

                                        Text("\(entry.durationMs) ms")
                                            .font(.system(size: 11, design: .monospaced))
                                            .foregroundStyle(.tertiary)
                                            .frame(width: 52, alignment: .trailing)

                                        Text(entry.timestamp, style: .time)
                                            .font(.system(size: 10.5, design: .monospaced))
                                            .foregroundStyle(.tertiary)
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(Color.white.opacity(0.03), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                                }
                            }
                        }
                    }
                }
            }
            .padding(22)
        }
        .onAppear {
            server.startIfNeeded()
            hardware = HardwareGuardInspector.inspect()
            portString = String(server.primaryPort)
        }
        .onChange(of: server.requestLogs.count) { _, _ in
            hardware = HardwareGuardInspector.inspect()
        }
    }

    private func runSelfTest() async {
        let port = server.activePorts.first ?? server.primaryPort
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/tags") else { return }
        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                hardware = HardwareGuardInspector.inspect()
                showBanner("Loopback verified on 127.0.0.1:\(port)/api/tags (HTTP 200 OK)")
            }
        } catch {
            showBanner("Loopback test error: \(error.localizedDescription)")
        }
    }

    private func copyToClipboard(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    private func showBanner(_ msg: String) {
        copiedBanner = msg
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            if copiedBanner == msg {
                copiedBanner = nil
            }
        }
    }
}

struct HardwareMetricTile: View {
    let label: String
    let value: String
    let sub: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 9.5, weight: .bold, design: .rounded))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
                .lineLimit(1)
            Text(sub)
                .font(.system(size: 10.5))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
    }
}

struct PortStatusBadge: View {
    let port: Int
    let label: String
    let active: Bool

    var body: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(active ? Color.green : Color.gray)
                .frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 1) {
                Text(":\(String(port))")
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                Text(label)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(active ? Color.green.opacity(0.10) : Color.white.opacity(0.03))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(active ? Color.green.opacity(0.3) : Color.white.opacity(0.06), lineWidth: 1)
                )
        )
    }
}

struct EndpointRow: View {
    let method: String
    let path: String
    let desc: String

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Text(method)
                .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                .foregroundStyle(method == "POST" ? .orange : .green)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background((method == "POST" ? Color.orange : Color.green).opacity(0.14), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                .frame(width: 48, alignment: .center)

            Text(path)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)

            Text("— \(desc)")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.025), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

// MARK: - Codecaine Homebrew & Studio Packages Easy Installer View

struct PackagesInstallerView: View {
    @State private var manager = StudioPackageManager.shared
    @State private var selectedFilter: String = "All"
    @State private var searchQuery: String = ""
    @State private var copiedMessage: String? = nil

    private let filters = ["All", "Core Studio", "Installed", "Missing"]

    private var filteredTools: [StudioToolStatus] {
        manager.toolStatuses.filter { item in
            let matchesFilter: Bool
            switch selectedFilter {
            case "Core Studio":
                matchesFilter = item.spec.isCore
            case "Installed":
                matchesFilter = item.installed
            case "Missing":
                matchesFilter = !item.installed
            default:
                matchesFilter = true
            }
            guard matchesFilter else { return false }
            let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !q.isEmpty else { return true }
            return item.spec.name.lowercased().contains(q)
                || item.spec.what.lowercased().contains(q)
                || item.spec.brewFormula.lowercased().contains(q)
        }
    }

    private var installedCount: Int {
        manager.toolStatuses.filter(\.installed).count
    }

    private var missingCoreCount: Int {
        manager.toolStatuses.filter { $0.spec.isCore && !$0.installed }.count
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                DownloadedModelsCardView()
                homebrewHeroCard
                if let msg = copiedMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text(msg)
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.green.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                filterBarView
                packagesGridView
                consoleCardView
            }
            .padding(20)
        }
        .onAppear {
            manager.refreshStatus()
            DownloadedModelsManager.shared.refresh()
        }
        #if os(macOS)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            manager.refreshStatus()
            DownloadedModelsManager.shared.refresh()
        }
        #endif
    }

    @ViewBuilder
    private var homebrewHeroCard: some View {
        let brewReady = manager.brewInstalled
        let heroColors: [Color] = brewReady
            ? [Color.green.opacity(0.28), Color.teal.opacity(0.22)]
            : [Color.orange.opacity(0.32), Color.red.opacity(0.22)]
        let badgeTint: Color = brewReady ? .green : .orange

        AppleBentoCard(padding: 20) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 16) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(LinearGradient(colors: heroColors, startPoint: .topLeading, endPoint: .bottomTrailing))
                            .frame(width: 50, height: 50)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
                            )

                        Image(systemName: "shippingbox.fill")
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(badgeTint)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text("Homebrew & Codecaine Studio Packages")
                                .font(.system(size: 18, weight: .semibold))

                            Text(brewReady ? "✓ HOMEBREW INSTALLED" : "⚠️ HOMEBREW MISSING")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundStyle(badgeTint)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(badgeTint.opacity(0.16), in: Capsule())

                            if let ver = manager.brewVersion {
                                Text("v\(ver)")
                                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }

                        Text("One-click installer for Homebrew and optional studio tools (FFmpeg Read-Along MP4 export, Pandoc DOCX/EPUB extraction, Poppler PDF reader, Tesseract OCR, and Calibre).")
                            .font(.system(size: 12.5))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    HStack(spacing: 8) {
                        Button {
                            manager.installAllMissingCoreTools()
                        } label: {
                            HStack(spacing: 6) {
                                if manager.isInstalling {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Image(systemName: "sparkles")
                                }
                                Text(missingCoreCount > 0 ? "Install Core Suite (\(missingCoreCount))" : "Update Core Suite")
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(
                                LinearGradient(colors: [.blue, .indigo], startPoint: .top, endPoint: .bottom),
                                in: Capsule()
                            )
                            .foregroundStyle(.white)
                        }
                        .buttonStyle(.plain)
                        .disabled(manager.isInstalling)
                        .help("Install FFmpeg, Pandoc, Poppler & ImageMagick in one click")

                        Button {
                            manager.refreshStatus()
                            flashCopied("Re-scanned system PATH, /opt/homebrew/bin & /Applications")
                        } label: {
                            Label(manager.isScanning ? "Scanning…" : "Check Again", systemImage: "arrow.clockwise")
                                .font(.system(size: 12, weight: .medium))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color.white.opacity(0.08), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }

                homebrewBootstrapperBar(brewReady: brewReady, badgeTint: badgeTint)
            }
        }
    }

    @ViewBuilder
    private func homebrewBootstrapperBar(brewReady: Bool, badgeTint: Color) -> some View {
        let stepTitle = brewReady
            ? "STEP 1: HOMEBREW PACKAGE MANAGER READY (\(manager.brewPath ?? "/opt/homebrew/bin/brew"))"
            : "STEP 1: INSTALL HOMEBREW FIRST (REQUIRED ONCE ON MACOS)"
        let btnLabel = brewReady ? "Re-Run Homebrew Setup in Terminal" : "1-Click Install Homebrew in Terminal"
        let btnBg: Color = brewReady ? Color.white.opacity(0.09) : Color.orange.opacity(0.28)
        let btnFg: Color = brewReady ? Color.primary : Color.orange

        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(systemName: "terminal.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(badgeTint)
                    Text(stepTitle)
                        .font(.system(size: 10.5, weight: .bold, design: .rounded))
                        .foregroundStyle(badgeTint)
                }
                Text(StudioPackageManager.homebrewInstallCommand)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.primary.opacity(0.88))
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            Button {
                copyText(StudioPackageManager.homebrewInstallCommand)
                flashCopied("Copied official Homebrew install command to clipboard")
            } label: {
                Label("Copy Command", systemImage: "doc.on.doc")
                    .font(.system(size: 11.5, weight: .medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.white.opacity(0.07), in: Capsule())
            }
            .buttonStyle(.plain)

            Button {
                manager.installHomebrewInTerminal()
            } label: {
                Label(btnLabel, systemImage: "play.fill")
                    .font(.system(size: 11.5, weight: .semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(btnBg, in: Capsule())
                    .foregroundStyle(btnFg)
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(brewReady ? Color.black.opacity(0.24) : Color.orange.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(brewReady ? Color.white.opacity(0.08) : Color.orange.opacity(0.35), lineWidth: 0.8)
                )
        )
    }

    @ViewBuilder
    private var filterBarView: some View {
        HStack(spacing: 10) {
            ForEach(filters, id: \.self) { f in
                let active = selectedFilter == f
                Button {
                    selectedFilter = f
                } label: {
                    HStack(spacing: 5) {
                        Text(f)
                            .font(.system(size: 12, weight: active ? .semibold : .medium))
                        Text("\(filterCount(for: f))")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1.5)
                            .background(active ? Color.white.opacity(0.22) : Color.white.opacity(0.07), in: Capsule())
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(active ? Color.blue.opacity(0.32) : Color.white.opacity(0.045))
                            .overlay(
                                Capsule()
                                    .strokeBorder(active ? Color.blue.opacity(0.8) : Color.white.opacity(0.08), lineWidth: 0.8)
                            )
                    )
                }
                .buttonStyle(.plain)
            }

            Spacer()

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                TextField("Filter packages (ffmpeg, pandoc, ocr…)", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .frame(width: 220)
                if !searchQuery.isEmpty {
                    Button {
                        searchQuery = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color.white.opacity(0.05))
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                    )
            )
        }
    }

    private func filterCount(for filterName: String) -> Int {
        switch filterName {
        case "Core Studio": return manager.toolStatuses.filter(\.spec.isCore).count
        case "Installed": return installedCount
        case "Missing": return manager.toolStatuses.count - installedCount
        default: return manager.toolStatuses.count
        }
    }

    @ViewBuilder
    private var packagesGridView: some View {
        if filteredTools.isEmpty {
            AppleBentoCard(padding: 28) {
                VStack(spacing: 10) {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 30))
                        .foregroundStyle(.tertiary)
                    Text("No packages match the selected filter")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Button("Show All Packages") {
                        selectedFilter = "All"
                        searchQuery = ""
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .frame(maxWidth: .infinity)
            }
        } else {
            LazyVGrid(
                columns: [
                    GridItem(.flexible(), spacing: 14),
                    GridItem(.flexible(), spacing: 14)
                ],
                spacing: 14
            ) {
                ForEach(filteredTools) { item in
                    StudioPackageTileView(
                        item: item,
                        isInstalling: manager.isInstalling,
                        onCopy: {
                            copyText(item.installCommand)
                            flashCopied("Copied '\(item.installCommand)'")
                        },
                        onInstall: {
                            manager.installTool(item.spec)
                        },
                        onTerminal: {
                            manager.runCommandInTerminal(item.installCommand, title: "Install \(item.spec.name)")
                        }
                    )
                }
            }
        }
    }

    @ViewBuilder
    private var consoleCardView: some View {
        if !manager.consoleLog.isEmpty || manager.isInstalling {
            AppleBentoCard(padding: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label(
                            manager.isInstalling
                                ? "Installing \(manager.activeInstallTarget ?? "Package")…"
                                : "Homebrew Installer Console Log",
                            systemImage: "terminal.fill"
                        )
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.green)

                        if manager.isInstalling {
                            ProgressView()
                                .controlSize(.small)
                        }

                        Spacer()

                        Button("Clear Console") {
                            manager.consoleLog = ""
                        }
                        .font(.system(size: 11, weight: .medium))
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }

                    ScrollView {
                        Text(manager.consoleLog)
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .foregroundStyle(.green.opacity(0.9))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 150)
                    .padding(10)
                    .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
            }
        }
    }

    private func copyText(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    private func flashCopied(_ msg: String) {
        copiedMessage = msg
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            if copiedMessage == msg {
                copiedMessage = nil
            }
        }
    }
}

private extension StudioToolStatus {
    var whatSummary: String {
        spec.what
    }
}

struct StudioPackageTileView: View {
    let item: StudioToolStatus
    let isInstalling: Bool
    let onCopy: () -> Void
    let onInstall: () -> Void
    let onTerminal: () -> Void

    private var statusColor: Color {
        item.installed ? .green : .orange
    }

    var body: some View {
        AppleBentoCard(padding: 14) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 10) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(item.installed ? Color.green.opacity(0.16) : Color.blue.opacity(0.14))
                            .frame(width: 34, height: 34)
                        Image(systemName: item.spec.icon)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(item.installed ? .green : .blue)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(item.spec.name)
                                .font(.system(size: 13.5, weight: .semibold))
                                .lineLimit(1)

                            if item.spec.isCore {
                                Text("CORE")
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                                    .foregroundStyle(.blue)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1.5)
                                    .background(Color.blue.opacity(0.16), in: Capsule())
                            }
                        }

                        Text(item.whatSummary)
                            .font(.system(size: 11.5))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer(minLength: 4)

                    HStack(spacing: 4) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 6, height: 6)
                        Text(item.installed ? (item.version ?? "Installed") : "Not Installed")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .foregroundStyle(statusColor)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3.5)
                    .background(statusColor.opacity(0.14), in: Capsule())
                }

                HStack(spacing: 8) {
                    Text(item.installCommand)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.primary.opacity(0.85))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.28), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .lineLimit(1)

                    Button(action: onCopy) {
                        Image(systemName: "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                            .padding(6)
                            .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .help("Copy brew install command")

                    Button(action: onInstall) {
                        HStack(spacing: 4) {
                            Image(systemName: item.installed ? "arrow.triangle.2.circlepath" : "arrow.down.circle.fill")
                                .font(.system(size: 11))
                            Text(item.installed ? "Reinstall" : "Install")
                                .font(.system(size: 11.5, weight: .semibold))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5.5)
                        .background(
                            item.installed ? Color.white.opacity(0.08) : Color.blue.opacity(0.28),
                            in: Capsule()
                        )
                        .foregroundStyle(item.installed ? Color.primary : Color.white)
                    }
                    .buttonStyle(.plain)
                    .disabled(isInstalling)
                    .help("Install \(item.spec.name) directly via Homebrew")

                    Button(action: onTerminal) {
                        Image(systemName: "terminal")
                            .font(.system(size: 11, weight: .semibold))
                            .padding(6)
                            .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .help("Run '\(item.installCommand)' interactively in macOS Terminal.app")
                }
            }
        }
    }
}

// MARK: - Downloaded AI Models Manager Card & Sheet

struct DownloadedModelsSheetView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                DownloadedModelsCardView(isSheet: true)
                    .padding(18)
            }
            .navigationTitle("Downloaded AI Models")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 720, minHeight: 580)
        #endif
    }
}

struct DownloadedModelsCardView: View {
    var isSheet: Bool = false
    @State private var manager = DownloadedModelsManager.shared
    @State private var selectedCategory: DownloadedModelCategory = .all
    @State private var searchQuery: String = ""
    @State private var pendingDeleteItem: DownloadedModelItem? = nil

    private var filteredItems: [DownloadedModelItem] {
        manager.items.filter { item in
            let matchesCat: Bool
            switch selectedCategory {
            case .all:
                matchesCat = true
            case .incomplete:
                matchesCat = !item.hasWeights
            default:
                matchesCat = item.hasWeights && item.category == selectedCategory
            }
            guard matchesCat else { return false }
            let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !q.isEmpty else { return true }
            return item.displayName.lowercased().contains(q)
                || item.repoId.lowercased().contains(q)
                || item.category.rawValue.lowercased().contains(q)
        }
    }

    private func categoryCount(_ cat: DownloadedModelCategory) -> Int {
        switch cat {
        case .all:
            return manager.items.count
        case .incomplete:
            return manager.incompleteModelsCount
        default:
            return manager.items.filter { $0.hasWeights && $0.category == cat }.count
        }
    }

    var body: some View {
        AppleBentoCard(padding: 20) {
            VStack(alignment: .leading, spacing: 14) {
                // Header Row
                HStack(alignment: .center, spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [Color.blue.opacity(0.32), Color.purple.opacity(0.24)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 48, height: 48)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
                            )

                        Image(systemName: "internaldrive.fill")
                            .font(.system(size: 21, weight: .semibold))
                            .foregroundStyle(.blue)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text("Downloaded MLX AI Models")
                                .font(.system(size: 17.5, weight: .semibold))

                            Text("\(manager.completeModelsCount) MODELS · \(manager.totalSizeString) ON DISK")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundStyle(.green)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.green.opacity(0.16), in: Capsule())

                            if manager.incompleteModelsCount > 0 {
                                Text("\(manager.incompleteModelsCount) PARTIAL")
                                    .font(.system(size: 10, weight: .bold, design: .rounded))
                                    .foregroundStyle(.orange)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Color.orange.opacity(0.16), in: Capsule())
                            }
                        }

                        Text("Inspect and delete downloaded TTS, STT, Audio Lab, and Codec models stored in ~/.cache/huggingface/hub to reclaim Mac SSD storage.")
                            .font(.system(size: 12.5))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    HStack(spacing: 8) {
                        if manager.incompleteModelsCount > 0 {
                            Button {
                                manager.cleanIncompleteDownloads()
                            } label: {
                                Label("Clean Incomplete (\(manager.incompleteModelsCount))", systemImage: "sparkles")
                                    .font(.system(size: 11.5, weight: .semibold))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(Color.orange.opacity(0.22), in: Capsule())
                                    .foregroundStyle(.orange)
                            }
                            .buttonStyle(.plain)
                            .help("Remove incomplete or failed partial model downloads")
                        }

                        #if os(macOS)
                        Button {
                            ModelCacheHelper.revealCacheInFinder()
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                                .font(.system(size: 11.5, weight: .medium))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color.white.opacity(0.08), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        #endif

                        Button {
                            manager.refresh()
                        } label: {
                            Label("Rescan", systemImage: "arrow.clockwise")
                                .font(.system(size: 11.5, weight: .medium))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color.white.opacity(0.08), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }

                if let banner = manager.statusBanner {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text(banner)
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.green)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.green.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }

                // Category Filter Bar + Search
                HStack(spacing: 8) {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(DownloadedModelCategory.allCases) { cat in
                                let count = categoryCount(cat)
                                if cat == .all || count > 0 {
                                    let active = selectedCategory == cat
                                    Button {
                                        selectedCategory = cat
                                    } label: {
                                        HStack(spacing: 5) {
                                            Image(systemName: cat.icon)
                                                .font(.system(size: 10.5, weight: .semibold))
                                                .foregroundStyle(active ? .white : cat.tint)
                                            Text(cat.rawValue)
                                                .font(.system(size: 11.5, weight: active ? .semibold : .medium))
                                            Text("\(count)")
                                                .font(.system(size: 9.5, weight: .bold, design: .rounded))
                                                .padding(.horizontal, 5.5)
                                                .padding(.vertical, 1)
                                                .background(active ? Color.white.opacity(0.22) : Color.white.opacity(0.07), in: Capsule())
                                        }
                                        .padding(.horizontal, 11)
                                        .padding(.vertical, 5.5)
                                        .background(
                                            Capsule()
                                                .fill(active ? Color.blue.opacity(0.32) : Color.white.opacity(0.045))
                                                .overlay(
                                                    Capsule()
                                                        .strokeBorder(active ? Color.blue.opacity(0.8) : Color.white.opacity(0.08), lineWidth: 0.8)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }

                    Spacer()

                    HStack(spacing: 6) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        TextField("Filter downloaded models…", text: $searchQuery)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12))
                            .frame(width: 200)
                        if !searchQuery.isEmpty {
                            Button {
                                searchQuery = ""
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5.5)
                    .background(
                        Capsule()
                            .fill(Color.white.opacity(0.05))
                            .overlay(
                                Capsule()
                                    .strokeBorder(Color.white.opacity(0.10), lineWidth: 0.8)
                            )
                    )
                }

                // Models List
                if filteredItems.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "internaldrive")
                            .font(.system(size: 26))
                            .foregroundStyle(.tertiary)
                        Text(manager.items.isEmpty ? "No downloaded models found in cache" : "No downloaded models match the current filter")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 20)
                } else {
                    VStack(spacing: 7) {
                        ForEach(filteredItems) { item in
                            HStack(spacing: 10) {
                                Image(systemName: item.category.icon)
                                    .font(.system(size: 12.5, weight: .semibold))
                                    .foregroundStyle(item.category.tint)
                                    .frame(width: 28, height: 28)
                                    .background(item.category.tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 8, style: .continuous))

                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 6) {
                                        Text(item.displayName)
                                            .font(.system(size: 13, weight: .semibold))
                                            .lineLimit(1)

                                        Text(item.category.rawValue.uppercased())
                                            .font(.system(size: 9, weight: .bold, design: .rounded))
                                            .foregroundStyle(item.category.tint)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 1.5)
                                            .background(item.category.tint.opacity(0.14), in: Capsule())
                                    }

                                    Text(item.repoId)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }

                                Spacer()

                                Text(item.sizeString)
                                    .font(.system(size: 11.5, weight: .bold, design: .rounded))
                                    .foregroundStyle(item.hasWeights ? .green : .orange)
                                    .padding(.horizontal, 9)
                                    .padding(.vertical, 4)
                                    .background((item.hasWeights ? Color.green : Color.orange).opacity(0.14), in: Capsule())

                                #if os(macOS)
                                Button {
                                    ModelCacheHelper.revealModelInFinder(item.repoId)
                                } label: {
                                    Image(systemName: "folder")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .padding(6.5)
                                        .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                                }
                                .buttonStyle(.plain)
                                .help("Reveal \(item.repoId) in Finder")
                                #endif

                                Button {
                                    #if os(macOS)
                                    if NSEvent.modifierFlags.contains(.option) {
                                        manager.deleteModel(item.repoId, displayName: item.displayName)
                                        return
                                    }
                                    #endif
                                    pendingDeleteItem = item
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "trash.fill")
                                            .font(.system(size: 10.5))
                                        Text("Delete")
                                            .font(.system(size: 11.5, weight: .semibold))
                                    }
                                    .foregroundStyle(.red)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5.5)
                                    .background(Color.red.opacity(0.16), in: Capsule())
                                }
                                .buttonStyle(.plain)
                                .help("Delete \(item.displayName) (\(item.sizeString)) from disk (Option-click to delete immediately)")
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 11, style: .continuous)
                                    .fill(Color.black.opacity(0.22))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 11, style: .continuous)
                                            .strokeBorder(Color.white.opacity(0.07), lineWidth: 0.8)
                                    )
                            )
                        }
                    }
                }
            }
        }
        .alert(
            "Delete Downloaded Model?",
            isPresented: Binding(
                get: { pendingDeleteItem != nil },
                set: { if !$0 { pendingDeleteItem = nil } }
            ),
            presenting: pendingDeleteItem
        ) { target in
            Button("Delete Model (\(target.sizeString))", role: .destructive) {
                manager.deleteModel(target.repoId, displayName: target.displayName)
                pendingDeleteItem = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteItem = nil
            }
        } message: { target in
            Text("Remove '\(target.displayName)' (\(target.repoId)) from your local HuggingFace cache and free \(target.sizeString) of disk space? You can re-download it at any time.")
        }
    }
}
