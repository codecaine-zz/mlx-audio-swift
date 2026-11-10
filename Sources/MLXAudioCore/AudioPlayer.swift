import Foundation
import AVFoundation
import Combine

@MainActor
public class AudioPlayer: NSObject, ObservableObject {
    // Published properties for UI binding
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var isSpeaking: Bool = false
    @Published public private(set) var currentTime: TimeInterval = 0
    @Published public private(set) var duration: TimeInterval = 0
    @Published public private(set) var currentAudioURL: URL?
    @Published public var playbackRate: Float = 1.0 {
        didSet {
            if isStreaming, isNodeActivelyPlaying, let startWall = currentRunStartWallTime {
                let elapsed = max(0, (CFAbsoluteTimeGetCurrent() - startWall) * Double(oldValue))
                currentRunBaseAudioTime = min(duration, currentRunBaseAudioTime + elapsed)
                currentRunStartWallTime = CFAbsoluteTimeGetCurrent()
            }
            player?.enableRate = true
            player?.rate = playbackRate
            timePitchNode?.rate = playbackRate
        }
    }
    /// Hardware-accelerated pitch shift in semitones (-12 ... +12) via Apple's AVAudioUnitTimePitch (zero comb-filtering).
    @Published public var pitchSemitones: Float = 0.0 {
        didSet {
            timePitchNode?.pitch = pitchSemitones * 100.0
        }
    }

    /// Optional callback for speaking-state transitions during playback/streaming.
    public var onSpeakingStateChanged: ((Bool) -> Void)?
    /// Optional callback fired when a streaming source has finished and all queued audio is consumed.
    public var onDidFinishStreaming: (() -> Void)?

    private var player: AVAudioPlayer?
    private var timer: Timer?

    // Streaming playback components
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var timePitchNode: AVAudioUnitTimePitch?
    private var streamingFormat: AVAudioFormat?
    private var isStreaming: Bool = false
    private var scheduledFrames: Int = 0
    private var queuedBuffers: Int = 0
    private var consumedFramesBeforeCurrentRun: Int = 0
    private var completedFramesInCurrentRun: Int = 0
    private var currentRunBaseAudioTime: TimeInterval = 0
    private var currentRunStartWallTime: CFAbsoluteTime?
    private var isNodeActivelyPlaying: Bool = false
    private var streamFinished: Bool = false
    private var streamingTask: Task<Void, Never>?
    private var configurationChangeObserver: Task<Void, Never>?

    public override init() {
        super.init()
    }

    @MainActor deinit {
        stop()
    }

    // MARK: - Playback Control

    public func loadAudio(from url: URL) {
        // Stop any existing playback
        stop()

        // Setup audio session for iOS
        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        if let directPlayer = try? AVAudioPlayer(contentsOf: url), directPlayer.duration > 0 {
            player = directPlayer
            player?.enableRate = true
            player?.rate = playbackRate
            player?.delegate = self
            player?.prepareToPlay()
            currentAudioURL = url
            duration = directPlayer.duration
            currentTime = 0
            return
        }

        // Fallback: decode any audio/video container (.ogg, .opus, .mkv, .webm, .mp4, .mov, .wma, .m4b, etc.) to a temporary WAV
        do {
            let (sr, audioArray) = try loadAudioArray(from: url)
            let samples = audioArray.asArray(Float.self)
            let tempWAV = FileManager.default.temporaryDirectory
                .appendingPathComponent("player_decoded_\(UUID().uuidString).wav")
            try AudioUtils.writeWavFile(samples: samples, sampleRate: sr, fileURL: tempWAV)
            let fallbackPlayer = try AVAudioPlayer(contentsOf: tempWAV)
            player = fallbackPlayer
            player?.enableRate = true
            player?.rate = playbackRate
            player?.delegate = self
            player?.prepareToPlay()
            currentAudioURL = tempWAV
            duration = fallbackPlayer.duration
            currentTime = 0
        } catch {
            print("Failed to load audio: \(error.localizedDescription)")
            currentAudioURL = nil
            duration = 0
            currentTime = 0
        }
    }

    public func play() {
        guard let player = player else { return }

        player.enableRate = true
        player.rate = playbackRate
        player.play()
        isPlaying = true
        setSpeaking(true)
        startTimer()
    }

    public func pause() {
        if isStreaming {
            if isNodeActivelyPlaying, let startWall = currentRunStartWallTime {
                let elapsed = max(0, (CFAbsoluteTimeGetCurrent() - startWall) * Double(playbackRate))
                currentRunBaseAudioTime = min(duration, currentRunBaseAudioTime + elapsed)
            }
            currentRunStartWallTime = nil
            playerNode?.pause()
            isNodeActivelyPlaying = false
        } else {
            player?.pause()
        }
        isPlaying = false
        setSpeaking(false)
        stopTimer()
    }

    public func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            if isStreaming {
                if queuedBuffers > 0 {
                    currentRunStartWallTime = CFAbsoluteTimeGetCurrent()
                    playerNode?.play()
                    isNodeActivelyPlaying = true
                    isPlaying = true
                    setSpeaking(true)
                    startStreamingTimer()
                }
            } else {
                play()
            }
        }
    }

    public func stop() {
        if isStreaming {
            stopStreaming()
        } else {
            player?.stop()
            isPlaying = false
            setSpeaking(false)
            stopTimer()
            currentTime = 0
        }
    }

    public func unloadAudio() {
        stop()
        player = nil
        currentAudioURL = nil
        duration = 0
        currentTime = 0
    }

    public func seek(to time: TimeInterval) {
        if isStreaming, let url = currentAudioURL, streamFinished {
            loadAudio(from: url)
            player?.currentTime = max(0, min(time, duration))
            currentTime = player?.currentTime ?? 0
            play()
            return
        }
        guard let player = player else { return }
        player.currentTime = max(0, min(time, duration))
        currentTime = player.currentTime
    }

    // MARK: - Streaming Playback

    /// Start streaming playback - call this before scheduling chunks
    public func startStreaming(sampleRate: Double) {
        stop()

        // Setup audio session for iOS
        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        streamingFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
        scheduledFrames = 0
        queuedBuffers = 0
        streamFinished = false
        duration = 0

        do {
            try startStreamingEngine(connectionFormat: streamingFormat)
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    /// Play a stream of PCM buffers, converting source format to the active output format.
    public func play(stream: AsyncThrowingStream<AVAudioPCMBuffer, Error>) {
        stop()

        #if os(iOS)
        AudioSessionManager.shared.setupAudioSession()
        #endif

        do {
            try startStreamingEngine(connectionFormat: nil)
        } catch {
            print("Failed to start audio engine: \(error)")
            return
        }

        guard let engine = audioEngine else { return }
        let converter = PCMStreamConverter(outputFormat: engine.outputNode.inputFormat(forBus: 0))

        streamingTask = Task { @MainActor [weak self] in
            guard let self else { return }

            do {
                for try await inputBuffer in stream {
                    let convertedBuffers = try converter.push(inputBuffer)
                    for buffer in convertedBuffers {
                        scheduleStreamingBuffer(buffer)
                    }
                }

                let trailingBuffers = try converter.finish()
                for trailingBuffer in trailingBuffers {
                    scheduleStreamingBuffer(trailingBuffer)
                }

                streamFinished = true
                finishStreamIfDrained()
            } catch is CancellationError {
                // no-op
            } catch {
                print("Streaming playback failed: \(error)")
                stopStreaming()
            }
        }
    }

    /// Mark an open chunk stream as complete so completion can fire once queued audio drains.
    public func finishStreamingInput() {
        streamFinished = true
        finishStreamIfDrained()
    }

    /// Schedule audio samples for streaming playback with studio-grade raised-cosine edge de-clicking
    public func scheduleAudioChunk(_ samples: [Float], withCrossfade: Bool = true) {
        let count = samples.count
        guard isStreaming,
              count > 0,
              let format = streamingFormat,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else {
            return
        }

        buffer.frameLength = AVAudioFrameCount(count)

        if let channelData = buffer.floatChannelData {
            let dst = channelData[0]
            samples.withUnsafeBufferPointer { src in
                guard let base = src.baseAddress else { return }
                dst.update(from: base, count: count)
            }
            if withCrossfade && count > 64 {
                // 4ms raised-cosine (Hann) micro-taper at chunk boundaries prevents DC/phase clicks without ducking syllables
                let taperSamples = min(Int(format.sampleRate * 0.004), count / 4)
                if taperSamples > 1 {
                    let invTaper = Float.pi / Float(taperSamples)
                    for i in 0..<taperSamples {
                        let gain = 0.5 * (1.0 - cos(Float(i) * invTaper))
                        dst[i] *= gain
                        dst[count - 1 - i] *= gain
                    }
                }
            }
        }

        scheduleStreamingBuffer(buffer)
    }

    /// Stop streaming and clean up
    public func stopStreaming() {
        streamingTask?.cancel()
        streamingTask = nil

        configurationChangeObserver?.cancel()
        configurationChangeObserver = nil

        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        timePitchNode = nil
        streamingFormat = nil

        isStreaming = false
        isPlaying = false
        isNodeActivelyPlaying = false
        setSpeaking(false)

        scheduledFrames = 0
        queuedBuffers = 0
        consumedFramesBeforeCurrentRun = 0
        completedFramesInCurrentRun = 0
        currentRunBaseAudioTime = 0
        currentRunStartWallTime = nil
        streamFinished = false

        currentTime = 0
        duration = 0
        stopTimer()
    }

    /// Check if currently in streaming mode
    public var isStreamingMode: Bool {
        return isStreaming
    }

    // MARK: - Timer Management

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let player = self?.player else { return }
                self?.currentTime = player.currentTime
            }
        }
        timer?.tolerance = 0.004
    }

    private func startStreamingTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self,
                      self.isNodeActivelyPlaying,
                      let startWall = self.currentRunStartWallTime else { return }
                let wallElapsed = max(0, (CFAbsoluteTimeGetCurrent() - startWall) * Double(self.playbackRate))
                let totalSec = self.currentRunBaseAudioTime + wallElapsed
                self.currentTime = self.duration > 0 ? min(self.duration, totalSec) : totalSec
            }
        }
        timer?.tolerance = 0.004
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func startStreamingEngine(connectionFormat: AVAudioFormat?) throws {
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        let pitchNode = AVAudioUnitTimePitch()
        pitchNode.rate = playbackRate
        pitchNode.pitch = pitchSemitones * 100.0
        timePitchNode = pitchNode

        guard let engine = audioEngine, let node = playerNode else { return }

        engine.attach(node)
        engine.attach(pitchNode)
        engine.connect(node, to: pitchNode, format: connectionFormat)
        engine.connect(pitchNode, to: engine.mainMixerNode, format: connectionFormat)
        engine.prepare()
        try engine.start()
        // Do NOT call node.play() until the first buffer is queued in scheduleStreamingBuffer(_:)

        isStreaming = true
        isPlaying = false
        isNodeActivelyPlaying = false
        setSpeaking(false)
        scheduledFrames = 0
        queuedBuffers = 0
        consumedFramesBeforeCurrentRun = 0
        completedFramesInCurrentRun = 0
        currentRunBaseAudioTime = 0
        currentRunStartWallTime = nil
        streamFinished = false
        currentTime = 0
        duration = 0

        observeEngineConfigurationChanges()
    }

    private func observeEngineConfigurationChanges() {
        guard configurationChangeObserver == nil else { return }
        configurationChangeObserver = Task { @MainActor [weak self] in
            guard let self else { return }

            for await _ in NotificationCenter.default.notifications(named: .AVAudioEngineConfigurationChange) {
                guard isStreaming, let engine = audioEngine else { continue }
                if !engine.isRunning {
                    do {
                        try engine.start()
                        if isNodeActivelyPlaying && queuedBuffers > 0 {
                            playerNode?.play()
                        }
                    } catch {
                        print("Failed to restart audio engine after configuration change: \(error)")
                    }
                }
            }
        }
    }

    private func scheduleStreamingBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let node = playerNode else { return }

        let frameCount = Int(buffer.frameLength)
        let sr = buffer.format.sampleRate
        queuedBuffers += 1
        scheduledFrames += frameCount
        duration = sr > 0 ? Double(scheduledFrames) / sr : duration

        let completion: @Sendable (AVAudioPlayerNodeCompletionCallbackType) -> Void = { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleScheduledBufferConsumed(frameCount: frameCount, sampleRate: sr)
            }
        }
        // Use .dataPlayedBack (not .dataConsumed!) so the callback only fires after the audio has actually played through the hardware output
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack, completionHandler: completion)

        if !isNodeActivelyPlaying {
            currentRunBaseAudioTime = sr > 0 ? Double(consumedFramesBeforeCurrentRun) / sr : currentTime
            currentRunStartWallTime = CFAbsoluteTimeGetCurrent()
            node.play()
            isNodeActivelyPlaying = true
            isPlaying = true
            setSpeaking(true)
            startStreamingTimer()
        } else if !isSpeaking {
            setSpeaking(true)
        }
    }

    private func handleScheduledBufferConsumed(frameCount: Int, sampleRate: Double) {
        completedFramesInCurrentRun += frameCount
        queuedBuffers = max(queuedBuffers - 1, 0)
        if queuedBuffers == 0 {
            if streamFinished {
                finishStreamIfDrained()
            } else {
                // Pause (do not stop!) the hardware node if waiting for the next neural chunk
                playerNode?.pause()
                consumedFramesBeforeCurrentRun += completedFramesInCurrentRun
                completedFramesInCurrentRun = 0
                isNodeActivelyPlaying = false
                currentRunStartWallTime = nil
                if sampleRate > 0 {
                    let exactCompletedSec = Double(consumedFramesBeforeCurrentRun) / sampleRate
                    currentRunBaseAudioTime = exactCompletedSec
                    currentTime = exactCompletedSec
                }
                setSpeaking(false)
                stopTimer()
            }
        }
    }

    private func finishStreamIfDrained() {
        guard streamFinished, queuedBuffers == 0 else { return }
        isNodeActivelyPlaying = false
        isPlaying = false
        setSpeaking(false)
        stopTimer()
        onDidFinishStreaming?()
    }

    private func setSpeaking(_ speaking: Bool) {
        guard isSpeaking != speaking else { return }
        isSpeaking = speaking
        onSpeakingStateChanged?(speaking)
    }
}

// MARK: - AVAudioPlayerDelegate

@available(*, deprecated, renamed: "AudioPlayer", message: "Use AudioPlayer instead.")
public typealias AudioPlayerManager = AudioPlayer

extension AudioPlayer: @MainActor AVAudioPlayerDelegate {
    public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        setSpeaking(false)
        stopTimer()
        currentTime = 0
    }

    public func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("Audio decode error: \(error?.localizedDescription ?? "unknown")")
        isPlaying = false
        setSpeaking(false)
        stopTimer()
    }
}
