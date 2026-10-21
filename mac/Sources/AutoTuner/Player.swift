import AVFoundation
import SwiftUI

/// Listening to a result in the app: the original vocal or Teto singing the result, with or without the backing,
/// all on the same timeline. Switching what is heard keeps the position, so the two can be compared line by line.
@MainActor
final class Player: ObservableObject {
    enum Voice: String, CaseIterable, Identifiable {
        case original, teto
        var id: String { rawValue }
        var title: String { self == .original ? "원래 보컬" : "테토" }
    }

    @Published private(set) var playing = false
    @Published private(set) var time: Double = 0
    @Published var voice: Voice = .original { didSet { if voice != oldValue { restart() } } }
    @Published var withBacking = false { didSet { if withBacking != oldValue { restart() } } }
    /// Which files this result has.
    @Published private(set) var hasTeto = false
    @Published private(set) var hasBacking = false
    @Published private(set) var hasOriginal = false

    private var original: AVAudioPlayer?
    private var teto: AVAudioPlayer?
    private var backing: AVAudioPlayer?
    private var paths: [String?] = []
    /// Play only up to here (a section), then stop.
    private var stopAt: Double?
    private var ticker: Timer?
    private var started = Date()

    var duration: Double { [original, teto].compactMap { $0?.duration }.max() ?? 0 }

    /// Loads a result's files (again when they changed, e.g. after Teto sang the edits).
    func load(_ item: ResultItem?, force: Bool = false) {
        let next = [item?.guide, item?.teto, item?.backing]
        guard force || next != paths else { return }
        stop()
        paths = next
        original = Self.open(item?.guide)
        teto = Self.open(item?.teto)
        backing = Self.open(item?.backing)
        hasOriginal = original != nil
        hasTeto = teto != nil
        hasBacking = backing != nil
        if !hasTeto, voice == .teto { voice = .original }
        if !hasOriginal, hasTeto { voice = .teto }
        if !hasBacking { withBacking = false }
        time = 0
    }

    private static func open(_ path: String?) -> AVAudioPlayer? {
        guard let path, FileManager.default.fileExists(atPath: path) else { return nil }
        let p = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
        p?.prepareToPlay()
        return p
    }

    private var active: [AVAudioPlayer] {
        var list: [AVAudioPlayer] = []
        if let v = voice == .teto ? teto : original { list.append(v) }
        if withBacking, let backing { list.append(backing) }
        return list
    }

    func toggle() { playing ? pause() : play(from: time) }

    /// Plays from `start` (seconds), up to `end` when given.
    func play(from start: Double, to end: Double? = nil) {
        let list = active
        guard !list.isEmpty else { return }
        for p in [original, teto, backing].compactMap({ $0 }) { p.stop() }
        let t = max(0, min(start, max(0, duration - 0.05)))
        // Started together a moment from now so the vocal and the backing stay in step.
        let at = list[0].deviceCurrentTime + 0.05
        for p in list {
            p.currentTime = t
            p.play(atTime: at)
        }
        stopAt = end
        time = t
        started = Date()
        playing = true
        ticker?.invalidate()
        ticker = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func tick() {
        guard let p = active.first else { stop(); return }
        if p.isPlaying { time = p.currentTime }
        if let end = stopAt, time >= end {
            pause()
            time = end
        } else if !p.isPlaying, Date().timeIntervalSince(started) > 0.3 {
            pause()  // reached the end
        }
    }

    func pause() {
        for p in [original, teto, backing].compactMap({ $0 }) { p.pause() }
        ticker?.invalidate()
        ticker = nil
        playing = false
        stopAt = nil
    }

    func stop() {
        pause()
        for p in [original, teto, backing].compactMap({ $0 }) { p.stop() }
    }

    func seek(_ t: Double) {
        if playing { play(from: t, to: stopAt) } else { time = max(0, t) }
    }

    private func restart() {
        if playing { play(from: time, to: stopAt) }
    }
}
