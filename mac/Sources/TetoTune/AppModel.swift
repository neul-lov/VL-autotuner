import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Where the vocal comes from: a file the user picked, or a song split in StemMixer.
enum VocalSource: Equatable {
    case file(URL)
    case stem(StemSong)

    var title: String {
        switch self {
        case .file(let url): url.deletingPathExtension().lastPathComponent
        case .stem(let song): song.title
        }
    }

    /// Where its lyrics are saved.
    var key: String {
        switch self {
        case .file(let url): "file:" + url.path
        case .stem(let song): "stemmixer:" + song.id
        }
    }
}

struct Job {
    var label: String
    var step: String = "보내는 중"
    var progress: Double = 0
}

enum BreathMode: String, CaseIterable, Identifiable {
    case auto, voicebank, original, off
    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: "자동"
        case .voicebank: "테토 숨소리"
        case .original: "원래 숨소리"
        case .off: "안 넣기"
        }
    }
    var detail: String {
        switch self {
        case .auto: "음원에 숨소리(息 · 吸)가 있으면 테토 숨으로, 없으면 원래 보컬의 숨을 잘라 넣어요"
        case .voicebank: "숨 쉰 자리마다 숨 음표를 넣고 크기를 원래 숨에 맞춰요"
        case .original: "원래 보컬에서 숨소리만 잘라 별도 트랙으로 넣어요"
        case .off: "숨소리를 넣지 않아요"
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    enum ServerState: Equatable {
        case starting, ready, missing, failed(String)
        /// First run of the app's own engine: installing its Python packages.
        case installing(step: String, detail: String)
        /// No Python 3 on this Mac (Apple's Command Line Tools bring one).
        case needsPython
    }

    let backend = Backend()

    @Published var server: ServerState = .starting
    /// The log that explains the current failure (install log or engine log).
    @Published var failedLog: URL?
    @Published var status: ServerStatus?
    @Published var stemSongs: [StemSong] = []
    @Published var results: [ResultItem] = []

    @Published var source: VocalSource? {
        didSet {
            guard source != oldValue else { return }
            let saved = AppModel.savedLyrics()[source?.key ?? ""]
            if oldValue == nil, saved == nil, !lyrics.isEmpty {
                saveLyrics()  // typed before a vocal was chosen: they belong to this one
            } else {
                lyrics = saved ?? ""
            }
        }
    }
    /// Lyrics are kept per vocal, so another song's lyrics never end up on this one.
    @Published var lyrics = "" {
        didSet { saveLyrics() }
    }
    @Published var notesFile: URL?
    @Published var backingFile: URL?
    @Published var options: TuneOptions = AppModel.loadOptions() {
        didSet {
            saveOptions()
            if options.singer != oldValue.singer, server == .ready {
                Task { if let s = try? await backend.status(singer: options.singer) { status = s } }
            }
        }
    }

    @Published var job: Job?
    @Published var jobError: String?
    @Published var toast: String?
    @Published var toastIsSuccess = true

    /// The result shown on the right (the newest one after tuning).
    @Published var selected: ResultItem?
    @Published var preview: Preview?
    @Published var loadingPreview = false

    private var toastTask: Task<Void, Never>?

    // MARK: Server

    private var observing = false
    private var bootTask: Task<Void, Never>?
    private var stopping = false

    func start() {
        if !observing {
            observing = true
            // Pick up new StemMixer songs or voicebanks when coming back to the app.
            NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                                                   queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, self.server == .ready else { return }
                    Task { await self.refresh() }
                }
            }
        }
        server = .starting
        failedLog = nil
        bootTask?.cancel()
        bootTask = Task { await boot() }
    }

    private func fail(_ message: String, log: URL) {
        failedLog = log
        server = .failed(message)
    }

    /// Always starts the app's own engine (a new port and secret each time); see Backend.
    private func boot() async {
        if !backend.isInstalled {
            guard backend.bundled else {
                server = .missing
                return
            }
            do {
                try await backend.install { step, detail in
                    Task { @MainActor in self.server = .installing(step: step, detail: detail) }
                }
            } catch let error as BackendError where error.needsPython {
                server = .needsPython
                return
            } catch {
                fail(error.localizedDescription, log: backend.setupLog)
                return
            }
            guard !Task.isCancelled else { return }
            server = .starting
        }
        backend.onExit = { [weak self] code in
            Task { @MainActor in self?.engineExited(code) }
        }
        do { try backend.launch() } catch {
            fail(error.localizedDescription, log: backend.log)
            return
        }
        for _ in 0..<240 {
            guard !Task.isCancelled, server == .starting else { return }  // stopped, or the engine already quit
            if let s = try? await backend.status(singer: options.singer) {
                status = s
                server = .ready
                await refresh()
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        guard !Task.isCancelled, server == .starting else { return }
        fail("엔진이 응답하지 않아요. 로그를 확인해 주세요.", log: backend.log)
    }

    /// The engine quit by itself (it failed to start, crashed, or was stopped from outside).
    private func engineExited(_ code: Int32) {
        guard !stopping, server == .starting || server == .ready else { return }
        let tail = backend.logTail(3)
        fail("엔진이 멈췄어요 (종료 코드 \(code)). 다시 시도해 주세요." + (tail.isEmpty ? "" : "\n\n" + tail), log: backend.log)
    }

    /// Apple's installer for the Command Line Tools (brings python3).
    func installCommandLineTools() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/xcode-select")
        p.arguments = ["--install"]
        try? p.run()
    }

    func shutdown() {
        stopping = true
        bootTask?.cancel()
        backend.stop()
    }

    func refresh() async {
        if let s = try? await backend.status(singer: options.singer) { status = s }
        if let list = try? await backend.stemSongs() { stemSongs = list }
        if let list = try? await backend.results() {
            results = list
            if selected == nil, let first = list.first { select(first) }
        }
    }

    // MARK: Inputs

    static let audioTypes: [UTType] = [.audio, .wav, .mp3, .mpeg4Audio, .aiff,
                                       UTType(filenameExtension: "flac") ?? .audio]
    static let noteTypes: [UTType] = ["ustx", "ust", "mid", "midi", "json"].compactMap { UTType(filenameExtension: $0) }

    static func isNoteFile(_ url: URL) -> Bool {
        // json = StemMixer's notes file (midi/vocals.json), the same types the note picker takes
        ["ustx", "ust", "mid", "midi", "json"].contains(url.pathExtension.lowercased())
    }

    private func pick(_ types: [UTType], message: String) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.allowsMultipleSelection = false
        panel.message = message
        return panel.runModal() == .OK ? panel.url : nil
    }

    func chooseVocal() {
        if let url = pick(Self.audioTypes, message: "조교할 보컬 파일을 골라 주세요 (보컬만 있는 파일)") { useFile(url) }
    }

    func chooseNotes() {
        if let url = pick(Self.noteTypes, message: "음표 파일 (OpenUtau · UTAU · MIDI)") { notesFile = url }
    }

    func chooseBacking() {
        if let url = pick(Self.audioTypes, message: "반주 파일 (OpenUtau에 참고용 트랙으로 넣어요)") { backingFile = url }
    }

    /// A dropped or chosen file: notes go to the note slot, audio becomes the vocal.
    func useFile(_ url: URL) {
        withAnimation(.easeOut(duration: 0.15)) {
            if Self.isNoteFile(url) {
                notesFile = url
            } else {
                // Another song: the previous song's notes and backing don't belong to it
                // (kept when no vocal was chosen yet, so dropping the notes first still works).
                if let old = source, old != .file(url), notesFile != nil || backingFile != nil {
                    notesFile = nil
                    backingFile = nil
                    show("새 보컬이라 음표 · 반주 파일은 비웠어요", success: false)
                }
                source = .file(url)
            }
            jobError = nil
        }
    }

    func useStemSong(_ song: StemSong) {
        withAnimation(.easeOut(duration: 0.15)) {
            source = .stem(song)
            notesFile = nil
            backingFile = nil
            jobError = nil
        }
    }

    func resetOptions() {
        let singer = options.singer
        options = TuneOptions()
        options.singer = singer
    }

    private static func savedLyrics() -> [String: String] {
        UserDefaults.standard.dictionary(forKey: "lyricsBySource") as? [String: String] ?? [:]
    }

    private func saveLyrics() {
        guard let key = source?.key else { return }
        var all = AppModel.savedLyrics()
        all[key] = lyrics.isEmpty ? nil : lyrics
        UserDefaults.standard.set(all, forKey: "lyricsBySource")
    }

    private static func loadOptions() -> TuneOptions {
        guard let data = UserDefaults.standard.data(forKey: "options"),
              let saved = try? JSONDecoder().decode(TuneOptions.self, from: data) else { return TuneOptions() }
        return saved
    }

    private func saveOptions() {
        if let data = try? JSONEncoder().encode(options) { UserDefaults.standard.set(data, forKey: "options") }
    }

    // MARK: Tuning

    var canTune: Bool { server == .ready && job == nil && source != nil }

    var lyricLines: Int {
        lyrics.split(whereSeparator: \.isNewline).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
    }

    func tune() {
        guard canTune, let source else { return }
        jobError = nil
        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) { job = Job(label: source.title) }
        Task {
            do {
                var vocal: URL?, song: String?
                switch source {
                case .file(let url): vocal = url
                case .stem(let s): song = s.id
                }
                let id = try await backend.startTune(vocal: vocal, song: song, lyrics: lyrics, notes: notesFile,
                                                     backing: backingFile, options: options)
                try await follow(id)
            } catch let error as BackendError {
                withAnimation { job = nil; jobError = error.message }
            } catch {
                withAnimation { job = nil; jobError = "엔진과 연결이 끊겼어요. 다시 시도해 주세요." }
            }
        }
    }

    private func follow(_ id: String) async throws {
        while true {
            try await Task.sleep(for: .milliseconds(400))
            let s = try await backend.job(id)
            if s.status == "running" {
                job?.step = s.step ?? "조교하는 중"
                job?.progress = s.progress ?? 0
                continue
            }
            withAnimation { job = nil }
            if s.status == "error" {
                withAnimation { jobError = s.error ?? "조교하지 못했어요" }
                return
            }
            if let list = try? await backend.results() {
                results = list
                if let r = s.result, let item = list.first(where: { $0.folder == r.folder }) ?? list.first {
                    select(item)
                }
            }
            show("조교 끝! OpenUtau에서 열어 보세요")
            return
        }
    }

    // MARK: Results

    func select(_ item: ResultItem) {
        guard selected?.id != item.id || preview == nil else { return }
        selected = item
        preview = nil
        loadingPreview = true
        Task {
            let p = try? await backend.preview(item.id)
            guard selected?.id == item.id else { return }
            withAnimation(.easeOut(duration: 0.2)) {
                preview = p
                loadingPreview = false
            }
        }
    }

    /// Opens the project in OpenUtau. OpenUtau only opens a project passed as a launch argument, and a
    /// second copy quits right away, so when it's already running the file is shown in Finder instead.
    func openInOpenUtau(_ item: ResultItem) {
        let ws = NSWorkspace.shared
        if let running = NSRunningApplication.runningApplications(withBundleIdentifier: Backend.openUtauID).first {
            running.activate()
            reveal(item)
            show("OpenUtau가 이미 열려 있어요. 파일 → 열기로 project.ustx를 열어 주세요", success: false)
            return
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let app = ws.urlForApplication(withBundleIdentifier: Backend.openUtauID)
            ?? [URL(fileURLWithPath: "/Applications/OpenUtau.app"), home.appendingPathComponent("Applications/OpenUtau.app")]
                .first { FileManager.default.fileExists(atPath: $0.path) }
        guard let app else {
            show("OpenUtau를 찾지 못했어요. 응용 프로그램 폴더에 넣어 주세요", success: false)
            return
        }
        let config = NSWorkspace.OpenConfiguration()
        config.arguments = [item.ustx]
        ws.openApplication(at: app, configuration: config) { _, error in
            if error != nil { Task { @MainActor in self.show("OpenUtau를 열지 못했어요", success: false) } }
        }
    }

    func reveal(_ item: ResultItem) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.ustx)])
    }

    /// Moves a result folder to the Trash (it can hold the user's OpenUtau edits, so it can be put back).
    func delete(_ item: ResultItem) {
        let folder = URL(fileURLWithPath: item.folder).resolvingSymlinksInPath()
        // Only folders right inside the output folder (the list comes from the engine).
        guard let output = status.map({ URL(fileURLWithPath: $0.output).resolvingSymlinksInPath() }),
              folder.deletingLastPathComponent().path == output.path,
              FileManager.default.fileExists(atPath: folder.appendingPathComponent("report.json").path) else {
            show("지우지 못했어요", success: false)
            return
        }
        do {
            try FileManager.default.trashItem(at: folder, resultingItemURL: nil)
        } catch {
            show("휴지통으로 옮기지 못했어요", success: false)
            return
        }
        withAnimation(.easeOut(duration: 0.2)) {
            results.removeAll { $0.id == item.id }
            if selected?.id == item.id {
                selected = nil
                preview = nil
            }
        }
        if selected == nil, let first = results.first { select(first) }
        show("휴지통으로 옮겼어요")
    }

    func openOutputFolder() {
        guard let path = status?.output else { return }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.open(url)
    }

    // MARK: Toast

    func show(_ message: String, success: Bool = true) {
        toastTask?.cancel()
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            toast = message
            toastIsSuccess = success
        }
        toastTask = Task {
            try? await Task.sleep(for: .seconds(1.8))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.25)) { toast = nil }
        }
    }
}
