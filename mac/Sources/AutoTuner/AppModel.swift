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

    /// Listening to the result (original vocal · Teto · backing).
    let player = Player()
    /// The note being edited (its place in `preview.notes`).
    @Published var selectedNote: Int?
    /// The section picked on the roll (seconds).
    @Published var selection: ClosedRange<Double>?
    /// A time for the roll to scroll to.
    @Published var focusTime: Double?
    /// The preview before the last section re-tune, and that section: the roll draws the old pitch there to compare.
    @Published var before: Preview?
    @Published var compareRange: [Double]?
    /// An edit is being saved.
    @Published var editing = false
    /// A section re-tune or a Teto re-render running for the shown result.
    @Published var resultJob: Job?

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
            // Same rule as useFile: only another song's files are cleared.
            if let old = source, old != .stem(song), notesFile != nil || backingFile != nil {
                notesFile = nil
                backingFile = nil
                show("새 보컬이라 음표 · 반주 파일은 비웠어요", success: false)
            }
            source = .stem(song)
            jobError = nil
        }
    }

    /// The X on the vocal card: the notes and backing were picked for that vocal, so they go too.
    func clearSource() {
        withAnimation {
            source = nil
            notesFile = nil
            backingFile = nil
        }
    }

    func resetOptions() {
        // One assignment with the same singer: no extra status requests that could arrive out of order.
        var fresh = TuneOptions()
        fresh.singer = options.singer
        options = fresh
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
        guard resultJob == nil else {
            show("구간 다시 조교가 끝난 뒤에 다른 결과를 골라 주세요", success: false)
            return
        }
        selected = item
        preview = nil
        selectedNote = nil
        selection = nil
        before = nil
        compareRange = nil
        player.load(item)
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
        // OpenUtau skips the launch argument while it has an unsaved project to recover: say so up front
        if Self.openUtauHasRecovery() {
            reveal(item)
            show("OpenUtau가 먼저 복구할 작업을 물어봐요. 그 창을 닫은 뒤 파일 → 열기로 project.ustx를 열어 주세요", success: false)
        }
        let config = NSWorkspace.OpenConfiguration()
        config.arguments = [item.ustx]
        ws.openApplication(at: app, configuration: config) { opened, error in
            Task { @MainActor in
                if error != nil {
                    self.show("OpenUtau를 열지 못했어요", success: false)
                    return
                }
                // OpenUtau quits right away when another process named OpenUtau is running
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                if let opened, opened.isTerminated {
                    self.reveal(item)
                    self.show("OpenUtau가 켜지자마자 꺼졌어요. OpenUtau를 직접 연 뒤 파일 → 열기로 project.ustx를 열어 주세요", success: false)
                }
            }
        }
    }

    /// EasyUtau (음표를 손으로 고치는 앱)가 깔려 있나.
    static var easyUtauApp: URL? {
        guard let url = URL(string: "easyutau://open") else { return nil }
        return NSWorkspace.shared.urlForApplication(toOpen: url)
    }

    /// 지금 결과를 EasyUtau에서 연다 (seconds를 주면 그 자리의 섹터를 골라서).
    /// EasyUtau는 OpenUtau와 같은 project.ustx를 그대로 열고 고친다.
    func openInEasyUtau(_ item: ResultItem, at seconds: Double? = nil) {
        guard let app = Self.easyUtauApp else {
            show("EasyUtau가 없어요. 받아서 응용 프로그램 폴더에 넣어 주세요", success: false)
            NSWorkspace.shared.open(URL(string: "https://github.com/neul-lov/EasyUtau")!)
            return
        }
        var comps = URLComponents()
        comps.scheme = "easyutau"
        comps.host = "open"
        comps.queryItems = [URLQueryItem(name: "path", value: item.ustx)]
        if let seconds {
            comps.queryItems?.append(URLQueryItem(name: "time", value: String(format: "%.3f", max(0, seconds))))
        }
        guard let url = comps.url else { return }
        player.pause()
        let config = NSWorkspace.OpenConfiguration()
        config.activates = true
        NSWorkspace.shared.open([url], withApplicationAt: app, configuration: config) { _, error in
            Task { @MainActor in
                if error != nil { self.show("EasyUtau를 열지 못했어요", success: false) }
            }
        }
    }

    /// OpenUtau's prefs.json keeps RecoveryPath while an autosaved, unsaved project is waiting to be recovered.
    static func openUtauHasRecovery() -> Bool {
        let prefs = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/OpenUtau/prefs.json")
        guard let data = try? Data(contentsOf: prefs),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = json["RecoveryPath"] as? String, !path.trimmingCharacters(in: .whitespaces).isEmpty else { return false }
        return FileManager.default.fileExists(atPath: path)
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
                player.load(nil)
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

    // MARK: Fixing a result

    /// After a change to project.ustx: OpenUtau shows the file as it was when opened.
    private func edited(_ message: String) {
        if NSRunningApplication.runningApplications(withBundleIdentifier: Backend.openUtauID).first != nil {
            show(message + " · OpenUtau에서는 project.ustx를 다시 열어 주세요")
        } else {
            show(message)
        }
    }

    func undo() {
        guard let item = selected, !editing, resultJob == nil else { return }
        editing = true
        Task {
            defer { editing = false }
            do {
                let p = try await backend.undo(item.id)
                guard selected?.id == item.id else { return }
                withAnimation(.easeOut(duration: 0.15)) {
                    preview = p
                    before = nil
                    compareRange = nil
                    if let i = selectedNote, i >= p.notes.count { selectedNote = nil }
                }
                edited("되돌렸어요")
            } catch let error as BackendError {
                show(error.message, success: false)
            } catch {
                show("엔진과 연결이 끊겼어요", success: false)
            }
        }
    }

    /// Tunes the picked section again with other settings (`options`: tune.py names → values).
    func retune(_ range: ClosedRange<Double>, options: [String: Double], lyrics: String?, renote: Bool) {
        guard let item = selected, resultJob == nil else { return }
        player.pause()
        let old = preview
        withAnimation { resultJob = Job(label: "\(formatTime(range.lowerBound)) – \(formatTime(range.upperBound))") }
        Task {
            do {
                let id = try await backend.startRetune(item.id, start: range.lowerBound, end: range.upperBound,
                                                       options: options, lyrics: lyrics, renote: renote)
                guard let p = try await followResult(id) else { return }
                guard selected?.id == item.id else { return }
                withAnimation(.easeOut(duration: 0.2)) {
                    preview = p
                    before = old
                    compareRange = p.range
                    if let r = p.range, r.count == 2 { selection = r[0]...r[1] }
                    selectedNote = nil
                }
                await reloadAudio(item)
                if let w = p.warning { show(w, success: false) } else { edited("다시 조교했어요 (점선: 바꾸기 전)") }
            } catch let error as BackendError {
                withAnimation { resultJob = nil }
                show(error.message, success: false)
            } catch {
                withAnimation { resultJob = nil }
                show("엔진과 연결이 끊겼어요", success: false)
            }
        }
    }

    /// Sings the edited project with Teto again, so the Teto preview matches it.
    func renderTeto() {
        guard let item = selected, resultJob == nil else { return }
        player.pause()
        withAnimation { resultJob = Job(label: "고친 결과를 테토로", step: "테토로 불러 보는 중") }
        Task {
            do {
                let id = try await backend.startRender(item.id)
                guard let p = try await followResult(id), selected?.id == item.id else { return }
                withAnimation { preview = p }
                await reloadAudio(item)
                player.voice = .teto
                show("테토 미리 듣기를 새로 만들었어요")
            } catch let error as BackendError {
                withAnimation { resultJob = nil }
                show(error.message, success: false)
            } catch {
                withAnimation { resultJob = nil }
                show("엔진과 연결이 끊겼어요", success: false)
            }
        }
    }

    /// Follows a re-tune / re-render job; nil after an error (already shown).
    private func followResult(_ id: String) async throws -> Preview? {
        while true {
            try await Task.sleep(for: .milliseconds(400))
            let s = try await backend.previewJob(id)
            if s.status == "running" {
                resultJob?.step = s.step ?? "다시 조교하는 중"
                resultJob?.progress = s.progress ?? 0
                continue
            }
            withAnimation { resultJob = nil }
            if s.status == "error" || s.result == nil {
                show(s.error ?? "하지 못했어요", success: false)
                return nil
            }
            return s.result
        }
    }

    /// The Teto preview file changed (or appeared): read the list again and reopen the audio.
    private func reloadAudio(_ item: ResultItem) async {
        if let list = try? await backend.results() {
            results = list
            if let fresh = list.first(where: { $0.id == item.id }) { selected = fresh }
        }
        player.load(selected, force: true)
    }

    /// Plays a section with the voice picked in the player (a little before and after it).
    func play(_ start: Double, _ end: Double) {
        player.play(from: max(0, start - 0.4), to: end + 0.3)
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
