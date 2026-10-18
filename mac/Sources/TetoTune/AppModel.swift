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
    enum ServerState: Equatable { case starting, ready, missing, failed(String) }

    let backend = Backend()

    @Published var server: ServerState = .starting
    @Published var status: ServerStatus?
    @Published var stemSongs: [StemSong] = []
    @Published var results: [ResultItem] = []

    @Published var source: VocalSource?
    @Published var lyrics = UserDefaults.standard.string(forKey: "lyrics") ?? "" {
        didSet { UserDefaults.standard.set(lyrics, forKey: "lyrics") }
    }
    @Published var notesFile: URL?
    @Published var backingFile: URL?
    @Published var options: TuneOptions = AppModel.loadOptions() {
        didSet { saveOptions() }
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

    func start() {
        guard backend.isInstalled else {
            server = .missing
            return
        }
        Task {
            // A server left over from an earlier run can be reused as is.
            if (try? await backend.status()) == nil {
                do { try backend.launch() } catch {
                    server = .failed(error.localizedDescription)
                    return
                }
            }
            for _ in 0..<120 {
                if let s = try? await backend.status() {
                    status = s
                    server = .ready
                    await refresh()
                    return
                }
                try? await Task.sleep(for: .milliseconds(500))
            }
            server = .failed("엔진이 응답하지 않아요. 로그를 확인해 주세요.")
        }
        // Pick up new StemMixer songs or voicebanks when coming back to the app.
        NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.server == .ready else { return }
                Task { await self.refresh() }
            }
        }
    }

    func shutdown() {
        backend.stop()
    }

    func refresh() async {
        if let s = try? await backend.status() { status = s }
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
        ["ustx", "ust", "mid", "midi"].contains(url.pathExtension.lowercased())
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
            } catch {
                withAnimation { job = nil; jobError = error.localizedDescription }
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

    /// Opens the project in OpenUtau (or whatever app opens .ustx files).
    func openInOpenUtau(_ item: ResultItem) {
        let file = URL(fileURLWithPath: item.ustx)
        let candidates = [URL(fileURLWithPath: "/Applications/OpenUtau.app"),
                          FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/OpenUtau.app")]
        if let app = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            NSWorkspace.shared.open([file], withApplicationAt: app, configuration: NSWorkspace.OpenConfiguration()) { _, error in
                if error != nil { Task { @MainActor in self.show("OpenUtau를 열지 못했어요", success: false) } }
            }
        } else if !NSWorkspace.shared.open(file) {
            show("OpenUtau를 찾지 못했어요. 응용 프로그램 폴더에 넣어 주세요", success: false)
        }
    }

    func reveal(_ item: ResultItem) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.ustx)])
    }

    func delete(_ item: ResultItem) {
        Task {
            do {
                try await backend.deleteResult(item.id)
                withAnimation(.easeOut(duration: 0.2)) {
                    results.removeAll { $0.id == item.id }
                    if selected?.id == item.id {
                        selected = nil
                        preview = nil
                    }
                }
                if selected == nil, let first = results.first { select(first) }
                show("지웠어요")
            } catch {
                show(error.localizedDescription, success: false)
            }
        }
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
