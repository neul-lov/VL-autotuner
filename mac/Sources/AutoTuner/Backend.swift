import Foundation
import Security

/// Settings sent with each tuning job (same names as `Options` in tetotune/tune.py).
struct TuneOptions: Codable, Equatable {
    var bpm: Double = 0
    /// "auto" or semitones ("-12" … "12").
    var transpose: String = "auto"
    var pitchFollow: Double = 1.0
    var vibrato: Double = 1.0
    var correct: Double = 0.3
    var dynamics: Double = 0.7
    var breathiness: Double = 0.6
    var tension: Double = 0.4
    /// auto · voicebank · original · off
    var breaths: String = "auto"
    var breathVolume: Double = 1.0
    var pitchMethod: String = "auto"
    /// Voicebank folder; empty = find Teto.
    var singer: String = ""
    var defaultLyric: String = "ら"

    enum CodingKeys: String, CodingKey {
        case bpm, transpose, vibrato, correct, dynamics, breathiness, tension, breaths, singer
        case pitchFollow = "pitch_follow"
        case breathVolume = "breath_volume"
        case pitchMethod = "pitch_method"
        case defaultLyric = "default_lyric"
    }

    init() {}

    /// Older saved settings may miss newer keys; anything missing keeps its default.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TuneOptions()
        bpm = (try? c.decode(Double.self, forKey: .bpm)) ?? d.bpm
        transpose = (try? c.decode(String.self, forKey: .transpose)) ?? d.transpose
        pitchFollow = (try? c.decode(Double.self, forKey: .pitchFollow)) ?? d.pitchFollow
        vibrato = (try? c.decode(Double.self, forKey: .vibrato)) ?? d.vibrato
        correct = (try? c.decode(Double.self, forKey: .correct)) ?? d.correct
        dynamics = (try? c.decode(Double.self, forKey: .dynamics)) ?? d.dynamics
        breathiness = (try? c.decode(Double.self, forKey: .breathiness)) ?? d.breathiness
        tension = (try? c.decode(Double.self, forKey: .tension)) ?? d.tension
        breaths = (try? c.decode(String.self, forKey: .breaths)) ?? d.breaths
        breathVolume = (try? c.decode(Double.self, forKey: .breathVolume)) ?? d.breathVolume
        pitchMethod = (try? c.decode(String.self, forKey: .pitchMethod)) ?? d.pitchMethod
        singer = (try? c.decode(String.self, forKey: .singer)) ?? d.singer
        defaultLyric = (try? c.decode(String.self, forKey: .defaultLyric)) ?? d.defaultLyric
    }
}

struct SingerInfo: Decodable, Equatable {
    let name: String
    let path: String
    /// VCV (연속음) · CVVC · CV (단독음)
    let kind: String
    let breath_in: String?
    let breath_out: String?
    let phonemizer: String
}

struct SingerEntry: Decodable, Equatable, Identifiable {
    let id: String
    let name: String
    let path: String
}

struct ServerStatus: Decodable, Equatable {
    let version: String
    let singer: SingerInfo?
    let singers: [SingerEntry]
    let rmvpe: Bool
    let stemmixer: Bool
    let output: String
}

/// A song split in StemMixer.
struct StemSong: Decodable, Identifiable, Equatable {
    let id: String
    let title: String
    let duration: Double?
    let vocal: String
    /// StemMixer already read its notes (and maybe lyrics), which are used instead of guessing.
    let notes: Bool
    let lyrics: Bool
}

struct TuneResult: Decodable, Equatable {
    let folder: String
    let ustx: String
    let notes: Int
    let breaths: Int
    let transpose: Int
    let bpm: Double
    let singer: String?
    let phonemizer: String
    let pitch_method: String
    let warnings: [String]
}

struct TuneJob: Decodable {
    let status: String
    let step: String?
    let progress: Double?
    let result: TuneResult?
    let error: String?
}

/// A finished tuning in the output folder.
struct ResultItem: Decodable, Identifiable, Equatable {
    let id: String
    let folder: String
    let ustx: String
    let notes: Int?
    let breaths: Int?
    let warnings: [String]
    let singer: String?
    let transpose: Int?
    let created: Double

    /// The folder name is "<title> YYYY-MM-DD HHMMSS".
    var title: String {
        let parts = id.split(separator: " ")
        return parts.count > 2 ? parts.dropLast(2).joined(separator: " ") : id
    }
    var date: Date { Date(timeIntervalSince1970: created) }
}

/// What the preview draws: the original pitch, the tuned pitch, notes and breaths.
struct Preview: Decodable, Equatable {
    struct Note: Decodable, Equatable {
        let start: Double, end: Double, tone: Int, lyric: String
        init(from decoder: Decoder) throws {
            var c = try decoder.unkeyedContainer()
            start = try c.decode(Double.self); end = try c.decode(Double.self)
            tone = try c.decode(Int.self); lyric = (try? c.decode(String.self)) ?? ""
        }
    }
    /// Seconds per frame of `source` / `pitch` (the server sends every 2nd 5 ms frame).
    let hop: Double
    let bpm: Double
    let shift: Int
    let source: [Double?]
    let pitch: [Double]
    let dyn: [Double]
    let notes: [Note]
    let breaths: [[Double]]
    let duration: Double
    let phrases: Int

    var step: Double { hop * 2 }
}

/// Runs the Python engine (server.py) in the background and talks to it over HTTP.
///
/// The engine is either a checkout with its own `.venv` (built with start-mac.command / build.sh, the
/// `TetoTuneBackend` key), or the copy inside the app (`Resources/engine`), whose Python packages are
/// installed once into ~/Library/Application Support/TetoTune.
final class Backend {
    /// OpenUtau's bundle id (from its OpenUtau.csproj).
    static let openUtauID = "com.stakira.openutau"
    let folder: URL
    /// The engine shipped inside the app: packages live in `support`, not next to server.py.
    let bundled: Bool
    let support: URL
    private var process: Process?
    /// Each launch gets a free port and a new secret, so this app only ever talks to the engine it started
    /// (not another Mac user's, nor one left over from an older version), and nothing else can use it.
    private(set) var port = 0
    private let token = Backend.makeToken()
    private var base: URL { URL(string: "http://127.0.0.1:\(port)")! }
    /// Where the running engine's port and secret are written (only this Mac user can read it), for tools and tests.
    var serverInfo: URL { support.appendingPathComponent("server.json") }
    /// Called (on some thread) when the engine stops by itself.
    var onExit: (@Sendable (Int32) -> Void)?

    init() {
        let fm = FileManager.default
        support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TetoTune")
        let configured = (Bundle.main.object(forInfoDictionaryKey: "TetoTuneBackend") as? String).map { URL(fileURLWithPath: $0) }
        let inApp = Bundle.main.resourceURL?.appendingPathComponent("engine")
        if let configured, fm.isExecutableFile(atPath: configured.appendingPathComponent(".venv/bin/python").path) {
            folder = configured
            bundled = false
        } else if let inApp, fm.fileExists(atPath: inApp.appendingPathComponent("server.py").path) {
            folder = inApp
            bundled = true
        } else {
            folder = configured ?? URL(fileURLWithPath: fm.currentDirectoryPath)
            bundled = false
        }
    }

    var python: URL {
        bundled ? support.appendingPathComponent("venv/bin/python") : folder.appendingPathComponent(".venv/bin/python")
    }
    var log: URL { (bundled ? support : folder).appendingPathComponent("server-app.log") }
    var setupLog: URL { support.appendingPathComponent("setup.log") }
    private var requirements: URL { folder.appendingPathComponent("requirements.txt") }
    /// Packages the engine works without (RMVPE · kanji readings): a failure there only leaves a note in the log.
    private var extras: URL { folder.appendingPathComponent("requirements-extra.txt") }
    /// The requirement files the app's packages were installed from (reinstall when an update changes them).
    private var stamp: URL { support.appendingPathComponent("venv/.tetotune-requirements") }
    private var requirementsStamp: Data {
        [requirements, extras].compactMap { try? Data(contentsOf: $0) }.reduce(Data(), +)
    }

    var isInstalled: Bool {
        let fm = FileManager.default
        guard fm.isExecutableFile(atPath: python.path),
              fm.fileExists(atPath: folder.appendingPathComponent("server.py").path) else { return false }
        guard bundled else { return true }
        return (try? Data(contentsOf: stamp)) == requirementsStamp
    }

    func launch() throws {
        stop()
        let fm = FileManager.default
        try fm.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fm.createDirectory(at: support, withIntermediateDirectories: true)
        fm.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        guard let free = Self.freePort() else { throw BackendError(message: "엔진에 쓸 빈 포트를 찾지 못했어요") }
        port = free
        let p = Process()
        p.executableURL = python
        p.arguments = ["server.py"]
        p.currentDirectoryURL = folder
        p.standardOutput = handle
        p.standardError = handle
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["TETOTUNE_PORT"] = "\(port)"
        // In the environment, not the arguments: `ps` shows arguments to every user of the Mac.
        env["TETOTUNE_TOKEN"] = token
        // The engine quits by itself if the app is force-quit or crashes.
        env["TETOTUNE_PARENT_PID"] = "\(ProcessInfo.processInfo.processIdentifier)"
        // Never write .pyc files into the app bundle (that would break its signature).
        if bundled { env["PYTHONPYCACHEPREFIX"] = support.appendingPathComponent("pycache").path }
        p.environment = env
        let notify = onExit
        p.terminationHandler = { proc in
            try? handle.close()
            notify?(proc.terminationStatus)
        }
        try p.run()
        process = p
        let info = ["port": port, "pid": Int(p.processIdentifier), "token": token] as [String: Any]
        if let data = try? JSONSerialization.data(withJSONObject: info) {
            fm.createFile(atPath: serverInfo.path, contents: data, attributes: [.posixPermissions: 0o600])
        }
    }

    func stop() {
        guard let p = process else { return }
        p.terminationHandler = nil  // stopping on purpose is not a crash
        p.terminate()
        process = nil
        try? FileManager.default.removeItem(at: serverInfo)
    }

    /// The last lines of the engine's log (shown when it fails to start).
    func logTail(_ lines: Int = 6) -> String {
        guard let text = try? String(contentsOf: log, encoding: .utf8) else { return "" }
        return text.split(whereSeparator: \.isNewline).suffix(lines).joined(separator: "\n")
    }

    private static func makeToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        if SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) != errSecSuccess {
            bytes = Array((UUID().uuidString + UUID().uuidString).utf8)
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// A port nobody is listening on right now (the system picks it).
    private static func freePort() -> Int? {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let ok = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, len) == 0 && getsockname(fd, $0, &len) == 0 }
        }
        return ok ? Int(UInt16(bigEndian: addr.sin_port)) : nil
    }

    // MARK: First-run setup (engine inside the app)

    /// Python versions the engine's packages have Mac wheels for, best first. The app carries pyworld wheels for
    /// 3.9–3.14; a newer Python (Homebrew's python3 is 3.15 now) has no onnxruntime and would have to compile pyworld.
    static let pythonVersions = ["3.13", "3.12", "3.11", "3.10", "3.14", "3.9"]

    /// The best Python on this Mac: Homebrew, python.org, or Apple's Command Line Tools (3.9).
    /// /usr/bin/python3 is skipped because without the tools it only pops up an install dialog.
    func findPython() async -> (url: URL, version: String)? {
        var candidates: [String] = []
        for v in Self.pythonVersions {
            candidates += ["/opt/homebrew/bin/python\(v)", "/usr/local/bin/python\(v)",
                           "/Library/Frameworks/Python.framework/Versions/\(v)/bin/python\(v)"]
        }
        candidates += ["/opt/homebrew/bin/python3", "/usr/local/bin/python3",
                       "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
                       "/Library/Developer/CommandLineTools/usr/bin/python3",
                       "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"]
        var best: (rank: Int, url: URL, version: String)?
        var seen = Set<String>()
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            guard seen.insert((path as NSString).resolvingSymlinksInPath).inserted,
                  let v = await pythonVersion(path), let rank = Self.pythonVersions.firstIndex(of: v) else { continue }
            if best.map({ rank < $0.rank }) ?? true { best = (rank, URL(fileURLWithPath: path), v) }
            if rank == 0 { break }
        }
        return best.map { (url: $0.url, version: $0.version) }
    }

    /// "3.13" for a Python that also has venv, nil otherwise.
    private func pythonVersion(_ path: String) async -> String? {
        let out = Output()
        let code = await run(path, ["-c", "import sys, venv; print('%d.%d' % sys.version_info[:2])"]) { out.append($0) }
        // The last line: a warning printed on stderr comes through the same pipe.
        let v = out.text.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
        return code == 0 && !v.isEmpty ? v : nil
    }

    /// Makes the Python environment and installs the engine's packages (a few minutes, once).
    func install(step: @escaping @Sendable (String, String) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: support, withIntermediateDirectories: true)
        fm.createFile(atPath: setupLog.path, contents: nil)
        // An environment made with a Python the packages don't support (or whose Python was removed) is made again.
        var current: String?
        if fm.isExecutableFile(atPath: python.path) { current = await pythonVersion(python.path) }
        if current.map({ !Self.pythonVersions.contains($0) }) ?? true {
            guard let py = await findPython() else {
                throw BackendError(message: "Python 3.9~3.14가 필요해요", needsPython: true)
            }
            note("Python \(py.version): \(py.url.path)")
            step("Python 환경 만드는 중", "Python \(py.version)")
            // --clear: a venv whose Python was removed (e.g. a Homebrew upgrade) keeps dangling links otherwise
            guard await run(py.url.path, ["-m", "venv", "--clear", support.appendingPathComponent("venv").path]) == 0 else {
                throw BackendError(message: "Python 환경을 만들지 못했어요. 설치 기록을 확인해 주세요.")
            }
        }
        step("설치 도구 준비 중", "pip")
        _ = await run(python.path, ["-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"])
        step("필요한 프로그램 받는 중", "처음 한 번만 · 인터넷으로 몇 분 걸려요")
        let progress: @Sendable (String) -> Void = { line in
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("Collecting") || t.hasPrefix("Downloading") || t.hasPrefix("Building") || t.hasPrefix("Installing") {
                step("필요한 프로그램 받는 중", t)
            }
        }
        guard await run(python.path, pipArgs(requirements), line: progress) == 0 else {
            throw BackendError(message: "프로그램을 설치하지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요.")
        }
        if fm.fileExists(atPath: extras.path), await run(python.path, pipArgs(extras), line: progress) != 0 {
            note("추가 프로그램(RMVPE · 한자 읽기)은 설치하지 못했어요. 없어도 조교는 돼요.")
        }
        try requirementsStamp.write(to: stamp)
    }

    private func pipArgs(_ file: URL) -> [String] {
        // --prefer-binary: a newer source-only release on PyPI must not win over a ready-made wheel
        // (building it needs a compiler, which a Mac with only python.org Python doesn't have).
        var args = ["-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", "--prefer-binary",
                    "-r", file.path]
        // pyworld has no Mac wheels on PyPI; the app ships ones built for each Python version,
        // and constraints.txt pins pyworld to exactly that version.
        let wheels = folder.appendingPathComponent("wheels")
        if FileManager.default.fileExists(atPath: wheels.path) { args += ["--find-links", wheels.path] }
        let pins = wheels.appendingPathComponent("constraints.txt")
        if FileManager.default.fileExists(atPath: pins.path) { args += ["-c", pins.path] }
        return args
    }

    /// Adds a line of our own to the setup log.
    private func note(_ text: String) {
        guard let h = try? FileHandle(forWritingTo: setupLog) else { return }
        _ = try? h.seekToEnd()
        try? h.write(contentsOf: Data(("[AutoTuner] " + text + "\n").utf8))
        try? h.close()
    }

    /// Runs a program, appending its output to the setup log. `line` gets each output line. Returns the exit code.
    private func run(_ exe: String, _ args: [String], line: (@Sendable (String) -> Void)? = nil) async -> Int32 {
        let logURL = setupLog
        return await withCheckedContinuation { cont in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: exe)
            p.arguments = args
            var env = ProcessInfo.processInfo.environment
            env["PYTHONUNBUFFERED"] = "1"
            p.environment = env
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = pipe
            let logHandle = try? FileHandle(forWritingTo: logURL)
            _ = try? logHandle?.seekToEnd()
            // Finish once the program has ended AND all its output is read (EOF), so the last lines are never lost.
            let done = DispatchGroup()
            done.enter()
            done.enter()
            let eof = Output()
            pipe.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty else {
                    h.readabilityHandler = nil
                    if eof.markOnce() { done.leave() }
                    return
                }
                try? logHandle?.write(contentsOf: data)
                if let line, let text = String(data: data, encoding: .utf8) {
                    for l in text.split(whereSeparator: \.isNewline) { line(String(l)) }
                }
            }
            p.terminationHandler = { _ in done.leave() }
            do { try p.run() } catch {
                pipe.fileHandleForReading.readabilityHandler = nil
                try? logHandle?.close()
                done.leave()  // a group released with enters left open would crash
                done.leave()
                cont.resume(returning: -1)
                return
            }
            done.notify(queue: .global()) {
                try? logHandle?.close()
                cont.resume(returning: p.terminationStatus)
            }
        }
    }

    // MARK: Requests

    private func request(_ path: String, query: [URLQueryItem] = []) -> URLRequest {
        var url = base.appendingPathComponent(path)
        if !query.isEmpty, var c = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            c.queryItems = query
            // URLComponents leaves "+" as is and the server reads it as a space (same as post()).
            c.percentEncodedQuery = c.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B")
            url = c.url ?? url
        }
        var req = URLRequest(url: url)
        req.setValue(token, forHTTPHeaderField: "X-TetoTune-Token")
        return req
    }

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = [], as type: T.Type) async throws -> T {
        guard port != 0 else { throw URLError(.cannotConnectToHost) }
        let (data, response) = try await URLSession.shared.data(for: request(path, query: query))
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw error(from: data, "불러오지 못했어요") }
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post(_ path: String, _ items: [URLQueryItem]) async throws -> Data {
        var form = URLComponents()
        form.queryItems = items
        var req = request(path)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.httpBody = form.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B").data(using: .utf8)
        let (data, response) = try await URLSession.shared.data(for: req)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw error(from: data, "요청이 실패했어요") }
        return data
    }

    private func error(from data: Data, _ fallback: String) -> BackendError {
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return BackendError(message: json?["detail"] as? String ?? fallback)
    }

    /// `singer`: the voicebank folder chosen in the app ("" = find Teto).
    func status(singer: String = "") async throws -> ServerStatus {
        try await get("api/status", query: singer.isEmpty ? [] : [URLQueryItem(name: "singer", value: singer)],
                      as: ServerStatus.self)
    }
    func stemSongs() async throws -> [StemSong] { try await get("api/stemmixer/songs", as: [StemSong].self) }
    func results() async throws -> [ResultItem] { try await get("api/results", as: [ResultItem].self) }
    func job(_ id: String) async throws -> TuneJob { try await get("api/jobs/\(id)", as: TuneJob.self) }

    func preview(_ resultID: String) async throws -> Preview {
        try await get("api/results/\(resultID)/preview", as: Preview.self)
    }

    /// Starts tuning a vocal file or a StemMixer song. Returns the job id.
    func startTune(vocal: URL?, song: String?, lyrics: String, notes: URL?, backing: URL?,
                   options: TuneOptions) async throws -> String {
        let json = String(data: try JSONEncoder().encode(options), encoding: .utf8) ?? "{}"
        var items = [URLQueryItem(name: "lyrics", value: lyrics), URLQueryItem(name: "options", value: json)]
        if let vocal { items.append(URLQueryItem(name: "vocal", value: vocal.path)) }
        if let song { items.append(URLQueryItem(name: "song", value: song)) }
        if let notes { items.append(URLQueryItem(name: "notes", value: notes.path)) }
        if let backing { items.append(URLQueryItem(name: "backing", value: backing.path)) }
        let data = try await post("api/tune", items)
        guard let id = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["job_id"] as? String else {
            throw BackendError(message: "요청이 실패했어요")
        }
        return id
    }

}

/// Collects a program's output across the threads its lines arrive on.
private final class Output: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""
    private var marked = false
    func append(_ line: String) { lock.lock(); buffer += line + "\n"; lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return buffer }
    /// True the first time only.
    func markOnce() -> Bool { lock.lock(); defer { lock.unlock() }; if marked { return false }; marked = true; return true }
}

struct BackendError: LocalizedError {
    let message: String
    /// No usable Python: the app offers to install Apple's Command Line Tools.
    var needsPython = false
    var errorDescription: String? { message }
}
