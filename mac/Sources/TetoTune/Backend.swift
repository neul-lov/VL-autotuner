import Foundation

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
    static let port = 8766
    /// OpenUtau's bundle id (from its OpenUtau.csproj).
    static let openUtauID = "com.stakira.openutau"
    let folder: URL
    /// The engine shipped inside the app: packages live in `support`, not next to server.py.
    let bundled: Bool
    let support: URL
    private var process: Process?
    private let base = URL(string: "http://127.0.0.1:\(Backend.port)")!

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
    /// The requirements.txt the app's packages were installed from (reinstall when an update changes it).
    private var stamp: URL { support.appendingPathComponent("venv/.tetotune-requirements") }

    var isInstalled: Bool {
        let fm = FileManager.default
        guard fm.isExecutableFile(atPath: python.path),
              fm.fileExists(atPath: folder.appendingPathComponent("server.py").path) else { return false }
        guard bundled else { return true }
        return (try? Data(contentsOf: stamp)) == (try? Data(contentsOf: requirements))
    }

    func launch() throws {
        try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        let p = Process()
        p.executableURL = python
        p.arguments = ["server.py"]
        p.currentDirectoryURL = folder
        p.standardOutput = handle
        p.standardError = handle
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["TETOTUNE_PORT"] = "\(Backend.port)"
        // Never write .pyc files into the app bundle (that would break its signature).
        if bundled { env["PYTHONPYCACHEPREFIX"] = support.appendingPathComponent("pycache").path }
        p.environment = env
        try p.run()
        process = p
    }

    func stop() {
        process?.terminate()
        process = nil
    }

    // MARK: First-run setup (engine inside the app)

    /// Python 3.9+ on this Mac: Homebrew, python.org, or Apple's Command Line Tools.
    /// /usr/bin/python3 is skipped because without the tools it only pops up an install dialog.
    func findPython() async -> URL? {
        let candidates = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3",
                          "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
                          "/Library/Developer/CommandLineTools/usr/bin/python3",
                          "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            let ok = await run(path, ["-c", "import sys, venv; sys.exit(0 if sys.version_info >= (3, 9) else 1)"])
            if ok == 0 { return URL(fileURLWithPath: path) }
        }
        return nil
    }

    /// Makes the Python environment and installs the engine's packages (a few minutes, once).
    func install(step: @escaping @Sendable (String, String) -> Void) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: support, withIntermediateDirectories: true)
        fm.createFile(atPath: setupLog.path, contents: nil)
        guard let py = await findPython() else { throw BackendError(message: "Python 3이 필요해요", needsPython: true) }
        if !fm.isExecutableFile(atPath: python.path) {
            step("Python 환경 만드는 중", py.path)
            guard await run(py.path, ["-m", "venv", support.appendingPathComponent("venv").path]) == 0 else {
                throw BackendError(message: "Python 환경을 만들지 못했어요. 설치 기록을 확인해 주세요.")
            }
        }
        step("설치 도구 준비 중", "pip")
        _ = await run(python.path, ["-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"])
        step("필요한 프로그램 받는 중", "처음 한 번만 · 인터넷으로 몇 분 걸려요")
        var args = ["-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off", "-r", requirements.path]
        // pyworld has no Mac wheels on PyPI; the app ships ones built for each Python version.
        let wheels = folder.appendingPathComponent("wheels")
        if fm.fileExists(atPath: wheels.path) { args += ["--find-links", wheels.path] }
        let code = await run(python.path, args) { line in
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("Collecting") || t.hasPrefix("Downloading") || t.hasPrefix("Building") || t.hasPrefix("Installing") {
                step("필요한 프로그램 받는 중", t)
            }
        }
        guard code == 0 else {
            throw BackendError(message: "프로그램을 설치하지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요.")
        }
        try Data(contentsOf: requirements).write(to: stamp)
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
            pipe.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty else { return }
                try? logHandle?.write(contentsOf: data)
                if let line, let text = String(data: data, encoding: .utf8) {
                    for l in text.split(whereSeparator: \.isNewline) { line(String(l)) }
                }
            }
            p.terminationHandler = { p in
                pipe.fileHandleForReading.readabilityHandler = nil
                try? logHandle?.close()
                cont.resume(returning: p.terminationStatus)
            }
            do { try p.run() } catch {
                pipe.fileHandleForReading.readabilityHandler = nil
                cont.resume(returning: -1)
            }
        }
    }

    // MARK: Requests

    private func get<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        let (data, response) = try await URLSession.shared.data(from: base.appendingPathComponent(path))
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw error(from: data, "불러오지 못했어요") }
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post(_ path: String, _ items: [URLQueryItem]) async throws -> Data {
        var form = URLComponents()
        form.queryItems = items
        var req = URLRequest(url: base.appendingPathComponent(path))
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

    func status() async throws -> ServerStatus { try await get("api/status", as: ServerStatus.self) }
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

    func deleteResult(_ id: String) async throws {
        var req = URLRequest(url: base.appendingPathComponent("api/results/\(id)"))
        req.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: req)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw error(from: data, "지우지 못했어요") }
    }
}

struct BackendError: LocalizedError {
    let message: String
    /// No usable Python: the app offers to install Apple's Command Line Tools.
    var needsPython = false
    var errorDescription: String? { message }
}
