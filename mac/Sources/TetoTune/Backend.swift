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
final class Backend {
    static let port = 8766
    let folder: URL
    private var process: Process?
    private let base = URL(string: "http://127.0.0.1:\(Backend.port)")!

    init() {
        let configured = Bundle.main.object(forInfoDictionaryKey: "TetoTuneBackend") as? String
        folder = URL(fileURLWithPath: configured ?? FileManager.default.currentDirectoryPath)
    }

    var python: URL { folder.appendingPathComponent(".venv/bin/python") }
    var log: URL { folder.appendingPathComponent("server-app.log") }

    var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: python.path)
            && FileManager.default.fileExists(atPath: folder.appendingPathComponent("server.py").path)
    }

    func launch() throws {
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
        p.environment = env
        try p.run()
        process = p
    }

    func stop() {
        process?.terminate()
        process = nil
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
    var errorDescription: String? { message }
}
