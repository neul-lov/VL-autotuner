// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "TetoTune",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "TetoTune", path: "Sources/TetoTune")
    ]
)
