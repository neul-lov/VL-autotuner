// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "AutoTuner",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "AutoTuner", path: "Sources/AutoTuner")
    ]
)
