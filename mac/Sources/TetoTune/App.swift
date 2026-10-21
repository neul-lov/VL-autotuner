import AppKit
import SwiftUI

@main
struct TetoTuneApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("TetoTune", id: "main") {
            ContentView(model: delegate.model)
                .frame(minWidth: 940, minHeight: 640)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1080, height: 720)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        model.start()
    }

    /// While a song is being tuned, closing the window keeps the app (and the job) running.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { model.job == nil }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard model.job != nil else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "조교하는 중이에요"
        alert.informativeText = "지금 끄면 이 곡의 조교가 멈추고 결과가 남지 않아요."
        alert.addButton(withTitle: "계속 조교하기")
        alert.addButton(withTitle: "끄기")
        return alert.runModal() == .alertSecondButtonReturn ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) {
        model.shutdown()
    }
}
