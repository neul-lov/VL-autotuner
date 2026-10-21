import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var model: AppModel
    @State private var dropping = false

    var body: some View {
        HStack(spacing: 0) {
            Sidebar(model: model)
                .frame(width: 330)
                .background(Color.primary.opacity(0.03))
            Rectangle().fill(Color.primary.opacity(0.08)).frame(width: 0.5)
            MainView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .fontDesign(.rounded)
        .tint(.teto)
        .ignoresSafeArea()
        .overlay { if dropping { dropOverlay } }
        .overlay(alignment: .bottom) { toast }
        .onDrop(of: [.fileURL], isTargeted: $dropping.animation(.easeOut(duration: 0.15))) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in model.useFile(url) }
            }
            return true
        }
    }

    private var dropOverlay: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.teto.opacity(0.10))
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.teto.opacity(0.6), style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
            VStack(spacing: 10) {
                Image(systemName: "waveform.badge.plus")
                    .font(.system(size: 38, weight: .light))
                Text("보컬 파일이나 음표 파일(ustx · ust · mid · json)을 놓으세요")
                    .font(.system(size: 15, weight: .semibold))
            }
            .foregroundStyle(Color.teto)
        }
        .padding(12)
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private var toast: some View {
        if let message = model.toast {
            HStack(spacing: 8) {
                Image(systemName: model.toastIsSuccess ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .symbolEffect(.bounce, value: message)
                Text(message)
            }
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Capsule().fill(model.toastIsSuccess ? Color.green.gradient : Color.red.gradient))
            .shadow(color: .black.opacity(0.18), radius: 12, y: 4)
            .padding(.bottom, 40)
            .transition(.move(edge: .bottom).combined(with: .opacity).combined(with: .scale(scale: 0.9)))
        }
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            header
            sourceCard
            content
            footer
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(LinearGradient(colors: [.teto, .teto.opacity(0.7)],
                                         startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 26, height: 26)
                Image(systemName: "music.mic")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 0) {
                Text("AutoTuner")
                    .font(.system(size: 15, weight: .bold))
                Text(subtitle)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            settingsMenu
        }
        .padding(.horizontal, 16)
        .padding(.top, 38)  // clears the window's traffic lights
        .padding(.bottom, 10)
    }

    private var subtitle: String {
        switch model.server {
        case .starting: return "준비 중…"
        case .missing: return "설치가 필요해요"
        case .failed: return "엔진을 시작하지 못했어요"
        case .installing: return "처음 한 번 준비 중…"
        case .needsPython: return "Python이 필요해요"
        case .ready:
            guard let singer = model.status?.singer else { return "OpenUtau 음원을 찾지 못했어요" }
            return "\(singer.name) · \(kindLabel(singer.kind))"
        }
    }

    private func kindLabel(_ kind: String) -> String {
        switch kind {
        case "VCV": "연속음"
        case "CVVC": "CVVC"
        default: "단독음"
        }
    }

    private var settingsMenu: some View {
        Menu {
            if let singers = model.status?.singers, !singers.isEmpty {
                Picker("가수", selection: $model.options.singer) {
                    Text("자동 (테토 먼저)").tag("")
                    ForEach(singers) { Text($0.name).tag($0.path) }
                }
                Divider()
            }
            Button("보컬 파일 선택…") { model.chooseVocal() }
            Button("결과 폴더 열기") { model.openOutputFolder() }
            Button("엔진 로그 보기") { NSWorkspace.shared.open(model.backend.log) }
            if model.backend.bundled {
                Button("설치 기록 보기") { NSWorkspace.shared.open(model.backend.setupLog) }
            }
            Button("새로 고침") { Task { await model.refresh() } }
            Divider()
            Button("AutoTuner 종료") { NSApp.terminate(nil) }
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
                .background(Circle().fill(Color.primary.opacity(0.06)))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
    }

    /// The vocal to tune: drop zone when empty, the picked file or StemMixer song otherwise.
    private var sourceCard: some View {
        Button { model.chooseVocal() } label: {
            HStack(spacing: 10) {
                Image(systemName: sourceIcon)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(model.source == nil ? Color.secondary : Color.teto)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill((model.source == nil ? Color.primary : Color.teto).opacity(0.08)))
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.source?.title ?? "보컬 파일을 끌어다 놓거나 고르세요")
                        .font(.system(size: 12.5, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(sourceDetail)
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                if model.source != nil {
                    Button { model.clearSource() } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                    }
                    .buttonStyle(.plain)
                    .help("고른 보컬 빼기")
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.primary.opacity(0.05)))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(model.source == nil ? Color.primary.opacity(0.15) : Color.teto.opacity(0.4),
                                  style: StrokeStyle(lineWidth: 1, dash: model.source == nil ? [5, 4] : []))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
        .padding(.horizontal, 12)
        .padding(.bottom, 10)
    }

    private var sourceIcon: String {
        switch model.source {
        case .none: "waveform.badge.plus"
        case .file: "waveform"
        case .stem: "slider.vertical.3"
        }
    }

    private var sourceDetail: String {
        switch model.source {
        case .none: "보컬만 있는 파일 (wav · flac · mp3 · m4a)"
        case .file(let url): url.lastPathComponent
        case .stem(let s):
            s.lyrics ? "StemMixer 음표 · 가사를 써요 (가사 칸이 비어 있으면)" : s.notes ? "StemMixer · 음표도 가져와요" : "StemMixer · 반주도 같이 넣어요"
        }
    }

    @ViewBuilder
    private var content: some View {
        switch model.server {
        case .starting:
            placeholder(icon: "hourglass", title: "준비 중", text: "조교 엔진을 켜고 있어요.")
        case .missing:
            placeholder(icon: "shippingbox", title: "설치가 필요해요",
                        text: "\(model.backend.folder.path) 에서 start-mac.command를 한 번 실행해 주세요.")
        case .failed(let message):
            VStack(spacing: 0) {
                placeholder(icon: "exclamationmark.triangle", title: "엔진을 시작하지 못했어요", text: message)
                HStack(spacing: 8) {
                    Button("다시 시도") { model.start() }
                    Button("기록 보기") { NSWorkspace.shared.open(model.failedLog ?? model.backend.log) }
                }
                .font(.system(size: 12, weight: .semibold))
                .padding(.bottom, 40)
            }
        case .installing(let step, let detail):
            VStack(spacing: 10) {
                Spacer()
                ProgressView().controlSize(.regular)
                Text(step)
                    .font(.system(size: 15, weight: .semibold))
                Text(detail)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .padding(.horizontal, 24)
                Text("조교 엔진(음 높이 읽기 · WORLD)을 이 Mac에 설치하고 있어요. 처음 한 번만 해요.")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 30)
                Spacer()
            }
            .frame(maxWidth: .infinity)
        case .needsPython:
            VStack(spacing: 0) {
                placeholder(icon: "shippingbox", title: "Python 3이 필요해요",
                            text: "Apple의 개발자 도구(Command Line Tools)를 설치하면 같이 들어와요. 설치가 끝나면 다시 시도를 눌러 주세요. (python.org의 3.9~3.14도 돼요. 3.15는 아직 필요한 프로그램이 없어요)")
                HStack(spacing: 8) {
                    Button("개발자 도구 설치") { model.installCommandLineTools() }
                    Button("다시 시도") { model.start() }
                }
                .font(.system(size: 12, weight: .semibold))
                .padding(.bottom, 40)
            }
        case .ready where model.stemSongs.isEmpty && model.results.isEmpty:
            placeholder(icon: "music.note.list", title: "아직 조교한 곡이 없어요",
                        text: "보컬 파일을 끌어다 놓으세요. StemMixer에서 분리한 곡은 여기 저절로 나와요.")
        case .ready:
            ScrollView {
                LazyVStack(spacing: 4) {
                    if !model.results.isEmpty {
                        SectionTitle(title: "조교한 결과", icon: "clock")
                        ForEach(model.results) { item in
                            ResultRow(item: item, current: model.selected?.id == item.id, model: model)
                        }
                    }
                    if !model.stemSongs.isEmpty {
                        SectionTitle(title: "StemMixer에서 분리한 곡", icon: "slider.vertical.3")
                        ForEach(model.stemSongs) { song in
                            StemRow(song: song, current: model.source == .stem(song), model: model)
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .scrollIndicators(.never)
        }
    }

    private func placeholder(icon: String, title: String, text: String) -> some View {
        VStack(spacing: 10) {
            Spacer()
            Image(systemName: icon)
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.tertiary)
            Text(title)
                .font(.system(size: 15, weight: .semibold))
            Text(text)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var footer: some View {
        HStack(spacing: 14) {
            KeyHint("⌘⏎", "조교")
            Spacer()
            KeyHint("끌어다 놓기", "보컬 · 음표")
        }
        .frame(height: 34)
        .padding(.horizontal, 16)
        .overlay(alignment: .top) { Rectangle().fill(Color.primary.opacity(0.06)).frame(height: 0.5) }
    }
}

private struct ResultRow: View {
    let item: ResultItem
    let current: Bool
    @ObservedObject var model: AppModel
    @State private var hovering = false
    @State private var confirmDelete = false

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "music.note")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(current ? .white : Color.teto)
                .frame(width: 28, height: 28)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(current ? AnyShapeStyle(Color.teto.gradient) : AnyShapeStyle(Color.teto.opacity(0.12))))
            VStack(alignment: .leading, spacing: 1) {
                Text(item.title)
                    .font(.system(size: 12.5, weight: .semibold))
                    .lineLimit(1)
                Text(detail)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if hovering {
                RowAction(icon: "arrow.up.forward.app", help: "OpenUtau에서 열기") { model.openInOpenUtau(item) }
                RowAction(icon: "folder", help: "Finder에서 보기") { model.reveal(item) }
                RowAction(icon: "trash", help: "휴지통으로", tint: .red) { confirmDelete = true }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(current ? Color.teto.opacity(0.12) : hovering ? Color.primary.opacity(0.05) : .clear))
        .contentShape(Rectangle())
        .onTapGesture { model.select(item) }
        .onHover { h in withAnimation(.easeOut(duration: 0.12)) { hovering = h } }
        .contextMenu {
            Button("OpenUtau에서 열기") { model.openInOpenUtau(item) }
            Button("Finder에서 보기") { model.reveal(item) }
            Divider()
            Button("휴지통으로 옮기기…", role: .destructive) { confirmDelete = true }
        }
        .confirmationDialog("‘\(item.title)’ 결과를 휴지통으로 옮길까요?", isPresented: $confirmDelete) {
            Button("휴지통으로 옮기기", role: .destructive) { model.delete(item) }
        } message: {
            Text("OpenUtau에서 고친 project.ustx도 같이 옮겨져요. 휴지통에서 되살릴 수 있어요.")
        }
    }

    private var detail: String {
        var parts = [item.date.formatted(date: .abbreviated, time: .shortened)]
        if let n = item.notes { parts.append("음표 \(n)개") }
        if let b = item.breaths, b > 0 { parts.append("숨 \(b)번") }
        return parts.joined(separator: " · ")
    }
}

private struct StemRow: View {
    let song: StemSong
    let current: Bool
    @ObservedObject var model: AppModel
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(Color.source)
                .frame(width: 28, height: 28)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.source.opacity(0.12)))
            VStack(alignment: .leading, spacing: 1) {
                Text(song.title)
                    .font(.system(size: 12.5, weight: .semibold))
                    .lineLimit(1)
                Text(detail)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if current {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.teto)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(current ? Color.teto.opacity(0.08) : hovering ? Color.primary.opacity(0.05) : .clear))
        .contentShape(Rectangle())
        .onTapGesture { model.useStemSong(song) }
        .onHover { h in withAnimation(.easeOut(duration: 0.12)) { hovering = h } }
        .help("이 곡의 보컬로 조교하기")
    }

    private var detail: String {
        var parts: [String] = []
        if let d = song.duration { parts.append(formatTime(d)) }
        parts.append(song.lyrics ? "음표 · 가사 있음" : song.notes ? "음표 있음" : "보컬")
        return parts.joined(separator: " · ")
    }
}

// MARK: - Main

struct MainView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                LyricsSection(model: model)
                OptionsSection(model: model)
                actionRow
                if let job = model.job { JobCard(job: job) }
                if let error = model.jobError { errorCard(error) }
                if let item = model.selected { ResultSection(item: item, model: model) }
            }
            .padding(.horizontal, 28)
            .padding(.top, 38)
            .padding(.bottom, 28)
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .scrollIndicators(.automatic)
    }

    private var actionRow: some View {
        HStack(spacing: 12) {
            Button { model.tune() } label: {
                PrimaryCapsule(title: "자동 조교 시작", icon: "wand.and.stars", trailing: "⌘⏎", enabled: model.canTune)
            }
            .buttonStyle(PressableStyle())
            .disabled(!model.canTune)
            .keyboardShortcut(.return, modifiers: .command)
            Text(hint)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var hint: String {
        if model.server != .ready { return "엔진이 준비되면 시작할 수 있어요" }
        if model.source == nil { return "왼쪽에서 보컬 파일이나 StemMixer 곡을 고르세요" }
        if model.status?.singer == nil { return "OpenUtau에 테토 음원이 없어서 가수는 OpenUtau에서 골라야 해요" }
        return "결과는 OpenUtau 프로젝트(.ustx)로 저장돼요"
    }

    private func errorCard(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.system(size: 12, weight: .medium))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button { withAnimation { model.jobError = nil } } label: {
                Image(systemName: "xmark").font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.red.opacity(0.1)))
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

/// A rounded card holding one section of the main view.
struct Card<Content: View>: View {
    let title: String
    let icon: String
    var trailing: AnyView?
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(title, systemImage: icon)
                    .font(.system(size: 13, weight: .bold))
                Spacer()
                if let trailing { trailing }
            }
            content
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Color.primary.opacity(0.04)))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Color.primary.opacity(0.06), lineWidth: 1))
    }
}

private struct LyricsSection: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Card(title: "가사 (발음)", icon: "text.quote",
             trailing: AnyView(Text(count).font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary))) {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $model.lyrics)
                    .font(.system(size: 14, weight: .medium))
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(minHeight: 120, maxHeight: 220)
                if model.lyrics.isEmpty {
                    Text("한 줄에 한 소절씩 적어 주세요 (숨 쉬는 곳에서 줄 바꿈)\n예) あいたいよ  또는  보고 싶어\n비워 두면 음 높이만 따라가고 가사는 「ら」로 넣어요")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                        .allowsHitTesting(false)
                }
            }
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(nsColor: .textBackgroundColor).opacity(0.6)))
            Text("히라가나 · 가타카나 · 한글 · 로마자를 읽어요. 장음 「ー」나 「+」는 앞 글자를 늘여요. 한글 가사는 일본어 음원에 맞게 바꿔 불러요.")
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                FileChip(title: "음표 파일", placeholder: "없으면 보컬에서 음을 읽어요", icon: "pianokeys",
                         url: model.notesFile, choose: { model.chooseNotes() }) { model.notesFile = nil }
                FileChip(title: "반주", placeholder: stemBacking ? "StemMixer 반주를 넣어요" : "있으면 같이 넣어요",
                         icon: "speaker.wave.2", url: model.backingFile, choose: { model.chooseBacking() }) { model.backingFile = nil }
            }
        }
    }

    private var stemBacking: Bool {
        if case .stem = model.source { return true }
        return false
    }

    private var count: String {
        let n = model.lyricLines
        return n == 0 ? "비어 있음" : "\(n)줄"
    }
}

private struct FileChip: View {
    let title: String
    let placeholder: String
    let icon: String
    let url: URL?
    let choose: () -> Void
    let clear: () -> Void

    var body: some View {
        Button(action: choose) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(url == nil ? Color.secondary : Color.teto)
                VStack(alignment: .leading, spacing: 0) {
                    Text(title).font(.system(size: 11, weight: .bold))
                    Text(url?.lastPathComponent ?? placeholder)
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
                if url != nil {
                    Button(action: clear) { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                        .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .frame(maxWidth: .infinity)
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.primary.opacity(0.05)))
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
    }
}

private struct OptionsSection: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Card(title: "조교 설정", icon: "slider.horizontal.3",
             trailing: AnyView(Button("기본값") { withAnimation { model.resetOptions() } }
                .buttonStyle(.plain).font(.system(size: 11, weight: .semibold)).foregroundStyle(Color.teto))) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                row("음 높이 따라가기", "원래 보컬의 꺾기 · 흔들림을 얼마나 따라 할지", $model.options.pitchFollow, 0...1)
                row("비브라토", "빠른 떨림의 세기", $model.options.vibrato, 0...1.5, mark: 1)
                row("음정 보정", "길게 끄는 음 가운데에서 틀린 음정을 바로잡아요", $model.options.correct, 0...1)
                row("셈여림", "크게 · 작게 부른 곳을 따라가요", $model.options.dynamics, 0...1.5, mark: 1)
                row("숨 섞인 소리", "속삭이듯 부른 곳 (BREC)", $model.options.breathiness, 0...1.5, mark: 1)
                row("힘 준 소리", "세게 내지른 곳 (TENC)", $model.options.tension, 0...1.5, mark: 1)
                row("숨소리 크기", "숨 음표의 크기 (원래 숨 크기 기준)", $model.options.breathVolume, 0...2, mark: 1)
            }
            Divider().padding(.vertical, 2)
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("숨소리").font(.system(size: 12, weight: .semibold))
                    Segmented(selection: $model.options.breaths, items: BreathMode.allCases.map { ($0.rawValue, $0.title) })
                    Text(BreathMode(rawValue: model.options.breaths)?.detail ?? "")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("음역").font(.system(size: 12, weight: .semibold))
                    Segmented(selection: $model.options.transpose,
                              items: [("auto", "자동"), ("-12", "−1옥타브"), ("0", "그대로"), ("12", "+1옥타브")])
                    Text("자동: 남자 목소리처럼 낮으면 한 옥타브 올려요")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("빠르기").font(.system(size: 12, weight: .semibold))
                    HStack(spacing: 4) {
                        TextField("자동", value: $model.options.bpm, format: .number.precision(.fractionLength(0...2)))
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 64)
                        Text("BPM").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                    }
                    Text("0 = 음표 파일 빠르기, 없으면 120")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            Divider().padding(.vertical, 2)
            VStack(alignment: .leading, spacing: 6) {
                Text("테토로 불러 보고 다듬기").font(.system(size: 12, weight: .semibold))
                Segmented(selection: $model.options.refine, items: [("auto", "다듬기"), ("off", "안 하기")])
                Text(refineDetail)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var refineDetail: String {
        if model.options.refine == "off" { return "조교만 하고 바로 끝내요 (앱에서 테토 소리는 못 들어요)" }
        if model.status?.openutau == false {
            return "OpenUtau가 응용 프로그램 폴더에 없어서 지금은 건너뛰어요. 넣어 두면 테토 소리로 확인하고 다듬어요."
        }
        return "조교한 뒤 OpenUtau로 테토가 부르게 해서, 끊기거나 음이 어긋난 곳을 원래 보컬과 비교해 고쳐요. 시간이 더 걸려요."
    }

    private func row(_ title: String, _ help: String, _ value: Binding<Double>, _ range: ClosedRange<Double>,
                     mark: Double? = nil) -> some View {
        GridRow {
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.system(size: 12, weight: .semibold))
                Text(help).font(.system(size: 10, weight: .medium)).foregroundStyle(.secondary).lineLimit(1)
            }
            .frame(width: 230, alignment: .leading)
            CapsuleSlider(value: value, range: range, tint: .teto, mark: mark)
            Text("\(Int((value.wrappedValue * 100).rounded()))%")
                .font(.system(size: 11, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
    }
}

/// StemMixer-style pill segmented control.
struct Segmented: View {
    @Binding var selection: String
    let items: [(String, String)]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items, id: \.0) { item in
                let value = item.0
                Button {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) { selection = value }
                } label: {
                    Text(item.1)
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(selection == value ? .primary : .secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            Capsule().fill(selection == value ? Color(nsColor: .controlBackgroundColor) : .clear)
                                .shadow(color: .black.opacity(selection == value ? 0.12 : 0), radius: 2, y: 1)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(Capsule().fill(Color.primary.opacity(0.06)))
        .fixedSize()
    }
}

private struct ResultSection: View {
    let item: ResultItem
    @ObservedObject var model: AppModel
    @State private var retuning = false

    var body: some View {
        Card(title: item.title, icon: "checkmark.seal.fill",
             trailing: AnyView(HStack(spacing: 6) {
                 CircleIconButton(icon: "folder", help: "Finder에서 보기") { model.reveal(item) }
                 Button { model.openInOpenUtau(item) } label: {
                     PrimaryCapsule(title: "OpenUtau에서 열기", icon: "arrow.up.forward.app")
                 }
                 .buttonStyle(PressableStyle())
             })) {
            HStack(spacing: 16) {
                stat("음표", item.notes.map { "\($0)개" } ?? "–")
                stat("숨", item.breaths.map { "\($0)번" } ?? "–")
                stat("옮김", item.transpose.map { $0 == 0 ? "그대로" : "\($0 > 0 ? "+" : "")\($0)반음" } ?? "–")
                stat("가수", item.singer ?? "OpenUtau에서 고르기")
                Spacer()
            }
            if let p = model.preview {
                TransportBar(model: model, player: model.player)
                PreviewView(preview: p, before: model.before, compareRange: model.compareRange,
                            selectedNote: $model.selectedNote, selection: $model.selection, focus: $model.focusTime,
                            player: model.player)
                    .frame(height: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                legend
                if let job = model.resultJob {
                    JobCard(job: job)
                } else if let i = model.selectedNote, i < p.notes.count {
                    NoteEditorBar(model: model, preview: p, index: i)
                } else if let r = model.selection {
                    RangeBar(model: model, range: r, retuning: $retuning)
                } else {
                    Text("음표를 누르면 가사 · 음절을 고칠 수 있고, 끌어서 구간을 고르면 그 구간만 다시 조교할 수 있어요. 빈 곳을 누르면 거기부터 들어요.")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                FlagList(model: model, flags: p.marks)
            } else if model.loadingPreview {
                ProgressView().controlSize(.small).frame(maxWidth: .infinity, minHeight: 120)
            }
            ForEach(item.warnings, id: \.self) { w in
                Label(w, systemImage: "lightbulb")
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text("OpenUtau에서 열면 테토 트랙 아래에 원래 보컬(음소거) 트랙이 있어요. 비교해 들을 때만 켜 보세요. 여기서 고친 뒤에는 OpenUtau에서 파일을 다시 열어 주세요.")
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .sheet(isPresented: $retuning) {
            if let r = model.selection { RetuneSheet(model: model, range: r) }
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 13, weight: .semibold)).lineLimit(1)
        }
    }

    private var legend: some View {
        HStack(spacing: 14) {
            swatch(Color.source.opacity(0.6), "원래 보컬 음 높이")
            swatch(Color.teto, "테토 조교 음 높이")
            swatch(Color.primary.opacity(0.25), "음표")
            swatch(Color.green.opacity(0.5), "숨")
            swatch(Color.orange.opacity(0.8), "확인할 곳")
            if model.compareRange != nil { swatch(Color.primary.opacity(0.45), "바꾸기 전 (점선)") }
            Spacer()
        }
        .font(.system(size: 10.5, weight: .medium))
        .foregroundStyle(.secondary)
    }

    private func swatch(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            Capsule().fill(color).frame(width: 14, height: 4)
            Text(label)
        }
    }
}

struct JobCard: View {
    let job: Job

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 1) {
                    Text(job.step)
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(2)
                    Text(job.label)
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
                if job.progress > 0 {
                    Text("\(Int(job.progress * 100))%")
                        .font(.system(size: 11, weight: .bold))
                        .monospacedDigit()
                        .foregroundStyle(Color.teto)
                        .contentTransition(.numericText())
                }
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.08))
                    Capsule()
                        .fill(LinearGradient(colors: [.teto.opacity(0.7), .teto],
                                             startPoint: .leading, endPoint: .trailing))
                        .frame(width: max(6, geo.size.width * job.progress))
                }
            }
            .frame(height: 6)
            .animation(.easeOut(duration: 0.4), value: job.progress)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.teto.opacity(0.1)))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.teto.opacity(0.3), lineWidth: 1))
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}
