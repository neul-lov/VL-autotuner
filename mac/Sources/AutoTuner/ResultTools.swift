import SwiftUI

/// Play / pause, what is heard (original vocal · Teto, with or without the backing) and where the playhead is.
struct TransportBar: View {
    @ObservedObject var model: AppModel
    @ObservedObject var player: Player

    var body: some View {
        HStack(spacing: 10) {
            Button { player.toggle() } label: {
                Image(systemName: player.playing ? "pause.fill" : "play.fill")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill(Color.teto))
            }
            .buttonStyle(PressableStyle())
            .help("재생 · 멈춤")
            .disabled(!player.hasOriginal && !player.hasTeto)

            Text("\(formatTime(player.time)) / \(formatTime(player.duration))")
                .font(.system(size: 11, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(.secondary)

            if player.hasTeto {
                Segmented(selection: Binding(get: { player.voice.rawValue },
                                             set: { player.voice = Player.Voice(rawValue: $0) ?? .original }),
                          items: Player.Voice.allCases.map { ($0.rawValue, $0.title) })
                    .help("같은 자리에서 바꿔 들어 보세요")
            } else {
                Text(model.status?.openutau == true ? "원래 보컬만 들을 수 있어요 (테토로 불러 보기를 하면 테토도 들려요)"
                                                    : "원래 보컬만 들을 수 있어요 (OpenUtau가 있으면 테토로도 들어 볼 수 있어요)")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            if player.hasBacking {
                Toggle("반주 같이", isOn: $player.withBacking)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .font(.system(size: 11, weight: .semibold))
            }
            Spacer(minLength: 0)
            if model.preview?.teto_stale == true, model.status?.openutau == true {
                Button { model.renderTeto() } label: {
                    Label("고친 결과로 테토 다시 듣기", systemImage: "arrow.triangle.2.circlepath")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.teto)
                .help("고친 뒤라 지금 테토 소리는 예전 것이에요")
                .disabled(model.resultJob != nil)
            }
            if (model.preview?.history ?? 0) > 0 {
                Button { model.undo() } label: {
                    Label("되돌리기", systemImage: "arrow.uturn.backward")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("마지막으로 고친 것을 되돌려요")
                .disabled(model.editing || model.resultJob != nil)
            }
        }
    }
}

/// The clicked note: listen to it, see why it was flagged, and hand it to EasyUtau to fix by hand.
struct NoteBar: View {
    @ObservedObject var model: AppModel
    let preview: Preview
    let index: Int

    private var note: Preview.Note { preview.notes[index] }
    private var held: Bool { note.lyric == "+" }
    private var hasEasyUtau: Bool { AppModel.easyUtauApp != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("음표 \(index + 1)")
                    .font(.system(size: 12, weight: .bold))
                Text("\(formatTime(note.start)) · \(held ? "앞 음절을 늘여 불러요 (+)" : "「\(note.lyric)」")")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                CircleIconButton(icon: "play.fill", help: "이 음표 들어 보기", size: 24) {
                    model.play(note.start, note.end)
                }
                CircleIconButton(icon: "xmark", help: "닫기", size: 24) { model.selectedNote = nil }
            }
            HStack(spacing: 8) {
                Button {
                    model.selected.map { model.openInEasyUtau($0, at: note.start) }
                } label: {
                    PrimaryCapsule(title: hasEasyUtau ? "EasyUtau에서 고치기" : "EasyUtau 받기",
                                   icon: hasEasyUtau ? "slider.horizontal.3" : "arrow.down.circle")
                }
                .buttonStyle(PressableStyle())
                .disabled(model.selected == nil)
                .help(hasEasyUtau
                      ? "이 음표를 EasyUtau에서 열어요. 가사 · 음 높이 · 떨림 · 발음을 손으로 고치고, 들어 보면서 바로 확인할 수 있어요"
                      : "음표를 손으로 고치는 앱이에요. 받아서 응용 프로그램 폴더에 넣으면 여기서 바로 넘어가요")
                Text(hasEasyUtau
                     ? "가사 · 음 높이 · 떨림 · 발음은 EasyUtau에서 고쳐요 (같은 project.ustx를 쓰니 고친 건 그대로 남아요)"
                     : "자동 조교 결과를 손으로 다듬으려면 EasyUtau가 필요해요")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            let reasons = preview.marks.filter { $0.start < note.end && $0.end > note.start }
            ForEach(reasons, id: \.self) { f in
                Label(f.reason, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(f.level >= 2 ? .red : .orange)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.teto.opacity(0.07)))
    }
}

/// The section picked on the roll: listen to it, or tune it again.
struct RangeBar: View {
    @ObservedObject var model: AppModel
    let range: ClosedRange<Double>
    @Binding var retuning: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "selection.pin.in.out").foregroundStyle(Color.source)
            Text("고른 구간 \(formatTime(range.lowerBound)) – \(formatTime(range.upperBound))")
                .font(.system(size: 12, weight: .semibold))
                .monospacedDigit()
            Button { model.play(range.lowerBound, range.upperBound) } label: {
                Label("들어 보기", systemImage: "play.fill").font(.system(size: 11, weight: .semibold))
            }
            .controlSize(.small)
            Button { retuning = true } label: {
                Label("이 구간 다시 조교하기…", systemImage: "wand.and.stars").font(.system(size: 11, weight: .semibold))
            }
            .controlSize(.small)
            .disabled(model.resultJob != nil || model.editing)
            if model.compareRange != nil {
                Button { withAnimation { model.before = nil; model.compareRange = nil } } label: {
                    Text("점선(바꾸기 전) 지우기").font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            Spacer()
            CircleIconButton(icon: "xmark", help: "구간 지우기", size: 24) { model.selection = nil }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.source.opacity(0.08)))
    }
}

/// "확인할 곳": the marked places as a list, so each can be jumped to and heard.
struct FlagList: View {
    @ObservedObject var model: AppModel
    let flags: [Preview.Flag]
    @State private var showAll = false

    var body: some View {
        if flags.isEmpty {
            Label("틀렸을 만한 곳을 찾지 못했어요", systemImage: "checkmark.circle")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("확인할 곳 \(flags.count)군데")
                        .font(.system(size: 12, weight: .bold))
                    Text("빨강: 틀렸을 가능성이 커요 · 주황: 들어 보세요")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if flags.count > 5 {
                        Button(showAll ? "접기" : "모두 보기") { withAnimation { showAll.toggle() } }
                            .buttonStyle(.plain)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Color.teto)
                    }
                }
                ForEach(Array((showAll ? flags : Array(flags.prefix(5))).enumerated()), id: \.offset) { _, f in
                    row(f)
                }
            }
        }
    }

    private func row(_ f: Preview.Flag) -> some View {
        HStack(spacing: 8) {
            Circle().fill(f.level >= 2 ? Color.red : Color.orange).frame(width: 7, height: 7)
            Text(formatTime(f.start))
                .font(.system(size: 11, weight: .bold))
                .monospacedDigit()
                .frame(width: 40, alignment: .leading)
            Text(f.kindTitle)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.secondary)
                .frame(width: 52, alignment: .leading)
            Text(f.reason)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 4)
            RowAction(icon: "play.fill", help: "이 구간 들어 보기") {
                model.focusTime = f.start
                model.play(f.start, f.end)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            model.focusTime = f.start
            model.selectedNote = nil
            model.selection = f.start...max(f.end, f.start + 0.05)
        }
        .help("누르면 그 구간을 골라요")
    }
}

/// Settings for tuning the picked section again.
struct RetuneSheet: View {
    @ObservedObject var model: AppModel
    let range: ClosedRange<Double>
    @Environment(\.dismiss) private var dismiss
    @State private var opts = TuneOptions()
    @State private var renote = false
    @State private var lyrics = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("구간 다시 조교하기")
                .font(.system(size: 15, weight: .bold))
            Text("\(formatTime(range.lowerBound)) – \(formatTime(range.upperBound)) · 앞뒤 소절 경계까지 넓혀서 그 부분만 바꿔요. 나머지는 그대로예요.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 9) {
                row("음 높이 따라가기", $opts.pitchFollow, 0...1)
                row("비브라토", $opts.vibrato, 0...1.5)
                row("음정 보정", $opts.correct, 0...1)
                row("셈여림", $opts.dynamics, 0...1.5)
                row("숨 섞인 소리", $opts.breathiness, 0...1.5)
                row("힘 준 소리", $opts.tension, 0...1.5)
            }
            Toggle("음표도 다시 나누기", isOn: $renote)
                .font(.system(size: 12, weight: .semibold))
            if renote {
                VStack(alignment: .leading, spacing: 4) {
                    TextField("이 구간 가사 (비우면 지금 가사 그대로)", text: $lyrics, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                    Text("음표 나누기부터 다시 해요. 가사는 숨 쉬는 곳마다 줄을 바꿔 써 주세요.")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("음표 · 가사는 그대로 두고 음 높이 · 크기 곡선만 새 설정으로 다시 만들어요.")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            if model.status?.openutau == true, (model.preview?.options?.refine ?? "auto") != "off" {
                Text("다시 조교한 구간도 테토로 불러 보고 다듬어요.")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("취소") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("다시 조교하기") {
                    let values: [String: Double] = [
                        "pitch_follow": opts.pitchFollow, "vibrato": opts.vibrato, "correct": opts.correct,
                        "dynamics": opts.dynamics, "breathiness": opts.breathiness, "tension": opts.tension,
                    ]
                    let text = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
                    model.retune(range, options: values, lyrics: renote && !text.isEmpty ? text : nil, renote: renote)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 440)
        .onAppear { opts = model.preview?.options ?? model.options }
    }

    private func row(_ title: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        GridRow {
            Text(title).font(.system(size: 12, weight: .semibold)).frame(width: 110, alignment: .leading)
            CapsuleSlider(value: value, range: range, tint: .teto, mark: range.upperBound > 1 ? 1 : nil)
            Text("\(Int((value.wrappedValue * 100).rounded()))%")
                .font(.system(size: 11, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
    }
}
