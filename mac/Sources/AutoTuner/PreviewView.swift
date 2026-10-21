import SwiftUI

/// Piano-roll style picture of a tuning: notes with lyrics, the original vocal's pitch (blue),
/// the pitch Teto will sing (crimson) and the breaths (green). Scrolls sideways for long songs.
///
/// Places that may be wrong are marked on top (orange: check, red: likely wrong). Click a note to edit it,
/// drag across the roll to pick a section, click empty space to move the playhead there.
struct PreviewView: View {
    let preview: Preview
    /// Before the last section re-tune: its pitch is drawn dashed in `compareRange`, to compare.
    var before: Preview?
    var compareRange: [Double]?
    @Binding var selectedNote: Int?
    @Binding var selection: ClosedRange<Double>?
    /// Set to scroll that time into view (then cleared).
    @Binding var focus: Double?
    /// Not observed here, so the roll isn't redrawn on every playhead tick (PlayheadLine is).
    let player: Player
    /// Horizontal zoom (points per second).
    @State private var zoom: Double = 70
    @State private var dragging = false

    var body: some View {
        GeometryReader { geo in
            // Capped so a long song zoomed in doesn't make one huge drawing layer.
            let width = max(geo.size.width, min(preview.duration * zoom, 16000))
            let map = Mapping(preview: preview, size: CGSize(width: width, height: geo.size.height))
            ScrollViewReader { proxy in
                ScrollView(.horizontal) {
                    ZStack(alignment: .topLeading) {
                        Canvas { context, size in draw(context, map) }
                            .frame(width: width, height: geo.size.height)
                            .contentShape(Rectangle())
                            .gesture(drag(map))
                        anchors(map)
                        PlayheadLine(player: player, map: map)
                    }
                    .frame(width: width, height: geo.size.height)
                }
                .scrollIndicators(.visible)
                .onChange(of: focus) { _, t in
                    guard let t else { return }
                    withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(Self.anchorID(t), anchor: .center) }
                    focus = nil
                }
            }
            .overlay(alignment: .topTrailing) { zoomButtons }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
    }

    /// One invisible point per second, so a time can be scrolled to.
    private func anchors(_ map: Mapping) -> some View {
        ForEach(0...max(0, Int(preview.duration)), id: \.self) { s in
            Color.clear.frame(width: 1, height: 1)
                .id(Self.anchorID(Double(s)))
                .position(x: map.x(Double(s)), y: 1)
        }
        .allowsHitTesting(false)
    }

    private static func anchorID(_ t: Double) -> String { "t\(max(0, Int(t)))" }

    private var zoomButtons: some View {
        HStack(spacing: 4) {
            CircleIconButton(icon: "minus.magnifyingglass", help: "작게 보기", size: 24) { zoom = max(20, zoom / 1.5) }
            CircleIconButton(icon: "plus.magnifyingglass", help: "크게 보기", size: 24) { zoom = min(400, zoom * 1.5) }
        }
        .padding(8)
    }

    /// A click picks a note (or moves the playhead); a drag picks a section.
    private func drag(_ map: Mapping) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { v in
                guard abs(v.translation.width) > 4 || dragging else { return }
                dragging = true
                let a = map.t(min(v.startLocation.x, v.location.x)), b = map.t(max(v.startLocation.x, v.location.x))
                selectedNote = nil
                selection = max(0, a)...min(preview.duration, max(b, a + 0.01))
            }
            .onEnded { v in
                defer { dragging = false }
                guard !dragging else { return }
                if let i = map.note(at: v.location) {
                    selection = nil
                    selectedNote = selectedNote == i ? nil : i
                } else {
                    selectedNote = nil
                    selection = nil
                    player.seek(map.t(v.location.x))
                }
            }
    }

    private func draw(_ context: GraphicsContext, _ map: Mapping) {
        let size = map.size
        let (lo, hi) = (map.lo, map.hi)
        let x = map.x, y = map.y

        // Octave lines (C) and a light line for every semitone.
        for m in Int(lo.rounded(.up))...Int(hi.rounded(.down)) {
            let yy = y(Double(m))
            let isC = m % 12 == 0
            var line = Path()
            line.move(to: CGPoint(x: 0, y: yy))
            line.addLine(to: CGPoint(x: size.width, y: yy))
            context.stroke(line, with: .color(.primary.opacity(isC ? 0.14 : 0.04)), lineWidth: isC ? 1 : 0.5)
            if isC {
                context.draw(Text("C\(m / 12 - 1)").font(.system(size: 9, weight: .bold)).foregroundColor(.secondary),
                             at: CGPoint(x: 4, y: yy - 6), anchor: .leading)
            }
        }

        // Places that may be wrong: a light band and a bar on top.
        for f in preview.marks {
            let color: Color = f.level >= 2 ? .red : .orange
            let rect = CGRect(x: x(f.start), y: 0, width: max(3, x(f.end) - x(f.start)), height: size.height)
            context.fill(Path(rect), with: .color(color.opacity(f.level >= 2 ? 0.08 : 0.05)))
            context.fill(Path(roundedRect: CGRect(x: rect.minX, y: 0, width: rect.width, height: 4), cornerRadius: 2),
                         with: .color(color.opacity(0.8)))
        }

        // The picked section.
        if let s = selection {
            let rect = CGRect(x: x(s.lowerBound), y: 0, width: max(2, x(s.upperBound) - x(s.lowerBound)), height: size.height)
            context.fill(Path(rect), with: .color(Color.source.opacity(0.12)))
            for edge in [rect.minX, rect.maxX] {
                var line = Path()
                line.move(to: CGPoint(x: edge, y: 0))
                line.addLine(to: CGPoint(x: edge, y: size.height))
                context.stroke(line, with: .color(Color.source.opacity(0.7)), lineWidth: 1)
            }
        }

        // Breaths.
        for b in preview.breaths where b.count >= 2 {
            let rect = CGRect(x: x(b[0]), y: 0, width: max(2, x(b[1]) - x(b[0])), height: size.height)
            context.fill(Path(rect), with: .color(.green.opacity(0.14)))
            context.draw(Text("숨").font(.system(size: 9, weight: .bold)).foregroundColor(.green),
                         at: CGPoint(x: rect.midX, y: 12))
        }

        // Notes and lyrics.
        for (i, n) in preview.notes.enumerated() {
            let rect = map.rect(n)
            let shape = Path(roundedRect: rect, cornerRadius: min(4, rect.height / 2))
            let picked = i == selectedNote
            context.fill(shape, with: .color(picked ? Color.teto.opacity(0.3) : .primary.opacity(n.lyric == "+" ? 0.08 : 0.16)))
            if picked { context.stroke(shape, with: .color(.teto), lineWidth: 1.5) }
            if rect.width > 12, n.lyric != "+" {
                context.draw(Text(n.lyric).font(.system(size: 10, weight: .semibold)).foregroundColor(picked ? .teto : .primary),
                             at: CGPoint(x: rect.minX + 3, y: rect.minY - 7), anchor: .leading)
            }
        }

        let step = preview.step

        // Original vocal: blue, broken where it's silent.
        var source = Path()
        var drawing = false
        for (i, v) in preview.source.enumerated() {
            guard let v else { drawing = false; continue }
            let p = CGPoint(x: x(Double(i) * step), y: y(v))
            if drawing { source.addLine(to: p) } else { source.move(to: p) }
            drawing = true
        }
        context.stroke(source, with: .color(Color.source.opacity(0.55)), lineWidth: 1.5)

        // Before the section re-tune: dashed, only in that section.
        if let before, let r = compareRange, r.count == 2 {
            let old = tunedPath(before, x: x, y: y, from: r[0], to: r[1])
            context.stroke(old, with: .color(.primary.opacity(0.45)), style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
        }

        // Tuned pitch: crimson, only where a note is sung.
        context.stroke(tunedPath(preview, x: x, y: y), with: .color(.teto),
                       style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
    }

    private func tunedPath(_ p: Preview, x: (Double) -> CGFloat, y: (Double) -> CGFloat,
                           from: Double = 0, to: Double = .infinity) -> Path {
        var path = Path()
        let step = p.step
        for n in p.notes where n.end > from && n.start < to {
            let a = Int(max(n.start, from) / step), b = min(p.pitch.count, Int(min(n.end, to) / step) + 1)
            guard a < b else { continue }
            for i in a..<b {
                let pt = CGPoint(x: x(Double(i) * step), y: y(p.pitch[i]))
                if i == a { path.move(to: pt) } else { path.addLine(to: pt) }
            }
        }
        return path
    }
}

/// Seconds and semitones ↔ points in the roll.
struct Mapping {
    let preview: Preview
    let size: CGSize
    let lo: Double
    let hi: Double

    init(preview: Preview, size: CGSize) {
        self.preview = preview
        self.size = size
        // Lowest and highest semitone shown, with a little room around the notes and the pitch line.
        var values = preview.notes.map { Double($0.tone) }
        values += preview.source.compactMap { $0 }
        if let a = values.min(), let b = values.max() {
            let mid = (a + b) / 2
            let half = max(6, (b - a) / 2 + 2)
            (lo, hi) = (mid - half, mid + half)
        } else {
            (lo, hi) = (55, 79)
        }
    }

    var rowHeight: CGFloat { size.height / CGFloat(hi - lo) }
    func x(_ t: Double) -> CGFloat { CGFloat(t / max(preview.duration, 0.1)) * size.width }
    func t(_ x: CGFloat) -> Double { Double(x / max(size.width, 1)) * max(preview.duration, 0.1) }
    func y(_ m: Double) -> CGFloat { size.height - CGFloat((m - lo) / (hi - lo)) * size.height }

    func rect(_ n: Preview.Note) -> CGRect {
        CGRect(x: x(n.start), y: y(Double(n.tone) + 0.5), width: max(2, x(n.end) - x(n.start) - 1), height: max(4, rowHeight))
    }

    /// The note under a point (a little taller than drawn, so short notes are easy to hit).
    func note(at p: CGPoint) -> Int? {
        preview.notes.indices.first { rect(preview.notes[$0]).insetBy(dx: 0, dy: -6).contains(p) }
    }
}

/// The playhead: a thin line that follows the player (its own view, so only it redraws while playing).
struct PlayheadLine: View {
    @ObservedObject var player: Player
    let map: Mapping

    var body: some View {
        Rectangle()
            .fill(Color.primary.opacity(player.playing || player.time > 0 ? 0.75 : 0))
            .frame(width: 1.5, height: map.size.height)
            .offset(x: map.x(player.time))
            .allowsHitTesting(false)
    }
}
