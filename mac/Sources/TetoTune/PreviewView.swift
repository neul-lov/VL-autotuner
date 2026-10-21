import SwiftUI

/// Piano-roll style picture of a tuning: notes with lyrics, the original vocal's pitch (blue),
/// the pitch Teto will sing (crimson) and the breaths (green). Scrolls sideways for long songs.
struct PreviewView: View {
    let preview: Preview
    /// Horizontal zoom (points per second).
    @State private var zoom: Double = 70

    var body: some View {
        GeometryReader { geo in
            // Capped so a long song zoomed in doesn't make one huge drawing layer.
            let width = max(geo.size.width, min(preview.duration * zoom, 16000))
            ScrollView(.horizontal) {
                Canvas { context, size in draw(context, size) }
                    .frame(width: width, height: geo.size.height)
            }
            .scrollIndicators(.visible)
            .overlay(alignment: .topTrailing) { zoomButtons }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
    }

    private var zoomButtons: some View {
        HStack(spacing: 4) {
            CircleIconButton(icon: "minus.magnifyingglass", help: "작게 보기", size: 24) { zoom = max(20, zoom / 1.5) }
            CircleIconButton(icon: "plus.magnifyingglass", help: "크게 보기", size: 24) { zoom = min(400, zoom * 1.5) }
        }
        .padding(8)
    }

    /// Lowest and highest semitone shown, with a little room around the notes and the pitch line.
    private var range: (lo: Double, hi: Double) {
        var values = preview.notes.map { Double($0.tone) }
        values += preview.source.compactMap { $0 }
        guard let lo = values.min(), let hi = values.max() else { return (55, 79) }
        let mid = (lo + hi) / 2
        let half = max(6, (hi - lo) / 2 + 2)
        return (mid - half, mid + half)
    }

    private func draw(_ context: GraphicsContext, _ size: CGSize) {
        let (lo, hi) = range
        let x = { (t: Double) -> CGFloat in CGFloat(t / max(preview.duration, 0.1)) * size.width }
        let y = { (m: Double) -> CGFloat in size.height - CGFloat((m - lo) / (hi - lo)) * size.height }
        let rowHeight = size.height / CGFloat(hi - lo)

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

        // Breaths.
        for b in preview.breaths where b.count >= 2 {
            let rect = CGRect(x: x(b[0]), y: 0, width: max(2, x(b[1]) - x(b[0])), height: size.height)
            context.fill(Path(rect), with: .color(.green.opacity(0.14)))
            context.draw(Text("숨").font(.system(size: 9, weight: .bold)).foregroundColor(.green),
                         at: CGPoint(x: rect.midX, y: 10))
        }

        // Notes and lyrics.
        for n in preview.notes {
            let rect = CGRect(x: x(n.start), y: y(Double(n.tone) + 0.5),
                              width: max(2, x(n.end) - x(n.start) - 1), height: max(4, rowHeight))
            let shape = Path(roundedRect: rect, cornerRadius: min(4, rect.height / 2))
            context.fill(shape, with: .color(.primary.opacity(n.lyric == "+" ? 0.08 : 0.16)))
            if rect.width > 12, n.lyric != "+" {
                context.draw(Text(n.lyric).font(.system(size: 10, weight: .semibold)).foregroundColor(.primary),
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

        // Tuned pitch: crimson, only where a note is sung.
        var tuned = Path()
        for n in preview.notes {
            let a = Int(n.start / step), b = min(preview.pitch.count, Int(n.end / step) + 1)
            guard a < b else { continue }
            for i in a..<b {
                let p = CGPoint(x: x(Double(i) * step), y: y(preview.pitch[i]))
                if i == a { tuned.move(to: p) } else { tuned.addLine(to: p) }
            }
        }
        context.stroke(tuned, with: .color(.teto), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
    }
}
