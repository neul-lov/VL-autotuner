using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using OpenUtau.Core;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Format;
using OpenUtau.Classic;
using Serilog;

// OpenUtau(Mac · 리눅스판)로 ustx를 열어 음소를 확인하고 트랙별 wav로 내보낸다. 소리는 재생하지 않는다.
class Harness {
    static int Main(string[] args) {
        var ustx = args[0];
        var outWav = args[1];
        Serilog.Log.Logger = new Serilog.LoggerConfiguration().MinimumLevel.Warning().WriteTo.Console().CreateLogger();
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        ToolsManager.Inst.Initialize();
        SingerManager.Inst.Initialize();
        DocManager.Inst.Initialize(Thread.CurrentThread, TaskScheduler.Default);
        var queue = new BlockingCollection<Action>();
        DocManager.Inst.PostOnUIThread = a => queue.Add(a);
        void Pump(Func<bool> done) {
            while (!done()) {
                if (queue.TryTake(out var a, 20)) a();
            }
        }
        SingerManager.Inst.SearchAllSingers();
        Console.WriteLine("singers: " + string.Join(", ", SingerManager.Inst.Singers.Keys));
        var project = Ustx.Load(ustx);
        DocManager.Inst.ExecuteCmd(new LoadProjectNotification(project));
        var track = project.tracks[0];
        Console.WriteLine($"singer={(track.Singer?.Id ?? "null")} found={track.Singer?.Found} phonemizer={track.Phonemizer?.GetType().Name} renderer={track.RendererSettings.renderer}");
        var part = project.parts.OfType<UVoicePart>().First();
        var t0 = DateTime.Now;
        Pump(() => (part.phonemes.Count > 0 && part.PhonemesUpToDate) || (DateTime.Now - t0).TotalSeconds > 20);
        Console.WriteLine($"notes={part.notes.Count} phonemes={part.phonemes.Count} curves={string.Join(",", part.curves.Select(c => c.abbr + ":" + c.xs.Count))}");
        int errors = part.phonemes.Count(p => p.Error);
        Console.WriteLine($"phoneme_errors={errors}");
        foreach (var p in part.phonemes.Take(int.Parse(args.Length > 2 ? args[2] : "12"))) {
            Console.WriteLine($"  {p.position} {p.phoneme} oto={(p.oto?.Alias ?? "-")} err={p.Error}");
        }
        var task = PlaybackManager.Inst.RenderToFiles(project, outWav);
        Pump(() => task.IsCompleted);
        while (queue.TryTake(out var a2)) a2();
        Console.WriteLine("rendered");
        return errors > 0 ? 2 : 0;
    }
}
