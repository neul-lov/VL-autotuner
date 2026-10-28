// AutoTuner가 사용자의 OpenUtau로 조교 결과를 불러 보는(렌더하는) 작은 프로그램. 소리는 재생하지 않고 wav로만 쓴다.
// OpenUtau 프로그램 폴더를 복사한 곳에서 OpenUtau.dll 자리에 놓고 돌린다 (tetotune/openutau.py).
// OpenUtau의 어느 버전이든 쓰도록 OpenUtau 코드는 이름으로만 찾아 부른다 (컴파일할 때 OpenUtau를 참조하지 않는다).
//
// 사용: OpenUtau RenderHost.dll <project.ustx> <render.wav>   → <render>_<트랙 이름>.wav, 끝나면 "rendered"
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

static class RenderHost {
    static Assembly core;

    static Type[] types;

    // 이름공간이 버전마다 옮겨 다녀서, 전체 이름으로 없으면 마지막 이름으로 찾는다
    static Type T(string name) {
        var t = core.GetType(name);
        if (t != null) return t;
        if (types == null) {
            try { types = core.GetTypes(); } catch (ReflectionTypeLoadException e) { types = e.Types.Where(x => x != null).ToArray(); }
        }
        var shortName = name.Substring(name.LastIndexOf('.') + 1);
        t = types.FirstOrDefault(x => x.Name == shortName && x.IsPublic);
        if (t == null) throw new Exception("OpenUtau에서 " + name + "을(를) 찾지 못했어요");
        return t;
    }

    static object Inst(string type) {
        var t = T(type);
        var p = t.GetProperty("Inst", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (p != null) return p.GetValue(null);
        var f = t.GetField("Inst", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (f != null) return f.GetValue(null);
        throw new Exception(type + ".Inst가 없어요");
    }

    static object Call(object target, string method, params object[] args) {
        var type = target as Type ?? target.GetType();
        var flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.FlattenHierarchy;
        // 버전마다 뒤에 기본값 있는 인자가 붙기도 해서: 앞 인자가 맞고 나머지는 기본값이 있는 것
        var m = type.GetMethods(flags).Where(x => x.Name == method).Select(x => (x, ps: x.GetParameters()))
            .Where(c => c.ps.Length >= args.Length && c.ps.Skip(args.Length).All(p => p.HasDefaultValue)
                && c.ps.Take(args.Length).Select((p, i) => args[i] == null || p.ParameterType.IsInstanceOfType(args[i])).All(ok => ok))
            .OrderBy(c => c.ps.Length).FirstOrDefault();
        if (m.x == null) throw new Exception(type.Name + "." + method + "을(를) 찾지 못했어요");
        var full = m.ps.Select((p, i) => i < args.Length ? args[i] : p.DefaultValue).ToArray();
        return m.x.Invoke(target is Type ? null : target, full);
    }

    static object Get(object target, string name) {
        var t = target.GetType();
        var p = t.GetProperty(name);
        if (p != null) return p.GetValue(target);
        var f = t.GetField(name);
        if (f != null) return f.GetValue(target);
        throw new Exception(t.Name + "." + name + "이(가) 없어요");
    }

    static void Set(object target, string name, object value) {
        var t = target.GetType();
        var p = t.GetProperty(name);
        if (p != null) { p.SetValue(target, value); return; }
        var f = t.GetField(name);
        if (f != null) { f.SetValue(target, value); return; }
        throw new Exception(t.Name + "." + name + "이(가) 없어요");
    }

    static int Main(string[] args) {
        if (args.Length < 2) {
            Console.Error.WriteLine("사용: RenderHost <project.ustx> <render.wav>");
            return 64;
        }
        try {
            return Run(args[0], args[1]);
        } catch (Exception e) {
            var inner = e is TargetInvocationException && e.InnerException != null ? e.InnerException : e;
            Console.Error.WriteLine("render_error: " + inner.GetType().Name + ": " + inner.Message);
            return 1;
        }
    }

    static int Run(string ustx, string outWav) {
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        core = Assembly.Load("OpenUtau.Core");
        Console.WriteLine("openutau_core=" + core.GetName().Version);
        try {  // OpenUtau 기록은 경고만 (없으면 그냥 둔다)
            var serilog = Assembly.Load("Serilog");
            var config = Activator.CreateInstance(serilog.GetType("Serilog.LoggerConfiguration"));
            var min = Get(config, "MinimumLevel");
            config = Call(min, "Warning");
            var sinks = Assembly.Load("Serilog.Sinks.Console");
            var write = Get(config, "WriteTo");
            var ext = sinks.GetTypes().First(t => t.Name == "ConsoleLoggerConfigurationExtensions");
            var console = ext.GetMethods().First(m => m.Name == "Console" && m.GetParameters()[0].ParameterType.IsInstanceOfType(write));
            var ps = console.GetParameters();
            config = console.Invoke(null, ps.Select((p, i) => i == 0 ? write : p.HasDefaultValue ? p.DefaultValue : null).ToArray());
            var logger = Call(config, "CreateLogger");
            serilog.GetType("Serilog.Log").GetProperty("Logger").SetValue(null, logger);
        } catch (Exception) { }

        // 렌더 캐시는 AutoTuner 전용 폴더로 (TETOTUNE_OU_CACHE): 소리 없는 렌더 때 캐시를 비워도 사용자 OpenUtau 캐시는 그대로
        var cache = Environment.GetEnvironmentVariable("TETOTUNE_OU_CACHE");
        if (!string.IsNullOrEmpty(cache)) {
            try {
                System.IO.Directory.CreateDirectory(cache);
                var pm = Inst("OpenUtau.Core.PathManager");
                var prop = pm.GetType().GetProperty("CachePath", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                prop.GetSetMethod(true).Invoke(pm, new object[] { cache });
                Console.Error.WriteLine("render_cache: " + Get(pm, "CachePath"));
            } catch (Exception e) {
                Console.Error.WriteLine("render_cache_default: " + e.Message);
            }
        }
        Call(Inst("OpenUtau.Core.ToolsManager"), "Initialize");
        var singers = Inst("OpenUtau.Core.SingerManager");
        Call(singers, "Initialize");
        var docs = Inst("OpenUtau.Core.DocManager");
        Call(docs, "Initialize", Thread.CurrentThread, TaskScheduler.Default);
        var queue = new BlockingCollection<Action>();
        Set(docs, "PostOnUIThread", (Action<Action>)(a => queue.Add(a)));
        void Pump(Func<bool> done, double seconds) {
            var t0 = DateTime.Now;
            while (!done() && (DateTime.Now - t0).TotalSeconds < seconds) {
                if (queue.TryTake(out var a, 20)) a();
            }
        }
        Call(singers, "SearchAllSingers");

        var project = Call(T("OpenUtau.Core.Format.Ustx"), "Load", ustx);
        var notification = Activator.CreateInstance(T("OpenUtau.Core.LoadProjectNotification"), project);
        Call(docs, "ExecuteCmd", notification);
        var tracks = (IList)Get(project, "tracks");
        var track = tracks[0];
        var singer = Get(track, "Singer");
        Console.WriteLine("singer=" + (singer == null ? "null" : Get(singer, "Id")) + " found=" + (singer == null ? "False" : Get(singer, "Found")));
        try {  // 소리가 안 날 때 까닭을 찾으려고: 음소 변환기 · 렌더러
            var ph = Get(track, "Phonemizer");
            var rs = Get(track, "RendererSettings");
            Console.WriteLine("phonemizer=" + (ph == null ? "null" : ph.GetType().Name) + " renderer=" + Get(rs, "renderer"));
            Console.WriteLine("resampler=" + (Get(rs, "resampler") ?? "null"));
        } catch (Exception) { }
        var voicePart = T("OpenUtau.Core.Ustx.UVoicePart");
        var part = ((IEnumerable)Get(project, "parts")).Cast<object>().First(p => voicePart.IsInstanceOfType(p));
        Pump(() => ((ICollection)Get(part, "phonemes")).Count > 0 && (bool)Get(part, "PhonemesUpToDate"), 30);
        var phonemes = ((IEnumerable)Get(part, "phonemes")).Cast<object>().ToList();
        int errors = phonemes.Count(p => (bool)Get(p, "Error"));
        Console.WriteLine("phonemes=" + phonemes.Count + " phoneme_errors=" + errors);

        var task = (Task)Call(Inst("OpenUtau.Core.PlaybackManager"), "RenderToFiles", project, outWav);
        Pump(() => task.IsCompleted, 3600);
        while (queue.TryTake(out var rest)) rest();
        if (!task.IsCompleted) throw new Exception("렌더가 끝나지 않았어요");
        if (task.IsFaulted) throw task.Exception.GetBaseException();
        // 예전 OpenUtau(2025년 판 등)는 RenderToFiles가 렌더를 뒤에서 돌려 놓고 바로 끝난다: 파일이 다 써질 때까지 기다린다
        var outDir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(outWav));
        long Size() => System.IO.Directory.GetFiles(outDir, "*.wav").Sum(f => new System.IO.FileInfo(f).Length);
        Pump(() => Size() > 0, 180);
        for (long last = -1, now = Size(); now != last; last = now, now = Size()) {
            Pump(() => false, 2);
        }
        try {  // 무엇을 내보냈는지 (트랙이 음소거됐거나 렌더러가 없으면 파일이 안 생긴다)
            int i = 0;
            foreach (var t in tracks) {
                object rs = null;
                try { rs = Get(Get(t, "RendererSettings"), "renderer"); } catch (Exception) { }
                Console.WriteLine("track" + i++ + " muted=" + Get(t, "Muted") + " renderer=" + (rs ?? "null"));
            }
            var dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(outWav));
            var files = System.IO.Directory.GetFiles(dir, "*.wav");
            Console.WriteLine("exported=" + files.Length + " " + string.Join(" | ", files.Select(f => System.IO.Path.GetFileName(f) + ":" + new System.IO.FileInfo(f).Length)));
        } catch (Exception e) {
            Console.WriteLine("exported=? " + e.Message);
        }
        Console.WriteLine("rendered");
        return 0;
    }
}
