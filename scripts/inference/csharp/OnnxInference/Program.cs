/**
 * C# ONNX Runtime inference runner for the energy measurement study.
 *
 * Energy/RAPL measurement is handled externally by scripts/measure.py.
 * This program only performs inference and signals the measurement boundary.
 *
 * Usage (called by measure.py):
 *   ./OnnxInference <model_path> <inputs_dir> <volume> <phase> <model_short>
 *
 * READY protocol:
 *   Cold  — prints "READY" after session load, reads 1 byte stdin, runs inference
 *   Steady— prints "READY" after warmup,       reads 1 byte stdin, runs R×K inference
 *
 * Stderr output (JSON):
 *   {"init_ms": <float|null>, "inference_ms": <float>,
 *    "total_ms": <float|null>}
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// ---------------------------------------------------------------------------
// Wall-clock in ms (high-resolution)
// ---------------------------------------------------------------------------
static double NowMs() =>
    (double)System.Diagnostics.Stopwatch.GetTimestamp() /
    System.Diagnostics.Stopwatch.Frequency * 1000.0;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Signal READY and wait for 1 byte on stdin
// ---------------------------------------------------------------------------
static void SignalReady() {
    Console.WriteLine("READY");
    Console.Out.Flush();
    Console.Read();
}

// ---------------------------------------------------------------------------
// Minimal .npy loader (float32 + int64, C-contiguous, little-endian)
// ---------------------------------------------------------------------------
static NpyArray NpyLoad(string path) {
    var raw = File.ReadAllBytes(path);
    int off = 6; // skip magic
    int major = raw[off++];
    int minor = raw[off++];
    int hlen;
    if (major == 1) { hlen = raw[off] | (raw[off+1] << 8); off += 2; }
    else            { hlen = raw[off] | (raw[off+1]<<8) | (raw[off+2]<<16) | (raw[off+3]<<24); off += 4; }
    string hdr = Encoding.UTF8.GetString(raw, off, hlen);
    off += hlen;

    bool isFloat = true;
    int dtPos = hdr.IndexOf("'descr'"); if (dtPos < 0) dtPos = hdr.IndexOf("\"descr\"");
    if (dtPos >= 0) {
        int lt = hdr.IndexOf('<', dtPos), gt = hdr.IndexOf('>', dtPos), eq = hdr.IndexOf('=', dtPos);
        int q = new[]{ lt < 0 ? int.MaxValue : lt,
                       gt < 0 ? int.MaxValue : gt,
                       eq < 0 ? int.MaxValue : eq }.Min();
        if (q < int.MaxValue && q+1 < hdr.Length && (hdr[q+1]=='i'||hdr[q+1]=='u'))
            isFloat = false;
    }

    long[] shape;
    int shpPos = hdr.IndexOf("'shape'"); if (shpPos < 0) shpPos = hdr.IndexOf("\"shape\"");
    if (shpPos >= 0) {
        int lp = hdr.IndexOf('(', shpPos), rp = hdr.IndexOf(')', shpPos);
        string inner = hdr.Substring(lp+1, rp-lp-1).Trim();
        shape = inner.Split(',')
            .Select(s => s.Trim()).Where(s => !string.IsNullOrEmpty(s))
            .Select(long.Parse).ToArray();
    } else { shape = Array.Empty<long>(); }

    long nelems = shape.Length == 0 ? 0 : shape.Aggregate(1L, (a,b)=>a*b);
    var arr = new NpyArray { Shape = shape, IsFloat = isFloat };
    if (isFloat) {
        arr.FloatData = new float[nelems];
        Buffer.BlockCopy(raw, off, arr.FloatData, 0, (int)nelems * 4);
    } else {
        arr.LongData = new long[nelems];
        Buffer.BlockCopy(raw, off, arr.LongData, 0, (int)nelems * 8);
    }
    return arr;
}

static DenseTensor<float> ToFloatTensor(NpyArray arr) =>
    new DenseTensor<float>(arr.FloatData, arr.Shape.Select(d=>(int)d).ToArray());

static DenseTensor<long> ToLongTensor(NpyArray arr) =>
    new DenseTensor<long>(arr.LongData, arr.Shape.Select(d=>(int)d).ToArray());

static DenseTensor<long> OnesLike(NpyArray rf) {
    long[] ones = new long[rf.LongData!.Length];
    Array.Fill(ones, 1L);
    return new DenseTensor<long>(ones, rf.Shape.Select(d=>(int)d).ToArray());
}

static DenseTensor<long> ZerosLike(NpyArray rf) {
    long[] zeros = new long[rf.LongData!.Length];
    return new DenseTensor<long>(zeros, rf.Shape.Select(d=>(int)d).ToArray());
}

// ---------------------------------------------------------------------------
// Report timing JSON to stderr
// ---------------------------------------------------------------------------
static void Report(double? initMs, double inferenceMs, double? totalMs) {
    string Fm(double v) => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);

    var sb = new StringBuilder("{");
    sb.Append(initMs.HasValue ? $"\"init_ms\": {Fm(initMs.Value)}" : "\"init_ms\": null");
    sb.Append($", \"inference_ms\": {Fm(inferenceMs)}");
    sb.Append(totalMs.HasValue ? $", \"total_ms\": {Fm(totalMs.Value)}" : ", \"total_ms\": null");
    sb.Append("}");
    Console.Error.WriteLine(sb);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
if (args.Length != 5) {
    Console.Error.WriteLine("Usage: Program <model_path> <inputs_dir> <volume> <phase> <model_short>");
    return 1;
}

string modelPath  = args[0], inputsDir = args[1];
int    K          = int.Parse(args[2]);
string phase      = args[3];
string modelShort = args[4];

bool isGpt2   = modelShort == "gpt2";
bool isFc     = modelShort == "fashionclip";
bool isSteady = phase == "steady";
int  warmup   = isSteady ? 300 : 0;
int  estMs    = modelShort == "minilm" ? 15 : modelShort == "gpt2" ? 70 : modelShort == "fashionclip" ? 100 : 80;
int  R        = isSteady ? Math.Max(1, (100 + K*estMs - 1) / (K*estMs)) : 1;

NpyArray[] am = new NpyArray[10], at = new NpyArray[10];
for (int i = 0; i < 10; i++) {
    if (isFc) { am[i] = NpyLoad($"{inputsDir}/fashionclip/image_{i}.npy");
                at[i] = NpyLoad($"{inputsDir}/fashionclip/text_{i}.npy"); }
    else      { am[i] = NpyLoad($"{inputsDir}/{modelShort}/{i}.npy"); }
}

var so = new SessionOptions {
    IntraOpNumThreads = 1, InterOpNumThreads = 1,
    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
};

double tStart=0, tReady=0, tI0=0, tI1=0;

if (!isSteady) tStart = NowMs();
using var session = new InferenceSession(modelPath, so);
if (!isSteady) tReady = NowMs();

var inpNames = session.InputMetadata.Keys.ToList();
var outNames = session.OutputMetadata.Keys.ToList();

int iPixel=-1, iIds=-1, iMask=-1, iTtype=-1;
for (int i=0;i<inpNames.Count;i++) {
    var n = inpNames[i].ToLower();
    if (n.Contains("pixel"))      iPixel = i;
    if (n.Contains("input_id"))   iIds   = i;
    if (n.Contains("attention")||n.Contains("mask")) iMask = i;
    if (n.Contains("token_type")) iTtype = i;
}

List<NamedOnnxValue> MakeFeed(int j) {
    var feed = new List<NamedOnnxValue>();
    for (int i=0;i<inpNames.Count;i++) {
        var nm = inpNames[i];
        if (i==iPixel)  feed.Add(NamedOnnxValue.CreateFromTensor(nm, ToFloatTensor(am[j])));
        else if (i==iIds)   { var src = isFc&&at[j]!=null?at[j]:am[j]; feed.Add(NamedOnnxValue.CreateFromTensor(nm, ToLongTensor(src))); }
        else if (i==iMask)  { var rf = isFc&&at[j]!=null?at[j]:am[j]; feed.Add(NamedOnnxValue.CreateFromTensor(nm, OnesLike(rf))); }
        else if (i==iTtype) { feed.Add(NamedOnnxValue.CreateFromTensor(nm, ZerosLike(am[j]))); }
        else { if (am[j].IsFloat) feed.Add(NamedOnnxValue.CreateFromTensor(nm, ToFloatTensor(am[j]))); else feed.Add(NamedOnnxValue.CreateFromTensor(nm, ToLongTensor(am[j]))); }
    }
    return feed;
}

void RunLoop(int count) {
    if (isGpt2) {
        string idName = inpNames.FirstOrDefault(n => n.ToLower().Contains("input_id")) ?? inpNames[0];
        string? maskName = inpNames.FirstOrDefault(n => {
            var ln = n.ToLower();
            return ln.Contains("attention") || ln.Contains("mask");
        });
        string? ttypeName = inpNames.FirstOrDefault(n => n.ToLower().Contains("token_type"));

        for (int j=0;j<count;j++) {
            var ids = ((long[])am[j%10].LongData!.Clone()).ToList();
            for (int t=0;t<20;t++) {
                long[] arr = ids.ToArray();
                var tensor = new DenseTensor<long>(arr, new[]{1,arr.Length});
                var feed = new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor(idName, tensor)};
                if (maskName != null) {
                    var mask = Enumerable.Repeat(1L, arr.Length).ToArray();
                    var maskTensor = new DenseTensor<long>(mask, new[]{1, arr.Length});
                    feed.Add(NamedOnnxValue.CreateFromTensor(maskName, maskTensor));
                }
                if (ttypeName != null) {
                    var ttype = new long[arr.Length];
                    var ttypeTensor = new DenseTensor<long>(ttype, new[]{1, arr.Length});
                    feed.Add(NamedOnnxValue.CreateFromTensor(ttypeName, ttypeTensor));
                }
                using var res = session.Run(feed);
                var logits = res.First().AsTensor<float>().ToArray();
                int vocab = logits.Length/arr.Length, base_ = (arr.Length-1)*vocab, best=0;
                for (int v=1;v<vocab;v++) if (logits[base_+v]>logits[base_+best]) best=v;
                ids.Add((long)best);
            }
        }
    } else {
        for (int j=0;j<count;j++) { using var res = session.Run(MakeFeed(j%10)); }
    }
}

if (isSteady) {
    RunLoop(warmup);
    SignalReady();                    // measure.py reads RAPL_0 here
    tI0 = NowMs();
    for (int r=0;r<R;r++) RunLoop(K);
    tI1 = NowMs();
    double inferMs = (tI1 - tI0) / R;
    Report(null, inferMs, null);
} else {
    SignalReady();                    // measure.py reads RAPL_mid here
    tI0 = NowMs();
    RunLoop(K);
    tI1 = NowMs();
    double initMs  = tReady - tStart;
    double inferMs = tI1 - tI0;
    double totalMs = tI1 - tStart;
    Report(initMs, inferMs, totalMs);
}

return 0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
class NpyArray {
    public float[]? FloatData;
    public long[]?  LongData;
    public long[]   Shape = Array.Empty<long>();
    public bool     IsFloat;
}
