package com.onnxstudy;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.SessionOptions;

import java.io.*;
import java.nio.file.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.*;

/**
 * Java ONNX Runtime inference runner for the energy measurement study.
 *
 * Energy/RAPL measurement is handled externally by scripts/measure.py.
 * This program only performs inference and signals the measurement boundary.
 *
 * Usage (called by measure.py):
 *   java -XX:CompileThreshold=100 -jar onnx-inference.jar
 *        <model_path> <inputs_dir> <volume> <phase> <model_short>
 *
 * READY protocol:
 *   Cold  — prints "READY" after session load, reads 1 byte stdin, runs inference
 *   Steady— prints "READY" after warmup,       reads 1 byte stdin, runs R×K inference
 *
 * Stderr output (JSON):
 *   {"init_ms": <float|null>, "inference_ms": <float>,
 *    "total_ms": <float|null>}
 */
public class Infer {

    // -----------------------------------------------------------------------
    // Wall-clock in ms
    // -----------------------------------------------------------------------
    private static double nowMs() {
        return System.nanoTime() / 1_000_000.0;
    }

    // -----------------------------------------------------------------------
    // Signal READY and wait for 1 byte on stdin
    // -----------------------------------------------------------------------
    private static void signalReady() throws IOException {
        System.out.println("READY");
        System.out.flush();
        System.in.read();
    }

    // -----------------------------------------------------------------------
    // Minimal .npy loader (float32 + int64, C-contiguous)
    // -----------------------------------------------------------------------
    static class NpyArray {
        final Object data;       // float[] or long[]
        final long[] shape;
        final boolean isFloat;

        NpyArray(Object data, long[] shape, boolean isFloat) {
            this.data = data; this.shape = shape; this.isFloat = isFloat;
        }
    }

    private static NpyArray npyLoad(String path) throws IOException {
        byte[] raw = Files.readAllBytes(Path.of(path));
        int off = 0;

        // magic + version
        off += 6; // \x93NUMPY
        int major = raw[off++] & 0xFF;
        int minor = raw[off++] & 0xFF;

        int hlen;
        if (major == 1) {
            hlen = (raw[off] & 0xFF) | ((raw[off+1] & 0xFF) << 8); off += 2;
        } else {
            hlen = (raw[off] & 0xFF) | ((raw[off+1] & 0xFF) << 8)
                 | ((raw[off+2] & 0xFF) << 16) | ((raw[off+3] & 0xFF) << 24);
            off += 4;
        }
        String hdr = new String(raw, off, hlen, "UTF-8");
        off += hlen;

        // dtype
        boolean isFloat = true;
        int dtPos = hdr.indexOf("'descr'"); if (dtPos < 0) dtPos = hdr.indexOf("\"descr\"");
        if (dtPos >= 0) {
            int lt = hdr.indexOf('<', dtPos), gt = hdr.indexOf('>', dtPos), eq = hdr.indexOf('=', dtPos);
            int q = Math.min(lt < 0 ? Integer.MAX_VALUE : lt,
                    Math.min(gt < 0 ? Integer.MAX_VALUE : gt,
                             eq < 0 ? Integer.MAX_VALUE : eq));
            if (q != Integer.MAX_VALUE && q + 1 < hdr.length()) {
                char c = hdr.charAt(q + 1);
                if (c == 'i' || c == 'u') isFloat = false;
            }
        }

        // shape
        long[] shape;
        int shpPos = hdr.indexOf("'shape'"); if (shpPos < 0) shpPos = hdr.indexOf("\"shape\"");
        if (shpPos >= 0) {
            int lp = hdr.indexOf('(', shpPos), rp = hdr.indexOf(')', shpPos);
            String inner = hdr.substring(lp + 1, rp).trim();
            if (inner.isEmpty()) {
                shape = new long[0];
            } else {
                String[] parts = inner.split(",");
                List<Long> dims = new ArrayList<>();
                for (String p : parts) {
                    p = p.trim();
                    if (!p.isEmpty()) dims.add(Long.parseLong(p));
                }
                shape = dims.stream().mapToLong(Long::longValue).toArray();
            }
        } else {
            shape = new long[]{};
        }

        long nelems = 1;
        for (long d : shape) nelems *= d;

        ByteBuffer buf = ByteBuffer.wrap(raw, off, raw.length - off)
                                   .order(ByteOrder.LITTLE_ENDIAN);
        if (isFloat) {
            float[] data = new float[(int)nelems];
            buf.asFloatBuffer().get(data);
            return new NpyArray(data, shape, true);
        } else {
            long[] data = new long[(int)nelems];
            buf.asLongBuffer().get(data);
            return new NpyArray(data, shape, false);
        }
    }

    // -----------------------------------------------------------------------
    // Build OrtTensor from NpyArray (using Buffer API for ORT 1.21+)
    // -----------------------------------------------------------------------
    private static OnnxTensor makeTensor(OrtEnvironment env, NpyArray arr)
            throws OrtException {
        if (arr.isFloat) {
            FloatBuffer buf = FloatBuffer.wrap((float[])arr.data);
            return OnnxTensor.createTensor(env, buf, arr.shape);
        } else {
            LongBuffer buf = LongBuffer.wrap((long[])arr.data);
            return OnnxTensor.createTensor(env, buf, arr.shape);
        }
    }

    private static OnnxTensor makeOnesLike(OrtEnvironment env, NpyArray ref)
            throws OrtException {
        long n = 1; for (long d : ref.shape) n *= d;
        long[] ones = new long[(int)n];
        Arrays.fill(ones, 1L);
        LongBuffer buf = LongBuffer.wrap(ones);
        return OnnxTensor.createTensor(env, buf, ref.shape);
    }

    private static OnnxTensor makeZerosLike(OrtEnvironment env, NpyArray ref)
            throws OrtException {
        long n = 1; for (long d : ref.shape) n *= d;
        long[] zeros = new long[(int)n];  // Java initialises to 0
        LongBuffer buf = LongBuffer.wrap(zeros);
        return OnnxTensor.createTensor(env, buf, ref.shape);
    }

    // -----------------------------------------------------------------------
    // Helper interface
    // -----------------------------------------------------------------------
    interface ThrowingIntConsumer { void run(int count) throws Exception; }

    // -----------------------------------------------------------------------
    // Main
    // -----------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: Infer <model_path> <inputs_dir> <volume> <phase> <model_short>");
            System.exit(1);
        }

        String modelPath   = args[0];
        String inputsDir   = args[1];
        int    K           = Integer.parseInt(args[2]);
        String phase       = args[3];
        String modelShort  = args[4];

        boolean isgpt2        = modelShort.equals("gpt2");
        boolean isfashionclip = modelShort.equals("fashionclip");
        boolean isSteady      = phase.equals("steady");
        int     warmupCount   = isSteady ? 300 : 0;

        int estMs = 80;
        if (modelShort.equals("minilm")) estMs = 15;
        else if (modelShort.equals("gpt2")) estMs = 70;
        else if (modelShort.equals("fashionclip")) estMs = 100;

        int R = 1;
        if (isSteady) {
            int denom = K * estMs;
            R = Math.max(1, (100 + denom - 1) / denom);
        }

        // Load inputs
        NpyArray[] arrMain = new NpyArray[10];
        NpyArray[] arrText = new NpyArray[10];
        for (int i = 0; i < 10; i++) {
            if (isfashionclip) {
                arrMain[i] = npyLoad(inputsDir + "/fashionclip/image_" + i + ".npy");
                arrText[i] = npyLoad(inputsDir + "/fashionclip/text_"  + i + ".npy");
            } else {
                arrMain[i] = npyLoad(inputsDir + "/" + modelShort + "/" + i + ".npy");
            }
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        SessionOptions opts = new SessionOptions();
        opts.setIntraOpNumThreads(1);
        opts.setInterOpNumThreads(1);
        opts.setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL);
        opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);

        // Timing
        double tStart = 0, tSessionReady = 0, tInferStart = 0, tInferEnd = 0;

        if (!isSteady) tStart = nowMs();

        OrtSession session = env.createSession(modelPath, opts);

        if (!isSteady) tSessionReady = nowMs();

        // Discover input names
        List<String> inpNames = new ArrayList<>(session.getInputNames());
        List<String> outNames = new ArrayList<>(session.getOutputNames());

        int idxPixel = -1, idxIds = -1, idxMask = -1, idxTtype = -1;
        for (int i = 0; i < inpNames.size(); i++) {
            String n = inpNames.get(i).toLowerCase();
            if (n.contains("pixel"))      idxPixel = i;
            if (n.contains("input_id"))   idxIds   = i;
            if (n.contains("attention") || n.contains("mask")) idxMask = i;
            if (n.contains("token_type")) idxTtype = i;
        }

        // Final copies needed for lambda capture
        final int fIdxPixel = idxPixel;
        final int fIdxIds   = idxIds;
        final int fIdxMask  = idxMask;
        final int fIdxTtype = idxTtype;

        ThrowingIntConsumer runLoop = (count) -> {
            for (int j = 0; j < count; j++) {
                int idx = j % 10;
                Map<String, OnnxTensor> feed = new LinkedHashMap<>();
                for (int i = 0; i < inpNames.size(); i++) {
                    OnnxTensor t;
                    if (i == fIdxPixel) {
                        t = makeTensor(env, arrMain[idx]);
                    } else if (i == fIdxIds) {
                        NpyArray src = (isfashionclip && arrText[idx] != null) ? arrText[idx] : arrMain[idx];
                        t = makeTensor(env, src);
                    } else if (i == fIdxMask) {
                        NpyArray ref = (isfashionclip && arrText[idx] != null) ? arrText[idx] : arrMain[idx];
                        t = makeOnesLike(env, ref);
                    } else if (i == fIdxTtype) {
                        t = makeZerosLike(env, arrMain[idx]);
                    } else {
                        t = makeTensor(env, arrMain[idx]);
                    }
                    feed.put(inpNames.get(i), t);
                }
                try (OrtSession.Result res = session.run(feed)) { /* discard */ }
                feed.values().forEach(OnnxTensor::close);
            }
        };

        // GPT-2 autoregressive loop
        ThrowingIntConsumer gpt2Loop = (count) -> {
            String idName = inpNames.get(0);
            String maskName = null;
            String ttypeName = null;
            for (String n : inpNames) {
                String l = n.toLowerCase();
                if ((l.contains("attention") || l.contains("mask")) && maskName == null) {
                    maskName = n;
                }
                if (l.contains("token_type") && ttypeName == null) {
                    ttypeName = n;
                }
            }
            for (int j = 0; j < count; j++) {
                long[] ids = Arrays.copyOf((long[])arrMain[j % 10].data,
                                           ((long[])arrMain[j % 10].data).length);
                int curLen = ids.length;
                for (int t = 0; t < 20; t++) {
                    long[] shape = {1, curLen};
                    long[] slice = Arrays.copyOf(ids, curLen);
                    LinkedHashMap<String, OnnxTensor> feed = new LinkedHashMap<>();

                    LongBuffer idBuf = LongBuffer.wrap(slice);
                    OnnxTensor idsTensor = OnnxTensor.createTensor(env, idBuf, shape);
                    feed.put(idName, idsTensor);

                    if (maskName != null) {
                        long[] maskVals = new long[curLen];
                        Arrays.fill(maskVals, 1L);
                        LongBuffer maskBuf = LongBuffer.wrap(maskVals);
                        OnnxTensor maskTensor = OnnxTensor.createTensor(env, maskBuf, shape);
                        feed.put(maskName, maskTensor);
                    }

                    if (ttypeName != null) {
                        long[] ttypeVals = new long[curLen];
                        LongBuffer ttypeBuf = LongBuffer.wrap(ttypeVals);
                        OnnxTensor ttypeTensor = OnnxTensor.createTensor(env, ttypeBuf, shape);
                        feed.put(ttypeName, ttypeTensor);
                    }

                    try (OrtSession.Result res = session.run(feed)) {
                        Object out = ((OnnxTensor)res.get(0)).getValue();
                        float[] lastLogits;
                        if (out instanceof float[][][]) {
                            lastLogits = ((float[][][]) out)[0][curLen - 1];
                        } else if (out instanceof float[]) {
                            float[] flat = (float[]) out;
                            int vocab = flat.length / curLen;
                            int base = (curLen - 1) * vocab;
                            lastLogits = Arrays.copyOfRange(flat, base, base + vocab);
                        } else {
                            throw new IllegalStateException(
                                "Unexpected GPT-2 logits type: " + out.getClass().getName()
                            );
                        }
                        int best  = 0;
                        for (int v = 1; v < lastLogits.length; v++)
                            if (lastLogits[v] > lastLogits[best]) best = v;
                        ids = Arrays.copyOf(ids, curLen + 1);
                        ids[curLen++] = (long)best;
                    }
                    for (OnnxTensor tns : feed.values()) tns.close();
                }
            }
        };

        ThrowingIntConsumer effectiveLoop = isgpt2 ? gpt2Loop : runLoop;

        double initMs = 0, inferMs = 0, totalMs = 0;
        boolean hasInit = false;

        if (isSteady) {
            // Warmup: run exactly warmupCount inferences
            effectiveLoop.run(warmupCount);

            signalReady();              // measure.py reads RAPL_0 here
            tInferStart = nowMs();
            for (int r = 0; r < R; r++) effectiveLoop.run(K);
            tInferEnd = nowMs();

            inferMs = (tInferEnd - tInferStart) / R;
        } else {
            signalReady();              // measure.py reads RAPL_mid here
            tInferStart = nowMs();
            effectiveLoop.run(K);
            tInferEnd = nowMs();

            initMs   = tSessionReady - tStart;
            inferMs  = tInferEnd - tInferStart;
            totalMs  = tInferEnd - tStart;
            hasInit  = true;
        }

        // Emit JSON to stderr
        StringBuilder sb = new StringBuilder("{");
        if (hasInit) {
            sb.append(String.format("\"init_ms\": %.3f", initMs));
        } else {
            sb.append("\"init_ms\": null");
        }
        sb.append(String.format(", \"inference_ms\": %.3f", inferMs));
        if (hasInit) {
            sb.append(String.format(", \"total_ms\": %.3f", totalMs));
        } else {
            sb.append(", \"total_ms\": null");
        }
        sb.append("}");
        System.err.println(sb);

        session.close();
        env.close();
    }
}
