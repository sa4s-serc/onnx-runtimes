/**
 * C++ ONNX Runtime inference runner for the energy measurement study.
 *
 * Energy/RAPL measurement is handled externally by scripts/measure.py.
 * This program only performs inference and signals the measurement boundary.
 *
 * Usage (called by measure.py):
 *   ./infer <model_path> <inputs_dir> <volume> <phase> <model_short>
 *
 * READY protocol:
 *   Cold  — prints "READY\n" after session load, reads 1 byte stdin, runs inference
 *   Steady— prints "READY\n" after warmup,       reads 1 byte stdin, runs R×K inference
 *
 * Stderr output (JSON):
 *   {"init_ms": <float|null>, "inference_ms": <float>,
 *    "total_ms": <float|null>}
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <cctype>

#include "onnxruntime_cxx_api.h"

// ---------------------------------------------------------------------------
// Wall-clock
// ---------------------------------------------------------------------------
static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Simple .npy loader (float32 and int64, C-contiguous)
// ---------------------------------------------------------------------------
struct NpyArray {
    std::vector<uint8_t> data;
    std::vector<int64_t> shape;
    char dtype; // 'f' = float32, 'i' = int64
};

static NpyArray npy_load(const std::string &path) {
    NpyArray arr;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint8_t magic[6]; f.read((char*)magic, 6);
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);

    uint32_t hlen;
    if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
    else { f.read((char*)&hlen, 4); }

    std::string hdr(hlen, ' ');
    f.read(&hdr[0], hlen);

    arr.dtype = 'f';
    auto dt_pos = hdr.find("'descr'");
    if (dt_pos == std::string::npos) dt_pos = hdr.find("\"descr\"");
    if (dt_pos != std::string::npos) {
        auto lt = hdr.find('<', dt_pos), gt = hdr.find('>', dt_pos), eq = hdr.find('=', dt_pos);
        size_t q = std::min({lt, gt, eq});
        if (q != std::string::npos && q + 1 < hdr.size()) {
            if (hdr[q+1] == 'i' || hdr[q+1] == 'u') arr.dtype = 'i';
        }
    }

    auto shp_pos = hdr.find("'shape'");
    if (shp_pos == std::string::npos) shp_pos = hdr.find("\"shape\"");
    if (shp_pos != std::string::npos) {
        auto lp = hdr.find('(', shp_pos), rp = hdr.find(')', shp_pos);
        if (lp != std::string::npos && rp != std::string::npos) {
            std::istringstream ss(hdr.substr(lp+1, rp-lp-1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                auto s = tok.find_first_not_of(" \t");
                if (s != std::string::npos) {
                    auto e = tok.find_last_not_of(" \t,)");
                    std::string t = tok.substr(s, e - s + 1);
                    if (!t.empty()) arr.shape.push_back(std::stoll(t));
                }
            }
        }
    }

    size_t nelems = 1;
    for (auto d : arr.shape) nelems *= (size_t)d;
    size_t esize = (arr.dtype == 'f') ? 4 : 8;
    arr.data.resize(nelems * esize);
    f.read((char*)arr.data.data(), arr.data.size());
    return arr;
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------
static Ort::Value make_tensor(const NpyArray &arr, Ort::MemoryInfo &mem_info) {
    if (arr.dtype == 'f') {
        return Ort::Value::CreateTensor<float>(
            mem_info, (float*)arr.data.data(),
            arr.data.size() / 4, arr.shape.data(), arr.shape.size());
    } else {
        return Ort::Value::CreateTensor<int64_t>(
            mem_info, (int64_t*)arr.data.data(),
            arr.data.size() / 8, arr.shape.data(), arr.shape.size());
    }
}

static NpyArray make_ones_like(const NpyArray &ref) {
    NpyArray ones;
    ones.dtype = 'i';
    ones.shape = ref.shape;
    size_t n = ref.data.size() / 8;
    ones.data.resize(n * 8);
    int64_t *p = (int64_t*)ones.data.data();
    for (size_t i = 0; i < n; i++) p[i] = 1;
    return ones;
}

// ---------------------------------------------------------------------------
// Signal READY and wait for 1 byte on stdin
// ---------------------------------------------------------------------------
static void signal_ready() {
    std::cout << "READY\n" << std::flush;
    std::cin.get();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <inputs_dir> <volume> <phase> <model_short>\n";
        return 1;
    }

    std::string model_path   = argv[1];
    std::string inputs_dir   = argv[2];
    int         K            = std::stoi(argv[3]);
    std::string phase        = argv[4];
    std::string model_short  = argv[5];

    bool is_gpt2        = (model_short == "gpt2");
    bool is_fashionclip = (model_short == "fashionclip");
    bool is_steady      = (phase == "steady");
    int  warmup_count   = is_steady ? 300 : 0;

    int est_ms = 80;
    if (model_short == "minilm") est_ms = 15;
    else if (model_short == "gpt2") est_ms = 70;
    else if (model_short == "fashionclip") est_ms = 100;

    int R = 1;
    if (is_steady) {
        int denom = K * est_ms;
        R = std::max(1, (100 + denom - 1) / denom);
    }

    std::vector<NpyArray> arr_main(10), arr_text(10);
    for (int i = 0; i < 10; i++) {
        if (is_fashionclip) {
            arr_main[i] = npy_load(inputs_dir + "/fashionclip/image_" + std::to_string(i) + ".npy");
            arr_text[i] = npy_load(inputs_dir + "/fashionclip/text_"  + std::to_string(i) + ".npy");
        } else {
            arr_main[i] = npy_load(inputs_dir + "/" + model_short + "/" + std::to_string(i) + ".npy");
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpp_infer");
    Ort::SessionOptions sopts;
    sopts.SetIntraOpNumThreads(1);
    sopts.SetInterOpNumThreads(1);
    sopts.SetExecutionMode(ORT_SEQUENTIAL);
    sopts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    double t_start = 0, t_session_ready = 0, t_infer_start = 0, t_infer_end = 0;

    if (!is_steady) t_start = now_ms();

    Ort::Session session(env, model_path.c_str(), sopts);

    if (!is_steady) t_session_ready = now_ms();

    Ort::AllocatorWithDefaultOptions allocator;
    size_t n_inputs  = session.GetInputCount();
    size_t n_outputs = session.GetOutputCount();

    std::vector<std::string> inp_name_strs(n_inputs), out_name_strs(n_outputs);
    std::vector<const char*>  inp_names(n_inputs),    out_names(n_outputs);
    for (size_t i = 0; i < n_inputs;  i++) {
        inp_name_strs[i] = session.GetInputNameAllocated(i, allocator).get();
        inp_names[i] = inp_name_strs[i].c_str();
    }
    for (size_t i = 0; i < n_outputs; i++) {
        out_name_strs[i] = session.GetOutputNameAllocated(i, allocator).get();
        out_names[i] = out_name_strs[i].c_str();
    }

    int idx_pixel = -1, idx_ids = -1, idx_mask = -1, idx_ttype = -1;
    for (size_t i = 0; i < n_inputs; i++) {
        if (inp_name_strs[i].find("pixel")      != std::string::npos) idx_pixel = (int)i;
        if (inp_name_strs[i].find("input_id")   != std::string::npos) idx_ids   = (int)i;
        if (inp_name_strs[i].find("attention")  != std::string::npos ||
            inp_name_strs[i].find("mask")       != std::string::npos) idx_mask  = (int)i;
        if (inp_name_strs[i].find("token_type") != std::string::npos) idx_ttype = (int)i;
    }

    auto make_zeros_like = [](const NpyArray &ref) -> NpyArray {
        NpyArray zeros;
        zeros.dtype = 'i';
        zeros.shape = ref.shape;
        size_t n = ref.data.size() / 8;
        zeros.data.resize(n * 8, 0);
        return zeros;
    };

    using FeedResult = std::pair<std::vector<Ort::Value>, std::vector<NpyArray>>;
    auto make_feed = [&](int j) -> FeedResult {
        std::vector<Ort::Value> vals;
        std::vector<NpyArray>   tmp_arrays;
        vals.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; i++) {
            if ((int)i == idx_pixel) {
                vals.push_back(make_tensor(arr_main[j], mem_info));
            } else if ((int)i == idx_ids) {
                auto &src = (is_fashionclip && !arr_text[j].data.empty()) ? arr_text[j] : arr_main[j];
                vals.push_back(make_tensor(src, mem_info));
            } else if ((int)i == idx_mask) {
                auto &ref = (is_fashionclip && !arr_text[j].data.empty()) ? arr_text[j] : arr_main[j];
                tmp_arrays.push_back(make_ones_like(ref));
                vals.push_back(make_tensor(tmp_arrays.back(), mem_info));
            } else if ((int)i == idx_ttype) {
                tmp_arrays.push_back(make_zeros_like(arr_main[j]));
                vals.push_back(make_tensor(tmp_arrays.back(), mem_info));
            } else {
                vals.push_back(make_tensor(arr_main[j], mem_info));
            }
        }
        return {std::move(vals), std::move(tmp_arrays)};
    };

    auto run_loop = [&](int count) {
        if (is_gpt2) {
            const char* id_name = inp_names[0];
            const char* mask_name = nullptr;
            const char* ttype_name = nullptr;
            for (size_t i = 0; i < n_inputs; i++) {
                std::string lname = inp_name_strs[i];
                std::transform(lname.begin(), lname.end(), lname.begin(),
                               [](unsigned char c) { return (char)std::tolower(c); });
                if (!mask_name && (lname.find("attention") != std::string::npos ||
                                   lname.find("mask") != std::string::npos)) {
                    mask_name = inp_names[i];
                }
                if (!ttype_name && lname.find("token_type") != std::string::npos) {
                    ttype_name = inp_names[i];
                }
                if (lname.find("input_id") != std::string::npos) {
                    id_name = inp_names[i];
                }
            }

            for (int j = 0; j < count; j++) {
                auto &src = arr_main[j % 10];
                std::vector<int64_t> ids((int64_t*)src.data.data(),
                                         (int64_t*)src.data.data() + src.data.size()/8);
                for (int t = 0; t < 20; t++) {
                    int64_t cur_len = (int64_t)ids.size();
                    std::vector<int64_t> shape = {1, cur_len};
                    auto ids_v = Ort::Value::CreateTensor<int64_t>(
                        mem_info, ids.data(), ids.size(), shape.data(), 2);

                    std::vector<const char*> gpt2_inp_names;
                    std::vector<Ort::Value>  gpt2_inp_vals;
                    gpt2_inp_names.reserve(3);
                    gpt2_inp_vals.reserve(3);
                    gpt2_inp_names.push_back(id_name);
                    gpt2_inp_vals.push_back(std::move(ids_v));

                    std::vector<int64_t> mask_vals;
                    if (mask_name) {
                        mask_vals.assign((size_t)cur_len, 1);
                        auto mask_v = Ort::Value::CreateTensor<int64_t>(
                            mem_info, mask_vals.data(), mask_vals.size(), shape.data(), 2);
                        gpt2_inp_names.push_back(mask_name);
                        gpt2_inp_vals.push_back(std::move(mask_v));
                    }

                    std::vector<int64_t> ttype_vals;
                    if (ttype_name) {
                        ttype_vals.assign((size_t)cur_len, 0);
                        auto ttype_v = Ort::Value::CreateTensor<int64_t>(
                            mem_info, ttype_vals.data(), ttype_vals.size(), shape.data(), 2);
                        gpt2_inp_names.push_back(ttype_name);
                        gpt2_inp_vals.push_back(std::move(ttype_v));
                    }

                    auto outs = session.Run(Ort::RunOptions{nullptr},
                        gpt2_inp_names.data(), gpt2_inp_vals.data(), gpt2_inp_vals.size(),
                        out_names.data(), 1);
                    float *logits = outs[0].GetTensorMutableData<float>();
                    auto info = outs[0].GetTensorTypeAndShapeInfo();
                    auto dims = info.GetShape();
                    int vocab = (int)dims.back();
                    float *row = logits + (cur_len - 1) * vocab;
                    int best = (int)(std::max_element(row, row + vocab) - row);
                    ids.push_back((int64_t)best);
                }
            }
        } else {
            for (int j = 0; j < count; j++) {
                auto [feed, tmps] = make_feed(j % 10);
                session.Run(Ort::RunOptions{nullptr},
                    inp_names.data(), feed.data(), n_inputs,
                    out_names.data(), n_outputs);
            }
        }
    };

    double init_ms = 0, infer_ms = 0, total_ms = 0;

    if (is_steady) {
        run_loop(warmup_count);
        signal_ready();                 // measure.py reads RAPL_0 here
        t_infer_start = now_ms();
        for (int r = 0; r < R; r++) run_loop(K);
        t_infer_end = now_ms();
        infer_ms = (t_infer_end - t_infer_start) / R;

        std::cerr << std::fixed;
        std::cerr << "{\"init_ms\": null, \"inference_ms\": " << infer_ms
                  << ", \"total_ms\": null}\n";
    } else {
        signal_ready();                 // measure.py reads RAPL_mid here
        t_infer_start = now_ms();
        run_loop(K);
        t_infer_end = now_ms();
        init_ms  = t_session_ready - t_start;
        infer_ms = t_infer_end - t_infer_start;
        total_ms = t_infer_end - t_start;

        std::cerr << std::fixed;
        std::cerr << "{\"init_ms\": " << init_ms
                  << ", \"inference_ms\": " << infer_ms
                  << ", \"total_ms\": " << total_ms << "}\n";
    }

    return 0;
}
