/*
 * C ONNX Runtime inference runner for the energy measurement study.
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

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "onnxruntime_c_api.h"

static const OrtApi *g_ort = NULL;

#define ORT_CHECK(expr) \
    do { OrtStatus *_s = (expr); \
         if (_s) { fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(_s)); \
                   g_ort->ReleaseStatus(_s); exit(1); } } while(0)

/* Wall-clock in ms --------------------------------------------------------- */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* .npy loader (float32 + int64, C-contiguous) ------------------------------ */
typedef struct { void *data; size_t nbytes; int64_t shape[8]; int ndim; char dtype; } NpArr;

static NpArr npy_load(const char *path) {
    NpArr a = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    uint8_t hdr[10]; fread(hdr, 1, 10, f);
    int major = hdr[6];
    uint32_t hlen;
    if (major == 1) {
        hlen = (uint32_t)hdr[8] | ((uint32_t)hdr[9] << 8);
    } else {
        uint8_t extra[2]; fread(extra, 1, 2, f);
        hlen = (uint32_t)hdr[8] | ((uint32_t)hdr[9] << 8) |
               ((uint32_t)extra[0] << 16) | ((uint32_t)extra[1] << 24);
    }
    char *hdr_str = malloc(hlen + 1);
    fread(hdr_str, 1, hlen, f); hdr_str[hlen] = '\0';

    a.dtype = 'f';
    char *dt = strstr(hdr_str, "'descr'"); if (!dt) dt = strstr(hdr_str, "\"descr\"");
    if (dt) {
        char *q = strchr(dt, '<'); char *q2 = strchr(dt, '>'); char *q3 = strchr(dt, '=');
        char *qm = q  && (!q2||q<q2)  && (!q3||q<q3)  ? q :
                   q2 && (!q3||q2<q3) ? q2 : q3;
        if (qm && (qm[1]=='i'||qm[1]=='u')) a.dtype = 'i';
    }
    char *sp = strstr(hdr_str, "'shape'"); if (!sp) sp = strstr(hdr_str, "\"shape\"");
    if (sp) {
        char *lp = strchr(sp, '('), *rp = strchr(sp, ')');
        if (lp && rp) {
            char buf[512]; int n = (int)(rp-lp-1); if (n>0&&n<511){
                strncpy(buf,lp+1,n); buf[n]='\0';
                char *tok=strtok(buf,","); while(tok&&a.ndim<8){
                    char *t=tok; while(*t==' ')t++;
                    if(*t) a.shape[a.ndim++]=atoll(t);
                    tok=strtok(NULL,",");
                }
            }
        }
    }
    free(hdr_str);
    size_t nelems=1; for(int i=0;i<a.ndim;i++) nelems*=(size_t)a.shape[i];
    size_t es=(a.dtype=='f')?4:8; a.nbytes=nelems*es;
    a.data=malloc(a.nbytes); fread(a.data,1,a.nbytes,f); fclose(f);
    return a;
}

/* Global session state ----------------------------------------------------- */
static OrtSession       *g_session   = NULL;
static OrtMemoryInfo    *g_mem_info  = NULL;
static OrtAllocator     *g_alloc     = NULL;
static char            **g_inp_names = NULL;
static char            **g_out_names = NULL;
static size_t            g_n_inp     = 0;
static size_t            g_n_out     = 0;
static int               g_idx_pixel = -1;
static int               g_idx_ids   = -1;
static int               g_idx_mask  = -1;
static int               g_idx_ttype = -1;

static NpArr g_arr_main[10];
static NpArr g_arr_text[10];
static int   g_is_fashionclip = 0;
static int   g_is_gpt2        = 0;

static void run_one(int j) {
    OrtValue *inp_vals[16] = {0};
    void *tmp_bufs[16] = {0};

    NpArr *m = &g_arr_main[j%10];
    NpArr *t = g_is_fashionclip ? &g_arr_text[j%10] : m;

    for (size_t i = 0; i < g_n_inp; i++) {
        ONNXTensorElementDataType etype;
        void *data; size_t nbytes; int64_t *shape; int ndim;

        if ((int)i == g_idx_pixel) {
            etype=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            data=m->data; nbytes=m->nbytes; shape=m->shape; ndim=m->ndim;
        } else if ((int)i == g_idx_ids) {
            etype=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            data=t->data; nbytes=t->nbytes; shape=t->shape; ndim=t->ndim;
        } else if ((int)i == g_idx_mask) {
            etype=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            size_t n = t->nbytes/8;
            int64_t *p = malloc(n*8);
            for(size_t k=0;k<n;k++) p[k]=1;
            tmp_bufs[i]=p;
            data=p; nbytes=n*8; shape=t->shape; ndim=t->ndim;
        } else if ((int)i == g_idx_ttype) {
            etype=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            size_t n = t->nbytes/8;
            tmp_bufs[i]=calloc(n, 8);
            data=tmp_bufs[i]; nbytes=n*8; shape=t->shape; ndim=t->ndim;
        } else {
            etype=(m->dtype=='f')?ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            data=m->data; nbytes=m->nbytes; shape=m->shape; ndim=m->ndim;
        }
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(g_mem_info,data,nbytes,shape,ndim,etype,&inp_vals[i]));
    }
    OrtValue *out_vals[32] = {0};
    ORT_CHECK(g_ort->Run(g_session,NULL,(const char*const*)g_inp_names,
                          (const OrtValue*const*)inp_vals,g_n_inp,
                          (const char*const*)g_out_names,g_n_out,out_vals));
    for(size_t i=0;i<g_n_inp;i++) {
        if(inp_vals[i]) g_ort->ReleaseValue(inp_vals[i]);
        if(tmp_bufs[i]) { free(tmp_bufs[i]); tmp_bufs[i]=NULL; }
    }
    for(size_t i=0;i<g_n_out;i++) if(out_vals[i]) g_ort->ReleaseValue(out_vals[i]);
}

static void run_gpt2_one(int j) {
    NpArr *src = &g_arr_main[j%10];
    size_t n0 = src->nbytes/8;
    int64_t *ids = malloc((n0+20)*8);
    memcpy(ids, src->data, src->nbytes);
    size_t cur = n0;
    char *id_name   = g_inp_names[g_idx_ids   >= 0 ? g_idx_ids  : 0];
    char *mask_name = (g_idx_mask >= 0) ? g_inp_names[g_idx_mask] : NULL;
    for (int t = 0; t < 20; t++) {
        int64_t shape2[2] = {1, (int64_t)cur};
        OrtValue *inp_id = NULL, *inp_mask = NULL, *outp = NULL;
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(g_mem_info,ids,cur*8,
                  shape2,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&inp_id));
        size_t nfeed = 1;
        const char *fnames[2] = {id_name, NULL};
        const OrtValue *fvals[2] = {inp_id, NULL};
        if (mask_name) {
            int64_t *ones = calloc(cur, 8);
            for(size_t k=0;k<cur;k++) ones[k]=1;
            ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(g_mem_info,ones,cur*8,
                      shape2,2,ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&inp_mask));
            free(ones);
            fnames[1] = mask_name; fvals[1] = inp_mask; nfeed = 2;
        }
        ORT_CHECK(g_ort->Run(g_session,NULL,(const char*const*)fnames,
                              (const OrtValue*const*)fvals,nfeed,
                              (const char*const*)g_out_names,1,&outp));
        float *logits;
        ORT_CHECK(g_ort->GetTensorMutableData(outp,(void**)&logits));
        OrtTensorTypeAndShapeInfo *info;
        ORT_CHECK(g_ort->GetTensorTypeAndShape(outp,&info));
        int64_t dims[8]; size_t ndim;
        ORT_CHECK(g_ort->GetDimensionsCount(info,&ndim));
        ORT_CHECK(g_ort->GetDimensions(info,dims,ndim));
        int vocab = (int)dims[ndim-1];
        float *row = logits + (cur-1)*vocab;
        int best=0; for(int v=1;v<vocab;v++) if(row[v]>row[best]) best=v;
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        g_ort->ReleaseValue(outp);
        g_ort->ReleaseValue(inp_id);
        if(inp_mask) g_ort->ReleaseValue(inp_mask);
        ids[cur++] = (int64_t)best;
    }
    free(ids);
}

static void do_loop(int count) {
    for (int j = 0; j < count; j++) {
        if (g_is_gpt2) run_gpt2_one(j);
        else           run_one(j);
    }
}

/* Signal READY and wait for 1 byte on stdin -------------------------------- */
static void signal_ready(void) {
    fputs("READY\n", stdout);
    fflush(stdout);
    getchar();
}

/* Report timing JSON to stderr --------------------------------------------- */
static void report(double init_ms, int has_init,
                   double inference_ms, double total_ms, int has_total) {
    if (has_init)
        fprintf(stderr,
            "{\"init_ms\": %.3f, \"inference_ms\": %.3f, \"total_ms\": %.3f}\n",
            init_ms, inference_ms, total_ms);
    else
        fprintf(stderr,
            "{\"init_ms\": null, \"inference_ms\": %.3f, \"total_ms\": null}\n",
            inference_ms);
}

/* Main --------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr,"Usage: %s <model_path> <inputs_dir> <volume> <phase> <model_short>\n",
                argv[0]);
        return 1;
    }
    const char *model_path  = argv[1];
    const char *inputs_dir  = argv[2];
    int         K           = atoi(argv[3]);
    const char *phase       = argv[4];
    const char *model_short = argv[5];

    g_is_gpt2        = (strcmp(model_short,"gpt2")==0);
    g_is_fashionclip = (strcmp(model_short,"fashionclip")==0);
    int is_steady    = (strcmp(phase,"steady")==0);
    int warmup       = is_steady ? 300 : 0;

    int est_ms = 80;
    if(!strcmp(model_short,"minilm")) est_ms=15;
    else if(!strcmp(model_short,"gpt2")) est_ms=70;
    else if(!strcmp(model_short,"fashionclip")) est_ms=100;
    int R = 1;
    if (is_steady) { int d=K*est_ms; R=(100+d-1)/d; if(R<1)R=1; }

    /* Load inputs */
    for (int i = 0; i < 10; i++) {
        char path[2048];
        if (g_is_fashionclip) {
            snprintf(path,sizeof(path),"%s/fashionclip/image_%d.npy",inputs_dir,i);
            g_arr_main[i]=npy_load(path);
            snprintf(path,sizeof(path),"%s/fashionclip/text_%d.npy",inputs_dir,i);
            g_arr_text[i]=npy_load(path);
        } else {
            snprintf(path,sizeof(path),"%s/%s/%d.npy",inputs_dir,model_short,i);
            g_arr_main[i]=npy_load(path);
        }
    }

    /* ORT init */
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"c_infer",&env));
    OrtSessionOptions *sopts;
    ORT_CHECK(g_ort->CreateSessionOptions(&sopts));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(sopts,1));
    ORT_CHECK(g_ort->SetInterOpNumThreads(sopts,1));
    ORT_CHECK(g_ort->SetSessionExecutionMode(sopts,ORT_SEQUENTIAL));
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(sopts,ORT_ENABLE_ALL));
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&g_mem_info));

    double t_start=0, t_session_ready=0, t_i0=0, t_i1=0;

    if(!is_steady) t_start=now_ms();

    ORT_CHECK(g_ort->CreateSession(env,model_path,sopts,&g_session));

    if(!is_steady) t_session_ready=now_ms();

    ORT_CHECK(g_ort->GetAllocatorWithDefaultOptions(&g_alloc));
    ORT_CHECK(g_ort->SessionGetInputCount(g_session,&g_n_inp));
    ORT_CHECK(g_ort->SessionGetOutputCount(g_session,&g_n_out));
    g_inp_names=malloc(g_n_inp*sizeof(char*));
    g_out_names=malloc(g_n_out*sizeof(char*));
    for(size_t i=0;i<g_n_inp;i++) ORT_CHECK(g_ort->SessionGetInputName(g_session,i,g_alloc,&g_inp_names[i]));
    for(size_t i=0;i<g_n_out;i++) ORT_CHECK(g_ort->SessionGetOutputName(g_session,i,g_alloc,&g_out_names[i]));
    for(size_t i=0;i<g_n_inp;i++){
        if(strstr(g_inp_names[i],"pixel"))                                g_idx_pixel=(int)i;
        if(strstr(g_inp_names[i],"input_id"))                            g_idx_ids=(int)i;
        if(strstr(g_inp_names[i],"attention")||strstr(g_inp_names[i],"mask")) g_idx_mask=(int)i;
        if(strstr(g_inp_names[i],"token_type"))                          g_idx_ttype=(int)i;
    }

    double init_ms=0, infer_ms=0, total_ms=0;

    if(is_steady){
        do_loop(warmup);
        signal_ready();           /* measure.py reads RAPL_0 here */
        t_i0=now_ms();
        for(int r=0;r<R;r++) do_loop(K);
        t_i1=now_ms();
        infer_ms=(t_i1-t_i0)/R;
        report(0, 0, infer_ms, 0, 0);
    } else {
        signal_ready();           /* measure.py reads RAPL_mid here */
        t_i0=now_ms();
        do_loop(K);
        t_i1=now_ms();
        init_ms  = t_session_ready - t_start;
        infer_ms = t_i1 - t_i0;
        total_ms = t_i1 - t_start;
        report(init_ms, 1, infer_ms, total_ms, 1);
    }

    for(size_t i=0;i<g_n_inp;i++) g_alloc->Free(g_alloc,g_inp_names[i]);
    for(size_t i=0;i<g_n_out;i++) g_alloc->Free(g_alloc,g_out_names[i]);
    free(g_inp_names); free(g_out_names);
    g_ort->ReleaseSession(g_session);
    g_ort->ReleaseSessionOptions(sopts);
    g_ort->ReleaseMemoryInfo(g_mem_info);
    g_ort->ReleaseEnv(env);
    return 0;
}
