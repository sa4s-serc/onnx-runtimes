//! Rust ONNX Runtime inference runner for the energy measurement study.
//!
//! Energy/RAPL measurement is handled externally by scripts/measure.py.
//! This program only performs inference and signals the measurement boundary.
//!
//! Usage (called by measure.py):
//!   ./onnx_inference <model_path> <inputs_dir> <volume> <phase> <model_short>
//!
//! READY protocol:
//!   Cold  — prints "READY\n" after session load, reads 1 byte stdin, runs inference
//!   Steady— prints "READY\n" after warmup,       reads 1 byte stdin, runs R×K inference
//!
//! Stderr output (JSON):
//!   {"init_ms": <float|null>, "inference_ms": <float>,
//!    "total_ms": <float|null>}

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::time::Instant;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Tensor, DynValue};
use serde_json::json;

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn signal_ready() {
    print!("READY\n");
    std::io::stdout().flush().unwrap();
    let mut buf = [0u8; 1];
    std::io::stdin().read_exact(&mut buf).ok();
}

#[derive(Clone)]
enum Nd { F32(Vec<f32>), I64(Vec<i64>) }
#[derive(Clone)]
struct Npy { d: Nd, s: Vec<i64> }

fn npy_load(path: &str) -> AnyResult<Npy> {
    let mut raw = Vec::new();
    fs::File::open(path)?.read_to_end(&mut raw)?;
    let mut off = 6usize;
    let major = raw[off]; off += 2;
    let hlen = if major == 1 {
        let h = u16::from_le_bytes([raw[off], raw[off+1]]) as usize; off += 2; h
    } else {
        let h = u32::from_le_bytes([raw[off],raw[off+1],raw[off+2],raw[off+3]]) as usize; off += 4; h
    };
    let hdr = std::str::from_utf8(&raw[off..off+hlen])?.to_owned(); off += hlen;
    let is_float = hdr.find("'descr'").or_else(|| hdr.find("\"descr\"")).map(|p| {
        let sub = &hdr[p..];
        let q = ['<','>','='].iter().filter_map(|c| sub.find(*c)).min().unwrap_or(usize::MAX);
        q < sub.len()-1 && { let c = sub.as_bytes()[q+1]; c != b'i' && c != b'u' }
    }).unwrap_or(true);
    let shape: Vec<i64> = hdr.find("'shape'").or_else(|| hdr.find("\"shape\"")).map(|p| {
        let lp = hdr[p..].find('(').map(|x| p+x+1).unwrap_or(p);
        let rp = hdr[p..].find(')').map(|x| p+x).unwrap_or(p+1);
        hdr[lp..rp].split(',').filter_map(|s| s.trim().parse().ok()).collect()
    }).unwrap_or_default();
    let nelems: usize = shape.iter().map(|&x| x as usize).product::<usize>().max(1);
    let rest = &raw[off..];
    let d = if is_float {
        Nd::F32((0..nelems).map(|i| f32::from_le_bytes(rest[i*4..i*4+4].try_into().unwrap())).collect())
    } else {
        Nd::I64((0..nelems).map(|i| i64::from_le_bytes(rest[i*8..i*8+8].try_into().unwrap())).collect())
    };
    Ok(Npy { d, s: shape })
}

fn f32_dyn(arr: &Npy) -> AnyResult<DynValue> {
    let Nd::F32(ref d) = arr.d else { return Err("need f32".into()); };
    Ok(Tensor::from_array((arr.s.clone(), d.clone()))?.into())
}
fn i64_dyn(arr: &Npy) -> AnyResult<DynValue> {
    let Nd::I64(ref d) = arr.d else { return Err("need i64".into()); };
    Ok(Tensor::from_array((arr.s.clone(), d.clone()))?.into())
}
fn ones_dyn(shape: &[i64]) -> AnyResult<DynValue> {
    let n: usize = shape.iter().map(|&x| x as usize).product();
    Ok(Tensor::from_array((shape.to_vec(), vec![1i64; n]))?.into())
}
fn zeros_dyn(shape: &[i64]) -> AnyResult<DynValue> {
    let n: usize = shape.iter().map(|&x| x as usize).product();
    Ok(Tensor::from_array((shape.to_vec(), vec![0i64; n]))?.into())
}

fn extract_best_token(outputs: &ort::session::SessionOutputs, cur: usize) -> AnyResult<i64> {
    let t0 = outputs[0].try_extract_tensor::<f32>()?;
    let sl: &[f32] = t0.1;
    let vocab = sl.len() / cur;
    let row = &sl[(cur-1)*vocab..cur*vocab];
    let best = row.iter().enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    Ok(best as i64)
}

fn run_loop(
    session: &mut Session,
    arr_main: &[Npy],
    arr_text: &[Option<Npy>],
    inp_names: &[String],
    idx_pixel: Option<usize>,
    idx_ids: Option<usize>,
    idx_mask: Option<usize>,
    idx_ttype: Option<usize>,
    is_gpt2: bool,
    count: usize,
) -> AnyResult<()> {
    if is_gpt2 {
        let id_idx  = idx_ids.unwrap_or(0);
        let msk_idx = idx_mask;
        for j in 0..count {
            let Nd::I64(ref base) = arr_main[j%10].d else { return Err("need i64".into()); };
            let mut ids = base.clone();
            for _ in 0..20 {
                let cur = ids.len();
                let shp = vec![1i64, cur as i64];
                let mut vals: Vec<(String, DynValue)> = Vec::new();
                for i in 0..inp_names.len() {
                    let v = if i == id_idx {
                        Tensor::from_array((shp.clone(), ids.clone()))?.into()
                    } else if Some(i) == msk_idx {
                        ones_dyn(&shp)?
                    } else if Some(i) == idx_ttype {
                        zeros_dyn(&shp)?
                    } else {
                        Tensor::from_array((shp.clone(), ids.clone()))?.into()
                    };
                    vals.push((inp_names[i].clone(), v));
                }
                let nr: Vec<(&str, &DynValue)> = vals.iter().map(|(n,v)| (n.as_str(),v)).collect();
                let outputs = session.run(nr)?;
                let best = extract_best_token(&outputs, cur)?;
                ids.push(best);
            }
        }
    } else {
        for j in 0..count {
            let m = &arr_main[j%10];
            let t = arr_text[j%10].as_ref();
            let mut vals: Vec<(String, DynValue)> = Vec::new();
            for i in 0..inp_names.len() {
                let v = if Some(i) == idx_pixel {
                    f32_dyn(m)?
                } else if Some(i) == idx_ids {
                    i64_dyn(t.unwrap_or(m))?
                } else if Some(i) == idx_mask {
                    ones_dyn(&t.unwrap_or(m).s)?
                } else if Some(i) == idx_ttype {
                    zeros_dyn(&t.unwrap_or(m).s)?
                } else {
                    match &m.d { Nd::F32(_)=>f32_dyn(m)?, Nd::I64(_)=>i64_dyn(m)? }
                };
                vals.push((inp_names[i].clone(), v));
            }
            let nr: Vec<(&str, &DynValue)> = vals.iter().map(|(n,v)| (n.as_str(),v)).collect();
            session.run(nr)?;
        }
    }
    Ok(())
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err(format!(
            "Usage: {} <model_path> <inputs_dir> <volume> <phase> <model_short>",
            args[0]
        ).into());
    }
    let model_path  = &args[1];
    let inputs_dir  = &args[2];
    let k: usize    = args[3].parse()?;
    let phase       = &args[4];
    let model_short = &args[5];

    let is_gpt2   = model_short == "gpt2" || model_short == "pythia";
    let is_fc     = model_short == "fashionclip";
    let is_steady = phase == "steady";
    let warmup    = if is_steady { 300usize } else { 0 };
    let est_ms: usize = match model_short.as_str() {
        "minilm" => 15, "gpt2" => 70, "pythia" => 90, "fashionclip" => 100, _ => 80
    };
    let r_rep: usize = if is_steady {
        ((100 + k*est_ms - 1) / (k*est_ms)).max(1)
    } else { 1 };

    let arr_main: Vec<Npy> = (0..10usize).map(|i| {
        if is_fc { npy_load(&format!("{}/fashionclip/image_{}.npy", inputs_dir, i)) }
        else     { npy_load(&format!("{}/{}/{}.npy", inputs_dir, model_short, i)) }
    }).collect::<AnyResult<_>>()?;
    let arr_text: Vec<Option<Npy>> = (0..10usize).map(|i| {
        if is_fc { Some(npy_load(&format!("{}/fashionclip/text_{}.npy", inputs_dir, i)).unwrap()) }
        else { None }
    }).collect();

    ort::init().with_name("rust_infer").commit();

    let build_session = || -> AnyResult<Session> {
        let mut b = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::All)?;
        Ok(b.commit_from_file(model_path)?)
    };

    let inp_info = |session: &Session| -> (Vec<String>, Option<usize>, Option<usize>, Option<usize>, Option<usize>) {
        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_owned()).collect();
        let mut idx_pixel = None;
        let mut idx_ids   = None;
        let mut idx_mask  = None;
        let mut idx_ttype = None;
        for (i, n) in names.iter().enumerate() {
            let nl = n.to_lowercase();
            if nl.contains("pixel")                            { idx_pixel = Some(i); }
            if nl.contains("input_id")                        { idx_ids   = Some(i); }
            if nl.contains("attention")||nl.contains("mask") { idx_mask  = Some(i); }
            if nl.contains("token_type")                      { idx_ttype = Some(i); }
        }
        (names, idx_pixel, idx_ids, idx_mask, idx_ttype)
    };

    if is_steady {
        let mut session = build_session()?;
        let (inp_names, idx_pixel, idx_ids, idx_mask, idx_ttype) = inp_info(&session);
        run_loop(&mut session, &arr_main, &arr_text, &inp_names,
                 idx_pixel, idx_ids, idx_mask, idx_ttype, is_gpt2, warmup)?;

        signal_ready();                // measure.py reads RAPL_0 here
        let t0 = Instant::now();
        for _ in 0..r_rep {
            run_loop(&mut session, &arr_main, &arr_text, &inp_names,
                     idx_pixel, idx_ids, idx_mask, idx_ttype, is_gpt2, k)?;
        }
        let infer_ms = t0.elapsed().as_secs_f64() * 1000.0 / r_rep as f64;

        let null_f: Option<f64> = None;
        eprintln!("{}", serde_json::to_string(&json!({
            "init_ms":      null_f,
            "inference_ms": infer_ms,
            "total_ms":     null_f,
        }))?);
    } else {
        let ts = Instant::now();
        let mut session = build_session()?;
        let init_ms = ts.elapsed().as_secs_f64() * 1000.0;
        let (inp_names, idx_pixel, idx_ids, idx_mask, idx_ttype) = inp_info(&session);

        signal_ready();                // measure.py reads RAPL_mid here
        let ti = Instant::now();
        run_loop(&mut session, &arr_main, &arr_text, &inp_names,
                 idx_pixel, idx_ids, idx_mask, idx_ttype, is_gpt2, k)?;
        let infer_ms = ti.elapsed().as_secs_f64() * 1000.0;
        let total_ms = ts.elapsed().as_secs_f64() * 1000.0;

        eprintln!("{}", serde_json::to_string(&json!({
            "init_ms":      init_ms,
            "inference_ms": infer_ms,
            "total_ms":     total_ms,
        }))?);
    }

    Ok(())
}
