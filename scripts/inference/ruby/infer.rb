#!/usr/bin/env ruby
# frozen_string_literal: true

# Ruby ONNX Runtime inference runner for the energy measurement study.
#
# Energy/RAPL measurement is handled externally by scripts/measure.py.
# This script only performs inference and signals the measurement boundary.
#
# Usage (called by measure.py):
#   ruby infer.rb <model_path> <inputs_dir> <volume> <phase> <model_short>
#
# READY protocol:
#   Cold  — prints "READY" after session load, reads 1 byte stdin, runs inference
#   Steady— prints "READY" after warmup,       reads 1 byte stdin, runs R×K inference
#
# Stderr output (JSON):
#   {"init_ms": <float|null>, "inference_ms": <float>,
#    "total_ms": <float|null>}

require 'onnxruntime'
require 'json'

# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------

def now_ms
  Process.clock_gettime(Process::CLOCK_MONOTONIC) * 1000.0
end

# Signal READY and wait for 1 byte on stdin
def signal_ready
  $stdout.puts 'READY'
  $stdout.flush
  $stdin.read(1)
end

def report(init_ms, inference_ms, total_ms)
  $stderr.puts JSON.generate({
    init_ms:      init_ms,
    inference_ms: inference_ms,
    total_ms:     total_ms,
  })
end

# ---------------------------------------------------------------------------
# Minimal .npy loader (float32 + int64, C-contiguous, little-endian)
# ---------------------------------------------------------------------------

NpyArray = Struct.new(:data, :shape, :dtype)  # dtype: :float32 | :int64

def npy_load(path)
  raw = File.binread(path)
  off = 6
  major = raw.getbyte(off); off += 1
  _minor = raw.getbyte(off); off += 1
  if major == 1
    hlen = raw[off, 2].unpack1('v'); off += 2
  else
    hlen = raw[off, 4].unpack1('V'); off += 4
  end
  hdr = raw[off, hlen].encode('UTF-8', invalid: :replace, undef: :replace)
  off += hlen

  is_float = true
  if (m = hdr.match(/['"]descr['"]\s*:\s*['"]([<>=|])([a-z])(\d+)['"]/i))
    is_float = false if %w[i u].include?(m[2])
  end

  shape = []
  if (m = hdr.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/))
    shape = m[1].split(',').map(&:strip).reject(&:empty?).map(&:to_i)
  end

  nelems = shape.reduce(1, :*)
  dtype  = is_float ? :float32 : :int64
  esize  = is_float ? 4 : 8
  pack   = is_float ? 'e*' : 'q<*'
  data   = raw[off, nelems * esize].unpack(pack)
  NpyArray.new(data, shape, dtype)
end

def reshape(flat, shape)
  return flat if shape.size <= 1
  chunk = shape[1..].reduce(1, :*)
  flat.each_slice(chunk).map { |sub| reshape(sub, shape[1..]) }
end

def build_feed(session, arr_main, arr_text, j, model_short)
  m = arr_main[j % 10]
  t = arr_text[j % 10]
  inp_names = session.inputs.map { |i| i[:name] }
  feed = {}
  inp_names.each do |name|
    nl = name.downcase
    val = if nl.include?('pixel')
      [reshape(m.data, m.shape[1..])]
    elsif nl.include?('input_id')
      src = t || m
      [src.data]
    elsif nl.include?('attention') || nl.include?('mask')
      ref = t || m
      seq_len = ref.shape.last
      [[1] * seq_len]
    elsif nl.include?('token_type')
      seq_len = m.shape.last
      [[0] * seq_len]
    else
      m.dtype == :float32 ? [reshape(m.data, m.shape[1..])] : [m.data]
    end
    feed[name] = val
  end
  feed
end

def run_loop(session, arr_main, arr_text, count, model_short)
  if model_short == 'gpt2'
    input_names = session.inputs.map { |i| i[:name] }
    id_name = input_names.find { |n| n.downcase.include?('input_id') } || input_names.first
    mask_name = input_names.find { |n| n.downcase.include?('attention') || n.downcase.include?('mask') }
    count.times do |j|
      ids = arr_main[j % 10].data.dup
      20.times do
        feed = { id_name => [ids] }
        feed[mask_name] = [Array.new(ids.length, 1)] if mask_name
        out = session.predict(feed).values
        logits = out.first.flatten
        vocab  = logits.size / ids.size
        row    = logits[((ids.size - 1) * vocab), vocab]
        best   = row.each_with_index.max_by { |v, _| v }[1]
        ids << best
      end
    end
  else
    count.times do |j|
      feed = build_feed(session, arr_main, arr_text, j, model_short)
      session.predict(feed)
    end
  end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ARGV.size != 5
  warn "Usage: #{$PROGRAM_NAME} <model_path> <inputs_dir> <volume> <phase> <model_short>"
  exit 1
end

model_path, inputs_dir, volume, phase, model_short = ARGV

k         = volume.to_i
is_steady = phase == 'steady'
warmup    = is_steady ? 300 : 0
est_ms    = case model_short
            when 'minilm'      then 15
            when 'gpt2'        then 70
            when 'fashionclip' then 100
            else                    80
            end
r_rep = is_steady ? [1, (100.0 / (k * est_ms)).ceil].max : 1

arr_main = Array.new(10) do |i|
  path = model_short == 'fashionclip' ?
    "#{inputs_dir}/fashionclip/image_#{i}.npy" :
    "#{inputs_dir}/#{model_short}/#{i}.npy"
  npy_load(path)
end
arr_text = Array.new(10) do |i|
  model_short == 'fashionclip' ?
    npy_load("#{inputs_dir}/fashionclip/text_#{i}.npy") : nil
end

OnnxRuntime.ffi_lib = ENV.fetch('ORT_DYLIB_PATH',
  File.expand_path('../../../lib/onnxruntime/lib/libonnxruntime.so', __dir__))

opts = {
  intra_op_num_threads: 1,
  inter_op_num_threads: 1,
  execution_mode:           :sequential,
  graph_optimization_level: :all,
}

t_start = t_ready = t_i0 = t_i1 = 0.0
init_ms = inference_ms = total_ms = nil

if is_steady
  session = OnnxRuntime::Model.new(model_path, **opts)
  run_loop(session, arr_main, arr_text, warmup, model_short)

  signal_ready             # measure.py reads RAPL_0 here
  t_i0 = now_ms
  r_rep.times { run_loop(session, arr_main, arr_text, k, model_short) }
  t_i1 = now_ms

  inference_ms = (t_i1 - t_i0) / r_rep
else
  t_start = now_ms
  session = OnnxRuntime::Model.new(model_path, **opts)
  t_ready = now_ms

  signal_ready             # measure.py reads RAPL_mid here
  t_i0 = now_ms
  run_loop(session, arr_main, arr_text, k, model_short)
  t_i1 = now_ms

  init_ms      = t_ready - t_start
  inference_ms = t_i1 - t_i0
  total_ms     = t_i1 - t_start
end

report(init_ms, inference_ms, total_ms)
