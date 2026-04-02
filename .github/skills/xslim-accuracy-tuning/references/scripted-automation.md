# Scripted Automation Reference

Use this file when the user wants the repetitive setup work to be scripted rather than hand-written.

## Bundled Script

`scripts/bootstrap_xslim_project.py`

## What the Script Does

Given a user ONNX model path, the script:

1. reads the ONNX graph inputs
2. skips initializer tensors so only real model inputs are analyzed
3. infers a starter profile from input rank, dtype, shape, and common tensor names
4. creates a runnable XSlim project scaffold
5. writes a starter quantization config
6. writes `demo.py`, `demo.sh`, and `README.md`
7. creates calibration list placeholder files for INT8 workflows

## Recommended Invocation

```bash
python /absolute/path/to/.github/skills/xslim-accuracy-tuning/scripts/bootstrap_xslim_project.py \
  --model /path/to/model.onnx \
  --output-dir /path/to/generated_project \
  --mode int8
```

Mode options:

- `int8` - generate a calibration-based starter config
- `dynq` - generate a dynamic quantization starter config
- `fp16` - generate an FP16 starter config

## Generated Artifacts

The generated project contains:

- `xslim_config.json`
- `demo.py`
- `demo.sh`
- `README.md`
- `calib_data/*.txt` placeholder list files for INT8 mode

## Heuristics

### Image-like inputs

If an input is float-like and rank-4 with channel count 1/3/4, the script prefers:

- `file_type: "img"`
- `color_format: "bgr"`
- `preprocess_file: "PT_IMAGENET"`
- starter INT8 profile with `precision_level: 0`, `finetune_level: 1`

### Token / feature inputs

If inputs are integer-like or names contain tokens such as `input_ids`, `token`, `mask`, or `segment`, the script prefers:

- `file_type: "npy"`
- more conservative starter INT8 profile such as `precision_level: 1`, `finetune_level: 2`

### Unknown / generic tensor inputs

For non-image float tensors, the script defaults to `file_type: "npy"` and preserves dtype and inferred shape.

## How to Use the Output

After generation:

1. fill the calibration list files with real sample paths if using INT8
2. adjust preprocessing fields to exactly match training and deployment
3. run `demo.sh` or `demo.py`
4. inspect the generated report and continue with the tuning workflow from `SKILL.md`

## When to Prefer the Script

Prefer the script when:

- the user provides a specific ONNX file
- the task repeats across many models
- a quick starter project is more useful than a hand-written explanation
- the user wants a demo scaffold together with the config

Do not rely on the script alone for final accuracy tuning. It creates a strong starting point, not a fully optimized final configuration.
