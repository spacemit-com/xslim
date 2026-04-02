#!/usr/bin/env python3
"""Scaffold a starter XSlim project from an ONNX model."""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path
from typing import Dict, List


DTYPE_FALLBACK = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
}

TOKEN_HINTS = ("token", "mask", "segment", "input_ids", "attention", "position")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a starter XSlim config and demo project from an ONNX model.")
    parser.add_argument("--model", required=True, help="Path to the input ONNX model.")
    parser.add_argument("--output-dir", required=True, help="Directory where the scaffolded project will be created.")
    parser.add_argument(
        "--mode",
        choices=("int8", "dynq", "fp16"),
        default="int8",
        help="Starter quantization mode to generate.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite files in the output directory if it already exists.")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "input"


def tensor_dtype_name(onnx_module, elem_type: int) -> str:
    helper = getattr(onnx_module, "helper", None)
    if helper is not None and hasattr(helper, "tensor_dtype_to_np_dtype"):
        try:
            return helper.tensor_dtype_to_np_dtype(elem_type).name
        except Exception:
            pass
    return DTYPE_FALLBACK.get(elem_type, "float32")


def load_model_inputs(model_path: Path) -> List[Dict[str, object]]:
    import onnx

    model = onnx.load(str(model_path))
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs: List[Dict[str, object]] = []
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value and dim.dim_value > 0:
                shape.append(int(dim.dim_value))
            else:
                shape.append(1)
        inputs.append(
            {
                "name": value_info.name,
                "shape": shape,
                "dtype": tensor_dtype_name(onnx, tensor_type.elem_type),
            }
        )
    if not inputs:
        raise RuntimeError("No non-initializer graph inputs were found in the ONNX model.")
    return inputs


def is_float_dtype(dtype_name: str) -> bool:
    return dtype_name.startswith("float")


def is_integer_dtype(dtype_name: str) -> bool:
    return dtype_name.startswith(("int", "uint"))


def is_image_like(input_info: Dict[str, object]) -> bool:
    shape = input_info["shape"]
    return is_float_dtype(str(input_info["dtype"])) and len(shape) == 4 and shape[1] in (1, 3, 4)


def is_token_like(input_info: Dict[str, object]) -> bool:
    name = str(input_info["name"]).lower()
    shape = input_info["shape"]
    dtype = str(input_info["dtype"]).lower()
    return is_integer_dtype(dtype) or any(hint in name for hint in TOKEN_HINTS) or len(shape) in (2, 3)


def choose_int8_profile(inputs: List[Dict[str, object]]) -> Dict[str, int]:
    if any(is_token_like(input_info) and not is_image_like(input_info) for input_info in inputs):
        return {"precision_level": 1, "finetune_level": 2}
    return {"precision_level": 0, "finetune_level": 1}


def build_input_parameter(project_dir: Path, input_info: Dict[str, object]) -> Dict[str, object]:
    input_name = str(input_info["name"])
    input_shape = list(input_info["shape"])
    dtype = str(input_info["dtype"])
    list_path = project_dir / "calib_data" / f"{sanitize_name(input_name)}_list.txt"

    parameter: Dict[str, object] = {
        "input_name": input_name,
        "input_shape": input_shape,
        "dtype": dtype,
        "data_list_path": str(list_path),
    }

    if is_image_like(input_info):
        parameter.update(
            {
                "file_type": "img",
                "color_format": "bgr",
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
                "preprocess_file": "PT_IMAGENET",
            }
        )
    else:
        parameter["file_type"] = "npy"

    return parameter


def build_config(model_path: Path, project_dir: Path, mode: str, inputs: List[Dict[str, object]]) -> Dict[str, object]:
    model_stem = sanitize_name(model_path.stem)
    config: Dict[str, object] = {
        "model_parameters": {
            "onnx_model": str(model_path),
            "output_prefix": f"{model_stem}.q",
            "working_dir": str(project_dir / "output"),
        },
        "quantization_parameters": {
            "analysis_enable": True,
        },
    }

    if mode == "fp16":
        config["quantization_parameters"]["precision_level"] = 4
        return config

    if mode == "dynq":
        config["quantization_parameters"]["precision_level"] = 3
        return config

    starter_profile = choose_int8_profile(inputs)
    config["quantization_parameters"].update(starter_profile)
    config["calibration_parameters"] = {
        "calibration_step": 200 if any(is_image_like(item) for item in inputs) else 100,
        "calibration_device": "cuda",
        "calibration_type": "default",
        "input_parametres": [build_input_parameter(project_dir, input_info) for input_info in inputs],
    }
    return config


def write_json(path: Path, content: Dict[str, object]) -> None:
    path.write_text(json.dumps(content, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def write_demo_py(path: Path) -> None:
    path.write_text(
        """from pathlib import Path

import xslim


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "xslim_config.json"
    xslim.quantize_onnx_model(str(config_path))
""",
        encoding="utf-8",
    )


def write_demo_sh(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m xslim -c "${SCRIPT_DIR}/xslim_config.json"
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_readme(path: Path, model_path: Path, mode: str, inputs: List[Dict[str, object]]) -> None:
    input_lines = "\n".join(
        f"- `{item['name']}`: shape={item['shape']}, dtype={item['dtype']}" for item in inputs
    )
    calibration_note = (
        "Populate the files under `calib_data/` with one sample path per line before running INT8 quantization."
        if mode == "int8"
        else "No calibration data is required for this generated mode."
    )
    path.write_text(
        f"""# Generated XSlim Demo Project

This project was scaffolded from:

- ONNX model: `{model_path}`
- mode: `{mode}`

## Detected Inputs

{input_lines}

## Files

- `xslim_config.json` - starter XSlim configuration
- `demo.py` - Python API entry
- `demo.sh` - CLI entry
- `output/` - generated output directory

## Next Steps

1. Review `xslim_config.json` and adjust preprocessing to match training exactly.
2. {calibration_note}
3. Run `./demo.sh` or `python demo.py`.
4. Inspect the generated quantization report and refine `precision_level`, `finetune_level`, or `custom_setting` as needed.
""",
        encoding="utf-8",
    )


def write_calibration_placeholders(project_dir: Path, inputs: List[Dict[str, object]]) -> None:
    calib_dir = project_dir / "calib_data"
    calib_dir.mkdir(parents=True, exist_ok=True)
    for input_info in inputs:
        list_path = calib_dir / f"{sanitize_name(str(input_info['name']))}_list.txt"
        list_path.write_text("", encoding="utf-8")


def ensure_output_dir(project_dir: Path, force: bool) -> None:
    if project_dir.exists() and any(project_dir.iterdir()) and not force:
        raise FileExistsError(f"{project_dir} already exists and is not empty. Use --force to overwrite.")
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "output").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    project_dir = Path(args.output_dir).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ensure_output_dir(project_dir, args.force)
    inputs = load_model_inputs(model_path)
    config = build_config(model_path, project_dir, args.mode, inputs)

    write_json(project_dir / "xslim_config.json", config)
    write_demo_py(project_dir / "demo.py")
    write_demo_sh(project_dir / "demo.sh")
    write_readme(project_dir / "README.md", model_path, args.mode, inputs)

    if args.mode == "int8":
        write_calibration_placeholders(project_dir, inputs)


if __name__ == "__main__":
    main()
