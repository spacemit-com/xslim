# Targeted Tuning Reference

Use this file when Graphwise Analysis already identified problem operators or subgraphs.

## Method 1: Raise Local Precision with `custom_setting`

Use this when a specific subgraph has high SNR or low Cosine.

```json
"quantization_parameters": {
  "precision_level": 0,
  "custom_setting": [
    {
      "input_names": ["problem_layer_input_tensor"],
      "output_names": ["problem_layer_output_tensor"],
      "precision_level": 2
    }
  ]
}
```

## Method 2: Adjust Calibration Strategy

Use this when float range and quantized range suggest clipping or outliers.

### Percentile-based adjustment

```json
"calibration_parameters": {
  "calibration_type": "percentile"
},
"quantization_parameters": {
  "max_percentile": 0.9995
}
```

### Min-max calibration

```json
"calibration_parameters": {
  "calibration_type": "minmax"
}
```

### Local override inside `custom_setting`

```json
"quantization_parameters": {
  "custom_setting": [
    {
      "input_names": ["problem_layer_input"],
      "output_names": ["problem_layer_output"],
      "calibration_type": "percentile",
      "max_percentile": 0.9995
    }
  ]
}
```

## Method 3: Exclude Sensitive Operators

Use this when the same operator type repeatedly shows high error.

### Exclude by operator type

```json
"quantization_parameters": {
  "ignore_op_types": ["Softmax", "LayerNormalization"]
}
```

### Exclude by operator name

```json
"quantization_parameters": {
  "ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
}
```

## Method 4: Raise `finetune_level`

Use this when many layers show moderate error and no single hotspot dominates.

```json
"quantization_parameters": {
  "finetune_level": 2
}
```

## Symptom-to-Action Mapping

| Symptom | Likely Cause | Recommended Action |
|---|---|---|
| Large drop right after quantization | preprocessing mismatch | verify `mean_value`, `std_value`, `color_format` |
| Small global drop | insufficient calibration or too-low global precision | increase `calibration_step`, raise `precision_level` or `finetune_level` |
| One layer has very high SNR | local activation issue | `custom_setting` with higher local precision |
| Float range far exceeds quantized range | clipping | switch calibration mode or adjust percentile |
| Softmax / LayerNorm repeatedly fail | operator sensitivity | use `ignore_op_types` or local exclusion |
| INT8 stays unacceptable | model too quantization-sensitive | switch to dynamic quantization or FP16 |

## Combined Example

```json
{
  "model_parameters": {
    "onnx_model": "models/my_model.onnx",
    "working_dir": "./output"
  },
  "calibration_parameters": {
    "calibration_step": 500,
    "calibration_type": "percentile",
    "input_parameters": [
      {
        "mean_value": [103.94, 116.78, 123.68],
        "std_value": [57.0, 57.0, 57.0],
        "color_format": "bgr",
        "preprocess_file": "PT_IMAGENET",
        "data_list_path": "./calib_data/img_list.txt"
      }
    ]
  },
  "quantization_parameters": {
    "precision_level": 0,
    "finetune_level": 2,
    "max_percentile": 0.9999,
    "analysis_enable": true,
    "ignore_op_types": ["Softmax"],
    "custom_setting": [
      {
        "input_names": ["problem_layer_input"],
        "output_names": ["problem_layer_output"],
        "precision_level": 2,
        "calibration_type": "minmax"
      }
    ]
  }
}
```
