# Step 3: Calibration Dataset Size

[← Step 2](step2.md) | [Next: Step 4 →](step4.md)

---

## Why Calibration Data Size Matters

The number of calibration samples directly affects the quality of activation range statistics and therefore quantization accuracy.

## Recommended Counts

| Scenario | Recommended samples | `calibration_step` reference (batch_size=1) |
|---|:---:|:---:|
| Quick validation | 50–100 | 50–100 |
| General (CNN, classification) | 100–500 | 100–500 |
| Complex models (Transformer, detection) | 500–1000 | 500–1000 |
| High-accuracy requirements | ≥ 1000 | ≥ 1000 |

> **Note:** `calibration_step` counts dataloader iterations, so effective sample count ≈ `calibration_step × batch_size`. The default is `500`.

## Calibration Data Quality Requirements

- **Diversity:** Calibration data should cover the typical input distribution the model will encounter at inference, not just a single category or scene.
- **Representativeness:** Sample randomly from a validation or test set. Avoid using duplicate samples from the training set.
- **Realism:** Use real data that resembles the actual deployment scenario rather than synthetic images.

## Symptoms of Insufficient Calibration Data

- Accuracy varies noticeably across runs (due to random sampling of calibration data)
- Graphwise report shows extremely uneven `F.Hist` distributions (values concentrated in only a few bins)

## Example Configuration

```json
{
    "calibration_parameters": {
        "calibration_step": 500,
        "input_parametres": [
            {
                "data_list_path": "./calib_img_list.txt"
            }
        ]
    }
}
```

## ✅ Skill Check

1. You have a BERT-based NLP model. How many calibration samples do you recommend?
2. What symptom suggests your calibration dataset is too small?

<details>
<summary>Answers</summary>

1. 500–1000 samples (complex Transformer model). Use `calibration_step: 500` or higher.
2. Accuracy varies noticeably across runs, or the Graphwise report shows very uneven `F.Hist` distributions with values concentrated in only a few bins.

</details>

---

[← Step 2 – Check Input Preprocessing](step2.md) | [Next: Step 4 – Adjust precision_level and finetune_level →](step4.md)
