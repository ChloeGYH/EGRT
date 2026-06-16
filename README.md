# EGRT

This repository contains the public dataset constructed for the associated paper.

## Dataset Files

The PLUS dataset constructed in our work is stored under `data/`:

| File | Records | Description |
| --- | ---: | --- |
| `data/FakeSV_PLUS.json` | 3,624 | FakeSV-based samples with video metadata, OCR/ASR flags, extracted claims, web evidence outputs, and overall assessments. |
| `data/FakeTT_PLUS.json` | 1,992 | FakeTT-based samples with video-level analysis, text analysis, cross-modal verification, and overall assessments. |

Both files are JSON arrays. Each item represents one short-video sample.

The original text-modality files from the FakeSV and FakeTT datasets are provided under `reproduction_data/`:

| File | Description |
| --- | --- |
| `reproduction_data/FakeSV/data_complete.json` | Original FakeSV text-modality data used for reproduction. |
| `reproduction_data/FakeTT/data_complete.json` | Original FakeTT text-modality data used for reproduction. |
| `reproduction_data/FakeSV/data_split/*.txt` | Original FakeSV train/validation/test split files. |
| `reproduction_data/FakeTT/data_split/*.txt` | Original FakeTT train/validation/test split files. |

## Field Overview

`FakeSV_PLUS.json` includes fields such as:

- `video_id`
- `keywords`
- `annotation`
- `title`
- `publish_time_norm`
- engagement and author metadata
- `ocr`
- `needOCR`
- `needASR`
- `claims`
- `overall_assessment`

`FakeTT_PLUS.json` includes fields such as:

- `video_id`
- `annotation`
- `title`
- `Video Analysis`
- `Text Analysis`
- `Cross-Modal Verification`
- `Overall Assessment`

## Usage

```python
import json
from pathlib import Path

with Path("data/FakeSV_PLUS.json").open(encoding="utf-8") as f:
    fakesv_plus = json.load(f)

with Path("data/FakeTT_PLUS.json").open(encoding="utf-8") as f:
    fakett_plus = json.load(f)

print(len(fakesv_plus), len(fakett_plus))
```

## Models and Checkpoints

The released reproduction code and best checkpoints are under `models/`:

| Directory | Checkpoint | Test Accuracy |
| --- | --- | ---: |
| `models/fakesv/` | `weights/best_model_sv_0.8616.pth` | 0.8616 |
| `models/fakett/` | `weights/best_model_fake_TT_bst_0.8462.pth` | 0.8462 |

The checkpoint files are stored with Git LFS.

The original FakeSV/FakeTT text-modality and split files are included under `reproduction_data/`. The extracted multimodal feature files are large and are not included in this repository. They are available from the authors upon request by email. See `FEATURES.md` for required feature filenames, expected paths, and contact information.

Environment details are provided in `ENVIRONMENT.md`.

Example evaluation commands:

```bash
export EGRT_FAKESV_DATA_DIR=/path/to/FakeSV
export EGRT_FAKESV_FEATURE_DIR=/path/to/FakeSV/features/features_original
python models/fakesv/test_best.py --model_path models/fakesv/weights/best_model_sv_0.8616.pth

export EGRT_FAKETT_DATA_DIR=/path/to/FakeTT
export EGRT_FAKETT_FEATURE_DIR=/path/to/FakeTT/features/features_original
python models/fakett/test_best.py --model_path models/fakett/weights/best_model_fake_TT_bst_0.8462.pth
```

## Citation

Citation information will be added when the paper metadata is finalized.

## License

This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You may share and adapt the dataset for non-commercial purposes, provided that appropriate credit is given. See `LICENSE` or <https://creativecommons.org/licenses/by-nc/4.0/> for details.
