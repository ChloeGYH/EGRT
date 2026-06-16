# EGRT

This repository contains the public dataset constructed for the associated paper.

## Dataset Files

The dataset is stored under `data/`:

| File | Records | Description |
| --- | ---: | --- |
| `data/FakeSV_PLUS.json` | 3,624 | FakeSV-based samples with video metadata, OCR/ASR flags, extracted claims, web evidence outputs, and overall assessments. |
| `data/FakeTT_PLUS.json` | 1,992 | FakeTT-based samples with video-level analysis, text analysis, cross-modal verification, and overall assessments. |

Both files are JSON arrays. Each item represents one short-video sample.

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

## Citation

Citation information will be added when the paper metadata is finalized.

## License

This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You may share and adapt the dataset for non-commercial purposes, provided that appropriate credit is given. See `LICENSE` or <https://creativecommons.org/licenses/by-nc/4.0/> for details.
