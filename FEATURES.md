# Reproduction Files

The original FakeSV/FakeTT text-modality files and split files are included in this repository under `reproduction_data/`.

The extracted multimodal feature files are large and are not included in this GitHub repository. They are available from the authors upon request by email: aheadgyh@163.com.

For each dataset, the model scripts expect this structure:

```text
data_complete.json
data_split/train.txt
data_split/val.txt
data_split/test.txt
features/features_original/<feature files listed below>
```

Place the feature files under the following default directories:

```text
reproduction_data/FakeSV/features/features_original/
reproduction_data/FakeTT/features/features_original/
```

Alternatively, set these environment variables before running evaluation:

```bash
export EGRT_FAKESV_DATA_DIR=/path/to/FakeSV
export EGRT_FAKESV_FEATURE_DIR=/path/to/FakeSV/features/features_original
export EGRT_FAKETT_DATA_DIR=/path/to/FakeTT
export EGRT_FAKETT_FEATURE_DIR=/path/to/FakeTT/features/features_original
```

## Required FakeSV Features

| File | Approx. Size |
| --- | ---: |
| `audio_hubert_feats_30s.pkl` | 22 MB |
| `audio_vggish.pkl` | 107 MB |
| `title_ocr_concat_bert.pkl` | 33 MB |
| `title_ocr_concat_xclip.pkl` | 22 MB |
| `video_c3d.hdf5` | 5.7 GB |
| `video_xclip.h5` | 86 MB |
| `internet_bert_qudiaoduoyuzifu.pkl` | 11 MB |
| `internet_xclip_qudiaoduoyuzifu.pkl` | 7.3 MB |
| `internet_claim_bert_qudiaoduoyuzifu.pkl` | 11 MB |
| `internet_claim_xclip_qudiaoduoyuzifu.pkl` | 7.3 MB |
| `non_internet_summary_bert.pkl` | 11 MB |
| `non_internet_summary_xclip.pkl` | 7.3 MB |

## Required FakeTT Features

| File | Approx. Size |
| --- | ---: |
| `audio_hubert_feats_av_filled.pkl` | 7.9 MB |
| `audio_vggish_filled.pkl` | 48 MB |
| `title_ocr_bert.pkl` | 6.0 MB |
| `title_ocr_concat_xclip.pkl` | 7.9 MB |
| `video_c3d.hdf5` | 2.0 GB |
| `video_xclip.h5` | 32 MB |
| `internet_bert.pkl` | 6.0 MB |
| `internet_xclip.pkl` | 4.0 MB |
| `internet_claim_bert.pkl` | 6.0 MB |
| `internet_claim_xclip.pkl` | 4.0 MB |
| `non_internet_summary_bert.pkl` | 6.0 MB |
| `non_internet_summary_xclip.pkl` | 4.0 MB |
