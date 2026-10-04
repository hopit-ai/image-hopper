---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
pipeline_tag: image-text-to-text
tags:
- image-decisions
- calibration
- multiple-choice
---

# Image Hopper

Image Hopper answers multiple-choice and yes/no questions about images with a calibrated probability for every option,
in one forward pass and without generating text. It is built on `Qwen/Qwen3.5-9B`
(revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`) and is designed for the Image JevBench `/v1/systemone` contract.

## How it works

- **One base, two routes.** A fixed text rule (`router-rules/v1`) reads only the question and option text. Questions
  about on-screen elements (markers, buttons, menus, windows, …) or geometry (angles, triangles, circles, lengths, …)
  use a LoRA adapter; every other question uses the unmodified base model. Both routes share one loaded model and one
  forward pass.
- **Readout.** Options are lettered A, B, C, …; the answer distribution is the FP32 softmax of the next-token logits
  for those letters at the last prompt position.
- **Calibration.** Logits are divided by a temperature chosen by route and answer type (fitted by maximum likelihood
  with shrinkage toward a global temperature). Calibration never changes which option ranks first.
- **Images** use the base processor's native resolution route; up to four images per request.

| Route / answer type | Temperature |
|---|---:|
| default / 4-option | 1.2338 |
| default / 5-marker | 1.9765 |
| default / yes/no | 1.3353 |
| screen_geometry / 4-option | 3.1396 |
| screen_geometry / 5-marker | 1.3292 |
| screen_geometry / yes/no | 0.6572 |

## Training and calibration data

The adapter (LoRA r64, one epoch, about 5,900 examples) was trained on real images with their datasets' own labels —
desktop screenshots from GroundCUA (training applications only), document pages from DocLayNet, industrial-defect
photos from VisA and openly licensed photo collections — plus original procedural renders (charts, documents, tables,
geometry, interface mock-ups). No benchmark items were used for training or calibration, and every training image was
checked against the benchmark's reference images for near-duplicates.

| Source | Licence of admitted material |
|---|---|
| GroundCUA (training applications) | MIT |
| DocLayNet | CDLA-Permissive-1.0 |
| Visual Anomaly (VisA) | CC BY 4.0 |
| Amazon Berkeley Objects | CC BY 4.0 |
| Public Domain 12M | CDLA-Permissive-2.0 metadata; admitted images public domain or CC0 |
| AgriFreshNET | CC BY 4.0 |
| SHARD shelf-management dataset | CC BY 4.0 |
| PHELE physical-hazards dataset | CC BY 4.0 |
| Original procedural renders | Apache-2.0 (ours) |

The default-route temperatures were fitted on a held-out calibration split of our own data; the screen/geometry
temperatures on held-out real screenshots and geometry diagrams that were never used for training.

## Usage

    pip install -r requirements.txt
    python -m open_decisions.image_jev.release.server --offline --adapter adapter --port 8080

`POST /v1/systemone`:

    {"images": ["data:image/png;base64,..."],
     "questions": {"decision": {"type": "choice", "instructions": "Which shelf is empty?",
       "criteria": {"top": "Top shelf", "bottom": "Bottom shelf"}}}}

The response has `model`, token `usage` and `answers`; a `choice` answer holds the selected option and a probability
per option, a `noul` answer holds P(yes).

## Evaluation

Image JevBench results will be added here once the benchmark maintainer publishes them.

## Intended use and limitations

For research and evaluation of image-grounded decisions where callers supply the options. Not a safety system or a
substitute for human review in high-impact decisions. Weak at dense counting; the adapter route is specialised for
screens and geometry; probabilities cover only the supplied options (no abstain answer); calibration reflects our
data mix and may transfer imperfectly; evaluated on English questions with thinking mode off. Native-resolution cost
and latency grow with image size.

## Licence

Code, adapter and calibration: Apache-2.0. The base model `Qwen/Qwen3.5-9B` is Apache-2.0 and is not redistributed
here. See `LICENSE`, `BASE-MODEL-LICENSE` and `NOTICE`; third-party data licences as listed above.
