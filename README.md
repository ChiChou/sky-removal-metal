# sky-removal-metal

Standalone local sky segmentation for macOS, using Core ML (CPU/GPU) and Metal-backed Core Image. The inference model and guided edge refinement run in this process. A host editor needs only a file-based subprocess client; there is no library dependency on the host application.

Licensed under **AGPL-3.0-or-later**; see [LICENSE](LICENSE). The reference project is [OpenDroneMap/SkyRemoval](https://github.com/OpenDroneMap/SkyRemoval), whose source is AGPL-3.0. No model weights or photographs are included in this repository. Model conversion does not change the upstream weights' licensing terms. This repository documents a technical process boundary, not a determination about the licensing of any particular combined distribution.

## Build and use

Requires macOS 14+, a Metal-capable GPU, and the Apple command-line developer tools. Tested on Apple Silicon with macOS 27.

```sh
swift build -c release
swift test
.build/release/sashimi-sky --model models/SkyRemoval.mlmodel --input photo.png --output sky.png
```

The protocol is `--model PATH --input PATH --output PATH`, in that order. The model may be a `.mlmodel` or compiled `.mlmodelc`. Input is an orientation-aware RGB image of at most 40 megapixels; the host normally provides an sRGB PNG with a 3000-pixel long edge. Output is a same-size 16-bit grayscale PNG: white means sky, black means excluded. The CLI exits 0 on success; failures exit 1 with a short stderr message. It neither accesses the network nor downloads models. `--version` prints its version and license. The caller controls temporary files, cancellation, and model selection.

Before storing a model preference, a host can run `sashimi-sky --validate-model PATH` (available since 1.1.0). This compiles/loads the model and checks the required input/output shapes without an image or output file. It exits 0 on success, or 1 with a stderr error. Model validation and inference both stay in the helper process.

## Convert the reference model

Create a Python environment with `numpy`, `onnx`, and `coremltools`. Obtain and unzip the upstream v1.0.6 [model archive](https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.6/model.zip) separately, observing its licensing terms.

```sh
python tools/convert.py model.onnx models/SkyRemoval.mlmodel
```

The converter folds the fixed-shape coordinate-channel construction and maps the remaining 59 convolutions, 54 ReLUs, 19 additions, 7 concatenations, 5 nearest-neighbor upsamplings, max pooling, and sigmoid to Core ML. Input is float RGB NCHW `[1,3,384,384]` in `[0,1]`; output is `[1,1,384,384]`. It rejects unknown dynamic operators. No PyTorch, Python, ONNX runtime, or MLX dependency ships in the CLI.

The Swift pipeline uses Lanczos downsampling, numeric sRGB input, Core ML configured for CPU/GPU, Lanczos upsampling, and a blue-channel guided filter with radius 20 and epsilon 0.01. Four separately compiled Metal kernels plus Core Image box blurs implement refinement. It preserves soft sky probabilities rather than the reference project's inverted binary photogrammetry mask. Lanczos input resizing and replicated-edge box filtering differ from OpenCV area resizing and truncated-boundary normalization in the reference, so the complete pipelines are not pixel-identical.

## Validation

The Metal guided-filter regression verifies a constant mask through image boundaries. The host integration tests validate sky/foreground orientation against a real RAW photograph. A seeded synthetic RGB tensor run through both ONNX Runtime CPU and the converted Core ML CPU/GPU model gave mean absolute error **0.00114**, max **0.00601**; one cold prediction took **0.89 s** on the development Mac. These are one-machine measurements, not a throughput benchmark. GPU arithmetic can differ from CPU FP32. On a real-photo tensor, mean error was **0.000152**, max **0.00304**, and the warmed prediction took **0.088 s**; the same preprocessing tensor was supplied to both engines.

For independent numerical checks:

```sh
swift tools/compare.swift models/SkyRemoval.mlmodel input.f32 output.f32 gpu
```

The input file is little-endian float32 NCHW. A PNG may be supplied instead; the script then writes the exact preprocessed tensor to `output.f32.input.f32` for comparison with ONNX Runtime. Mask quality remains limited by the 384-pixel network and its training data; inspect fine tree/building boundaries and refine manually in the host editor.
