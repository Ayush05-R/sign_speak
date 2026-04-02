# Inference Guide

## Live Single-Label Inference

```bash
python -m ml.pipeline.inference.run_static
```

Performance tuning (two-hand friendly):
```bash
python -m ml.pipeline.inference.run_static --detector-max-side 480 --frame-skip 1 --max-hands 2
```

Higher accuracy:
```bash
python -m ml.pipeline.inference.run_static --frame-skip 0 --history 7 --stable-frames 4 --min-det-conf 0.5 --min-pres-conf 0.5
```

If your dataset requires both hands, enable:
```bash
python -m ml.pipeline.inference.run_static --require-two-hands
```

Tip: `run_static` does not auto-launch the sentence builder. If you want that chaining, use:
```bash
python -m ml.pipeline.inference.run_static --run-sentence-builder
```

## Live Sentence Builder (All-in-one)

This window shows the current prediction and builds a sentence using `space` and `full-stop` labels.

```bash
python -m ml.pipeline.inference.run_sentence_builder
```

Common flags:
```bash
--history 7 --stable-frames 6 --cooldown-frames 12 --fade-frames 18
```

Performance tuning:
```bash
python -m ml.pipeline.inference.run_sentence_builder --detector-max-side 480 --frame-skip 1 --max-hands 2
```

Require both hands (for two-hand datasets):
```bash
python -m ml.pipeline.inference.run_sentence_builder --require-two-hands
```

Optional overlays:
```bash
python -m ml.pipeline.inference.run_sentence_builder --show-fps --draw-landmarks
```

