# TGR-41 ยท Day 3 ยท Coffee bean instance segmentation + 7 classes (Raspberry Pi 5)

## Run

```sh
python infer.py --input <test_folder/> --weights weights/ --out outputs/
python client.py --config config.json                 # HTTP GET -> infer -> HTTP POST + MQTT (parallel)
python client.py --config config.json --local-input test/ --dry-run   # offline check
```

`config.json` = copy of `config.example.json` filled with the organisers' URLs / MQTT credentials / topic.

Outputs: `outputs/seg_mask/<stem>.png` (0 = bg, 1..N), `outputs/predictions.json` (= `results.json`),
`outputs/mqtt/<stem>.json` (exact payload published), `outputs/vis/<stem>.jpg`.

## Install (Pi 5, 64-bit OS)

```sh
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Method

1. Reduced JPEG decode (1/2 of 3072x4096), EXIF-aware.
2. Bean region from a 256 px thumbnail (darker/more coloured than white paper) -> crop with the same margin
   used for training crops -> beans are ~2x larger at model input (val: P 27.3 vs 25.4 /30 without crop).
3. YOLO-seg (NCNN on the Pi) imgsz 640, conf 0.15 -> one pixel one bean (highest score first) ->
   per-bean bilinear upsample to full size -> largest part + holes filled -> PNG and polygon (CHAIN_APPROX_NONE).
4. MQTT counts are computed from the same objects written to seg_mask/predictions.json, so they always agree.

## Train

```sh
python tools/build_dataset.py            # yolo7/ + val_day3/ (4 Day 3 images held out by image)
python train.py --model <init.pt> --name run --epochs 70 --imgsz 640
python tools/build_dataset.py --all-train && python train.py ...   # final model
python train.py --export runs/segment/<name>/weights/best.pt --imgsz 640   # ONNX + NCNN
```

Data: Day 3 `labels.json` (16 images, 7 classes) + Day 1-2 single-class images with auto masks +
Day 2 Roboflow piles (good/shell). Class ids follow labels.json (0 husk โ€ฆ 6 broken); names are output.

## Self-check

```sh
python infer.py --input val_day3/images --weights weights/ --out out_val
python tools/score.py --pred out_val --gt-dir val_day3/gt --latency <median s on Pi>
```
