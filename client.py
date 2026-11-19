"""Day 3 client (organisers' API_DOCS.md): per image
  1. GET  /api/v1/image   -> bytes + X-Image-ID + X-Image-Name
  2. infer (model already loaded, warm)
  3. POST /api/v1/submit  {"img_id", "predictions": {...}}   <- sent FIRST, server clock stops here
  4. MQTT publish (background thread, persistent WSS connection) {"image","latency_ms","counts","defect_ratio"}
  then seg_mask/vis/mqtt files are written (after the POST, so they do not add to the measured time).
At the end: outputs/predictions.json (+ results.json copy) and team_<team>_Day3.zip.

python client.py --config config.json --count 16          # rehearsal (train mode)
python client.py --config config.json --mode test --count 5
python client.py --config config.json --local-input val_day3/images --dry-run   # offline
Credentials live in config.json (not packed into the zip); copy config.example.json.
"""
import argparse
import json
import os
import queue
import shutil
import ssl
import sys
import threading
import time
import uuid
import zipfile

import infer

CONFIG = {
    "http_url": "",                      # announced at the rehearsal, e.g. https://xxx  (no trailing /)
    "http_token": "",                    # team WEBSITE LOGIN PASSWORD (Bearer)
    "team_id": "TEAM_41",                # MQTT username
    "mqtt_key": "",                      # MQTT key generated on the website (password + topic)
    "mqtt_host": "tesa-score.c333m.site",
    "mqtt_port": 443,
    "mqtt_path": "/mqtt/",
    "track": "EdgeAI",
    "event": "1",
    "weights": "weights",
    "imgsz": 640,
    "conf": 0.15,
    "eps": 1.0,                          # polygon simplification (full-res px); smaller JSON = faster POST
    "out_dir": "outputs",
    "img_dir": "downloaded_images",
    "zip_team": "41",
}
ZIP_INCLUDE = ["README.md", "requirements.txt", "train.py", "infer.py", "client.py", "config.example.json",
               "tools", "weights", "outputs"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class Publisher(threading.Thread):
    """Keeps one MQTT v5 WebSocket-TLS connection; publishes queued payloads with QoS 1 and waits for PUBACK."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        import paho.mqtt.client as mqtt
        self.mqtt, self.cfg, self.q = mqtt, cfg, queue.Queue()
        self.ok, self.fail = [], []
        self.topic = f"/event/{cfg['mqtt_key']}/{cfg['track']}/{cfg['event']}"
        self.connected = threading.Event()
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{cfg['team_id']}:python-{uuid.uuid4().hex[:12]}",
                        transport="websockets", protocol=mqtt.MQTTv5)
        c.username_pw_set(cfg["team_id"], cfg["mqtt_key"])
        c.ws_set_options(path=cfg["mqtt_path"])
        c.tls_set_context(ssl.create_default_context())
        c.on_connect = lambda cl, u, fl, rc, pr: (log(f"MQTT connect: {rc}"), self.connected.set() if not rc.is_failure else None)
        c.on_disconnect = lambda cl, u, fl, rc, pr: (self.connected.clear(), log(f"MQTT disconnected: {rc}"))
        c.reconnect_delay_set(1, 5)
        self.c = c
        c.connect_async(cfg["mqtt_host"], int(cfg["mqtt_port"]), keepalive=30)
        c.loop_start()

    def run(self):
        while True:
            pay = self.q.get()
            if pay is None:
                return
            data = json.dumps(pay, ensure_ascii=False).encode("utf-8")
            for attempt in range(5):
                try:
                    if not self.connected.wait(15):
                        raise RuntimeError("not connected")
                    info = self.c.publish(self.topic, data, qos=1, retain=False)
                    info.wait_for_publish(timeout=15)
                    if info.is_published():
                        self.ok.append(pay["image"])
                        log(f"MQTT ok   {pay['image']}")
                        break
                    raise RuntimeError("no PUBACK")
                except Exception as e:
                    log(f"MQTT retry {pay['image']} ({attempt + 1}/5): {e}")
                    time.sleep(1)
            else:
                self.fail.append(pay["image"])
                log(f"MQTT FAIL {pay['image']}")

    def close(self):
        self.q.put(None)
        self.join(timeout=120)
        self.c.disconnect()
        self.c.loop_stop()


def make_zip(cfg):
    name = f"team_{cfg['zip_team']}_Day3.zip"
    root = f"team_{cfg['zip_team']}_day3"
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for item in ZIP_INCLUDE:
            if os.path.isfile(item):
                z.write(item, f"{root}/{item}")
            elif os.path.isdir(item):
                for dp, _, fs in os.walk(item):
                    if "__pycache__" not in dp:
                        for f in fs:
                            p = os.path.join(dp, f)
                            z.write(p, f"{root}/{os.path.relpath(p).replace(os.sep, '/')}")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mode", choices=["train", "test"], default=None, help="?mode= for GET (default: server's)")
    ap.add_argument("--count", type=int, default=1, help="images to request (stops early on non-200)")
    ap.add_argument("--local-input", default=None, help="offline: read images from this folder instead of GET")
    ap.add_argument("--dry-run", action="store_true", help="no network (needs --local-input)")
    ap.add_argument("--skip-mqtt", action="store_true")
    a = ap.parse_args()
    cfg = dict(CONFIG)
    if os.path.exists(a.config):
        cfg.update(json.load(open(a.config, encoding="utf-8-sig")))
    online = not a.dry_run
    if online and not a.local_input and not (cfg["http_url"] and cfg["http_token"]):
        sys.exit(f"set http_url and http_token in {a.config}")
    if a.dry_run and not a.local_input:
        sys.exit("--dry-run needs --local-input")

    import requests
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {cfg['http_token']}"
    base = cfg["http_url"].rstrip("/")

    inf = infer.Inferencer(cfg["weights"], cfg["imgsz"], cfg["conf"], crop=True, vis=True, eps=cfg["eps"])
    log(f"model ready: {inf.wpath}")
    pub = None
    if online and not a.skip_mqtt:
        if not cfg["mqtt_key"]:
            sys.exit(f"set mqtt_key in {a.config} (or use --skip-mqtt)")
        pub = Publisher(cfg)
        pub.start()

    out = cfg["out_dir"]
    shutil.rmtree(out, ignore_errors=True)
    infer.prepare_out(out)
    os.makedirs(cfg["img_dir"], exist_ok=True)

    if a.local_input:
        jobs = [(None, f, os.path.join(a.local_input, f)) for f in infer.list_images(a.local_input)]
        jobs = iter(jobs)
    else:
        jobs = None

    results, lat, done_names = [], [], set()
    for k in range(a.count if jobs is None else 10 ** 6):
        # ---- 1. GET
        if jobs is None:
            params = {"mode": a.mode} if a.mode else None
            try:
                r = sess.get(f"{base}/api/v1/image", params=params, timeout=60)
            except requests.RequestException as e:
                log(f"GET error: {e}")
                break
            if r.status_code != 200:
                log(f"GET stop: HTTP {r.status_code} {r.text[:200]}")
                break
            t0 = time.perf_counter()
            img_id, name = r.headers.get("X-Image-ID"), r.headers.get("X-Image-Name", f"image_{k}.jpg")
            name = os.path.basename(name)
            path, data = os.path.join(cfg["img_dir"], name), r.content  # decoded from memory, saved after POST
        else:
            nxt = next(jobs, None)
            if nxt is None:
                break
            img_id, name, path = nxt
            data = None
            t0 = time.perf_counter()
        # ---- 2. infer
        res = inf.predict(path, name=name, t0=t0, data=data)
        n_obj = len(res["entry"]["objects"])
        lat.append(res["mqtt"]["latency_ms"])
        # ---- 3. POST (first: the server stops its clock here)
        if online and img_id is not None:
            try:
                pr = sess.post(f"{base}/api/v1/submit", json={"img_id": img_id, "predictions": res["entry"]}, timeout=60)
                info = pr.json() if pr.headers.get("content-type", "").startswith("application/json") else pr.text[:200]
                log(f"POST {name}: HTTP {pr.status_code} {info if pr.status_code != 200 else 'server ' + str(info.get('duration_seconds')) + ' s'}")
            except requests.RequestException as e:
                log(f"POST error {name}: {e}")
        # ---- 4. MQTT (async)
        if pub:
            pub.q.put(res["mqtt"])
        log(f"{name}: {n_obj} beans, local {res['mqtt']['latency_ms']} ms, counts {res['mqtt']['counts']}")
        # ---- files (after the POST)
        if data is not None:
            with open(path, "wb") as fh:
                fh.write(data)
            log(f"GET was {name} id={img_id} mode={r.headers.get('X-Mode')} {len(data) / 1e6:.1f} MB")
        if name in done_names:
            log(f"WARNING duplicate image name {name}: outputs overwritten")
        done_names.add(name)
        results = [e for e in results if e["image_name"] != name] + [res["entry"]]
        inf.save(res, out)

    infer.write_predictions(out, results)
    shutil.copy(os.path.join(out, "predictions.json"), os.path.join(out, "results.json"))
    if lat:
        log(f"{len(lat)} images, median local latency {sorted(lat)[len(lat) // 2]} ms")
    if pub:
        pub.close()
        log(f"MQTT published {len(pub.ok)}/{len(lat)}  failed: {pub.fail}")
    if online and not a.local_input:
        try:
            log(f"stats: {sess.get(f'{base}/api/v1/stats', timeout=10).json()}")
        except Exception:
            pass
    z = make_zip(cfg)
    log(f"zip: {z} ({os.path.getsize(z) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
