"""Local mock of the organisers' HTTP API (API_DOCS.md) for testing client.py offline.
python tools/mock_server.py --images val_day3/images --port 8000 --token test
then: config.json {"http_url": "http://127.0.0.1:8000", "http_token": "test", ...}
Saves each submission to mock_submissions/<img_id>.json and checks its format.
"""
import argparse
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--token", default="test")
a = ap.parse_args()
FILES = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
STATE = {"i": 0, "sent": {}, "done": set()}
CLASSES = {"black", "broken", "good", "husk", "immature", "parchment", "shell"}
os.makedirs("mock_submissions", exist_ok=True)


class H(BaseHTTPRequestHandler):
    def _auth(self):
        if self.headers.get("Authorization") != f"Bearer {a.token}":
            self._json(401, {"detail": "bad token"})
            return False
        return True

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self._auth():
            return
        if self.path.startswith("/api/v1/image"):
            f = FILES[STATE["i"] % len(FILES)]
            STATE["i"] += 1
            data = open(os.path.join(a.images, f), "rb").read()
            iid = f"img_{uuid.uuid4().hex[:16]}"
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Image-ID", iid)
            self.send_header("X-Image-Name", f)
            self.send_header("X-Mode", "train")
            self.end_headers()
            self.wfile.write(data)
            STATE["sent"][iid] = (f, time.time())
        elif self.path.startswith("/api/v1/stats"):
            self._json(200, {"team_name": "MOCK", "train_mode": {"images_requested": STATE["i"],
                             "submissions_received": len(STATE["done"])}})
        else:
            self._json(404, {})

    def do_POST(self):
        if not self._auth():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        iid = body.get("img_id")
        if iid not in STATE["sent"]:
            return self._json(400, {"detail": "unknown img_id"})
        if iid in STATE["done"]:
            return self._json(409, {"detail": "duplicate"})
        p = body["predictions"]
        f, t = STATE["sent"][iid]
        assert p["image_name"] == f, (p["image_name"], f)
        for o in p["objects"]:
            assert isinstance(o["id"], int) and o["class"] in CLASSES and len(o["polygon"]) >= 3
        STATE["done"].add(iid)
        json.dump(body, open(f"mock_submissions/{iid}.json", "w"))
        dt = round(time.time() - t, 3)
        print(f"submit {f}: {len(p['objects'])} objects, {dt} s")
        self._json(200, {"status": "success", "img_id": iid, "image_name": f, "duration_seconds": dt})

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
