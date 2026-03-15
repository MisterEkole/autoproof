"""
Serve the proof tree viewer at http://localhost:7331
Run alongside the orchestrator to watch the tree update live.
"""
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
PORT = 7331


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/viewer.html":
            self._serve_file(ROOT / "viewer.html", "text/html")
        elif self.path == "/snapshots":
            files = sorted(f.name for f in LOGS.glob("tree_iter_*.json")) if LOGS.exists() else []
            self._json(files)
        elif self.path.startswith("/logs/"):
            name = self.path[len("/logs/"):]
            self._serve_file(LOGS / name, "application/json")
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(data))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass  # silence request logs


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"  Proof tree viewer → {url}")
    print(f"  Serving snapshots from: {LOGS}")
    print(f"  Ctrl+C to stop\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
