"""A stable-diffusion.cpp server kept loaded for a whole batch.

Loading the weights once and posting each image to its A1111-compatible
`/sdapi/v1/txt2img` endpoint avoids re-reading several GB per icon. Decoding goes
through TAESD/TAEF1, the tiny autoencoders: the full VAE decode was 100+ s of every
image on this CPU, and the detail it adds is lost at icon size anyway.

Runs at nice 19 — the machine is shared with other builds.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD_SERVER = ROOT / "tools" / "sd" / "sd-server"
WEIGHTS = ROOT / "model_weights"


@dataclass(frozen=True)
class Model:
    """The server's weight arguments and the sampler settings the model was distilled for."""

    files: tuple[str, ...]
    steps: int
    sampler: str


MODELS = {
    # Distilled to 4 steps and trained without CFG.
    "flux": Model(
        files=(
            "--diffusion-model", str(WEIGHTS / "flux" / "flux1-schnell-Q4_0.gguf"),
            "--t5xxl", str(WEIGHTS / "flux" / "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
            "--clip_l", str(WEIGHTS / "flux" / "clip_l.safetensors"),
            "--vae", str(WEIGHTS / "flux" / "ae.safetensors"),
            "--taesd", str(WEIGHTS / "flux" / "taef1.safetensors"),
        ),
        steps=4,
        sampler="euler",
    ),
    # Adversarially distilled for a single step, also without CFG.
    "turbo": Model(
        files=(
            "--model", str(WEIGHTS / "sd_turbo" / "sd_turbo.safetensors"),
            "--taesd", str(WEIGHTS / "sd_turbo" / "taesd.safetensors"),
        ),
        steps=1,
        sampler="euler_a",
    ),
}


class Renderer:
    """`with Renderer("turbo") as r: png_bytes = r.render(prompt, negative, seed)`."""

    def __init__(self, model: str, size: int = 512, port: int = 7861, log: Path | None = None):
        self.model = MODELS[model]
        self.size = size
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        self.log = log or ROOT / "src" / "icon_gen" / "output" / "sd-server.log"
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "Renderer":
        if not SD_SERVER.exists():
            raise SystemExit("sd-server not found — run `pixi run icons-setup` first")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        # One thread per physical core: hyperthreads add little to dense matrix work.
        threads = max(1, (os.cpu_count() or 2) // 2)
        self.process = subprocess.Popen(
            ["nice", "-n", "19", str(SD_SERVER), *self.model.files, "-t", str(threads), "--listen-port", str(self.port)],
            stdout=self.log.open("w"), stderr=subprocess.STDOUT,
            env={**os.environ, "LD_LIBRARY_PATH": str(SD_SERVER.parent)},
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise SystemExit(f"sd-server exited while loading; see {self.log}")
            try:
                urllib.request.urlopen(f"{self.url}/sdapi/v1/samplers", timeout=2)
                return self
            except OSError:
                time.sleep(2)
        raise SystemExit(f"sd-server did not come up in 15 minutes; see {self.log}")

    def __exit__(self, *_: object) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # At nice 19 on a loaded machine it may not get scheduled to handle
                # SIGTERM promptly; it holds several GB, so do not leave it behind.
                self.process.kill()
                self.process.wait()

    def render(self, prompt: str, negative: str, seed: int) -> bytes:
        body = json.dumps({
            "prompt": prompt, "negative_prompt": negative,
            "width": self.size, "height": self.size,
            "steps": self.model.steps, "cfg_scale": 1.0,
            "seed": seed, "sampler_name": self.model.sampler,
        }).encode()
        request = urllib.request.Request(
            f"{self.url}/sdapi/v1/txt2img", data=body, headers={"Content-Type": "application/json"},
        )
        # No timeout: under load a FLUX render has taken over an hour, and abandoning it
        # throws the work away. A dead server is caught by the connection dropping.
        with urllib.request.urlopen(request) as response:
            return base64.b64decode(json.load(response)["images"][0])
