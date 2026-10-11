"""Two picture backends with one interface: `render(prompt, negative, seed) -> png bytes`.

**Local** — a stable-diffusion.cpp server kept loaded for a whole batch.

Loading the weights once and posting each image to its A1111-compatible
`/sdapi/v1/txt2img` endpoint avoids re-reading several GB per icon. Decoding goes
through TAESD/TAEF1, the tiny autoencoders: the full VAE decode was 100+ s of every
image on this CPU, and the detail it adds is lost at icon size anyway.

Runs at nice 19 — the machine is shared with other builds.

**Cloudflare** — FLUX.1 [schnell] on Workers AI: the same model as the local FLUX,
seconds per image instead of half an hour, within a free daily allowance. Needs
CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
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


class QuotaExhausted(Exception):
    """The backend's allowance is spent for now; the batch stops cleanly and resumes later."""


class PromptRefused(Exception):
    """The backend's content filter declined this prompt. Only this icon is skipped; the
    filter misfires on horror vocabulary, so the fix is rewording its concept."""


class CloudflareRenderer:
    """FLUX.1 [schnell] on Workers AI.

    The endpoint takes no seed (it rejects one) and no negative prompt, so both are
    ignored: a re-roll is simply a fresh request, and the build's cache is what makes
    an interrupted run resumable.
    """

    MODEL = "@cf/black-forest-labs/flux-1-schnell"

    def __init__(self) -> None:
        try:
            account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
            self.token = os.environ["CLOUDFLARE_API_TOKEN"]
        except KeyError as missing:
            raise SystemExit(f"{missing.args[0]} is not set — see the icon_gen section of the README")
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{self.MODEL}"

    def __enter__(self) -> "CloudflareRenderer":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def render(self, prompt: str, negative: str, seed: int) -> bytes:
        body = json.dumps({"prompt": prompt[:2048], "steps": 4}).encode()
        refusals = 0
        for attempt in range(8):
            request = urllib.request.Request(self.url, data=body, headers={
                "Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                # The free allowance resets daily at 00:00 UTC; anything else that is
                # 429 is a short-lived rate limit worth waiting out.
                if error.code == 429 and ("neuron" in detail.lower() or "allocation" in detail.lower()):
                    raise QuotaExhausted(detail[:300]) from error
                if error.code == 400 and '"code":8007' in detail.replace(" ", ""):
                    # The filter is not deterministic: the same prompt it refuses once
                    # usually passes on a retry, so only a repeated refusal skips it.
                    refusals += 1
                    if refusals < 3:
                        continue
                    raise PromptRefused(detail[:300]) from error
                if error.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise SystemExit(f"Workers AI refused the request (HTTP {error.code}): {detail[:400]}")
            except urllib.error.URLError:
                if attempt < 5:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise
        if not data.get("success"):
            raise SystemExit(f"Workers AI returned an error: {data.get('errors')}")
        # A JPEG; the composer reads any format Pillow does.
        return base64.b64decode(data["result"]["image"])
