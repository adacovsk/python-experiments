#!/bin/bash
# Fetch the stable-diffusion.cpp CPU binary and the model weights icon_gen uses.
# Everything lands in gitignored paths: tools/sd/ and model_weights/{flux,sd_turbo}/.
# Re-running skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Pinned to a release a couple of weeks old rather than whatever is latest.
SD_RELEASE="master-920-2f88688"
if [ ! -x tools/sd/sd-cli ]; then
    mkdir -p tools/sd
    gh release download "$SD_RELEASE" -R leejet/stable-diffusion.cpp \
        -p '*Linux-Ubuntu-24.04-x86_64.zip' -D tools/sd --clobber
    (cd tools/sd && unzip -o -q ./*.zip && rm ./*.zip)
fi

fetch() {  # fetch <dest> <url>
    [ -s "$1" ] && return 0
    mkdir -p "$(dirname "$1")"
    echo "Downloading $1"
    curl -fL --retry 3 -o "$1.part" "$2" && mv "$1.part" "$1"
}

HF=https://huggingface.co
# FLUX.1 [schnell] — Apache 2.0, so its output is safe to ship. Q4_0 fits in RAM on CPU.
fetch model_weights/flux/flux1-schnell-Q4_0.gguf "$HF/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_0.gguf"
fetch model_weights/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf "$HF/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf"
fetch model_weights/flux/clip_l.safetensors "$HF/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
# The FLUX VAE. Black Forest Labs gates its own copy behind a login; ComfyUI's
# repackage is the same file (same size, and the hash below matches a second mirror).
fetch model_weights/flux/ae.safetensors "$HF/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors"
echo "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38  model_weights/flux/ae.safetensors" | sha256sum -c -
# SD-Turbo — Stability Community License (free commercial use under $1M/yr revenue).
# One-step model: fast enough to iterate on prompts before committing FLUX time.
fetch model_weights/sd_turbo/sd_turbo.safetensors "$HF/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors"
# Tiny autoencoders: the full VAE decode dominated each image on this CPU (~100 s
# of ~150 s), and the detail it adds is lost at icon size.
fetch model_weights/sd_turbo/taesd.safetensors "$HF/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors"
fetch model_weights/flux/taef1.safetensors "$HF/madebyollin/taef1/resolve/main/diffusion_pytorch_model.safetensors"
echo "icon_gen setup complete"
