# DreamPhysics (PhysGaussian-based) research tool image.
# Built via GitHub Actions (x86). CUDA exts compile ahead-of-time via
# TORCH_CUDA_ARCH_LIST + FORCE_CUDA (no GPU needed at build).
#
# Base image is DreamPhysics's own pinned torch/CUDA combo (torch==2.0.0,
# cu117) -- their diff-gaussian-rasterization/simple-knn/tiny-cuda-nn CUDA
# extensions are built against this exact version, so we don't fight it with
# a mismatched base like the rest of the anchorflow project uses (torch 2.5.1).
#
# nvdiffrast/nerfacc/envlight/xatlas/libigl/pysdf/trimesh/wandb/gradio are in
# DreamPhysics's upstream requirements.txt (copied wholesale from threestudio)
# but are NEVER actually imported by ms_simulation.py/svd_simulation.py or
# anything they pull in (verified by grepping the whole repo) -- skipped here
# to avoid building nvdiffrast's CUDA extension for nothing. tiny-cuda-nn IS
# needed (utils/threestudio_utils.py imports tinycudann unconditionally).
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build libglm-dev ffmpeg libgl1 libglib2.0-0 \
        tmux openssh-client curl unzip rclone \
    && rm -rf /var/lib/apt/lists/*

# DreamPhysics + its gaussian-splatting submodule (diff-gaussian-rasterization,
# simple-knn), baked into the image at a fixed path.
RUN git clone --depth 1 https://github.com/tyhuang0428/DreamPhysics /opt/DreamPhysics \
    && cd /opt/DreamPhysics \
    && git clone --depth 1 --recursive https://github.com/graphdeco-inria/gaussian-splatting \
    && pip install --no-cache-dir -e gaussian-splatting/submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir -e gaussian-splatting/submodules/simple-knn

# PhysGaussian core requirements.
RUN pip install --no-cache-dir \
        h5py==3.10.0 "numpy==1.24.1" opencv_python==4.8.1.78 opencv_python_headless==4.9.0.80 \
        Pillow==10.2.0 plyfile==1.0.3 PyMCubes==0.1.4 pymeshlab==2023.12.post1 \
        scipy==1.12.0 setuptools==68.0.0 taichi==1.5.0 tqdm==4.66.1 warp-lang==0.10.1

# Minimal slice of threestudio's requirements actually exercised by
# ms_guidance.py/svd_guidance.py/prompt_processors.py (verified by grep, not
# threestudio's full 3D-mesh-generation stack -- that needs nvdiffrast/
# nerfacc/envlight/xatlas/libigl/pysdf which nothing here imports).
RUN pip install --no-cache-dir \
        pytorch-lightning==2.0.0 omegaconf==2.3.0 jaxtyping typeguard \
        "diffusers<0.20" transformers==4.30.2 accelerate \
        tensorboard matplotlib "imageio>=2.28.0" "imageio[ffmpeg]" torchmetrics

# tiny-cuda-nn: only used for utils.threestudio_utils.cleanup()'s
# tcnn.free_temporary_memory() call -- still a hard unconditional import.
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

WORKDIR /workspace
