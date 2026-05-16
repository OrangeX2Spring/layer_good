
<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/spaces/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<div align="center">
    <img src="assets/teaser.png" alt="Teaser image of Pixal3D"/>
</div>

**Pixal3D** generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

---

## ✨ News

- **May 2026**: Release the improved version based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone. 💪
- **May 2026**: Release inference code and online demo. 🤗
- **Apr 2026**: Our paper is accepted to SIGGRAPH 2026! 🎉

## 📌 Branches

| Branch | Description |
|--------|-------------|
| `main` | **Latest version** — improved implementation based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone with better performance. |
| `paper` | **Paper version** — original implementation based on [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), corresponding to results reported in our SIGGRAPH 2026 paper. |

> If you want to reproduce the results in our paper, please switch to the `paper` branch.

## 🎮 Try It Online

You can try Pixal3D directly in your browser without any installation via our Hugging Face Gradio demo:

👉 [**Launch Demo**](https://huggingface.co/spaces/TencentARC/Pixal3D)

## 🚀 Getting Started

### Installation

#### Step 1: Follow TRELLIS.2 Installation

Please first follow the installation guide of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) to set up the base environment.

#### Step 2: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Install utils3d

```bash
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

> **Note**: `requirements-hfdemo.txt` is for the Hugging Face Spaces demo (H-series GPU architecture) and may not be compatible with other architectures.

### Usage

#### Inference

Generate a GLB mesh from a single image:

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
```

**Low-VRAM mode** (reduces peak VRAM by loading models on-demand):

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
```

By default, the pipeline resolution is **1536** (standard mode) or **1024** (low-VRAM mode). You can override this with `--resolution`:

```bash
# Force 1536 even in low-VRAM mode
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram --resolution 1536

# Force 1024 in standard mode
python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1024
```

**Tip**: If you don't have `flash_attn` installed, you can use PyTorch's built-in SDPA backend instead:
> ```bash
> ATTN_BACKEND=sdpa python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
> ```

### Web Demo

We provide a Gradio web demo for Pixal3D, which allows you to generate 3D meshes from images interactively.

```bash
python app.py 
```

Low-VRAM mode is also available for the web demo. The frontend default resolution will automatically switch to 1024 in low-VRAM mode (1536 otherwise), but can be changed manually in the UI.

```bash
python app.py --low_vram
# or via environment variable:
LOW_VRAM=1 python app.py
```

## 🤗 Acknowledgements

This project is heavily built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2) and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We sincerely thank the authors for their outstanding work on scalable 3D generation , which serves as the foundation of our codebase and model architecture.

We also thank the following repos for their great contributions:

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```

