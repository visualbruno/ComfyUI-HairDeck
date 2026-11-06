# ComfyUI-HairDeck

ComfyUI nodes for [HairDeck](https://github.com/hyz317/HairDeck), which generates stylized hair cards and the
spring-bone chains that animate them from a single character image
([paper](https://arxiv.org/abs/2610.40599), [project page](https://hyzcluster.github.io/HairDeck)).

HairDeck's color model predicts 16 colors along every card, from root to tip, so highlights, tips and ribbons in the
image carry over to the hair. With the color model off, the hair gets one color measured from the image.

The nodes save the hair three ways:

- **GLB**: the hair cards as a static mesh with UVs and the per-card colors.
- **Rigged GLB**: the hair alone, with an armature and skinning weights, ready to animate or to attach to your own
  character.
- **VRM**: HairDeck's avatar export. The hair goes on the bundled avatar with VRM spring-bone physics.

The Rigged GLB and VRM nodes need Blender, but not a Blender install: they use the `bpy` Python package. A Blender
executable also works.

## Nodes

| Node | In | Out |
| --- | --- | --- |
| **HairDeck Model Loader** | offload mode, chains and colors on/off, `auto_download` | `HAIRDECK_MODEL` |
| **HairDeck Generate Hair** | `HAIRDECK_MODEL`, `IMAGE`, optional `MASK`, seed, steps, cfg, chains and colors on/off | `HAIRDECK_HAIR`, `TRIMESH`, preprocessing panel, model crop, info |
| **HairDeck Save GLB** | `HAIRDECK_HAIR`, head proxy and chain display toggles | `FILE_3D_GLB`, path |
| **HairDeck Rigged GLB** | `HAIRDECK_HAIR`, Blender backend | `FILE_3D_GLB`, path, info |
| **HairDeck Export VRM** | `HAIRDECK_HAIR`, Blender backend | `FILE_3D_GLB` (a preview copy), VRM path, info |

The `TRIMESH` output is a `trimesh.Trimesh`, the same socket as ComfyUI-Hunyuan3DWrapper and ComfyUI-Hi3D, so the
hair can go straight into their remesh, UV and export nodes. The three save nodes show their result in the node, and
their `model_3d` output connects to **Preview 3D**.

An example workflow is in [example_workflows/HairDeck.json](example_workflows/HairDeck.json). Put your character
image in ComfyUI's `input/` folder and select it in **Load Image**.

ComfyUI stores a node's settings in a saved workflow by position. Version 1.0.1 inserted the color switches in the
middle of the Model Loader and Generate Hair settings; they are now the last setting of each node. Workflows saved
with 1.0.0 load correctly. In a workflow saved with 1.0.1, re-add those two nodes or load the example workflow again.

### Model Loader

| Setting | Default | Meaning |
| --- | --- | --- |
| `offload` | on | `on`: the models wait in system RAM and visit the GPU only for their stage, which leaves the GPU to the rest of the workflow. `auto`: HairDeck decides from the free VRAM. `off`: everything stays on the GPU (fastest, about 10 GB). |
| `load_chains` | on | Loads the chain model. Rigging and VRM need chains. |
| `load_colors` | on | Loads the color model. Off: one hair color for the whole hairstyle. |
| `auto_download` | on | Fetches missing weights into `models/hairdeck` (see [Weights](#weights)). |
| `copy_from_workspace` | – | An existing HairDeck workspace (the folder holding `ckpt/`, `data/`, `third_party/`) to copy missing files from instead of downloading them. |
| `gated_source` | huggingface | Where SAM 3 and DINOv3 come from: `huggingface` or `modelscope`. Both are checked against the same sha256. |

### Generate Hair

| Setting | Default | Meaning |
| --- | --- | --- |
| `seed` | 0 | Seed of the cards. The chains use the same seed. Each seed gives a different hairstyle. |
| `steps` | 50 | Euler steps of the image model. |
| `cfg` | 2.0 | Classifier-free guidance of the image model. |
| `generate_chains` | on | Generates the spring-bone chains. Some seeds produce no chains; that hair then only follows the head. |
| `colors` | on | Colors every card with 16 colors from root to tip (needs `load_colors`). Off: one hair color, measured from the image's hair mask. |
| `filename_prefix` | HairDeck/hair | Result folder under `output/`, such as `output/HairDeck/hair_00001_/`. |
| `mask` / `invert_mask` | – | Optional foreground mask. HairDeck finds the character by itself (SAM 3 + BiRefNet), so a mask only helps on busy backgrounds. Turn `invert_mask` on for the MASK of the core Load Image node. |

Each run writes HairDeck's usual result folder: the input, the preprocessing panel, `cards_seed<s>.json` (with each
card's `colors`, sRGB 0–1 from root to tip), `chains_seed<s>.json`, the GLBs and `info.json`. The Rigged GLB and VRM
nodes work from that folder.

With colors, the `TRIMESH` output and the GLBs carry them as vertex colors (`COLOR_0`) on a white base color.

### Save GLB

| Setting | Default | Meaning |
| --- | --- | --- |
| `include_head_proxy` | off | Adds HairDeck's grey head ellipsoid, a fitting aid. |
| `show_chains` | off | Draws the spring-bone chains as orange tubes. These are display geometry, not bones. |

The mesh is in HairDeck's head frame: Y up, the face toward +Z, metres.

### Rigged GLB

The hair alone, rigged:

- **Bones:** a `Head` root bone, plus one bone chain per generated spring chain (`HairGen_c00_00`, `HairGen_c00_01`,
  …, ending in `HairGen_c00_end`).
- **Weights:** HairDeck's own skinning. Each vertex is weighted between `Head` and up to two chain bones, and cards
  that no chain drives stay on `Head`.
- **Material:** double-sided. With colors, the base color is a small color texture (one 8-pixel column per card,
  embedded in the GLB) that HairDeck builds for its VRM export, and the mesh UVs point into it. Without colors, the
  material has the single hair color and the UVs are per card.
- **Stiffness:** the predicted stiffness of each chain is stored in the armature's glTF extras
  (`hairdeck_stiffness`, with the root bones in `hairdeck_chain_roots`), for a spring or jiggle-bone setup in your
  engine.

The hair is placed where the bundled avatar's head is, about 1.4 m up. To use it on your own character, parent the
`Head` bone to your character's head bone.

A GLB has no standard way to store spring physics, so the hair doesn't move on its own. Animate the bones, or add
spring bones in your engine.

### Export VRM

This runs HairDeck's VRM 0.x export unchanged:

1. It removes the bundled avatar's own hair.
2. It adds the generated hair with an MToon material, skinned to new chain bones under the avatar's head bone. With
   colors, the per-card color texture is both the lit and the shade texture.
3. It writes one VRM spring-bone group per chain, using the predicted stiffness, drag 0.4, no gravity and the
   avatar's colliders. Chain bones that start inside the head stay rigid.
4. It validates the file with HairDeck's own checks.

The VRM file contains no animation. The physics runs in VRM players and tools: UniVRM in Unity, three-vrm, VSeeFace,
warudo, VRoid Hub. Blender's VRM Add-on doesn't simulate VRM 0.x spring bones.

## Blender: `bpy` or an executable

The `blender` setting on the Rigged GLB and VRM nodes:

- `bpy` (also what `auto` picks when the package is installed): the
  [bpy](https://pypi.org/project/bpy/) package in ComfyUI's own Python. It runs in a child process, so Blender's
  state and libraries never enter the ComfyUI process. Install the version that matches your Python:
  ```
  python -m pip install bpy==5.2.3      # Python 3.13 (bpy 5.1+)
  ```
  `bpy` 4.x and 5.0 are for Python 3.11. Installing `bpy` 5.2.3 into a Python 3.13 ComfyUI adds only `bpy`, `Cython`
  and `cattrs`.
- `executable`: a Blender install, 4.2 LTS or later. It is found from `blender_path`, then `HAIRDECK_BLENDER`, then
  the newest `Program Files\Blender Foundation\Blender *`, then `blender` on the PATH.

With either backend, the VRM node installs the VRM Add-on for Blender that HairDeck pins (4.10.2, sha256-checked).
It goes into `models/hairdeck/blender_user`, and Blender runs with that folder as its user profile. So your own
Blender profile and add-ons are never read or changed, and you don't need to install the VRM Add-on yourself. Before
the export, the node also patches two things at run time:

- **Blender 5.2's glTF exporter** returns three values where the add-on expects two, which made every material fail
  to export.
- **The MToon conversion** never marks the material as MToon in a background run, so the hair was exported as a
  plain glTF material and lost its MToon colors and color texture.

The patches are in [blender/bpy_runner.py](blender/bpy_runner.py).

## Weights

The Model Loader puts HairDeck's workspace in `ComfyUI/models/hairdeck` (or uses `HAIRDECK_ROOT` when that
environment variable is set). The files come from HairDeck's own downloader, each checked against its sha256:

| Item | Size | Source |
| --- | --- | --- |
| HairDeck checkpoints and data (including the color model) | 2.2 GB | [hyz317/HairDeck](https://huggingface.co/hyz317/HairDeck) |
| SAM 3 | 3.3 GB | [facebook/sam3](https://huggingface.co/facebook/sam3) (gated) |
| DINOv3 ViT-L/16 | 1.2 GB | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (gated) |
| BiRefNet-matting | 0.85 GB | [ZhengPeng7/BiRefNet-matting](https://huggingface.co/ZhengPeng7/BiRefNet-matting) |
| SAM 3 and DINOv3 code | 0.1 GB | GitHub, at pinned commits |

SAM 3 and DINOv3 are gated. Accept their licences on Hugging Face and log in (`hf auth login`) before the first
run, or set `gated_source` to `modelscope`. If you already ran HairDeck on its own, set `copy_from_workspace` to
that workspace instead of downloading again.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/visualbruno/ComfyUI-HairDeck
python -m pip install -r ComfyUI-HairDeck/requirements.txt
python -m pip install bpy==5.2.3      # optional: Rigged GLB and VRM without a Blender install (Python 3.13)
```

Most requirements are already in a ComfyUI install. HairDeck needs an NVIDIA GPU. HairDeck's own code is bundled,
unmodified, in [HairDeck/](HairDeck/UPSTREAM).

Other node packs may bundle their own SAM 3 or DINOv3 under the same module names `sam3` and `dinov3`. One example
is ComfyUI-RMBG, which imports its `sam3` at startup. While a HairDeck node runs, HairDeck's own copies take those
names, and the other pack's modules are put back afterwards, so the packs work side by side.

## Performance

Measured on an RTX 3080 Laptop GPU (16 GB) with Python 3.13, torch 2.10 and CUDA 13.0, offload `on`:

| Step | Time |
| --- | --- |
| Model Loader (first run) | 15 s |
| Generate Hair (preprocessing 4.1 s, cards 1.7 s, colors 1.2 s, chains 1.1 s) | 8 s |
| Rigged GLB (`bpy`; 7 s through a Blender executable) | 1.5 s |
| Export VRM (`bpy`; 27 s through a Blender executable) | 20 s |

Peak VRAM with offload `on` is about 6.7 GB, during matting.

## Credits and licence

HairDeck by Yuze He, Yixiang Wei, Wang Zhao, Chuhao Chen, Yanning Zhou and Yong-Jin Liu (Tsinghua University, Carnegie
Mellon University, The Chinese University of Hong Kong). If you use HairDeck, cite the paper:

```bibtex
@misc{hairdeck2026,
  title         = {HairDeck: Animation-Ready Stylized Hair Cards from a Single Image},
  author        = {Yuze He and Yixiang Wei and Wang Zhao and Chuhao Chen and Yanning Zhou and Yong-Jin Liu},
  year          = {2026},
  eprint        = {2610.40599},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.40599},
  url           = {https://arxiv.org/abs/2610.40599}
}
```

This node pack and HairDeck's code are licensed under [Apache-2.0](LICENSE). The downloaded models keep their own
licences: SAM 3 (SAM License), DINOv3 (DINOv3 License), BiRefNet (MIT), the VRM Add-on for Blender (MIT). See
[NOTICE](NOTICE).
