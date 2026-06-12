# Haptic Neural Fields

This is the official repository for the paper *Haptic Neural Fields: Bringing Tactile Interactions to 3D Rendered Scenes*.

Visit the [project website](https://mmlab-cv.github.io/HapticNeuralFields/).

## First Run

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository includes a training example for the *Bubble Envelope* material in the indoor table scene.

The `Digit` directory contains haptic maps classified as *Bubble Envelope* by the proposed classifier. The `1-RecordedData_Friction` directory contains the corresponding acceleration, position, and force measurements.

Run a short CPU smoke test with one material and one training epoch:

```bash
python main.py \
  --device=cpu \
  --num_of_materials=1 \
  --num_epochs=1 \
  --validation_interval=1 \
  --checkpoint_interval=1 \
  --batch_size=8 \
  --num_workers=0
```

Training outputs are saved in a timestamped directory under `output/`.

For a full training run, adjust the flags as needed. For example:

```bash
python main.py
```

## Generate Novel Actions

Example actions are available in the `novel_actions` directory. After training a model, generate a signal for an action with:

```bash
python generate_novel_action.py \
  --action=Scratch_LeftRight_Strong \
  --image=Digit/BubbleEnvelope/00487.jpg
```

By default, the script uses the newest checkpoint under `output/` and automatically selects CUDA when available. Use `--checkpoint`, `--action`, `--image`, or `--device` to override these defaults.

## References

1. Dou, Yiming, et al. [*Tactile-Augmented Radiance Fields*](https://openaccess.thecvf.com/content/CVPR2024/papers/Dou_Tactile-Augmented_Radiance_Fields_CVPR_2024_paper.pdf), Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
2. Culbertson, Heather, Juan José López Delgado, and Katherine J. Kuchenbecker. [*One Hundred Data-Driven Haptic Texture Models and Open-Source Methods for Rendering on 3D Objects*](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6775475&casa_token=zOuADNnmP4oAAAAA:SZqGqFxesT819U8sFcaBdiJJBJH8QPXjt5OfH2Gc7x1ys_2LNv9N3t31PMnV3AGoIthVWL9s), 2014 IEEE haptics symposium (HAPTICS). IEEE, 2014.
3. Heravi, Negin, et al. [*Development and evaluation of a learning-based model for real-time haptic texture rendering*](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10480578&casa_token=vcXtN_KlMkEAAAAA:rcBwkMSHlLLfpAPm4Md8HZy8ZqvPfU5J_f_TjPY1paODsO-C8QerlLj4PI4jvmQdtWcWvpt2&tag=1), IEEE transactions on haptics 17.4 (2024): 705-716.
4. Stefani, Antonio Luigi, et al. [*Haptic Neural Fields: Bringing Tactile Interactions to 3D Rendered Scenes*](https://openaccess.thecvf.com/content/CVPR2026/papers/Stefani_Haptic_Neural_Fields_Bringing_Tactile_Interactions_to_3D_Rendered_Scenes_CVPR_2026_paper.pdf), Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2026.
