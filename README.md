<div align="center">

<h1>Generative Powers of Ten</h1>

Implementation of <a href="https://powers-of-10.github.io/">Generative Powers of Ten</a>, using <a href="http://stability.ai/">Stability AI</a>'s <a href="https://github.com/deep-floyd/IF">DeepFloyd IF</a>.

<video src="./assets/animation_teaser.mp4" width="75%"></video>
![teaser](./assets/forest_teaser.png)

</div>

## 🛠️ Usage

First, refer to the `examples` folder to see how to structure your prompts and zoom level `p` in a `metadata.json` file.

Then, simply run:

```bash
python generate.py --name <name> --seed <seed>
```

You can customize various generation parameters (e.g., `steps`, `cfg`, `negative`, etc.) as well.

To generate a video animation based on the generated images, run:

```bash
python make_animation.py <generated_images_folder>
```

## Other Examples

*from: "A aerial photo capturing Hawaii's islands..."*
*to: "... a man standing on the edge of a volcano's caldera, waving at the camera"*
![teaser](./assets/hawaii_teaser.png)

*from: "A sunflower field from afar"*
*to: "... the honey bee sipping nectar and transferring pollen"*
![teaser](./assets/sunflower_teaser.png)

## 📑 Citation

If you find this repository useful, consider giving it a star 🌟 and refer to the original paper:

```bibtex
@article{wang2023generativepowers,
    title={Generative Powers of Ten},
    author={Xiaojuan Wang and Janne Kontkanen and Brian Curless and Steve Seitz and Ira Kemelmacher 
        and Ben Mildenhall and Pratul Srinivasan and Dor Verbin and Aleksander Holynski},
    journal={arXiv preprint arXiv:2312.02149},
    year={2023}
}
```

## 🤝 Acknowledgements

Thanks to [Yifan Zhou](https://github.com/SingleZombie) for providing the foundation to reproducing this work.
