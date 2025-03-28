<div align="center">

<h1>Generative Powers of Ten</h1>

*<div align="left">from: "Path leading to the dense forest..."</div>*<br>
![teaser](./assets/forest_teaser.png)
*<div align="right">to: "... close-up of tree bark showing small cracks, lichen, and insects"</div>*

Implementation / replication of <a href="https://powers-of-10.github.io/">Generative Powers of Ten</a>, using <a href="http://stability.ai/">Stability AI</a>'s <a href="https://github.com/deep-floyd/IF">DeepFloyd IF</a>.

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

## 🌄 Other Examples

*<div align="left">from: "A aerial photo capturing Hawaii's islands..."</div>*<br>
![teaser](./assets/hawaii_teaser.png)
*<div align="right">to: "... a man standing on the edge of a volcano's caldera, waving at the camera"</div>*

*<div align="left">from: "A sunflower field from afar..."</div>*<br>
![teaser](./assets/sunflower_teaser.png)
*<div align="right">to: "... the honey bee sipping nectar and transferring pollen"</div>*

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
