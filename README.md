# Neuromorphic Imaging with Density-Based Spatiotemporal Denoising

[![doi](https://img.shields.io/badge/Journal-IEEE_TCI-blue)](https://doi.org/10.1109/TCI.2023.3281202)
[![HKU](https://img.shields.io/badge/PDF-HKU-b31b1b)](https://www.eee.hku.hk/optima/pub/journal/2305_TCI.pdf)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
[![Dataset](https://img.shields.io/badge/Dataset-Available-00b894)](https://bora.teracloud.jp/share/1222aea6cef93713)
![License](https://img.shields.io/badge/License-MIT-green)
[![News](https://img.shields.io/badge/News-IntelligentOptics-07C160)](https://mp.weixin.qq.com/s/vWEu9l1Zytw9iGqhb6FYHQ)

```
@article{zhang2023tci,
  title   = {Neuromorphic Imaging with Density-Based Spatiotemporal Denoising},
  author  = {Pei Zhang and Zhou Ge and Li Song and Edmund Y. Lam},
  journal = {IEEE Transactions on Computational Imaging},
  volume  = {9}, pages = {530--541},
  year    = {2023},
  doi     = {10.1109/TCI.2023.3281202}
}
```

## Implementation
Run the denoising demo on a directory of event files:
```bash
python main.py --data-dir /path/to/input_dir --output-dir /path/to/output_dir
```
This processes `.mat` and `.bin` files in the input directory, saves visual comparison results, and writes one denoised `.mat` file for each sample.

## Sample
Download our [samples](https://bora.teracloud.jp/share/1222aea6cef93713) for verification and further research.
