# Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping

**[ICLR 2025 Tackling Climate Change with Machine Learning Workshop - Accepted Paper]**

[![DOI](https://zenodo.org/badge/DOI/10.5281/ZENODO.15272540.svg)](https://doi.org/10.5281/ZENODO.15272540)
[![arXiv](https://img.shields.io/badge/arXiv-2504.14372-b31b1b.svg)](https://arxiv.org/abs/2504.14372)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Ocean%20Floor%20Bathymetry-blue)](https://huggingface.co/datasets/jomaminoza/global-ocean-floor-bathymetry-enhancement-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation of "Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping" (ICLR 2025 - Tackling Climate Change with ML Workshop). This study proposed a novel block-based uncertainty awareness mechanism during training, which when integrated with the Vector Quantized Variational Autoencoder (VQ-VAE) architecture, achieves superior bathymetric enhancement while preserving critical seafloor structures, enabling more accurate climate modeling and coastal hazard assessment.

![Model Comparison in 2D](assets/main.png)

## Abstract


Accurate ocean modeling and coastal hazard prediction depend on high-resolution bathymetric data; yet, current worldwide datasets are too coarse for exact numerical simulations. While recent deep learning advances have improved earth observation data resolution, existing methods struggle with the unique challenges of producing detailed ocean floor maps, especially in maintaining physical structure consistency and quantifying uncertainties. 

This work presents a novel uncertainty-aware mechanism using spatial blocks to efficiently capture local bathymetric complexity based on block-based conformal prediction. Using the Vector Quantized Variational Autoencoder (VQ-VAE) architecture, the integration of this uncertainty quantification framework yields spatially adaptive confidence estimates while preserving topographical features via discrete latent representations. With smaller uncertainty widths in well-characterized areas and appropriately larger bounds in areas of complex seafloor structures, the block-based design adapts uncertainty estimates to local bathymetric complexity. 

Compared to conventional techniques, experimental results over several ocean regions show notable increases in both reconstruction quality and uncertainty estimation reliability. This framework increases the reliability of bathymetric reconstructions by preserving structural integrity while offering spatially adaptive uncertainty estimates, opening the path for more solid climate modeling and coastal hazard assessment.

## Key Contributions

1. **VQ-VAE with Residual Attention**: An efficient adaptation of VQ-VAE with residual attention mechanisms for enhancing bathymetry, demonstrating superior empirical performance in capturing diverse topographic features while preserving structural consistency essential for accurate physical modeling of ocean events.

2. **Block-based Uncertainty Mechanism**: Block-wise uncertainty estimates are included into the loss function in a practical implementation that essentially solves the problem of spatially varying data quality with quantifiable confidence bound calibration (0.0138 calibration error vs. 0.0314-0.0374 for alternatives).

3. **Comprehensive Analysis**: Consistent performance improvements over conventional interpolation methods (26.88 dB vs. 15.85 dB PSNR) and other deep learning approaches in both reconstruction accuracy and uncertainty estimation.

## Methodology

### Vector Quantized Variational Autoencoder (VQ-VAE)

The VQ-VAE architecture maps input bathymetry x to a discrete latent space:

$$z = E(x), z_q = VQ(z)$$

where VQ performs vector quantization using a learned codebook $C = \{e_k\}_{k=1}^K$ of K embedding vectors. The quantization selects the nearest codebook entry:

$$z_q = e_k, k = \arg\min_i \|z - e_i\|^2$$

The decoder reconstructs the high-resolution bathymetry from the quantized representation:

$$\hat{x} = D(z_q)$$

### Block-based Uncertainty Awareness

The uncertainty tracking mechanism uses exponential moving averages (EMA) on spatial blocks:

$$EMA_i^{(t)} = \alpha EMA_i^{(t-1)} + (1 - \alpha) \frac{1}{|b_i|} \sum_{x,y \in b_i} |f(x,y) - \hat{f}(x,y)|$$

The block-wise uncertainty scores are normalized against historical error statistics:

$$U_i = \frac{\text{block\_error}_i}{EMA_{i,1-\alpha} + \epsilon}$$

### Integrated Loss Function

The training objective aggregates structural similarity preservation via SSIM loss with uncertainty-weighted reconstruction:


$$L = \sum_i U_i \cdot |D(z_q)_i - x_i|^2 + \lambda_s(1 - SSIM(D(z_q), x)) + \lambda_c L_{vq} + \lambda_d L_{div}$$

where $L_{vq}$ is the codebook commitment loss that ensures meaningful latent representations, and $L_{div}$ promotes codebook diversity to capture the full range of bathymetric features.

## Results

### Overall Model Performance

| Model | SSIM | PSNR | MSE | MAE | UWidth | CalErr |
|-------|------|------|-----|-----|--------|--------|
| Nearest | 0.6784 | 15.8114 | 0.0271 | 0.1140 | - | - |
| Bilinear | 0.7045 | 15.8568 | 0.0268 | 0.1131 | - | - |
| Bicubic | 0.7011 | 15.8271 | 0.0270 | 0.1135 | - | - |
| UA-SRCNN | 0.8128 | 18.7577 | 0.0137 | 0.0822 | 0.2966 | 0.0314 |
| UA-ESRGAN | 0.7582 | 19.2006 | 0.0123 | 0.0821 | 0.2691 | 0.0374 |
| UA-VQ-VAE | **0.9433** | **26.8779** | **0.0021** | **0.0317** | **0.1046** | **0.0138** |

### Regional Performance (UA-VQ-VAE)

| Region | SSIM | PSNR | MSE | MAE | UWidth | CalErr |
|--------|------|------|-----|-----|--------|--------|
| Eastern Atlantic Coast | 0.9419 | 26.3154 | 0.0024 | 0.0340 | 0.1049 | 0.0161 |
| Eastern Pacific Basin | 0.9525 | 27.0408 | 0.0020 | 0.0315 | 0.1048 | 0.0133 |
| Indian Ocean Basin | 0.9072 | 26.1573 | 0.0025 | 0.0346 | 0.1047 | 0.0171 |
| North Atlantic Basin | 0.9301 | 26.5595 | 0.0022 | 0.0328 | 0.1046 | 0.0147 |
| South Pacific Region | 0.9336 | 26.9907 | 0.0020 | 0.0310 | 0.1043 | 0.0133 |
| Western Pacific Region | 0.9385 | 27.3164 | 0.0019 | 0.0292 | 0.1041 | 0.0117 |

### Block Size Analysis (UA-VQ-VAE)

| Block Size | Avg. SSIM | Avg. PSNR | Avg. UWidth | Avg. CalErr |
|------------|-----------|-----------|-------------|-------------|
| 1×1 | 0.9186 | 26.0172 | 0.1242 | 0.0130 |
| 2×2 | 0.9242 | 26.0309 | 0.1199 | 0.0138 |
| 4×4 | **0.9340** | **26.7300** | 0.1046 | 0.0144 |
| 8×8 | 0.9307 | 26.6648 | 0.0954 | 0.0158 |
| 64×64 | 0.9337 | 26.8750 | **0.0441** | 0.0203 |

### Regional Dataset Distribution

| Region | Train | Validation | Events |
|--------|-------|------------|--------|
| Eastern Pacific Basin | 24000 | 6000 | Frequent tsunamis, submarine volcanism |
| Eastern Atlantic Coast | 14400 | 3600 | Tsunami-prone, coastal flooding |
| Western Pacific Region | 12000 | 3000 | Megathrust earthquakes, tsunamis |
| South Pacific Region | 6400 | 1600 | Cyclones, wave-driven inundation |
| North Atlantic Basin | 4000 | 1000 | Hurricanes, storm surges |
| Indian Ocean Basin | 720 | 180 | Tsunami risk, tectonic activity |
| **Total** | **61520 (80%)** | **15380 (20%)** | |

## Theoretical Foundations

### Block Size Trade-off

For a fixed block size k in bathymetric enhancement, we define:
- $E_{stat}(k)$: statistical estimation error
- $E_{feat}(k)$: feature preservation error

The fundamental trade-off is:

$$E_{total}(k) = \frac{\sigma^2}{k^2} + \lambda k$$

The optimal block size $k^*$ that minimizes this error is:

$$k^* = \left(\frac{\sigma^2}{2(\Delta z/\Delta x)^2}\right)^{1/4}$$

This represents a global compromise between measurement noise reduction and bathymetric structure preservation.



---

## Visual Results

### Model Comparison in 2D

The figure below presents a comprehensive visual comparison between different deep learning approaches for bathymetric enhancement. The visualization shows how each model handles the critical task of reconstructing high-resolution bathymetry while providing uncertainty estimates.

*Comparison of different models: Input, Ground Truth, and predictions from SRCNN, ESRGAN, and VQVAE with their respective uncertainty estimates.*

**_Model Comparison in 2D_**

![Model Comparison in 2D](https://github.com/JomaMinoza/Ocean-Floor-Mapping-with-Uncertainty-Aware-Deep-Learning/blob/d9cd1f094e48cc84b5e9346ad1785c9f25034e19/assets/Fig%202%20-%20Uncertainty%20Comparison%20of%20Models.png)

The 2D comparison reveals that while all models attempt to enhance bathymetric details, the VQ-VAE architecture produces reconstructions with significantly lower error rates (brighter regions in error maps indicate larger deviations) and more calibrated uncertainty estimates. Notice how the uncertainty blocks for VQ-VAE show appropriate confidence levels that correlate well with areas of bathymetric complexity.

### Detailed Model Results

#### Uncertainty-Aware SRCNN Results

![Uncertainty Aware SRCNN Results](assets/Fig%203%20-%20Uncertainty%20Plot%20-%20SRCNN.png)

The UA-SRCNN model shows moderate enhancement capabilities but struggles with complex bathymetric features. The uncertainty width ranges from 0.1793 to 0.3855, indicating relatively high uncertainty across the reconstruction. While providing some improvement over traditional interpolation methods, the CNN architecture's limited receptive field affects its ability to capture long-range dependencies in seafloor structures.

#### Uncertainty-Aware ESRGAN Results

![Uncertainty Aware ESRGAN Results](assets/Fig%204%20-%20Uncertainty%20Plot%20-%20ESRGAN.png)

The UA-ESRGAN model demonstrates improved visual quality compared to SRCNN, leveraging its generative adversarial architecture. However, with uncertainty widths ranging from 0.1706 to 0.3290, it still exhibits considerable uncertainty in its predictions. The adversarial training objective sometimes prioritizes perceptual quality over physical accuracy, which can be problematic for bathymetric applications requiring structural fidelity.

#### Uncertainty-Aware VQ-VAE Results by Block Size

The following visualizations demonstrate how different block sizes affect the reconstruction quality and uncertainty estimates in the VQ-VAE model:

**Block Size 1×1 Results**

![Uncertainty Aware VQVAE - Block Size 1 Results](assets/Fig%205%20-%20Uncertainty%20Plot%20-%20VQVAE%20-%20Block%20Size%201.png)

With the smallest 1×1 block size, the model captures fine-grained details but may occasionally lack global coherence. The uncertainty estimates are relatively consistent but do not fully adapt to local complexity.

**Block Size 2×2 Results**

![Uncertainty Aware VQVAE - Block Size 2 Results](assets/Fig%206%20-%20Uncertainty%20Plot%20-%20VQVAE%20-%20Block%20Size%202.png)

The 2×2 block size offers slightly improved performance over 1×1, with better balance between local detail and regional context. The uncertainty map shows more defined spatial variation.

**Block Size 4×4 Results**

![Uncertainty Aware VQVAE - Block Size 4 Results](assets/Fig%207%20-%20Uncertainty%20Plot%20-%20VQVAE%20-%20Block%20Size%204.png)

The 4×4 block size provides an optimal balance between local detail preservation and global structural coherence. With uncertainty widths ranging from approximately 0.08 to 0.20, this configuration demonstrates both high reconstruction accuracy and well-calibrated uncertainty estimates, particularly in regions with complex bathymetric features. The spatial adaptivity of the uncertainty estimates is clearly visible, with higher confidence in well-characterized areas and appropriately wider bounds in regions of structural complexity.

**Block Size 8×8 Results**

![Uncertainty Aware VQVAE - Block Size 8 Results](assets/Fig%208%20-%20Uncertainty%20Plot%20-%20VQVAE%20-%20Block%20Size%208.png)

With 8×8 blocks, the model maintains good reconstruction quality while further reducing uncertainty width. However, it may occasionally lose some fine details in highly complex regions.

**Block Size 64×64 Results**

![Uncertainty Aware VQVAE - Block Size 64 Results](assets/Fig%209%20-%20Uncertainty%20Plot%20-%20VQVAE%20-%20Block%20Size%2064.png)

The largest 64×64 block size produces the lowest overall uncertainty widths and excellent PSNR values, but at the cost of potentially missing localized features. The uncertainty map becomes more homogeneous, which may not accurately reflect local variations in prediction confidence.

These visualizations collectively demonstrate the superior performance of the VQ-VAE architecture with block-based uncertainty awareness, particularly with the 4×4 block size configuration that achieves an optimal balance between reconstruction accuracy and appropriate uncertainty quantification. This approach shows particular promise for applications in climate modeling and coastal hazard assessment, where both high-resolution bathymetry and reliable confidence estimates are crucial.

---


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{minoza2025learning,
  title={Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping},
  author={Minoza, Jose Marie Antonio},
  booktitle={ICLR 2025 Workshop on Tackling Climate Change with Machine Learning},
  url={https://www.climatechange.ai/papers/iclr2025/14},
  doi = {10.48550/arxiv.2504.14372}
  year={2025}
}

@misc{ocean-floor-mapping2025_modelzoo,
  doi = {10.5281/ZENODO.15272540},
  author = {Minoza, Jose Marie Antonio},
  title = {Model Zoo: Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping},
  publisher = {Zenodo},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GEBCO for providing bathymetric data
- The ICLR Tackling Climate Change with Machine Learning workshop organizers
- System Modelling and Simulation Laboratory, University of the Philippines
- Center for AI Research, Department of Education, Philippines

## Contact

For questions or collaborations, please contact:
- Jose Marie Antonio Minoza - jminoza@upd.edu.ph
