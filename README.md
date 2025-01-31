# Ocean Floor Mapping with Uncertainty-Aware Deep Learning

This repository contains the implementation of "Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping", a deep learning approach for enhancing bathymetric data resolution while providing calibrated uncertainty estimates.

## Abstract

High-resolution bathymetric data is crucial for accurate ocean modeling and coastal hazard prediction, yet current global datasets remain too coarse for precise numerical simulations. While recent deep learning advances have improved earth observation data resolution, existing methods struggle with the unique challenges of generating detailed ocean floor maps, particularly in maintaining physical structure consistency and quantifying uncertainties. This paper proposes a novel uncertainty-aware Vector Quantized Variational Autoencoder with block-based conformal prediction that brings the power of discrete latent representations to generate refined bathymetry images while localizing uncertainty estimates. The architecture proposed here incorporates residual attention mechanisms with learned codebooks, aiming to capture a range of topographical features and allowing spatially adaptive prediction intervals at the block level. Experimental results show significant improvements across ocean regions and depth ranges within the generated refined bathymetry and locally calibrated uncertainty quantification.

![Model Comparison in 2D](assets/Uncertainty Comparison of Models (2d).png)

![Model Comparison in 3D](assets/Uncertainty Comparison of Models (3d).png)

*Comparison of different models: Input, Ground Truth, and predictions from SRCNN, ESRGAN, and VQVAE with their respective uncertainty estimates.*

## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{ocean-floor-mapping2025,
  title={Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping},
  author={Jose Marie Antonio Minoza},
  booktitle={ICLR Workshop on Tackling Climate Change with Machine Learning},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GEBCO for providing bathymetric data
- Previous work in deep learning-based super-resolution

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
