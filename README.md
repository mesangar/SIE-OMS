# Semantic and Structural Image Segmentation for Prosthetic Vision

Prosthetic vision is being applied to partially recover the retinal stimulation of visually impaired people. However, the phosphenic images produced by the implants have very limited information bandwidth due to the poor resolution and lack of color or contrast. Here we prpose a new visual stimuli called SIE-OMS based on semantic segmentation for indoor scene understanding using visual prosthesis.



- To run the code:
```
generate_schematic_images.m

# Structural Informative Edges (SIE)
For Structural Informative Edges, we use the framework of CFL which uses a Fully Convolutional Network (FCN) for pixel classification to estimate probability maps representing the room structural edges. 

- To run the code:
```
persp_predict.py
```

# Object Mask and Silhouettes (OMS)
For Object Mask and Silhouettes, we perform instance segmentation of objects using the architecture of Mask R-CNN. Using selective search, it identifies a manageable number of bounding-box object region candidates (ROI). Then, it extracts CNN features from each region independently for classification and extract a binary mask in pixel level of the object.

- To run the code:
```
objects.py
```

# SIE-OMS
The final representation of the SIE-OMS method is a superposition of both parts, SIE and OMS, always assuming the edges as background and object masks as foreground. Finally, we use use a simulated phosphene map on a computer screen, but the same activation images could be directly applied to the retinal implant.
- To run the code:
```
generate_phosp_image.py
```


You can find more details in S1 Appendix [Supplementary material] (SIEOMS_supp.pdf).

## Related Research
Please cite these papers in your publications if it helps your research, ``[Sanchez-Garcia et al.]``.

```bibtex
@article{sanchez2020semantic,
  title={Semantic and structural image segmentation for prosthetic vision},
  author={Sanchez-Garcia, Melani and Martinez-Cantin, Ruben and Guerrero, Jose J},
  journal={PloS one},
  volume={15},
  number={1},
  pages={e0227677},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
For commercial purposes, please contact the authors.
