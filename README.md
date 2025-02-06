# Thickness-aware E(3)-Equivariant Mesh Neural Networks (T-EMNN)

## Abstract 
Mesh graph-based 3D static analysis methods have recently emerged as efficient alternatives to traditional computational numerical solvers, significantly reducing computational costs and runtime for various physics-based analyses. However, these methods primarily focus on surface topology and geometry, often overlooking the inherent thickness of real-world 3D objects, which exhibits high correlations and similar behavior between opposing surfaces. This limitation arises from the disconnected nature of these surfaces and the absence of internal edge connections within the mesh.
In this work, we propose a novel framework, the Thickness-aware E(3)-Equivariant Mesh Neural Network (T-EMNN), that effectively integrates the thickness of 3D objects while maintaining the computational efficiency of surface meshes. Additionally, we introduce data-driven coordinates that encode spatial information while preserving E(3)-equivariance and invariance properties, ensuring consistent and robust analysis. Evaluations on a real-world industrial dataset demonstrate the superior performance of T-EMNN in accurately predicting node-level 3D deformations, effectively capturing thickness effects while maintaining computational efficiency.

<p align="center">
<img width="2467" alt="image" src="https://github.com/user-attachments/assets/01bb9d35-21f4-47c0-917c-0dd6438c9ea3" />
</p>



## Requirements
- python=3.10.13
- pytorch=2.0.1
- torch-geometric=2.4.0
- trimesh 3.23.5

## Datasets
You can download the datasets using the following link:  
[Download datasets](https://drive.google.com/file/d/1f_3zYKPlfq1Umb8rZ3Jfd8LtCSCqN7Ym/view?usp=share_link)

After extracting `data.zip`, put the `data` folder in the same directory as the `main.py` file.

## How to run
```
python main.py
```
