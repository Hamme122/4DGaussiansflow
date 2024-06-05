# Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle

This is `Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle` unofficial implementation.

## ToDo list

In the rough
## Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
git clone https://github.com/Hamme122/4DGaussiansflow.git
cd 4DGaussiansflow
git submodule update --init --recursive
conda create -n Gaussiansflow python=3.7 
conda activate Gaussiansflow

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install taichi
```

In our environment, we use pytorch=1.13.1+cu116.

## Data Preparation

**For real dynamic scenes:**
[Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.

```
├── data
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```


## Training

For training dynerf scenes such as `cut_roasted_beef`, run

```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```



## Citation

The current implementation is based on 4DGaussian code. If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@article{wu20234dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei Wei and Liu, Wenyu and Tian, Qi and Wang Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

@article{lin2023gaussian,
  title={Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle},
  author={Lin, Youtian and Dai, Zuozhuo and Zhu, Siyu and Yao, Yao},
  journal={arXiv preprint arXiv:2312.03431},
  year={2023}
}
```
