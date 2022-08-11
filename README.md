# vcc-compression
Official repository of the ["Analysis of the Effect of Low-Overhead
Lossy Image Compression on the Performance of
Visual Crowd Counting for Smart City Applications"](https://arxiv.org/abs/2207.10155) paper.

# Instructions
1. Set up the environment. Example can be found [here](https://gitlab.au.dk/maleci/high-resolution-deep-learning)
2. Download the Shanghai Tech B [dataset](https://drive.google.com/drive/folders/13Sz1UNPN96cahwLubvS2aRxOGOUCyQ_i) 
and place it in the ```./data/shanghai_tech_cc``` folder
3. Unzip the file
4. Run ```python prepare_dataset.py --data_path ./shanghai_tech_cc/part_B_final```
5. For each experiment, modify `SELECTED_GPUS` to specify which GPUs to use (if you only have a single GPU available, 
set `SELECTED_GPUS = [0]`)

## Uniform downsampling experiment
1. Re-train SASNet using: train_test_resize_sasnet.py

## JPEG compression experiment
1. Run ```python create_compressed_shtb.py``` to generate a compressed dataset for the specified target compression ratio  
The ```config``` dictionary at the bottom of the file defines the prepared compressions. Compression of 0.05 means JPEG 
compression with 5% quality. The images in the original SHTB dataset are already compressed to 75%, so any value above 0.75 will 
effectively be equal to 0.75.
2. Re-train SASNet using: ```train_test_compressed_sasnet.py```

## Grayscale experiment
1. Re-train SASNet using: ```train_test_recolor.py```

#### Grayscale notes
* In original SASNet, each channel is normalized separately based on the mean and standard deviation of ImageNet.
* In the ```train_test_recolor.py``` the values provided are already calculated for grayscaled images.
* Since the pre-trained model is trained on images with 3 channels, to use those weights we need 3 channels in the first layer. 
We achieve it by copying the data in one grayscale channel to all 3 channels. Since our experiments focus on efficient 
transmission (not efficient processing), we assume this copying will be done after the transmission and does not 
affect our use case.

## Citation
### BibTeX
```bibtex
@article{bakhtiarnia2022,
  title = {Analysis of the Effect of Low-Overhead Lossy Image Compression on the Performance of Visual Crowd Counting for Smart City Applications},
  author = {Bakhtiarnia, Arian and Leporowski, Błażej and Esterle, Lukas and Iosifidis, Alexandros},
  url = {https://arxiv.org/abs/2207.10155},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
}

```

## Repo authors
Błażej Leporowski (bl@ece.au.dk) and Arian Bakhtiarnia (arianbakh@ece.au.dk)