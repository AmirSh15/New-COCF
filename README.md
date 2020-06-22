# FECNet

<img src="figures/2.png" />

This module contains code in support of the paper [An online algorithm for constrained face clustering in videos](https://tanayag.com/Pub_files/Kulshreshtha_Online_face.pdf). The experiment is implemented using the MXNet framework.

In this repository, I used the original implementation of this paper from [here](https://github.com/ankuPRK/COFC) and face recognition from [insightface](https://github.com/deepinsight/insightface)
### Dependencies

The code was successfully built and run with these versions:

```
mxnet-cu100
cudnn 7.6.5
cudatoolkit 10.0.130
opencv 3.4.2
scikit-learn 0.20.3

```
Note: You can also create the environment I've tested with by importing _environment.yml_ in conda.

### Testing
In this repo, I used the 512-D embedding feature from MobileFaceNe.

```
usage: run_COFC_on_video.py [-h] [-vp VID_PATH] [-sd SAVE_DIR]
                            [-ft FEAT_THRESH] [-ot OVERLAP_THRESH]
                            [-st SIM_THRESH] [--model MODEL_DIR]
                            [--gpu GPU_ID] 

optional arguments:
  -h, --help          show this help message and exit
  -vp VID_PATH        Path to the video file
  -sd SAVE_DIR        Directory path for saving the output
  -ft FEAT_THRESH     Threshold of distance bw features to belong to different
                      persons
  -ot OVERLAP_THRESH  Threshold of overlap above which two faces in
                      consecutive frames will belong to same track
  -st SIM_THRESH      Threshold of Similarity for facetracks to belong to a
                      cluster
  --model             Path to load model
  --gpu               The gpu id which you want to use
```



You can download the pretrained model [here](https://drive.google.com/drive/folders/10P9kIRYKodIGs7Vgv64aQYu9G1A3ofpC?usp=sharing). There are more pretrained model in [here](https://github.com/deepinsight/insightface/wiki/Model-Zoo)

Hence, you can run the algorithm on a video file. The output directory will contain one folder corresponding to each cluster, and then in each folder it will have all the faces belonging to that cluster. The algorithm is highly sensitive to SIM_THRESH. Its value ranges between (0.0, 4.0). Increasing beyond 3.2 will create a lot of clusters, each person will be split into multiple clusters. On the other hand, keeping it below say 2.8 will create less clusters but each cluster will have faces of multiple characters.


### References

If you found this repo useful give me a star!

```
@inproceedings{vemulapalli2019compact,
  title={A Compact Embedding for Facial Expression Similarity},
  author={Vemulapalli, Raviteja and Agarwala, Aseem},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5683--5692},
  year={2019}
}
```
