<h2 align="center">RT-DETR: DETRs Beat YOLOs on Real-time Object Detection</h2>

base papers 
- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)



| Model | Input shape | Dataset | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS)
|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
**RT-DETRv2-S** | 640 | COCO  | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 60 | 217 |
**RT-DETRv2-M**<sup>*<sup> | 640 | COCO  | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 92 | 161 |
**RT-DETRv2-M** | 640 | COCO | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 100 | 145 |
**RT-DETRv2-L** | 640 | COCO | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 136 | 108 |
**RT-DETRv2-X** | 640 | COCO | 54.3 | **72.8** <font color=green>(+0.1)</font>  | 76 | 259| 74 |

**Notes:**
- `COCO + Objects365` in the table means finetuned model on COCO using pretrained weights trained on Objects365.


## Citation
If you use `RT-DETR` or `RTDETRv2` in your work, please use the following BibTeX entries:
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```
