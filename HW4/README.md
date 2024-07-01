[![Static Badge](https://img.shields.io/badge/Desc-pdf-blue)](https://github.com/weberyoutoo/AD/blob/main/HW4/AD_HW4.pdf)
[![Static Badge](https://img.shields.io/badge/Report-pdf-green)](https://github.com/weberyoutoo/AD/blob/main/HW4/112062646-report4.pdf)
# Reference
Model : [https://github.com/gdwang08/Jigsaw-VAD](https://github.com/gdwang08/Jigsaw-VAD)  
Dataset : [Avenue Dataset for Abnormal Event Detection](https://drive.google.com/file/d/1LGAkgoqu5AQJzkqpR8s8R97xbXK5S9Mq/view)

### *Hint
> Peculiarly, it seems there are some bugs in the Jigsaw_VAD's source code. Especially, errors often occur when executing statements involving the data path.

1. During my experiments, I put the `avenue` dataset *(which should be a folder)* in the directory that Jigsaw-VAD *(which should also be a folder)* is located.
2. The data paths in the source code are absolute path like
   ```cpp
   data_dir = f"/irip/wangguodong_2020/projects/datasets/vad/{args.dataset}/training"
   ```
   I redefine the data path as relative path like
   ```cpp
   data_dir = f"../{args.dataset}/training"
   ```
   **there many similar path define in the source code, so I redefine all the data path definition.**

> After the rewriting above and dealing with some scattered bugs, the code can work correctly in my experiments.
