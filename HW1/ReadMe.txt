1. Requirement:

	此.py檔請執行在anaconda environment下。

	* env的pachage的建議版本如下:

	Package                       Version  
	----------------------------- ---------
	torch                         1.10.1
	torchvision                   0.11.2
	tqdm                          4.64.1
	numpy                         1.19.5
	scikit-learn                  0.24.2
	scipy                         1.5.4
	matplotlib                    3.3.4

2. Run the hw1.py:

	(1)請先進入已安裝適當版本package的anaconda environment。
		eg. conda activate [your_env]

	(2)移動到hw1.py所在directory
		eg. cd [path to HW1_112062646]\HW1_112062646

	(3)執行
		python hw1.py

3. 執行結果應如下:

	Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to MNIST/MNIST\raw\train-images-idx3-ubyte.gz
	9913344it [00:01, 6467496.21it/s]
	Extracting MNIST/MNIST\raw\train-images-idx3-ubyte.gz to MNIST/MNIST\raw

	Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to MNIST/MNIST\raw\train-labels-idx1-ubyte.gz
	29696it [00:00, 3826427.81it/s]
	Extracting MNIST/MNIST\raw\train-labels-idx1-ubyte.gz to MNIST/MNIST\raw

	Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to MNIST/MNIST\raw\t10k-images-idx3-ubyte.gz
	1649664it [00:00, 3741465.04it/s]
	Extracting MNIST/MNIST\raw\t10k-images-idx3-ubyte.gz to MNIST/MNIST\raw

	Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to MNIST/MNIST\raw\t10k-labels-idx1-ubyte.gz
	5120it [00:00, ?it/s]
	Extracting MNIST/MNIST\raw\t10k-labels-idx1-ubyte.gz to MNIST/MNIST\raw

	100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [04:37<00:00, 27.79s/it]
	Average ROC-AUC of KNN:           (k=1)    0.9658, (k=5) 0.9683, (k=10) 0.9669
	Average ROC-AUC of K-means:       (k=1)    0.9032, (k=5) 0.9487, (k=10) 0.9611
	Average ROC-AUC of Distance-Base: (Cosine) 0.9744, (r=1) 0.9483, (r=2)  0.9517, (r=inf) 0.9532, (mahalanobis) 0.9794
	Average ROC-AUC of LOF:  0.7644