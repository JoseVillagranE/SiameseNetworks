# Siamese Networks

### Content

This repo contain an implementation of siamese neural networks for clustering task. The training of the nueral network considering a tiplet loss and hard sample mining. TLP dataset handling is also implemented and some clustering metrics. It's possible to download the TLP dataset from [here](https://amoudgl.github.io/tlp/).

### Brief theorical introduction

Siamese neural networks are a type of neural networks that were created for the processing of two or more inputs. One of the reasons for it is the diferentiation and learning of some instances of the dataset to distinguish their characteristics in a deep way. For that, the neural networks implemented map each instance to a common two-dimensional Euclidean space for each class in the dataset. A graphic way to visualize this type of network is shown below [1]: 

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/SiameseNeural.png)

although the neural network is the most important aspect of this project, it's not the only one, since the loss function plays a fundamental role in the differentiation of instances. That is why the tiplet loss function is implemented, which aims to distance each negative instance from it's anchor, which is conveniently chosen, and fetch positive instances that belong to the same class. Both the loss function equation as an illustrative image that show the learning process [2] are shown below:

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/TripletLoss.png)

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/TripletLearning.png)

In addition to the above, it's common to find that the training of this neural networks implements hard sampling mining which aim to get the most difficult examples to learning from.

For tasks based on clustering, it is convenient to implement metrics that allow evaluating how accurate the examples are grouping. 
For this, the Silhouette coefficient method is implemented:

![a_i](https://latex.codecogs.com/gif.latex?a%28i%29%20%3D%20%5Cfrac%7B1%7D%7B%7CC_i%7C%20-%201%7D%20%5Csum_%7Bj%20%5Cin%20C_i%2C%20i%20%5Cneq%20j%7D%20d%28i%2C%20j%29)

![b_i](https://latex.codecogs.com/gif.latex?b%28i%29%20%3D%20%5Cmin_%7Bk%20%5Cneq%20i%7D%5Cfrac%7B1%7D%7B%7CC_k%7C%7D%20%5Csum_%7Bj%20%5Cin%20C_k%7D%20d%28i%2C%20j%29)

![s_i](https://latex.codecogs.com/gif.latex?S%28i%29%20%3D%20%5Cfrac%7Bb%28i%29%20-%20a%28i%29%7D%7B%5Cmax%28a%28i%29%2C%20b%28i%29%29%7D)

In short, this method evaluates how accurate an instance is relative to others in the same cluster or instances of the same class in other clusters.

### Dataset

This project contemplate the use of the dataset TLP [3]. This dataset counts with 50 different scenes of videos. Totaling a recording time of 400 minutes and 676k frames.

![TLP Dataset](https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/TLP.png)

### Results

Below are some results that were obtained for the clustering of 3 and 10 classes, considering an AlexNet Convolutional Neural Network as a images processor. First, it's shown the results of the network without training and after of that, the results of the clustering with 5 epochs of training:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWoutTraining_3_traindata.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWoutTraining_3_valdata.png" height="400" width ="400" />

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWTraining_3_traindata.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWTraining_3_valdata.png" height="400" width ="400" />

Now, it's presented the results with 10 classes:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWTraining_10_traindata.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/AlexnetWTraining_10_valdata.png" height="400" width ="400" />

Also, below is shown the reults of Silhoutte coefficient for 3 and 10 classes:

<img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/sil3.png" height="400" width ="400" /> <img src = "https://github.com/JoseVillagranE/SiameseNetworks/blob/master/Images/sil10.png" height="400" width ="400" />

### Referencias

[1] Koch, G., Zemel, R., Salakhutdinov, R.: Siamese neural networks for one-shot image recognition.In: ICML Deep Learning Workshop, vol. 2 (2015)

[2] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for facerecognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition(CVPR). doi:10.1109/cvpr.2015.7298682

[3] Abhinav Moudgil and Vineet Gandhi. Long-Term Visual Object Tracking Benchmark. CoRR. abs/1712.01358. 2017. Available at: http://arxiv.org/abs/1712.01358




