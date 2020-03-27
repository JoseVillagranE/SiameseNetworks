# Siamese Networks

### Contenido del repositorio

Se implementan redes neuronales siamesas con aprendizaje de tripletas y hard-mining. Además se implementa el manejo
del dataset TLP en pytorch y una métrica de evaluación de clusters con el objetivo de dar una evaluación cuantitativa de las técnicas ocupadas.
Este repositorio no trae el dataset para su descarga, sin embargo, este se puede descargar desde [acá](https://amoudgl.github.io/tlp/).

### Breve introducción teórica

La redes siamesas son un tipo de red neuronal que se crearon para el procesamiento de dos o más entradas diferentes. Uno de los motivos para ello es
la diferenciación y aprendizaje de ciertas instancias del dataset con el objetivo de distinguir sus caracteristicas de forma profunda. Para ello, la red implementada
mapea cada instancia a un espacio eucliadiano de dos dimensiones común para cada clase del dataset. Un forma gráfica de visualizar este tipo de redes, obtenida desde[1], se muestra a continuación:

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/Images/SiameseNeural.png)


Aunque la red neuronal es un de los aspactos más importantes del proyecto, no lo es únicamente, ya que la función de perdidad juega un rol importante en la diferenciación de instancias. Es por ello
que se implementa un función de perdida de tripletas (Triplet learning), la cual apunta a distanciar cada instancia negativa desde un anchor, elegido de forma conveniente, y acercar las instancias positivas o instancias común a una misma clase.
Tanto como la ecuación que gobierna dicha función de perdida, como una imagen ilustrativa, obtenidas desde[2], se muestran a continuación:

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/Images/TripletLoss.png)

![alt text](https://github.com/JoseVillagranE/SiameseNetworks/Images/TripletLearning.png)

Además, se implemento hard-mining, el cual se encarga de la busqueda del ejemplo o instancia más dificil de distinguir, con respecto al anchor, por parte de la red neuronal. 

### Dataset

Para este proyecto se ocupo el dataset de tracking de objetos reales de TLP[3]. Dicho dataset cuenta con 50 escenas diferentes de videos. Totalizando un tiempo de 400 minutos de grabación y 676k Frames.

### Referencias

[1] Koch, G., Zemel, R., Salakhutdinov, R.: Siamese neural networks for one-shot image recognition.In: ICML Deep Learning Workshop, vol. 2 (2015)

[2] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for facerecognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition(CVPR). doi:10.1109/cvpr.2015.7298682

[3] Abhinav Moudgil and Vineet Gandhi. Long-Term Visual Object Tracking Benchmark. CoRR. abs/1712.01358. 2017. Available at: http://arxiv.org/abs/1712.01358




