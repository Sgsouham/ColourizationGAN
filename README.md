# ChromaGAN

This is an attempt to reimplement the [CHROMAGAN](https://openaccess.thecvf.com/content_WACV_2020/papers/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.pdf) paper using pytorch.


## About

So the paper looks forward to tackle a simple probelem and that is to colourize black and white photos without any human intervention. The paper uses a Generative Adversarial Network to handle this. The Model training is based on the fact that for each given grayscale image L, the model tries to learn a mapping G:L -->(a,b) such that the image I = (L,a,b) is a plausible colour image. 

### For those who would want to reimplement this paper from scratch. 

- the easiest part of this project is to implement the model architecture

![Architecture](/images/1.png)
