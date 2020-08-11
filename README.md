# ChromaGAN

This is an attempt to reimplement the [CHROMAGAN](https://openaccess.thecvf.com/content_WACV_2020/papers/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.pdf) paper using pytorch.


## About

So the paper looks forward to tackle a simple probelem and that is to colourize black and white photos without any human intervention. The paper uses a Generative Adversarial Network to handle this. The Model training is based on the fact that for each given grayscale image L, the model tries to learn a mapping G:L -->(a,b) such that the image I = (L,a,b) is a plausible colour image. 

### For those who would want to reimplement this paper from scratch. 

- the easiest part of this project is to implement the model architecture

![Architecture](/images/1.png)

    As we can see, the architecture is denoted by several colour schemes and each colour scheme has different importance.
    - Yellow branch is a simple VGG network with no top layer
    - Small red boxes away from the main branches are modules consisting of Conv-BatchN-Relu blocks
    - Grey blocks are fc layers which extract the semantic meaning by classifying the classes.
    - Red blocks are again connected back to the main branch and is concatenated with the purple block.
    - the blue blocks are upsample conv blocks. 

    These blocks make the colourization_block/generator block(please check the paper for more details)

    The other block is the discriminator block which is based on a simple Markovian Discriminator (PatchGan used in Image translations tasks).

- Next up is with loss functions and the training.
        Accoriding to the author of the paper, they used multiple loss functions to learn the two chrominance value
        - Colour Error Loss ( L2 Norm Loss )
        - Class distribution loss (KL divergence )
        - WGAN loss (adverserial Wassertien GAN loss)
        
### Dataset
The original paper was trained on Imagenet
I trained it on a few images(50k)

### Results and Important links 

Results of their paper and every minute detail can be found in [https://github.com/pvitoria/ChromaGAN]


