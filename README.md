Concert of Whales
=========================

Generative Models to Synthesize Audio Waveforms Part II

The goal of this project was to synthesize a 2-second-long audio waveform of a right whale upcall by harnessing the tremendous advances in speech synthesis deep learning research in recent years. As detailed in Part I, a model based on [Tacotron](https://arxiv.org/pdf/1703.10135.pdf) was proposed as a viable option for tackling this problem, but ultimately yielded unsatisfactory results due to use of the Griffin-Lim algorithm to synthesize audio waveforms from magnitude spectrograms. 

In order to obtain more realistic-sounding results, the next model to be investigated was [SampleRNN](https://arxiv.org/pdf/1612.07837.pdf). A Three-Tier SampleRNN was implemented, in which each of the three modules in the hierarchy conditions the one below it so that the lowest module outputs sample-by-sample predictions. 

![threetier](https://github.com/cchinchristopherj/Concert-of-Whales/blob/master/threetier.png)

*Three-Tier SampleRNN Architecture. Image Source: [SampleRNN: An Unconditional End-To-End Neural Audio Generation Model.](https://arxiv.org/pdf/1612.07837.pdf)*

Concretely, the highest level module processes the previous 8 samples of the time series  using an RNN and passes a conditioning vector to the second-highest-level module, which in turn processes the previous 2 samples of the time series using an RNN and passes a conditioning vector to the lowest-level module. This module processes only the previous sample of the time series and outputs a q=256-way softmax over quantized values of the audio waveform. (As an additional design choice, which was proven to improve results, the lowest module passes the previous (quantized) value of the audio clip time series through an embedding layer, which maps each of the quantized values to a real-valued vector embedding. Further, to speed up training, a Multilayer Perceptron is used in the lowest-level module instead of an RNN to yield the final (quantized) prediction for the next sample). 

After training, the SampleRNN is tasked with synthesizing a new 2-second long right whale upcall sound. A representative sample can be found [here.](https://github.com/cchinchristopherj/Concert-of-Whales/blob/cchinchristopherj-patch-1/fake4.mp3)

For comparison, three representative examples of upcall sounds from the training set can be found below:
- [Real Upcall Sound 1](https://github.com/cchinchristopherj/Concert-of-Whales/blob/cchinchristopherj-patch-1/real1.mp3)
- [Real Upcall Sound 2](https://github.com/cchinchristopherj/Concert-of-Whales/blob/cchinchristopherj-patch-1/real2.mp3)
- [Real Upcall Sound 3](https://github.com/cchinchristopherj/Concert-of-Whales/blob/cchinchristopherj-patch-1/real3.mp3)

Web Application
=========================

In order to demonstrate the high degree of realism attained by SampleRNN with only a few epochs of training, an application was developed in which four whales are drawn to a canvas, three of which make “real” right whale upcall sounds from the training set, and the fourth makes the “fake” upcall sound created by the neural network. The task of the user is to identify the whale that made the "fake" right whale upcall sound.

The application begins by displaying the four baby whales on the canvas, each of which is a different size and swims in a different direction. 

![babywhales](https://github.com/cchinchristopherj/Concert-of-Whales/blob/master/Images/babywhales.png)

The whales will quickly grow to full size, as indicated by the progress bar, after which they can make their upcall sounds to each other. 

![finalscreen](https://github.com/cchinchristopherj/Concert-of-Whales/blob/master/Images/finalscreen.png)

In nature, these upcalls are made between whales as a greeting to other whales close by. To replicate this in the application, the upcall sounds associated with each individual whale are played when the center points of their shapes are close enough (in terms of Euclidean distance) to each other, thereby creating a chorus of sound when the whales randomly cluster together. The user also has the option of clicking the name of a whale to play their corresponding up call sound, in order to better inform their decision of which whale is the “fake.” 

Listening to each of the upcall sounds in turn reveals that the “fake” sound generated by SampleRNN is nearly indistinguishable from the sounds from the actual training set. With more training and computational resources, the SampleRNN could achieve even more realistic results. 

References
=========================

Mehri, Soroush, et al. ["SampleRNN: An Unconditional End-To-End Neural Audio Generation Model."](https://arxiv.org/pdf/1612.07837.pdf) *arXiv preprint arXiv:1612.07837*, 2017

Wang, Yuxuan, et al. ["Tacotron: Towards End-To-End Speech Synthesis."](https://arxiv.org/pdf/1703.10135.pdf) *arXiv preprint arXiv:1703.10135*, 2017
