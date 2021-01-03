# Face recognition using CNN
Face recognition problems commonly fall into two categories:

- Face Verification - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
- Face Recognition - "who is this person?". For example, the video lecture showed a face recognition video of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.

## Naive Face Verification
In Face Verification, you're given two images and you have to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person!

![Images](ImagesFR/Picture1.png)

- Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on.
- We can see that rather than using the raw image, we can learn an encoding
- By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

## Using a Convolutional Neural Net to compute encodings
The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning, first load weights that someone else has already trained. The network architecture follows the Inception model from Szegedy et al..

- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of  m  face images) as a tensor of shape  (m,nC,nH,nW)=(m,3,96,96) 
- It outputs a matrix of shape  (m,128)  that encodes each input face image into a 128-dimensional vector

By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings to compare two face images as follows:

So, an encoding is a good one if:

![Image](ImagesFR/Picture2.png)

- The encodings of two images of the same person are quite similar to each other.
- The encodings of two images of different persons are very different.

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart.

## The Triplet Loss
For an image  x , we denote its encoding  f(x) , where  f  is the function computed by the neural network.

![Images](ImagesFR/Picture3.png)

Training will use triplets of images  (A,P,N) :

- A is an "Anchor" image--a picture of a person.
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from our training dataset. We will write  (A(i),P(i),N(i))  to denote the  i -th training example.

We'd like to make sure that an image  A(i)  of an individual is closer to the Positive  P(i)  than to the Negative image  N(i) ) by at least a margin  α :

                              ||f(A(i))−f(P(i))|| ^ 2 + α < ||f(A(i))−f(N(i))|| ^ 2
                              
We'd thus like to minimize the following triplet cost                              

                              J=∑i=1 to m{||f(A(i))−f(P(i))|| ^ 2 - ||f(A(i))−f(N(i))|| ^ 2}
                              
  1. The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; we want this to be small.
  2. The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, we want this to be relatively large. It has a minus sign preceding it because minimizing the negative of the term is the same as maximizing that term.
  3. α  is called the margin. It is a hyperparameter that we pick manually. We will use  α=0.2 .
