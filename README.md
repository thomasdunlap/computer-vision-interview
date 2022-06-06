# Computer Vision Interview Resources

A collection of useful resources for Computer Vision interviews.

## Questions And Answers:

### Links with Pictures!
* https://www.mlstack.cafe/blog/computer-vision-interview-questions
* https://github.com/shafaypro/CrackingMachineLearningInterview - Great format with informative pictures.

### Computer Vision
* https://www.i2tutorials.com/computer-vision-interview-questions-answers/computer-vision-interview-questions-part-1/
* https://www.interviewquery.com/p/computer-vision-interview-questions

### Machine Learning
* https://github.com/andrewekhalel/MLQuestions
* https://huyenchip.com/ml-interviews-book/


1. How many types of image filters in OpenCV?

Image filters used in OpenCV are:
Bilateral Filter, Blur, Box Filter, Dilate, Build Pyramid, Erode, Filter2D, Gaussian Blur, Deriv, and Gabor Kernels, Laplacian, and Median Blur.

2. 4. What are face recognition algorithms?

The face recognition algorithm is basically the computer application that is used for tracking, detecting, identifying, or verifying the human faces simply from the image or the video that has been captured using the digital camera.
Some popular but evolving algorithms are:

    PCA- Principal Component Analysis
    LBPH- Local Binary Pattern Histograms
    k-NN (nearest neighbors) algorithm
    Eigen’s faces
    Fisher faces
    SIFT- Scale Invariant Feature Transform
    SURF- Speed Up Robust Features

3. Explain with an example why the inputs in computer vision problems can get huge. Provide a solution to overcome this challenge.

Consider a 500x500 pixel RGB image fed to a fully connected neural network for which the first hidden layer has just 1000 hidden units. For this image, the number of input features will be 500*500*3=750,000, i.e. the input vector will be 750,000 dimensional. The weight matrix at the first hidden layer will therefore be a 1000x750,000 dimensional matrix which is huge in size for both computations as well as storage. We can use convolution operation, which is the basis of convolutional neural networks, in order to address this challenge.

4. What are the features likely to be detected by the initial layers of a neural network used for Computer Vision? How is this different from what is detected by the later layers of the neural network?

The earlier layers of the neural network detect simple features of an image, such as edges or corners. As we go deeper into the neural network, the features become increasingly complex, detecting shapes and patterns. The later layers of the neural network are capable of detecting complex patterns such as complete objects.

5. Consider a filter [-1 -1 -1; 0 0 0; 1 1 1] used for convolution.What edges will this filter extract from the input image?

This filter will extract horizontal edges from the image. To get a more concrete understanding, consider a grayscale image represented by an array with the following pixel intensities: 

[0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 10 10 10 10 10 10; 10 10 10 10 10 10; ]. 

From the array, it should be apparent that the top half of the image is black, whereas the lower half is a lighter color forming an apparent edge at the center of the image. The convolution of the two will result in the array [0 0 0 0; 30 30 30 30; 30 30 30 30; 0 0 0 0;]. It can be observed from the values in the resultant array that the horizontal edge has been identified.

6. How do you address the issue of the edge pixels being used less than the central pixels during convolutional operation?

In order to address the issue of the filter or kernel extracting information from the edge pixels less in comparison to the central pixel, we can use padding. Padding is essentially adding one or more additional rows or columns of pixels along the boundary of the image. The padding forms the new edge pixels of the image and therefore results in insufficient extraction of information from the original edge pixels. Padding provides the added advantage of preventing the shrinking of an image as a result of the convolution operations.

7. For a 10x10 image used with a 5x5 filter, what should the padding be in order to obtain a resultant image of the same size as the original image?

For an image without padding, the size of the image nxn after convolution with a filter of size fxf is given as (n-f+1)x(n-f+1) (You can verify this with the example in question 3. of this section).

Now, if you add padding of 1 pixel, in effect, this will be like adding a border of one pixel around an image, consequently increasing the length and breadth of the image by 2. That is, for a padding p, the size of the original image becomes (n+2p)x(n+2p). To maintain the size of the image after convolution, you will need to choose p=f-12. 

For the given problem, therefore, we will need to choose p=(5-1)/2 =2.

8. Given a 5x5 image with a 3x3 filter and a padding p=1, what will the size of the resultant image be if a convolutional stride of s= 2 is used?

For an nxn image with an fxf, padding p, and stride length s, the size of the resultant image after convolution has the shape n+2p-fs+1 x n+2p-fs+1. Therefore for the given problem the size of the resultant image will be (((5+2*1-3)/2) +1) x (((5+2*1-3)/2) +1)= 3 x 3.

9. What will be the change in the size of the results of the convolution is performed with a 1x1 filter on a 10x10 image with no padding?

The size of the image will remain unchanged; that is, the height and width of the resultant image will be the same as the resultant image. However, the number of channels in the resultant can be changed with a 1x1 convolution, thus increasing or decreasing the dimensionality of the input

10. How many parameters are to be learned in the pooling layers?

No parameters are to be learned in the pooling layers. In general, the pooling layer has a set of hyperparameters describing the filter size and the stride length, which are set and work as a fixed computation. 

11) For max-pooling done on a 6x6x3 image with a filter of size f=2 and padding p=0 with stride s=1, what would be the size of the output image?

The size of the resultant image will be as per the formula n+2p-fs+1. Additionally, the number of channels in the resultant image will be the same as that in the input image. 

Therefore, for the given problem, the dimensions of the resultant image will be 

(6+2*0-2+1) x (6+2*0-2+1) x 3 = 5 x 5 x 3.

12) Suggest a way to train a convolutional neural network when you have a quite small dataset.

If you have a dataset that will not be enough to train a convolutional neural network sufficiently well ( and even otherwise sometimes in the interest of time or resource), it is suggested to use transfer learning to solve your machine learning problem. Depending on the size of the dataset and the budget of time and resources you have at your disposal, you can choose to train only the last classification layer or a few of the later layers. Alternatively, you could use transfer learning to train (when the size of the data set is large enough to allow this) all the layers with the parameters initialized from a previously trained model. Owing to the large availability of open-source pre-trained models, transfer learning for computer vision problems is both easy and strongly advised.

13) Explain why mirroring, random cropping, and shearing are some techniques that can help in a computer learning problem.

There is often limited data available to solve computer vision problems, which just isn't enough to train the neural networks. Techniques like mirroring, random cropping, and shearing can help augment the existing dataset and create more training data from the existing data, thereby ameliorating the issue of limited training data.

14) Mention a method that can be used to evaluate an object localization model. How does it work?

Intersection over Union (also known as IoU) is a commonly used method to evaluate the performance of the object localization model. The overlap between the ground truth bounding box and the predicted bounding box is checked, and the ratio of the intersection of the areas to the Union of the areas is calculated. If this ratio called IoU is found greater than some threshold (usually set to 0.5 or higher), the prediction of the model is considered correct.

15) How can IoU be used for resolving the issue of multiple detections of the same object?

The IoU method also helps in the non-max suppression technique for eliminating multiple detections of the same object. It is common for object localization models to predict multiple bounding boxes for the same object. In such cases, the box with the maximum probability among the overlapping boxes of a class is taken. The IoU of this box with the other overlapping boxes is then checked. If this IoU is above a certain threshold, the boxes are considered to be detecting the same object, and the lower probability boxes are eliminated.

16) Mention a scenario that would require the use of anchor boxes.

When detecting multiple classes of objects, there is a scenario where the center of two bounding boxes capturing objects of two different classes occurs on the same point or grid cell. Despite the overlap of the two boxes owing to the fact that both the boxes indicate different objects, they will both require to be retained (something a grid cell in the sliding window technique isn't ordinarily able to accomplish). For this purpose, a number of anchor boxes of different dimensions are used, and vectors similar to the original output vector of a grid cell are now provided for each anchor box in the new output vector. The best-fitting anchor box for a particular class of objects is used to indicate that the grid cell contains the center of the object, thus allowing for multiple overlapping object detection.

17) How does the Siamese Network help to address the one-shot learning problem?

Siamese Network works by encoding a given image with a learned set of parameters such that the distance between the encoding is large when the image is of two different people, and the distance is small when the two images compared are of the same purpose. Training this neural network, therefore, involves learning parameters such that these conditions of the encoding are satisfied when provided with images of two different people or two images of the same person.

18) What purpose does grayscaling serve?

Grayscaling helps to reduce the dimension of the image and thus allows for reduced computation time and effort. Further, it reduces the complexity of models and functions required for various operations. Some functions like edge and contour detection and machine learning problems Optical Character Recognition perform better or are implemented for working only with grayscale images.

19) What color to grayscale conversion algorithm does OpenCV employ? What is the logic behind this?

The color to grayscale algorithm in OpenCV uses the formula Y=0.299*R+0.587*G+0.114*B. This makes it similar to the luminosity method, which averages the color intensity values weighting them in accordance to human perception of different colors, i.e., it accounts for the fact that humans perceive green more strongly than red, and red more strongly than blue, which is apparent from the weightage given to each color's pixel intensity. Additionally, the OpenCV grayscaling algorithm takes into consideration the nonlinear operation used to encode images.

20) What is translational equivariance? What brings about this property in Convolutional Neural Networks?

Translational Equivariance is a property where the position of an object in an image will not affect its detection in the image. That is, if an image is shifted a few pixels to the right or to the left, the output will merely change its position equally and otherwise remain unaffected. Translational Equivariance is an important property of Convolutional neural networks and is brought about by the parameter sharing concept.

21) What is the basis of the popular EAST text detector? 

EAST text detector is based on a fully convolutional neural network adapted for text detection. It has gained immense popuarity because of its text detection accuracy in natural scene images.

22) What is the basis of the state-of-the-art object detection algorithm YOLO?

YOLO is an object detection algorithm that is based on a Convolutional Neural Network and capable of working in real-time. It provides accurate detections for a large variety of objects and can be used as a solution or a starting point for transfer learning for many computer vision problems.

23) Can you name some of the different types of image filters used in OpenCV?

Gaussian Blur, Bilateral Filter, Gabor Kernels, Median Blur, Dilate, Box Filter, and Erode.

Q: How do you project a 3D point to an image?

There is a trick to this question, and you’ll need to ask the follow-up question “What coordinate frame is the point represented?” You’ll want to know if this is the camera frame or if the point is projected into the image frame. To transform the point to another frame, you need to know the rigid transformation. To project the image, you need to know camera intrinsics such as the camera matrix and lens distortion.

Q: How do you make 3D measurements using 2D cameras/sensors?

Here you will need to demonstrate an understanding of epipolar geometry and essential matrix, in addition to the relationship between 2D and 3D points. You’ll need to know the basics that allow you to make measurements in a scene, and after understanding these parts of an image you can then start to use the sensor as a measurement device.

You can prepare for these types of questions by refreshing your knowledge of camera calibration/ representation, calibration, epipolar geometry, and PnP-based pose estimation, and homography/ transformations.

Q: What is the object, and what is its position & orientation based on the coordinate frame of a reference?

Advanced computer vision roles are not about detecting bounding boxes. As 3D pose estimation is about 3D translation and rotation of objects, it’s important to demonstrate that you can generate a rotation matrix. Geometry and appearance can define the origin of an object, and for any advanced role you will be expected to show your understanding of techniques using these formulas as well. Here’s a link for feature extraction + feature matching + homography-based pose estimation.



### References
* https://www.projectpro.io/article/computer-vision-engineer-interview-questions/450
* https://towardsdatascience.com/ace-your-computer-vision-job-interview-b0c61a144664





## Articles

### Computer Vision

* https://towardsdatascience.com/ace-your-computer-vision-job-interview-b0c61a144664
* https://whatdhack.medium.com/ 
