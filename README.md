# Scene Recognition
### Scene classification system created in Java with OpenIMAJ

Authors:
 * [tedigc](https://github.com/tedigc)
 * [marcossss3](https://github.com/marcossss3)
 
Created using Hare et al's [OpenIMAJ](http://www.openimaj.org/).

-------
The valid 'scenes' this system will try to classify as are:
* coast
* forest
* highway
* inside city
* mountain
* office
* open country
* street
* suburb
* tall building
* bedroom
* industrial
* kitchen
* living room
* store

There are three feature extractors and image classifiers.

---------
### "Tiny-Image" feature vectors with a KNN-Classifier
To acquire a 'tiny-image' feature vector, each image is cropped about the center to a fixed resolution of 16 by 16 pixels. After cropping, the image is normalised and each row of pixels is concatenated into a float feature vector, where each element represents a single pixel’s normalised grayscale value (all images are grayscale by default). A classifier is made using *all* feature vectors from the training set. 
##### Accuracy: 0.223
---------
### Dense Patch Sampling and LibLinearAnnotator classifier
In order to implement a linear classifier using the LiblinearAnnotator, we designed our own feature extractor that made use of a custom engine, which is responsible for sampling and normalising 8x8 patches every 4 pixels in the x and y directions. The engine inherits from OpenIMAJ’s Engine class and overrides the findFeatures method. This returns a LocalFeatureList of FloatKeypoints. To construct these FloatKeypoints, the pixel values of each row in a patch are concatenated into a row feature vector and passed to the FloatKeypoint constructor, along with the position of the feature. Rather than returning all the features found by the engine, the findFeatures() method returns a subset of keypoints, obtained by random selection.

The FeatureExtractor requires the construction of a HardAssigner in order to generate the bag-of-visual-words. This extracts features from the densely sampled pixel patches and uses K-Means to cluster them into separate classes. Initially, the suggested number of 500 clusters was used, however after testing the classifier with different numbers of clusters, it was found that using ~600 clusters maximised the accuracy.

Once the HardAssigner has been trained, the FeatureExtractor uses a BlockSpatialAggregator to append together the spatial histograms computed from the collection of visual words. The result of this is then normalised and returned in the form of a feature vector.

##### Accuracy: 0.269 **(This was significantly lower than the ~0.66 expected)**
---------
### Pyramid Dense SIFT With A Homogeneous Kernel Map

For the purpose of scene classification, using SIFT descriptors for patches within regular grid spaces has shown to be more suitable than descriptors of local interest points [2, 3]. Knowing this, we opted to use a dense SIFT engine to compute and describe a collection of 8x8 patches taken at a stepsize of 3 pixels. From the patches, a subset is taken by random selection, whose SIFT descriptors are used to generate visual bags of words. A PyramidSpatialAggregator is used rather than the BlockSpatialAggregator for computing the spatial histograms across the image. Additionally, by applying a homogeneous kernel map, the non-linearly separable data is projected into space where it can be analysed by a linear classifier. This is a highly accurate approximation of applying a nonlinear classifier.					

Rather than just using dense SIFT, this classifier makes use of spatial pyramid matching, which is regarded to be “one of the most successful methods in computer vision”[1]. The PyramidDenseSIFT class within OpenIMAJ takes a normal DenseSIFT instance and applies it to different sized windows on the regular sampling grid.

The FeatureExtractor for this classifier uses a PyramidSpatialAggregator to append the spatial histograms computed from the collection of visual words, returning the aggregated results in the form of a single vector. Unlike SpatialBlockAggregator, this approach groups the local features into fixed-size spatial blocks within a pyramid.

##### Accuracy: 0.793
