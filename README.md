Classification of histological images of colorectal cancer through texture and colour


For this work, I decided to implement a research paper called "Improved LBP and Discriminative LBP: Two novel local descriptors for Face Recognition" by Shekhar Karanwal. The article was published on July 30, 2022. link: https://ieeexplore.ieee.org/document/9915933

About ILBP and DLBP

ILBP stands for Improved Local Binary Patterns. It is an extension of the original Local Binary Patterns (LBP) texture descriptor, which is widely used in image analysis and computer vision tasks. ILBP enhances the performance of LBP by introducing several improvements. ILBP focuses on capturing "uniform" patterns in an image. Uniform patterns are defined as patterns that have at most two bitwise transitions (from 0 to 1 or vice versa) when traversing the circular neighborhood of a pixel. By considering uniform patterns, ILBP aims to capture more distinctive and discriminative texture information. It achieves rotation invariance by considering the minimum LBP value among the original pattern and its circular shifts. This eliminates redundancy caused by rotational variations in the image, ensuring consistent representations regardless of the rotation of the local neighborhood. It uses histograms to represent the distribution of different LBP patterns in an image. Instead of considering each LBP value individually, ILBP computes the histogram of LBP patterns. This histogram representation reduces the dimensionality of the feature space and allows for efficient feature comparison and classification. By incorporating these improvements, ILBP enhances the robustness and discriminative power of the LBP texture descriptor. It is commonly used in various applications such as texture classification, face recognition, and image retrieval, where capturing local texture information is important for accurate analysis and recognition.
Here's how ILBP operates:
•	Image Preprocessing: Convert the input image to grayscale. This step is necessary as ILBP operates on grayscale images.
•	Neighborhood Definition: Define a circular neighborhood around each pixel of the image. The neighborhood can be of any size, typically specified by a radius and the number of sampling points. For example, a common choice is an 8-pixel neighborhood with a radius of 1.  
•	With the result of the previous step, we can calculate the ILBP code, using the weigths as same as on image bellow for each position of the patch 
•	Determine the maximum value (MAX) within the patch.
•	Calculate a threshold value (T) using a newly introduced parameter, H. H can range between 0.1 and 0.9. While research suggests that the optimal value is 0.9, we will explore alternative values in this study to ensure consistency within this specific context.
•	Comparison with Neighbors: For each pixel in the image, compare its intensity value with the intensity values of its neighboring pixels within the defined neighborhood. This comparison results in a binary string, where each bit represents the result of the comparison (0 if the neighbor's intensity is less than or equal to the central pixel's intensity, and 1 otherwise).
•	Histogram Computation: Compute a histogram of the occurrence frequencies of the uniform patterns across the entire image. Each bin in the histogram corresponds to a unique uniform pattern label or code, and the bin count represents the frequency of occurrence.
•	Feature Extraction: The histogram of uniform patterns serves as a feature vector representing the local texture information in the image. This feature vector can be further processed or used directly for tasks such as texture classification, object recognition, or image retrieval.
 

DLBP stands for Discriminative Local Binary Patterns. It is an extension of the Improved Local Binary Patterns (ILBP) texture descriptor. DLBP incorporates class-specific information to enhance the discriminative power of the LBP representation for texture classification tasks. The idea behind this descriptor is when we call this class, to calculate the LBP and ILBP descriptors inside it and finally merge them.

Functions

get_pixel(img, center, x, y): Helper function that compares the intensity of a local neighborhood pixel with the center pixel value and returns 1 if it is greater than or equal, otherwise 0.
threshold_apply(pixel_to_analyze_value, threshold_pixel_value, threshold): Helper function that applies a threshold to a pixel value and returns 1 if the value minus the threshold pixel value is greater than or equal to 0, otherwise 0.
lbp_calculated_pixel(img, x, y, threshold): Function for calculating the Local Binary Pattern (LBP) value for a pixel in an image. It takes the grayscale image, coordinates of the pixel, and a threshold value. It compares the pixel's intensity with its surrounding neighborhood pixels, applies a threshold, and returns the LBP value.
LocalBinaryPatterns Class:
The class is defined with two parameters, numPoints and radius, which represent the number of points and radius used in the LBP computation.
The describe() method takes an input image and a threshold value. It computes the LBP representation of the image by iterating over each pixel and calling the lbp_calculated_pixel() function. It then builds the histogram of patterns using np.histogram(), specifying the number of bins and the range of values. The histogram is normalized by dividing it by its sum and returned as the result.
