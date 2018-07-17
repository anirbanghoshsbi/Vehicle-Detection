# # HOG Theory

the most important feature of a HOG descriptors are orientations , pixels_per_cell and cells_per_block
these three parameter along with the input image controls the dimensionality of the resulting featue vector.
Why HOG is used so extensively ?

The HOG parameters is used so extensively because local object appearance along with shape  can be characterised by
using the distribution of local intensity gradients . HOG is mainly used as a descriptor for object detection and that these 
descriptors can be fed into the machine learning clasifier.

What are HOG descripters used to describe ?

HOG descriptors are used to describe the structural shape and appearance of the image , making it a excellent descriptor for object classification. HOG also is a good texture descriptor as it covers the local intensity gradients and the edges quite nicely.

The HOG descriptor return a real-valued feature vector , the dimensionality of this feature vector depends on the parameters chosen like the orientations , pixel_per_cell , cells_per_block.

How do HOG Descriptors work?

the cornerstone of HOG descripters algorithm is that the appearance of the object can be modeled by the distribution of the intensity  gradients inside rectangular regions of an image.

Implementing thi requires one to divide the image into small connected regions called cells , and then for each cell compute a histogram of oriented gradients for the pixels within each cell and then accumulate these histograms to form our feature vector.

Block Normalization : Block normalization can be used to improve the performance of the algorithm.

What is Block Normalization?

We take a block overlapping cells , concanate thier histograms , calculate a normalizing value , and then contrast normalize the histograms. By nomalising over multiple overlapping blocks , the resulting descriptors is more robust to change in illumination and shadowing. Furthur these type of normalization also implies that each cell is represented in the final feature vector multiple times but normalized by a slightly different set of neighbouring cells 

steps for computing HOG descriptor.

Step1 : Normalizing image before description. 
this is an optional step but in some cases improve the performance of the HOG descriptor.
there are there ways of normalization :

@ Gamma / power law normalization . in this case we take the log(p) of each pixel p in the input inmage.

@ Square root normalization : here we take the sqrt(p) of each pixel p in the input image.By definition , square root normalization , compress the input pixel intensities far less than the gamma normalization and improve the accuracy.
@variance normalization : here we calculate the mean and the standard deviation of the input image , all pixels are mean centred by substracting the mean and then dividing by standard deviation of the input image.

In most cases it is worth to do no normalization or do square-root normalization.

Step 2 : Gradient Computation :
The actual step in the HOG descriptor is to compute the image gradient in both the x and y direction.We'll then apply a convolution
operation to obtain the gradient images:

$\G_x = I*D_x and G_y = I*D_y$
where I is the input image and D_x is our filter in the direction of x-direction and D_y is the filter in the y-direction.Now we obtain the 
gradient of the image on both the x and y direction |G| = sqrt(G_x^2 +G_y^2)

finally the orientation of the gradient for each pixel in the input image can then be computed by thetha = arctan2(G_y ,G_x).
Now given the |G| and thetha we can compute the histogram of oriented gradients , where the bin of the histograms is based on the thetha and the contribution or weights to a given bin is given by the |G|.

Step 3 : Weighted votes in each cell

Now that we have our gradients magnitude and orientations representations , we need to divide our image up into cells and blocks.A "cell" is a rectangular region defined by the number of pixels that belong to the cell. For example , if we hd a 128 * 128 image and a defined pixels_per_cell 4 * 4 thus we would  have 32*32 = 1024 cells:  (128/4=32).

if we defined our pixels_per_cell as 32 * 32 , we would have 4 * 4 =16 cells.Similarly if the pixel_per_cell is 128 then the number of cells is 1.

Now for each cell of ours we need to construct the histograms of oriented gradients using our gradient magnitude |G| and the orientations thetha.But before we have our gidtograms of oriented gradients we need to define the number of orientations as they would define the numbers of bins in the resulting histograms.gradient angle is taken between [0,180] for signed and [0,360] for unsigned.
Finally ,  each pixel contributes a weighted vote to the histogram - the weights of the votes is simply the gradient magnitude G at the given pixel.

Step 4 :Contrast normalization.

To account for changes in illumination and constrast we can normalize the gradient values locally. This requires grouping the cells together into larger , connecting blocks. It is common for these blocks to overlap , meaning that each cell contributed to the final feature vector more than once.For each of the cell in te current block we concateate their corresponding gradient histograms , followed by L1 or L2 normalization. Finaly all blocks are normalized and we get the final feature vector.



