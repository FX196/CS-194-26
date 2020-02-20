## Code Structure

All of my code is in `main.py`.

The core functions are `align` and `align_pyramid`. 
`align` is a naive search for the best offsets given two images, 
a base offset (for use in `align_pyramid`), search range,
crop percentage, and a choice of loss function.
 

Each of the loss functions defined in the module can be plugged into `align`, including L2 loss,
HOG loss, and Roberts loss, which computes an L2 loss on the edge maps of input images.

`align_pyramid` does a pyramid search for the best offset. The scaling ratio between each level is
tweakable, as is the search range for each level.

`procedure` wraps everything into a function to make it more 
convenient to run the script on all the input files at once.