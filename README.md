# This is a modified version specifically for spatially-guided style transfer based on Kolkin et. al, whose original paper (and code links) can be found at: https://arxiv.org/abs/1904.12785
# This version was only used for an assignment for 10-615, Art & Machine Learning at Carnegie Mellon University.

The largest updates to the original repo involve:
* Removed relative imports between modules
* Properly formatting resulting image array to ensure image is 0-255 and of type np.uint8
* Modifying how styleTransfer.py's __main__ function reads input parameters to the run_st function, where previously a bug in parsing command-line arguments caused the output path of the resulting image to save to the file name of the content_guidance image file.
