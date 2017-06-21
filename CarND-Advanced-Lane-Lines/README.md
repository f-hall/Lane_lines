
[//]: # (Image References)

[image1]: ./output_images/dist_chess.png "Chessboard"
[image2]: ./output_images/first.png "Testimage"
[image3]: ./output_images/dir_threshold.png "Binary Examples"
[image4]: ./output_images/combined_binary.png "Binary Result"
[image5]: ./output_images/warped_binary.png "Warped Binary and Source Points"
[image6]: ./output_images/histogram_lines.png "Histogram search and Image search"
[image7]: ./output_images/output.png "Output"
[video1]: ./output.mp4 "Video"

##Writeup for Project 4 - Advanced Lane Finding

###General Information
You can find the commented code for this Project in the notebook `P4.ipynb` or in the Python-file `P4.py`. All images from the notebook and in this writeup can be found in the directory `output_images`.  The result video `output_video.mp4`  is located in the main folder.

###Camera Calibration
For the calibration I used partly the code provided in the lessons for this project. Furthermore I used not only the original distorted chess images, instead I flipped them vertically, assuming the lens is also roughly symmetric in this sense,  to get more data.

I prepared the *object points*, which will be the (x,y,z) coordinates of the chessboard in the real world. We can assume that z=0, so that we are in the xy-plane. For each array of `objp` (which are every time the same points on the chessboard) we get the *image points*, that we are getting on each image, with the `cv2.findChessboardCorners`-function. Herefore we are using a 9x6 Chessboard. If we found the needed corners, we append `objp` the associated and the associated *image-corners* to the lists `objpoints ` and `imgpoints` respectively.

With the opencv function `cv2.calibrateCamera` we can use these lists to get the needed camera calibration. With the results `mtx` and `dist` from this function, we can use `cv2.undistort` to undistort any image.

In the following pictures, we can see a undistorted chessimage and a testimage from the video on the left (`project_video.mp4`). On the right are the undistorted results. I choose a quite complex picture (with the shadows) for this markup to test, if my pipeline is good enough to show an acceptable result here.

![alt text][image1]

![alt text][image2]

###Binary Image

I defined four threshold-functions and used a combined version of these to get a result binary picture. For the first two functions `dir_threshold ` and  `mag_threshold` I applied thresholds on the direction and magnitude of the gradient of the *gray image*. Furthermore I used the *HLS* Color Space, especially the S- and L-Channel to get binary images with the functions `hls_threshold` and `hls_threshold2`. For all the functions I tried different parameters to get appropriate results. Toprow are the binary images for `dir_threshold` and `mag_threshold` and below `hls_threshold` and `hls_threshold2`:

![alt text][image3]

To combined the information I used following logical operators between the binary images: (`dir_threshold` & `mag_threshold`) | (`hls_threshold` & `hls_threshold2`) = 1 with following result:

![alt text][image4]

###Perspective Transform

Now we want to get the binary image into a bird's eye view and herefore we use the opencv-functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective`. We need a source parameter `src` and a destination parameter `dst`. I choose `src = np.float32([[200,img_size[1]],[1113, img_size[1]],[724, 475],[559,475]])` and `dst = np.float32([[250,img_size[1]],[img_size[0]-250, img_size[1]], [img_size[0]-250, 0], [250, 0]])` with a bit of trial and error. Left is our warped testimage and on the right we can see the chosen ` src`-points building a rectangle on the undistorted testimage to verify them:

![alt text][image5]

###Identify the lane pixels

I use two different functions to find the correct lane lines. The first is named `Search_lines_with_histogram` and is using a histogram to decide the starting points of the two lane lines. For a detailed usage and description of the definition please look up the comments in the code. For the second function `Search_lines_with_image` we use the information from an already finished image, so we can get around searching the starting points with a histogram again and using stepwise searching after that. The idea behind that is, that for the video we assume that frame after frame should be relatively similar so that we can search the lane lines in the same area, hopefully saving some computing time. For testing this with images, I just used the same picture for `Search_lines_with_histogram` and with this additional information on `Search_lines_with_image`:

![alt text][image6]

To plot this images I also implemented the two functions `Draw_lines_with_histogram` and `Draw_lines_with_image`. I used the function `np.polyfit` to lay a 2nd-order polynom through the lane pixels found on the left and the right to get my two lines. Additionally I took the average between them to get a middle lane (seen in the image above).

###Calculating Radius and Distance from the Midpoint

To calculate the radius of my curve I picked the middle lane-line and used the code provided by udacity. To find the distance between my found mid-lane and the image center I just took the x-value where my middle line touches the bottom of the image (y=719) and subtracted it from the real x-centervalue (here 639). After that I just convert from pixel to meter with the given parameters from udacity-lessons. I wrote a function called `measure` for this.

###Back-Transform

Now we need to add our lane-lines together with our warped image (here from histogram-search) and transform this back into the original state. Herefore I use the the functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective` again. We just have to switch `src` and `dst`. Now we have our final image (On the right is the plot for just the found lane-lines):

![alt text][image7]

###Video

I wrote the function `process_image`, where I used all needed functions and code from before to build the result video (Please look at the commented code for more information, like the `sanity_check`.)

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

The writing of the code itself was fairly easy, the most difficult part was to choose the correct parameters and tweaking the functions. It was quite a lot of trial and error and I'm not fully satisfied with my results, but in the future I would like to make my code more stable. Maybe good enough for the other two videos (challenges...). But hopefully it is good enough to get a check.

P.S: Sorry for my bad english skills. Trying to get better...

Thanks in Advance for reviewing my project :)

Greetings Frank

