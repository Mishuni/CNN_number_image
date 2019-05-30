# CNN_number_image
AI that can classify the number 0~9 from the image by training through CNN algorithm
and print sum ot the numbers in image


# Image Processing (collect_image funtion)

1. Input image

2. Convert the Input image to Gray Image 
 
3. Gaussian smoothing (removing noise)

4. Histogram equalization (contrast enhancement)
 
5. Thresholding (make a region of number bright)
 
6. Extracting the region of number from the image


# AI Training (CNN algorithm)



# Explanation of files

main.py  - interface that connect user with inner program

AI_Number_function.py - this class have the function named 'AI_Number', which call the function of Extract_number and load the CNN and finally return the total sum of numbers

Extract_number.py - the class that have a function named data_collect, which extracts number images from user image

AnnClass.py	- this class is a CNN

cnn_test_model4.pth - this is the trained CNN from CNN_TEST_with_myimage.py

CNN_TEST_with_myimage.py	- the process of training CNN with MNIST data plus my data

cnn_test_model_r1.pth - this is the trained CNN from CNN_TEST_with_rotation_image.py

CNN_TEST_with_rotation_image.py - 	the process of training CNN with rotated data

number_data.txt - a set of data image 

number_label.txt - a set of target value 


# REFERENCES 
[1] PyTorch를 활용한 머신러닝, 딥러닝 철저 입문/ 코이즈미 사토시/ 위키북
