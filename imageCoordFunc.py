def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    outFlag, corners = cv2.findChessboardCorners(gray_image, (9,4), None)
    img_coord = np.reshape(corners,(36,2))

    cv2.drawChessboardCorners(image, (9,4), corners, True)
    cv2.imshow("corners", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return img_coord
