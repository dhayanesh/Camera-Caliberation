def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    bigMatrix = np.zeros([64, 12], dtype=float)
    pM = 0
    for i in range(0,32):
        X = world_coord[i,0]
        Y = world_coord[i,1]
        Z = world_coord[i,2]
        x = img_coord[i,0]
        y = img_coord[i,1]

        bigMatrix[pM,0] = X
        bigMatrix[pM,1] = Y
        bigMatrix[pM,2] = Z
        bigMatrix[pM,3] = 1
        bigMatrix[pM,4] = 0
        bigMatrix[pM,5] = 0
        bigMatrix[pM,6] = 0
        bigMatrix[pM,7] = 0
        bigMatrix[pM,8] = -1*x*X
        bigMatrix[pM,9] = -1*x*Y
        bigMatrix[pM,10] = -1*x*Z
        bigMatrix[pM,11] = -1*x

        pM+=1

        bigMatrix[pM,0] = 0
        bigMatrix[pM,1] = 0
        bigMatrix[pM,2] = 0
        bigMatrix[pM,3] = 0
        bigMatrix[pM,4] = X
        bigMatrix[pM,5] = Y
        bigMatrix[pM,6] = Z
        bigMatrix[pM,7] = 1
        bigMatrix[pM,8] = -1*y*X
        bigMatrix[pM,9] = -1*y*Y
        bigMatrix[pM,10] = -1*y*Z
        bigMatrix[pM,11] = -1*y

        pM+=1
    
    #print("bigMatrix:",bigMatrix)

    #Performing Singular Value Decomposition
    U, E, Vt = np.linalg.svd(bigMatrix, full_matrices=False)

    #print("Vt:",Vt)
    #last row of Vt is x
    xMatrix = np.zeros([12, 1], dtype=float)

    
    xMatrix = Vt[11]

    #print("xMatrix:", xMatrix)

    projMatrix = np.zeros([12, 1], dtype=float)

    #converting matrix to 3x4 form
    projMatrix = np.reshape(xMatrix, (3,4))
    
    #print("projMatrix:",projMatrix)

    projMatrix3 = np.zeros([3, 3], dtype=float)

    for i in range(0,3):
        for j in range(0,3):
            projMatrix3[i,j] = projMatrix[i,j]
    
    #print("projMatrix3:",projMatrix3)
    #performing QR decomposition to get Int Matrix and Rot Matrix
    K , R =  np.linalg.qr(projMatrix3)

    #print("K:",K)
    #print("R:",R)

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)


    # Your implementation
    bigMatrix = np.zeros([64, 12], dtype=float)
    pM = 0
    for i in range(0,32):
        X = world_coord[i,0]
        Y = world_coord[i,1]
        Z = world_coord[i,2]
        x = img_coord[i,0]
        y = img_coord[i,1]

        bigMatrix[pM,0] = X
        bigMatrix[pM,1] = Y
        bigMatrix[pM,2] = Z
        bigMatrix[pM,3] = 1
        bigMatrix[pM,4] = 0
        bigMatrix[pM,5] = 0
        bigMatrix[pM,6] = 0
        bigMatrix[pM,7] = 0
        bigMatrix[pM,8] = -1*x*X
        bigMatrix[pM,9] = -1*x*Y
        bigMatrix[pM,10] = -1*x*Z
        bigMatrix[pM,11] = -1*x

        pM+=1

        bigMatrix[pM,0] = 0
        bigMatrix[pM,1] = 0
        bigMatrix[pM,2] = 0
        bigMatrix[pM,3] = 0
        bigMatrix[pM,4] = X
        bigMatrix[pM,5] = Y
        bigMatrix[pM,6] = Z
        bigMatrix[pM,7] = 1
        bigMatrix[pM,8] = -1*y*X
        bigMatrix[pM,9] = -1*y*Y
        bigMatrix[pM,10] = -1*y*Z
        bigMatrix[pM,11] = -1*y

        pM+=1
    
    #print("bigMatrix:",bigMatrix)

    U, E, Vt = np.linalg.svd(bigMatrix, full_matrices=False)

    #print("Vt:",Vt)

    xMatrix = np.zeros([12, 1], dtype=float)

    #for i in range(0,12):
        #xMatrix[i,0] = Vt[i,11]
    xMatrix = Vt[11]

    #print("xMatrix:", xMatrix)

    projMatrix = np.zeros([12, 1], dtype=float)

    
    projMatrix = np.reshape(xMatrix, (3,4))
    
    #print("projMatrix:",projMatrix)

    projMatrix3 = np.zeros([3, 3], dtype=float)

    for i in range(0,3):
        for j in range(0,3):
            projMatrix3[i,j] = projMatrix[i,j]
    
    #print("projMatrix3:",projMatrix3)

    K , R =  np.linalg.qr(projMatrix3)

    #print("R:",R)

    Tz = projMatrix[2,3]
    OxTz = K[0,2] * Tz
    OyTz = K[1,2] * Tz

    Tx = (projMatrix[0,3] - OxTz) / K[0,0]
    Ty = (projMatrix[1,3] - OyTz) / K[1,1]

    T[0] = Tx
    T[1] = Ty
    T[2] = Tz

    return R, T
