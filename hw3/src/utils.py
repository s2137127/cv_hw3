import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = None
    for i in range(N):
        ui = u[i]
        vi = v[i]
        tmp = np.array([[ui[0], ui[1], 1, 0, 0, 0, -vi[0] * ui[0], -vi[0] * ui[1]], \
               [0, 0, 0, ui[0], ui[1], 1, -vi[1] * ui[0], -vi[1] * ui[1]]])
        if i == 0:
            A = tmp
            # A = A[:,np.newaxis]
        else:
            A = np.concatenate((A,tmp),axis=0)

    A = np.array(A)
    b = np.array(v).reshape(-1,1)
    # TODO: 2.solve H with A
    if (np.linalg.det(A)):

        H = np.dot(np.linalg.inv(A), b)
    else:
        H = np.dot(np.linalg.pinv(A), b)
    H = np.append(H,1).reshape(3,3)
    # print(H)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)
    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.linspace(xmin, xmax,xmax-xmin)
    y = np.linspace(ymin, ymax,ymax-ymin)
    mesh = np.array(np.meshgrid(x,y)).reshape(2,-1)
    mesh = np.concatenate((mesh,np.ones(shape=(1,mesh.shape[1]))),axis=0)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = np.dot(H_inv, mesh).T
        v = np.divide(v[:, :2], np.tile(v[:, 2], (2, 1)).T)
        v = v.reshape((ymax - ymin), (xmax - xmin), -1)
        v = np.round(v).astype(int)
        # print(v.shape)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        valid1 = np.logical_and(0 < v[:, :, 0], v[:, :, 0] < w_src)
        valid2 = np.logical_and(0 < v[:, :, 1], v[:, :, 1] < h_src)
        valid = np.logical_and(valid1, valid2)

        # TODO: 6. assign to destination image with proper masking
        # print(w_src)
        # v = np.multiply(np.tile(valid, 2).reshape(v.shape), v)
        # print(v.shape)
        v = v[valid]
        # src = np.multiply(np.tile(valid, 3).reshape(src.shape), src)
        # print(src[v[...,1],v[...,0]])
        # print(v.shape)
        # print(np.tile(valid,2).reshape(v.shape))
        # print(v.reshape(valid.shape))
        # print(dst[ymin:ymax,xmin:xmax][valid].shape)
        dst[ymin:ymax,xmin:xmax][valid] = src[v[...,1],v[...,0]]
        # for i in range(v.shape[0]):
        #     for j in range(v.shape[1]):
        #         if valid[i, j]:
        #             dst[i+ymin,j+xmin] = src[v[i,j,1],v[i,j,0]]
    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = np.dot(H,mesh).T
        # print(np.tile(v[:,2],(2,1)))
        v = np.divide(v[:,:2],np.tile(v[:,2],(2,1)).T)
        v = v.reshape((ymax-ymin),(xmax-xmin),-1)
        v= np.round(v).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # mask_xu = np.ones(v.shape[:2])*xmax
        # mask_xl = np.ones(v.shape[:2]) * xmin
        # mask_yu = np.ones(v.shape[:2])*ymax
        # mask_yl = np.ones(v.shape[:2]) * ymin

        # TODO: 5.filter the valid coordinates using previous obtained mask
        valid1 = np.logical_and(0 < v[:, :, 0] , v[:, :, 0]<w_dst)
        valid2 = np.logical_and(0 < v[:, :, 1] , v[:, :, 1]<h_dst)
        valid = np.logical_and(valid1,valid2)

        v = np.multiply(np.tile(valid,2).reshape(v.shape),v)
        src = np.multiply(np.tile(valid,3).reshape(src.shape),src)
        # TODO: 6. assign to destination image using advanced array indicing
        dst[v[..., 1], v[..., 0]] = src



    return dst
