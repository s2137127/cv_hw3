import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping


np.random.seed(999)


def Ransac(kp1, kp2, matches, n=4, th=500):

    matches = np.array([[matches[i].queryIdx, matches[i].trainIdx] for i in range(len(matches))])
    print('matches', matches.shape)
    p = 0.5
    P = 0.999
    k = np.ceil(np.log(1 - P) / np.log(1 - p ** n)).astype(int)
    H_arr, inlinear,inlinear_arr = [], [],[]
    for t in range(k):
        idx = np.random.randint(low=0, high=matches.shape[0], size=n,)
        id1, id2 = matches[idx, 0], matches[idx, 1]
        # for i in range(matches.shape[0]):
        #     a = kp1[matches[i, 0]]
        #     print(a)
        p1, p2 = np.array([list(kp1[matches[i, 0]]) + [1] for i in range(matches.shape[0]) if i not in idx]), \
                 np.array([list(kp2[matches[i, 1]]) + [1] for i in range(matches.shape[0]) if i not in idx])
        # print(p1)
        H = solve_homography(kp1[id1], kp2[id2])
        # print(H)
        p2_pred = np.dot(H, p1.T)

        err = np.subtract(p2_pred, p2.T)
        # print(err.shape)
        err = np.sqrt(np.square(err[0, :]) + np.square(err[1, :]))
        # print(np.where(err < th))
        inlinear_arr.append(np.where(err < th))
        inlinear.append(np.where(err < th)[0].shape[0])
        H_arr.append(H)
    idx = np.argmax(inlinear)
    print(H_arr[idx])
    return H_arr[idx],inlinear_arr[idx][0]


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    orb = cv2.ORB_create(nfeatures=4000)
    # for all images to be stitched:
    w = imgs[0].shape[1]
    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        im1_g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # TODO: 1.feature detection & matching
        kp1, f1 = orb.detectAndCompute(im1_g, None)
        kp2, f2 = orb.detectAndCompute(im2_g, None)


        # print(kp2.shape)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(f1, f2)
        matches = sorted(matches, key=lambda x: x.distance)[:300]
        pic = cv2.drawMatches(im1, kp1, im2, kp2, matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('ss', cv2.resize(pic,(800,800)))
        cv2.waitKey()
        kp1 = np.array([i.pt for i in kp1]).astype(int)
        kp2 = np.array([i.pt for i in kp2]).astype(int)
        # TODO: 2. apply RANSAC to choose best H
        H,inlinear = Ransac(kp2, kp1, matches)
        # print(inlinear)
        m = np.array(matches)[inlinear]
        idx = [i.queryIdx for i in m]
        # print(idx)
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H , H)
        # TODO: 4. apply warping
        kp = kp1[idx]

        # print(last_best_H)
        dst = warping(im2,dst, last_best_H, 0,h_max,0,w_max,
                      direction='b')
        cv2.imshow('ss', cv2.resize(dst, (800, 800)))
        cv2.waitKey()
        w += im2.shape[1]
    return dst


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
