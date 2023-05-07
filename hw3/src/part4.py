import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

np.random.seed(999)


def Ransac(kp1, kp2, matches, n=4, th=1500):
    matches = np.array([[matches[i].queryIdx, matches[i].trainIdx] for i in range(len(matches))])
    print('matches', matches.shape)
    p = 0.5
    P = 0.999
    k = np.ceil(np.log(1 - P) / np.log(1 - p ** n)).astype(int)
    H_arr, inlinear = [], []
    for t in range(k):
        idx = np.random.randint(low=0, high=matches.shape[0], size=n, )
        id1, id2 = matches[idx, 0], matches[idx, 1]

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
        # print(err)
        inlinear.append(np.where(err < th)[0].shape[0])
        H_arr.append(H)
    idx = np.argmax(inlinear)
    # print(inlinear)
    # print(H_arr[idx])
    return H_arr[idx]


def linearBlending(imgs,overlap_mask):

    dst,img_left, img_right = imgs
    (hl, wl) = dst.shape[:2]
    dst = dst.astype(float)
    img_right = img_right.astype(float)
    alpha_mask = np.ones((hl, wl),dtype=float)
    for i in range(overlap_mask.shape[0]):
        overlap = np.where(overlap_mask[i,:] == 1)
        if len(overlap[0]) == 0:
            continue
        idx_min = np.min(overlap)
        idx_max = np.max(overlap)
        alp = np.linspace(1,0,num = idx_max-idx_min)
        alpha_mask[i,idx_min:idx_max] = alp
    linearBlending_img = np.copy(dst)
    alpha_mask = np.tile(alpha_mask,3).reshape((hl, wl,3))

    linearBlending_img[overlap_mask] = (alpha_mask * img_left + (np.ones_like(alpha_mask) - alpha_mask) * img_right)[overlap_mask]
    return linearBlending_img.astype(np.uint8)


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
    out = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    orb = cv2.ORB_create(nfeatures=4000)
    # for all images to be stitched:
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

        matches = bf.match(f2, f1)

        # matches = sorted(matches, key=lambda x: x.distance)[:100]
        # print(len(matches))
        # pic = cv2.drawMatches(im2, kp2, im1, kp1, matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('ss', cv2.resize(pic,(800,800)))
        # cv2.waitKey()
        kp1 = np.array([i.pt for i in kp1]).astype(int)
        kp2 = np.array([i.pt for i in kp2]).astype(int)
        # TODO: 2. apply RANSAC to choose best H
        H = Ransac(kp2, kp1, matches)

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, H)
        # TODO: 4. apply warping
        # tmp = dst.copy()
        left_img = dst.copy()
        dst,mask = warping(im2, dst, last_best_H, 0, h_max, 0, w_max,
                      direction='b')
        right_img, _ = warping(im2, out, last_best_H, 0, h_max, 0, w_max,
                            direction='b')
        # cv2.imshow('.da' ,dst)
        # cv2.waitKey()
        dst = linearBlending([dst,left_img,right_img],mask)

        # cv2.imwrite('./db_%d.png' %idx, dst)
        # cv2.waitKey()
    return dst


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)