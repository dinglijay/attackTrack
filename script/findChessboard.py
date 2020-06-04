import cv2
import glob
import pickle
import numpy as np

from os.path import join

def findChessboard(img, cnrx, cnry):

    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(img, (cnrx, cnry), flags=chessboard_flags)
    if ret:
        print("Checkerboard found")
        # Refining corners position with sub-pixels based algorithm
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (5, 5), (-1, -1), subpix_criteria)
    else:
        print("Failed to parse checkerboard pattern in image")

    return ret, corners 

def main(img_root, save_fpath=None, display=True):
    im_names = sorted(glob.glob(join(img_root, '*.jp*')))

    if save_fpath:
        data_ret, data_corners = [], []
    for im_name in im_names:
        print(im_name)
        img = cv2.imread(im_name)
        cnrx, cnry = 7, 5
        ret, corners = findChessboard(img, cnrx=cnrx, cnry=cnry)

        if save_fpath:
            if not ret:
                corners = np.zeros((cnrx*cnry, 1, 2))
            data_ret.append(ret)
            data_corners.append(corners)

        if display:
            cv2.drawChessboardCorners(img, (cnrx, cnry), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    if save_fpath:
        data = {'ret': np.array(ret), 'corners': np.array(corners)}
        with open(save_fpath, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':

    import fire
    fire.Fire(main)
    # python script/findChessboard.py data/Human2/imgs/ --display=False --save_fpath=data/Human2/corners.dat
    
	    


