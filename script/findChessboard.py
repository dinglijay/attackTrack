import cv2
import os

def findChessboard(img, cnrx, cnry):
    print("Looking for {0:} x {1:} inside corners".format(cnrx, cnry))
    
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(img, (cnrx, cnry), flags=chessboard_flags)
    if ret:
        print("Checkerboard found")
        # Refining corners position with sub-pixels based algorithm
        # subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        # corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (5, 5), (-1, -1), subpix_criteria)
    else:
        print("Failed to parse checkerboard pattern in image")

    cv2.drawChessboardCorners(img, (cnrx, cnry), corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    return corners 

def main(img_root):
    im_names = os.listdir(img_root)

    for im_name in range(len(im_names)):
        img = cv2.imread(img_root + str(im_name+1) + '.jpg')

        print(im_name)

        corners = findChessboard(img, cnrx=7, cnry=5)
        cv2.waitKey(0)


if __name__ == '__main__':

    import fire
    fire.Fire(main)
    
	    


