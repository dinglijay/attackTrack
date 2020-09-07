
import os
import argparse

import cv2

import numpy as np
from glob import glob
from pathlib import Path



parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
parser.add_argument('--follow_up', default=True, type=bool, help='if follow on the gt file')
args = parser.parse_args()


def get_frames(video_name):
    if video_name.endswith('avi') or video_name.endswith('mp4')  or video_name.endswith('MP4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():

    # video_name = 'data/physical/Examples/Bottle_rpn_large/VID_20200825_182655.mp4'
    video_name = args.video_name
    window_name = "Lable Me"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

    # mk result dir
    dir_path = Path(video_name)
    Path(dir_path.parents[0]).joinpath('gtimgs').mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    v_path = str(dir_path.parents[0])+'/gt.avi'
    videoWriter = cv2.VideoWriter(v_path, fourcc, 29, (720, 1280))

    with open(os.path.join(dir_path.parents[0], 'groundtruth.txt')) as f:
        n_done = len(f.readlines())
    if n_done and args.follow_up:
        print('Follow Gt file on: ', os.path.join(dir_path.parents[0], 'groundtruth.txt'))
    else:
        print('Gt file to: ', os.path.join(dir_path.parents[0], 'groundtruth.txt'))

    c = 0
    for frame in get_frames(video_name):
        c = c + 1           
        if c <= n_done:
            continue

        # display frame number
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 2
        color = (0, 0, 255) 
        thickness = 3
        cv2.putText(frame, 'Frame #: {:4d}'.format(c), org, font,  fontScale, color, thickness, cv2.LINE_AA)

        while True:
            x, y, w, h = cv2.selectROI(window_name, frame, False, False)
            if cv2.waitKey(0) & 0xFF == ord('n'):
                break

        # save gt bbox
        with open(os.path.join(dir_path.parents[0], 'groundtruth.txt'), 'a') as file1:
            file1.write('{0:d},{1:d},{2:d},{3:d}\n'.format(x, y, w, h))

        # save image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        f_path = str(dir_path.parents[0])+'/gtimgs/' + '{0:05d}'.format(c) + '.jpg'
        out = cv2.imwrite(f_path, frame)
        print(out, c, f_path)
        
        # # write to video
        # videoWriter.write(frame)

        cv2.imshow(window_name, frame)

    # videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()