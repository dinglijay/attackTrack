import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

def main(img_root, out_path):

    HEIGHT, WIDTH = 720, 1280

    fps = 29.63
    fourcc = VideoWriter_fourcc('X', 'V', 'I', 'D')
    videoWriter = cv2.VideoWriter(out_path, fourcc, fps, (WIDTH, HEIGHT))

    im_names = os.listdir(img_root)
    for im_name in im_names:
        frame = cv2.imread(os.path.join(img_root, im_name))
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        print(im_name)
        videoWriter.write(frame)

    videoWriter.release()

if __name__ == "__main__":
    import fire
    fire.Fire(main)