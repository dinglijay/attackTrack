import cv2
from pathlib import Path

def main(video_file, resize=None, rotate=False):

    dir_path = Path(video_file)
    Path(dir_path.parents[0]).joinpath('img').mkdir(parents=True, exist_ok=True)

    vc = cv2.VideoCapture(video_file)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        if not rval:
            break
        img_h, img_w = frame.shape[:2]
        if resize:
            img_h = int(img_h * resize)
            img_w = int(img_w * resize)
            frame = cv2.resize(frame, (img_w, img_h))
        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
            img_h, img_w = frame.shape[:2]
        f_path = str(dir_path.parents[0])+'/img/' + '{0:04d}'.format(c) + '.jpg'
        out = cv2.imwrite(f_path, frame)
        print(out, c, f_path, ' Size:', img_h, '*', img_w)
        c = c + 1

    vc.release()

if __name__ == '__main__':
    import fire
    fire.Fire(main)


