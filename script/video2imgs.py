import cv2
from pathlib import Path

def main(video_file, rotate=False):

    dir_path = Path(video_file).stem
    Path(dir_path+'_imgs').mkdir(parents=True, exist_ok=True) 

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
        if rotate:
            frame = frame.transpose(1,0,2)
        out = cv2.imwrite(dir_path+'_imgs/' + '{0:04d}'.format(c) + '.jpg', frame)
        print(out, c, dir_path+'_imgs/' + '{0:04d}'.format(c) + '.jpg')
        c = c + 1

    vc.release()

if __name__ == '__main__':
    import fire
    fire.Fire(main)


