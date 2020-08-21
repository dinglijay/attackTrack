import numpy as np
import os
import cv2
import imagezmq


# STD_DIMENSIONS =  {
#     "480p": (640, 480),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
#     "4k": (3840, 2160),
# }

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


def main(filename):
    image_hub = imagezmq.ImageHub(open_port='tcp://*:6006')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    rpi_name, image = image_hub.recv_image()
    image_hub.send_reply(b'OK')
    cv2.namedWindow(rpi_name, cv2.WND_PROP_FULLSCREEN)
    out = cv2.VideoWriter(filename, fourcc, 25, (image.shape[1], image.shape[0]))

    record = False
    while True: 
        rpi_name, image = image_hub.recv_image()
        cv2.imshow(rpi_name, image) 
        image_hub.send_reply(b'OK')

        if cv2.waitKey(1) & 0xFF == ord('r'):
            record = True
            print('Start to record videos to', filename)
        if record:
            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Record ended')
                break
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    filename = 'video.avi'
    main(filename)