# srun this program on the Mac to display image streams from multiple RPis
import cv2
import imagezmq
image_hub = imagezmq.ImageHub(open_port='tcp://*:6006')
while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    print(image.shape)
    cv2.namedWindow(rpi_name, cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(rpi_name, image) # 1 window for each RPi
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')
