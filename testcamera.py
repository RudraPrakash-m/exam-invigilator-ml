import cv2

cap_usb = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap_wifi = cv2.VideoCapture("http://192.168.31.104:8080/video")

while True:
    ret1, frame1 = cap_usb.read()
    ret2, frame2 = cap_wifi.read()

    if ret1:
        cv2.imshow("USB Camera", frame1)
    if ret2:
        cv2.imshow("WiFi Camera", frame2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_usb.release()
cap_wifi.release()
cv2.destroyAllWindows()
