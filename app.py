import matplotlib.pyplot as plt
import cv2 
import subprocess
"""
In this file i will try to use open cv to check for faces in images 

later on i will try to make cropped photos based on photo position to meet the given passport standard of 600px by 600px
"""

#should take in the image as path
def detect_face(image:str):
    img = cv2.imread(image)
    #convert to grayscale for better detection 
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #load the detector/classifier 
    face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 
    face = face_detector.detectMultiScale(
    gray_img , scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))    

    #display selection
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    #display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,10))
    # plt.imsave("./images/detections/detection_test.jpg",img_rgb)
    # subprocess.call(["xdg-open","./images/detections/detection_test.jpg"])
    plt.axis('off')
    
    while True:
        cv2.imshow("face detector", img_rgb)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
image_path = "./images/potriat_test.jpg"
detect_face(image_path)
