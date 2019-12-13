import cv2
from joblib import load
from parse import image_parse, label_parse


def show_webcam(mirror=False):
    clf_sgd = load("clf-sgd.joblib")
    clf_lbfgs = load("newclf.joblib")
    i = 0
    cam = cv2.VideoCapture(1)
    newImage = cam.read()
    while True:
        ret_val, img = cam.read()
        # img, contonours, thresh = get_img_contour_thresh(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray, (35, 35), 0)
        img12 = cv2.GaussianBlur(gray, (11, 11), 0)
        img13 = cv2.GaussianBlur(gray, (51, 51), 0)
        img2 = cv2.blur(gray, (35, 35))
        ret, thresh1 = cv2.threshold(img12, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thresh2 = thresh1[25:25 + 450, 25:25 + 575]
        thresh2 = thresh1
        contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # contour = contours[0]
            if cv2.contourArea(contour) > 1000:
                print("Test " + str(i))
                i += 1
                x, y, w, h = cv2.boundingRect(contour)
                temp = adjustBox(x, y, w, h)
                x = int(temp[0])
                w = int(temp[1])
                cv2.rectangle(img, (x,y), (x+w, y+h), (200,255,200), 2)
                z = int(x + (h/2))
                newImage = thresh1[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (20,20))
                newImage = newImage / 255.0
                color = [0, 0, 0]
                newImage = cv2.copyMakeBorder(newImage, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
                print(newImage)
                # newImageUpdate = newImageUpdate / 255.0
                prediction_sgd = clf_sgd.predict(newImage.reshape(1, -1))
                prediction_lbfgs = clf_lbfgs.predict(newImage.reshape(1, -1))
                cv2.putText(img, "Prediction sgd: " + str(prediction_sgd), (50, 375), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
                cv2.putText(img, "Prediction new: " + str(prediction_lbfgs), (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
       
        cv2.imshow('original', img)
        # cv2.imshow('original2', img)
        cv2.imshow('threshold', thresh1)
        # cv2.imshow('threhold2', thresh2)
        # cv2.imshow('test images', newImage)
        cv2.imshow('predict image', newImage)
        # cv2.imshow('test', test)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def adjustBox(x, y, w, h):
    totalW = 650
    totalH = 400

    if h > w:
        delta = h - w
        newW = w + delta
        if x > delta/2:
            newX = x - (delta/2)
            if newX + newW < totalW:
                x = newX
                w = newW
    
    return (x, w)

def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()