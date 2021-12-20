import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def VThin(image, array):
    h,w= image.shape[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j-1] + image[i,j] + image[i, j+1] if 0<j<w-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if-1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def HThin(image, array):
    h, w = image.shape[:2]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1, j] + image[i, j] + image[i+1, j] if 0<i<h-1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k, j-1+l] == 255:
                                a[k*3 + l] = 1
                    sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def Xihua(binary, array, num=10):
    iXihua = binary.copy()
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua

array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

def Hough_Transform(img_skeleton, img_back):
    # Hough transform
    gray = cv2.cvtColor(img_skeleton, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 145, 150)
    lines = cv2.HoughLinesP(edges, 0.01, np.pi / 10, 5, maxLineGap=8, minLineLength=15)
    K = []
    lineh = []
    linel = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                lh = [x1, y1, x2, y2]
                cv2.line(img_back, (x1, y1), (x2, y2), (255, 147, 26), 2)
                lineh.append(lh)
            else:
                l = [x1, y1, x2, y2]
                k = (y2 - y1) / (x2 - x1)
                linel.append(l)
                K.append(k)

    # K-means
    K = [[i] for i in K]
    km = KMeans(n_clusters=3, max_iter=1000).fit(K)
    print(km.cluster_centers_)
    y = KMeans(n_clusters=3, max_iter=1000).fit_predict(K)
    df = pd.DataFrame(linel, columns=['x1', 'y1', 'x2', 'y2'])
    df['category'] = y
    category1 = np.array(df.loc[df['category'] == 0])
    category2 = np.array(df.loc[df['category'] == 1])
    category3 = np.array(df.loc[df['category'] == 2])
    for i in range(0, len(category1)):
        x1, y1, x2, y2, cat = category1[i]
        cv2.line(img_back, (x1, y1), (x2, y2), (0, 250, 250), 2)
    for i in range(0, len(category2)):
        x1, y1, x2, y2, cat = category2[i]
        cv2.line(img_back, (x1, y1), (x2, y2), (0, 0, 250), 2)
    for i in range(0, len(category3)):
        x1, y1, x2, y2, cat = category3[i]
        cv2.line(img_back, (x1, y1), (x2, y2), (250, 0, 250), 2)

    cv2.imwrite("result.png", img_back)

if __name__ == "__main__":

    img = cv2.imread('de-texturing.png',0)                             # read de-texturing image
    img_back = cv2.imread("TDOM.png")                                  # read background image
    plt.hist(img.ravel(),256,[0,256]);
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite('clahe.png',cl1)                                       # write histogram equalization image
    blur = cv2.GaussianBlur(img,(5,5),0)                               # Gaussian Blur
    ret,thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY)          # Binary Threshold
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)        # Opening Operation
    iThin = Xihua(opening, array)                                      # Extract the Skeleton

    cv2.imwrite("blur.png", blur)                                      # write the image
    cv2.imwrite("thresh.png", thresh)
    cv2.imwrite("opening.png", opening)
    cv2.imwrite("thin.png",iThin)

    img_skeleton = cv2.imread("thin.png")                              # read the skeleton image
    Hough_Transform(img_skeleton, img_back)                            # applying the hough transform
    cv2.waitKey(0)
