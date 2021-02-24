import cv2
import numpy as np
import matplotlib.pyplot as plt
crop_img="0406_X-3500-4000_Y-1000-1500.jpg"
img="20201125162132_IMG_0406.JPG"
#reading crop image
img1 = cv2.imread(crop_img)
cv2.imshow("img",img1)
copy1=img1.copy()
rich1 = img1.copy()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray1,None)
cv2.drawKeypoints(gray1, kp, outImage=copy1)
cv2.imshow('SIFT without Flag', copy1) #keypointleri gördük
cv2.drawKeypoints(gray1, kp, outImage=rich1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT with Flag', rich1)
#KEYPOINT DOSYAYA OKUMA-YAZMA
kpf="kp"+crop_img[:-4]+".txt"
f=open(kpf,"w+") #keypointleri dosyaya yazma modu
for point in kp:
    p = str(point.pt[0]) + "," + str(point.pt[1]) + "," + str(point.size) + "," + str(point.angle) + "," + str(
        point.response) + "," + str(point.octave) + "," + str(point.class_id) + "\n"
    f.write(p)
f.close()
kp1 = [] #dosyadan alınan keypointler kaydedilecek
lines = [line.strip() for line in open(kpf)] #dosyadan keypointleri çekiyoruz
for line in lines:
    list = line.split(',')
    temp = cv2.KeyPoint(x=float(list[0]), y=float(list[1]), _size=float(list[2]), _angle=float(list[3]),
                      _response=float(list[4]), _octave=int(list[5]), _class_id=int(list[6]))
    kp1.append(temp)
#DESCRIPTORS DOSYAYA OKUMA-YAZMA
desf="des"+crop_img[:-4]+".txt"
f=open(desf,"w+") #descriptors dosyaya yazdır
for row in des:
    np.savetxt(f, row,newline=',',delimiter=',')
    f.write('\n')
f.close()
des1=[]
f=open(desf)
lines=[line.strip() for line in open(desf)]
for line in lines:
    list = line[:-1].split(',')
    des1.append(list)  #dosyadan al
des1 = np.float32(des1)
#reading image
img2=cv2.imread(img)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
copy2 = img2.copy()
kp2, des2 = sift.detectAndCompute(gray2,None) #büyük resimdeki keypointleri aldık
#feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches using points taken from the files.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()