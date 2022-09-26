#Thực hiện khai báo thư viện
import time
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
start_time = time.time()

#Định nghĩa thuật toán phân đoạn cắt ngưỡng toàn cục
def phan_doan_bang_cat_nguong(img,nguong):
    m, n = img.shape
    img_phan_doan_cat_nguong = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if (img[i,j] < nguong):
                img_phan_doan_cat_nguong[i,j] = 0
            else:
                img_phan_doan_cat_nguong[i,j] = 225
    return img_phan_doan_cat_nguong

#Đọc dữ liệu hình ảnh
path = 'img/0200_04363_b.jpg'
Orginal_image = cv2.imread(path)
Final = Orginal_image.copy()
gray = cv2.cvtColor(Orginal_image,cv2.COLOR_BGR2GRAY)
image1 = gray.copy()
image2 = gray.copy()
cv2.imshow('Orginal_image',Orginal_image)
cv2.waitKey(0)

#Tiền xử lý ảnh
# Thuật toán co nở ảnh:
def ero_dila (img_phan_doan,n):
    if n==1:
        kernel = np.ones([3, 3])
        img_phan_doan = cv2.dilate(img_phan_doan, kernel, iterations=2)
        kernel = np.ones([5, 5])
        img_phan_doan = cv2.erode(img_phan_doan, kernel, iterations=5)
        kernel = np.ones([5, 5])
        img_phan_doan = cv2.dilate(img_phan_doan, kernel, iterations=2)
        return img_phan_doan

    if n==2:
        kernel = np.ones([5, 5])
        img_phan_doan = cv2.erode(img_phan_doan, kernel, iterations=10)
        img_phan_doan = cv2.dilate(img_phan_doan, kernel, iterations=15)

    return img_phan_doan


def preImg(image1,n):
    image1 =  cv2.bilateralFilter(image1,11,17,17)
    image1 = cv2.equalizeHist(image1)
    image1 = phan_doan_bang_cat_nguong(image1,110)
    image1 = ero_dila(image1,n).astype(np.uint8)
    image1 = cv2.Canny(image1,100,200)
    return image1
image1 = preImg(image1,1)
#cv2.imshow('Canny edge',image1)
#cv2.waitKey(0)

# Xác định đường bao:
cont,_= cv2.findContours(image1, mode=cv2.RETR_LIST, method= cv2.CHAIN_APPROX_SIMPLE)
img_drawcont = cv2.drawContours(image1,cont,-1,color=(233,150,122))
#cv2.imshow('img_cont',img_drawcont)

# Xác định đường bao chứa biển số xe:
    # Trường hợp tìm được đường bao
list=[]

def boundbox(cont):
    cont=sorted(cont, key = cv2.contourArea, reverse = True)[:10]
    for c in cont:
        peri = cv2.arcLength(c,True)
        temp = cv2.approxPolyDP(c,0.02 *peri,True)
        if 3 < len(temp) < 6:
            list.append(temp)
    return list
list = boundbox(cont)
    # Trường hợp không tìm được đường bao
    # Tiến hành xử lý lại ảnh
if len(list)==0:
    image2 = preImg(image2, 2)
    cont2, _ = cv2.findContours(image2, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    img_drawcont2 = cv2.drawContours(image2, cont2, -1, color=(233, 150, 122))
    cv2.imshow('img_cont2', img_drawcont2)
    list = boundbox(cont2)

# Vẽ bounding box:
NumberPlate = sorted(tuple(list), key=cv2.contourArea, reverse=True)

img_drawcont_final = cv2.drawContours(Orginal_image,[NumberPlate[0]],-1,color=(233,0,122),thickness=3)
#cv2.imshow('img_cont2',img_drawcont_final)
#cv2.waitKey(0)

# Crop numberplate
y_low = min(NumberPlate[0][:][:,:,0])
y_high = max(NumberPlate[0][:][:,:,0])

x_low = min(NumberPlate[0][:][:,:,1])
x_high = max(NumberPlate[0][:][:,:,1])
NP_Image = Final[x_low[0]-10:x_high[0]+10,y_low[0]-10:y_high[0]+10,:]
cv2.imshow('NumberPlate',NP_Image)
cv2.waitKey(0)
# Resize ảnh biển số
NP_Image = cv2.resize(NP_Image,(NP_Image.shape[0]*2,NP_Image.shape[1]*2),interpolation=cv2.INTER_AREA)
NP_Image = phan_doan_bang_cat_nguong(cv2.cvtColor(NP_Image,cv2.COLOR_BGR2GRAY),140)
cv2.imwrite('temp.jpg',NP_Image)
# Chuyển dữ liệu thành text
img= cv2.imread('temp.jpg')
text = pytesseract.image_to_string(img, lang='eng')
if len(text)==0:
    print("Vui lòng thử lại")
else: print("Number is :", text)

end_time = time.time()
print("Thời gian xử lý (s)", end_time-start_time)
cv2.waitKey(0)

