from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import cv2

def collect_image (filename, _margin_pixel, _areaLimit, _light):
    #_threshold
   
    img = cv2.imread(filename)
    
    #숫자부분만 떼어내기 위한 Process
    
    #이미지 gray scale 시키기 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #이미지 gaussian smoothing을 통해 Noise 제거
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #noise 제거 후 histogram equalization 한 다음에 
    #histogram에서 1 % 차지하는 밝기(threshold) 구하기
    img_equal = cv2.equalizeHist(img_blur)
    hist = cv2.calcHist([img_equal], [0],None,  [256], [0, 256])
    
    s = np.zeros(np.size(hist));
    s[0] = hist[0];
    ratio = 0.01;
    threshold = 0;
    
    for i in range(1,256):
        if (s[i-1]>=sum(hist)*ratio):
            threshold = i
            break
        s[i]=s[i-1]+hist[i]

    ret, img_th = cv2.threshold(img_equal, threshold, 255, cv2.THRESH_BINARY_INV)
    #임계점인 threshold 미만인 부분은 0으로 이상인 부분은 255의 밝기로 thresholding
    
    #이미지에서 사각형 뽑아내기
    contours,_ =cv2.findContours(img_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    
    #숫자만 담아낼 수 있도록, 사각형의 넒이가 _areaLimit 이상이고 
    # _areaLimit 의 10배 보다 작은 사각형만 저장
    rects = [cv2.boundingRect(each) for each in contours ]
    rects = [(x,y,w,h) for (x,y,w,h) in rects if (w*h>=_areaLimit and w*h<=_areaLimit*10)]
    
    img2 = img.copy() #뽑아낸 사각형이 그려진 이미지가 될 변수
    margin_pixel = _margin_pixel #이미지의 원래 사각형에서 더 추가할 범위 설정
    img_result = [] #margin까지 포함하는 각 사각형의 리스트를 가질 변수
    img_for_class = img.copy() #이미지에서 사각형 범위를 뽑아오기 위해 원본이미지를 copy
    
    for rect in rects:
        # 사각형의 시작위치에서 margin을 더하면 이미지의 범위를 벗어나는 경우
        if(margin_pixel>rect[1] or margin_pixel>rect[0] or rect[1]+rect[3]+margin_pixel>np.size(img,0) 
           or rect[0]+rect[2]+margin_pixel > np.size(img,1) ):
            continue;
        #[y-margin : y+h+margin, x-margin : x+w+margin]
        img_result.append(img_for_class[rect[1]-margin_pixel : rect[1]+rect[3]+margin_pixel,
                                        rect[0]-margin_pixel : rect[0]+rect[2]+margin_pixel])
    
        # 사각형 이미지에 그리기
        cv2.rectangle(img2, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 
        
    #사각형 그려진 그림 출력
    plt.figure(figsize=(10,8))
    plt.imshow(img2)
    plt.xticks([]), plt.yticks([])
    plt.title("Image with rectangles")
    print("The number of number characters is ",len(img_result))#사각형의 갯수 출력

    _data = [] #뽑아낸 이미지의 1차원 배열들이 담길 리스트
    
    for n in img_result:
            
        #뽑아낸 사진들을 28x28 행렬로 사이즈 바꾼다
        test_num = cv2.resize(n, (28,28))[:,:,1]
        #사진 반전 시키고(검은색인 숫자 부분 높은 숫자로) 배경은 수치를 0으로
        #_light 이상의 반전 밝기를 가진 부분만 0이 아닌 숫자로 만든다
        test_num = 255- test_num
        test_num = (test_num > _light) * test_num
        #28x28 행렬로 사이즈 바꾼 뽑아낸 이미지들을 출력
        test_num = test_num.astype('float32') / 255.
        _data.append(test_num.reshape(1,784).tolist())

    return _data
