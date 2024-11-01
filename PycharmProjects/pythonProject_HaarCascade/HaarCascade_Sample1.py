import cv2

# 顔検出用のHaar Cascadesモデルを読み込む
face_cascade = cv2.CascadeClassifier('C:haarcascade_frontalface_default.xml')
# 画像を読み込む
image = cv2.imread('image/IMG_1794.jpg')

# 画像をグレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出を実行
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 検出された顔に矩形を描画
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 結果の画像を表示
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()