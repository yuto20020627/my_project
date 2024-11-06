import cv2

# 顔認識用のHaar Cascade分類器のXMLファイルのパス
# OpenCVが提供する事前学習済みの分類器ファイルを使用します。
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_fullbody.xml')

# 画像の読み込み
image = cv2.imread('image/IMG_1785.JPG')  # 読み込む画像ファイルのパスを指定

# グレースケールに変換 (顔認識にはグレースケールが推奨されます)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔の検出
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 検出した顔に対して矩形を描画
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 青色の枠で矩形を描画

# 結果を表示
cv2.imshow('Detected Faces', image)

# 任意のキーを押すまで表示を保持
cv2.waitKey(0)

# ウィンドウを閉じる
cv2.destroyAllWindows()
