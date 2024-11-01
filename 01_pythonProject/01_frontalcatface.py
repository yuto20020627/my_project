import cv2

# カスケード分類器の読み込み（顔認識用）
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# ビデオキャプチャの開始（Webカメラを使用）
cap = cv2.VideoCapture(0)

while True:
    # フレームの読み込み
    ret, frame = cap.read()

    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔認識
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 認識された顔を四角で囲む
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # フレームを表示
    cv2.imshow('Video', frame)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの閉鎖
cap.release()
cv2.destroyAllWindows()
