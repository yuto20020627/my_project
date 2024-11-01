import cv2

# 顔認識用のHaar Cascade分類器を読み込む
face_cascade = cv2.CascadeClassifier('C:haarcascade_frontalface_default.xml')

# 動画ファイルを読み込む
cap = cv2.VideoCapture('image/IMG_2310.mp4')

# 動画が正常に読み込めたか確認
if not cap.isOpened():
    print("Error: 動画ファイルが読み込めません")
    exit()

while True:
    # フレームを1つずつ読み込む
    ret, frame = cap.read()

    # フレームが正常に読み込めなかった場合、ループを終了
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))
    #scaleFactor 画像の縮小、値を小さくすれば細かく検出、計算コスト上昇(1.1)
    #minNeighbors 隣接された場所を顔として認識するために必要な数(5)
    #minSize 検出する物体の最小の大きさ(30.30)

    # 検出した顔に枠を描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # フレームを表示
    cv2.imshow('Video', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
