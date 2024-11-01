import cv2

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/ScreenRecording_10-25-2024 15-45-20_1.mov')  # または 0 でカメラ起動

# 背景差分法（影除去を有効化したMOG2）
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをHSV色空間に変換
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 背景差分処理
    fg_mask = background_subtractor.apply(hsv_frame)

    # 影除去（影の領域はMOG2で通常灰色としてマスクされる）
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # マスクを使って元画像から前景部分を抽出
    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # 結果の表示
    cv2.imshow('Foreground (Weather-Resistant)', foreground)
    cv2.imshow('Original Video', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
