#天候の影響を受けやすい差分処理
import cv2

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/ScreenRecording_10-25-2024 15-45-20_1.mov')  # または 0 でカメラ起動

# 背景差分法（基本的なMOG2を使用）
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 背景差分による前景（動く物体）検出
    fg_mask = background_subtractor.apply(frame)

    # 結果の表示
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Original Video', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
