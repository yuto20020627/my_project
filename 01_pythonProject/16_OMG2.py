#形状フィルタリング、最も右側にいると考えられる縦の線を引く
import cv2

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.mov')  # または 0 でカメラ起動

# 背景差分法（基本的なMOG2を使用）
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 背景差分による前景（動く物体）検出
    fg_mask = background_subtractor.apply(frame)

    # 前景マスクから輪郭を抽出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最も右側のX座標を初期化
    max_right_x = 0

    # 各輪郭について形状とサイズをフィルタリング
    for cnt in contours:

        # 右端のX座標を更新
        right_x = x + w
        if right_x > max_right_x:
            max_right_x = right_x

    # 最も右側のX座標に基づいて縦線を描画
    if max_right_x > 0:
        cv2.line(frame, (max_right_x, 0), (max_right_x, frame.shape[0]), (255, 0, 0), 2)

    # 結果の表示
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Original Video with Detection', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()