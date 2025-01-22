import cv2
import numpy as np

# 動画の読み込み
cap = cv2.VideoCapture('video/IMG_2435_1.mov')  # または 0 でカメラ起動

# 背景差分法（基本的なMOG2を使用）
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# フレームサイズの取得
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 検出を行う多角形エリアを設定
roi_area = np.array([[0, frame_height], [0, frame_height // 2],
                     [frame_width // 4, frame_height // 2],
                     [frame_width - 1, frame_height // 4],
                     [frame_width - 1, frame_height]])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # マスクの作成
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_area], 255)

    # 背景差分で前景を抽出
    fg_mask = background_subtractor.apply(frame)

    # ノイズ除去
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # ROIマスクの適用
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)

    # 輪郭の抽出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最も右側のX座標を初期化
    max_right_x = 0

    # 各輪郭を処理
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > w and h > 50 and w > 30:  # サイズ条件
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
