#03_differencial_process2.pyの人間認識の緑枠をつけたバージョン カラーフィルタリング
import cv2
import numpy as np

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/ScreenRecording_10-25-2024 15-45-20_1.mov')  # または 0 でカメラ起動

# 背景差分法（基本的なMOG2を使用）
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# 暗い色（黒色）を検出する範囲 (HSVカラー空間)
lower_black = np.array([0, 0, 0])  # 黒色の範囲下限
upper_black = np.array([180, 255, 50])  # 黒色の範囲上限

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 背景差分による前景（動く物体）検出
    fg_mask = background_subtractor.apply(frame)

    # 前景マスクを適用して、元の画像とマスクを組み合わせ
    masked_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # 色空間をBGRからHSVに変換
    hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

    # カラーフィルタリング（暗い色を検出）
    color_mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    # フィルタリングされた色領域を輪郭抽出
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 最小外接矩形を取得
        x, y, w, h = cv2.boundingRect(cnt)

        # サイズ条件を確認
        if h > 50 and w > 30:
            # 条件を満たす領域を矩形で囲む
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 結果の表示
    cv2.imshow('Color Filtered Mask', color_mask)
    cv2.imshow('Original Video with Color Detection', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
