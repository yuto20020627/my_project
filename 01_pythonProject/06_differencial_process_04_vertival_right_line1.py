#形状フィルタリング、最も右側にいると考えられる縦の線を引く
import cv2
import numpy as np
import threading

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.mov')  # または 0 でカメラ起動


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break



        # マスクを適用して、指定エリア外の部分を黒くする
        fg_mask = cv2.bitwise_and(fg_mask, mask)

        # 前景から輪郭の抽出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 最も右側のX座標を初期化
    max_right_x = 0

    # 各輪郭について形状とサイズをフィルタリング
    for cnt in contours:
        # 最小外接矩形を取得
        x, y, w, h = cv2.boundingRect(cnt)

        # サイズとアスペクト比の条件を満たすか確認（例として高さ>幅、最小サイズ設定）
        if h > w and h > 50 and w > 30:
            # 検出した領域を矩形で囲む
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
