import cv2
import numpy as np

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/ScreenRecording_10-25-2024 15-45-20_1.mov')

# 最初のフレームを背景モデルとして取得
ret, background = cap.read()
if not ret:
    print("Failed to capture background")
    cap.release()
    exit()

# グレースケール化（差分計算を簡単にするため）
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# 設定
min_duration_frames = 5  # 1秒間（フレームレートが30FPSの場合）
min_contour_area = 10 * 10  # 最小の検出面積（50x50ピクセル）

# 検出したエリアを保持するための辞書
detected_areas = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 現在のフレームをグレースケール化
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 背景との差分を計算
    diff = cv2.absdiff(background, gray_frame)

    # 差分画像を二値化（しきい値を設定して、前景と背景を区別）
    _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # ノイズ除去（小さいブロブを除去）
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 輪郭抽出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最も右側のX座標を初期化
    max_right_x = 0

    # 各輪郭をループ処理
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # 面積が閾値以上の場合
        if area > min_contour_area:
            # エリアが辞書に存在しない場合、新規に追加
            if (x, y, w, h) not in detected_areas:
                detected_areas[(x, y, w, h)] = 1
            else:
                # フレームカウントを増加
                detected_areas[(x, y, w, h)] += 1

            # 継続フレーム数が閾値を超えた場合
            if detected_areas[(x, y, w, h)] >= min_duration_frames:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 右端のX座標を更新
            right_x = x + w
            if right_x > max_right_x:
                max_right_x = right_x

    # 検出エリアが閾値を満たさない場合はリセット
    for key in list(detected_areas.keys()):
        if detected_areas[key] < min_duration_frames:
            detected_areas[key] = 0

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
