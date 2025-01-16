#研究のために13_image_displayを改変
import cv2
import numpy as np

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.MOV')

# 動画のフレームの幅(3)、高さ(4)を設定、出力
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('幅：' + str(frame_width) + ' 高さ：' + str(frame_height))

# 任意の多角形エリアを設定
polygon_points = np.array([[0, 616], [0, 570], [1042, 85], [1042, 616]])

# 背景差分法のセットアップ（MOG2）
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=5000, varThreshold=10, detectShadows=True)

# グローバル変数（縦線の位置）
max_right_x = 0

# 動画処理開始
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("動画の読み込みが終了しました。")
        break

    # フレームと同じサイズの全黒マスクを作成
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # マスクに多角形を描画（白色で塗りつぶし）
    cv2.fillPoly(mask, [polygon_points], 255)

    # 背景差分でフレームに対する前景を抽出（背景：黒、前景：白）
    fg_mask = background_subtractor.apply(frame)

    # モルフォロジー変換を用いたノイズ除去
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # マスクを適用して、指定エリア外の部分を黒くする
    fg_mask = cv2.bitwise_and(fg_mask, mask)

    # 前景から輪郭の抽出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最も右側のX座標を更新
    max_right_x_local = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > w and h > 30 and w > 20:  # 条件に合う輪郭をフィルタリング
            right_x = x + w  # 右端の座標
            if right_x > max_right_x_local:
                max_right_x_local = right_x

    # max_right_xを更新
    max_right_x = max_right_x_local

    # マスク範囲を緑色で描画
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 検出結果を描画
    if max_right_x > 0:
        cv2.line(frame, (max_right_x, 0), (max_right_x, frame.shape[0]), (255, 0, 0), 2)

    # 処理結果を表示
    cv2.imshow('Processed Video', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()

