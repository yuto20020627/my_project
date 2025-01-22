import cv2
import numpy as np

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.mov')  # または 0 でカメラ起動

# 背景差分法（基本的なMOG2を使用）
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# フレームのサイズを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # 背景差分による前景（動く物体）検出
    fg_mask = background_subtractor.apply(frame)

    # モルフォロジー変換を用いたノイズ除去
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # マスクを適用して、指定エリア外の部分を黒くする
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)

    # 前景マスクから輪郭を抽出
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 各輪郭について形状とサイズをフィルタリング
    for cnt in contours:
        # 最小外接矩形を取得
        x, y, w, h = cv2.boundingRect(cnt)

        # サイズとアスペクト比の条件を満たすか確認（例として高さ>幅、最小サイズ設定）
        if h > w and h > 50 and w > 30:
            # 検出した領域を矩形で囲む
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # マスク範囲を緑色で描画
    cv2.polylines(frame, [roi_area], isClosed=True, color=(0, 255, 0), thickness=2)

    # 結果の表示
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Original Video with Detection', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
