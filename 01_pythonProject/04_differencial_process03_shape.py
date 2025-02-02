#03_differencial_process2.pyの人間認識の緑枠をつけたバージョン　形状とサイズのフィルタリング
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

    # 各輪郭について形状とサイズをフィルタリング
    for cnt in contours:
        # 最小外接矩形を取得
        x, y, w, h = cv2.boundingRect(cnt)

        # サイズとアスペクト比の条件を満たすか確認（例として高さ>幅、最小サイズ設定）
        if h > w and h > 50 and w > 30:
            # 検出した領域を矩形で囲む
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 結果の表示
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Original Video with Detection', frame)

    # ESCキーで終了
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
