import cv2
import numpy as np

# 動画の読み込み
cap = cv2.VideoCapture('video/IMG_2435_1.mov')

# 動画のフレームの幅、高さ、フレーム数、総フレーム数、再生時間を設定、出力
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_FPS = int(cap.get(cv2.CAP_PROP_FPS))
frame_FPS_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('幅：' + str(frame_width) + ' 高さ：' + str(frame_height))
print('フレーム数:' + str(frame_FPS) + ' 総フレーム数:' + str(frame_FPS_count))
print('動画の再生時間(秒):' + str(frame_FPS_count / frame_FPS))

# 検出を行う多角形エリアを設定
roi_area = np.array([[0, 616], [0, 570], [1042, 85], [1042, 616]])

# グローバル変数（縦線の位置）
max_right_x = 0
frame_count = 0
last_max_right_x = 0

def process_video():
    global max_right_x
    global frame_count
    global last_max_right_x

    # 背景差分法のセットアップ、MOG2のセットアップ
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=10, detectShadows=True)

    # while文1フレームの処理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("cap.read()===False")
            break

        # フレームと同じサイズの全黒マスクを作成（全体を黒にする）
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # マスクに多角形を描画（多角形内を白色で塗りつぶし）
        cv2.fillPoly(mask, [roi_area], 255)

        # 背景差分でフレームに対する前景を抽出
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
                right_x = x + w  # 左端＋幅で右端の座標
                if right_x > max_right_x_local:  # １つのフレームで同時に2つ以上の人間を検出した際、最も右の人に線を引くため
                    max_right_x_local = right_x

        # 変更があれば今回のフレームでのmax_right_xの更新、なければ更新しない
        if max_right_x_local > 0:
            max_right_x = max_right_x_local  # 今回のフレームでのmax_right_xの更新
            frame_count = 0 # カウントリセット
        else:
            frame_count = frame_count + 1

        if frame_count >= 90:
            last_max_right_x = max_right_x

        # マスク範囲を緑色で描画
        cv2.polylines(frame, [roi_area], isClosed=True, color=(0, 255, 0), thickness=2)

        # 検出結果を描画
        if max_right_x > 0:
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame_height), (255, 0, 0), 2)
            cv2.line(frame, (last_max_right_x, 0), (last_max_right_x, frame_height), (0, 0, 255), 2)

        # 結果の表示(出力時の名前、画像名)
        cv2.imshow('Queue_Video', frame)

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

