#17_image_display2を改変　処理後の黒白画像も出力
#
#中央値を用いた確定線を追加
import cv2
import numpy as np

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.MOV')

# 動画のフレームの幅、高さ、フレーム数、総フレーム数、再生時間を設定、出力
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_FPS = int(cap.get(cv2.CAP_PROP_FPS))
frame_FPS_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

number_of_seconds = 10000 #一時停止までの時間
pause_frame_count = frame_FPS * number_of_seconds

print('幅：' + str(frame_width) + ' 高さ：' + str(frame_height))
print('フレーム数:' + str(frame_FPS) + ' 総フレーム数:' + str(frame_FPS_count))
print('動画の再生時間(秒):' + str(frame_FPS_count / frame_FPS))

# 任意の多角形エリアを設定
polygon_points = np.array([[0, 616], [0, 570], [1042, 85], [1042, 616]])

# 背景差分法のセットアップ（MOG2）
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=10, detectShadows=True)

# グローバル変数（縦線の位置）
max_right_x = 0
frame_count = 0
confirmed_line_x = 0  # 確定線
history = []  # 過去のmax_right_xを保存するリスト
HISTORY_MAX_SIZE = 300  # 最大履歴サイズ
THRESHOLD_DIFF = 20  # 確定線更新のしきい値（許容誤差）

def process_video():
    global max_right_x
    global confirmed_line_x
    global history
    global frame_count
    # 動画処理開始
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("動画の読み込みが終了しました。")
            break

        frame_count += 1

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
        # 1フレームでcontours(検出された輪郭の数)繰り返す#########################
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w and h > 30 and w > 20:  # 条件に合う輪郭をフィルタリング
                right_x = x + w  # 右端の座標
                if right_x > max_right_x_local:
                    max_right_x_local = right_x
        #####################################################################
        # max_right_xを更新
        max_right_x = max_right_x_local

        # 確定ライン追加#######################################
        # 履歴リストの管理
        if max_right_x > 0:
            history.append(max_right_x)
            if len(history) > HISTORY_MAX_SIZE: #HISTORY_MAX_SIZE = 30
                history.pop(0)  # 古いデータを削除

        # 確定線の更新ロジック
        if history:
            median_right_x = int(np.median(history))  # 移動中央値
            if abs(median_right_x - confirmed_line_x) > THRESHOLD_DIFF: #THRESHOLD_DIFF = 20
                confirmed_line_x = median_right_x
        #####################################################

        # マスク範囲を緑色で描画
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # 検出結果を描画
        if max_right_x > 0:
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame.shape[0]), (255, 0, 0), 2)
        if confirmed_line_x > 0: #確定線
            cv2.line(frame, (confirmed_line_x, 0), (confirmed_line_x, frame.shape[0]), (0, 0, 255), 2)

        # 処理結果と背景差分マスクを表示
        cv2.imshow('Processed Video', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        # 一時停止処理 追加項目
        if frame_count == pause_frame_count:
            print("動画を一時停止します。ESCキー:終了、他のキー:再開")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESCキーで終了
                break

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # リソース解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video()
