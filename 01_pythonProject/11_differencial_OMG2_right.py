#MOG2を利用した背景差分法
# ノイズ除去（小さいブロブを除去）モロフォジー変換は行っていない
import cv2
import threading
from flask import Flask, render_template, jsonify

# Flaskアプリケーションのセットアップ

app = Flask(__name__)

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.MOV')

#動画のフレームの幅(3)、高さ(4)を設定、出力
min_left_x = cap.get(3)
height = cap.get(4)
print('幅：' + str(min_left_x) + '高さ：' + str(height))

# グローバル変数（縦線の位置）
max_right_x = 0
def process_video():
    global max_right_x
    # 背景差分法のセットアップ、MOG2のセットアップ
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    # while文1フレームの処理##################################################################################
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("cap.read()===False")
            break

        # 背景差分でフレームに対する前景を抽出、背景：黒（０）、前景：白（２５５）
        fg_mask = background_subtractor.apply(frame)

        # 前景から輪郭の抽出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最も右側のX座標を更新
        max_right_x_local = 0
        # 1フレームでcontours(検出された輪郭の数)繰り返す####################
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w and h > 50 and w > 30:  # 条件に合う輪郭をフィルタリング
                right_x = x + w #左端＋幅で右端の座標
                if right_x > max_right_x_local:#１つのフレームで同時に2つ以上の人間を検出した際、最も右の人に線を引くため
                    max_right_x_local = right_x
        ################################################################
        max_right_x = max_right_x_local#今回のフレームでのmax_right_xの更新

        # 検出結果を描画
        if max_right_x > 0:
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame.shape[0]), (255, 0, 0), 2)

        # 結果の表示
        cv2.imshow('Original Video with Detection', frame)

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break
    ###################################################################################################

    cap.release()
    cv2.destroyAllWindows()

# 縦線の位置をJSON形式で返す
@app.route('/get_bus_time')
def get_bus_time():
    return jsonify({'line': max_right_x})

# HTMLページをレンダリング
@app.route('/')
def index():
    return render_template('Queue_display_right.html')

# 動画処理を別スレッドで実行
thread = threading.Thread(target=process_video)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    app.run(threaded=True)