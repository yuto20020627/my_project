#MOG2を利用した背景差分法
#11_differencialにモルフォロジー変換を追加、MOG2のパラメータ調整
#画像認識の範囲を指定
#final_lineはない
import cv2
import numpy as np
import threading
from flask import Flask, render_template, jsonify

# Flaskアプリケーションのセットアップ
app = Flask(__name__)

# 動画の読み込み
cap = cv2.VideoCapture('video/IMG_2435_1.mov')

#動画のフレームの幅、高さ、フレーム数、総フレーム数、再生時間を設定、出力
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_FPS = int(cap.get(cv2.CAP_PROP_FPS))
frame_FPS_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('幅：' + str(frame_width) + ' 高さ：' + str(frame_height))
print('フレーム数:' + str(frame_FPS) + ' 総フレーム数:' + str(frame_FPS_count))
print('動画の再生時間(秒):' + str(frame_FPS_count / frame_FPS))

# 任意の多角形エリアを設定
roi_area = np.array([[0, 616], [0, 570], [1042, 85], [1042, 616]])

# グローバル変数（縦線の位置）
max_right_x = 0

def process_video():

    global max_right_x

    # 背景差分法のセットアップ、MOG2のセットアップ
    # history(500):背景モデルの更新回数　varThreshold(16):前景判断のしきい値　detectShadows(True):影の検出を有効にするか
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=10, detectShadows=True)

    # while文1フレームの処理##################################################################################
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("cap.read()===False")
            break

        # フレームと同じサイズの全黒マスクを作成（全体を黒にする）np.zeros すべての初期値0の配列
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # マスクに多角形を描画（多角形内を白色で塗りつぶし）
        cv2.fillPoly(mask, [roi_area], 255)

        # 背景差分でフレームに対する前景を抽出、背景：黒（０）、前景：白（２５５）
        fg_mask = background_subtractor.apply(frame)

        #モルフォロジー変換を用いたノイズ除去
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # マスクを適用して、指定エリア外の部分を黒くする
        fg_mask = cv2.bitwise_and(fg_mask, mask)

        # 前景から輪郭の抽出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最も右側のX座標を更新
        max_right_x_local = 0
        # 1フレームでcontours(検出された輪郭の数)繰り返す####################
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w and h > 30 and w > 20:  # 条件に合う輪郭をフィルタリング
                right_x = x + w #左端＋幅で右端の座標
                if right_x > max_right_x_local:#１つのフレームで同時に2つ以上の人間を検出した際、最も右の人に線を引くため
                    max_right_x_local = right_x
        ################################################################

        max_right_x = max_right_x_local#今回のフレームでのmax_right_xの更新

        # マスク範囲を緑色で描画
        # cv2.polylines(画像名、[すべての頂点]、isClosed=始点と終点を結ぶか、color=（色）、thickness=線の太さ)
        cv2.polylines(frame, [roi_area], isClosed=True, color=(0, 255, 0), thickness=2)

        # 検出結果を描画
        if max_right_x > 0:
            # cv2.line(画像名、（始点）、（終点）、（色）、線の太さ)
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame_height), (255, 0, 0), 2)

        # 結果の表示(出力時の名前、画像名)
        cv2.imshow('Queue_Video', frame)

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