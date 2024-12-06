# MOG2を利用した背景差分法
# 11_differencialにモルフォロジー変換を追加、MOG2のパラメータ調整
# 画像認識の範囲を指定
#常に最後尾の線が描写されるように変更
#14_を改変しリストを使い誤差範囲を決める
import cv2
import numpy as np
import threading
from flask import Flask, render_template, jsonify
import json
from datetime import datetime

# Flaskアプリケーションのセットアップ
app = Flask(__name__)

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
roi_area = np.array([[0, 616], [0, 510], [135, 510], [1042, 85], [1042, 616]])

# グローバル変数（縦線の位置）
max_right_x = 0
frame_count = 0
last_max_right_x = 0
position_history = []

def process_video():
    global max_right_x
    global frame_count
    global last_max_right_x
    global position_history

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

        # モルフォロジー変換を用いたノイズ除去
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # マスクを適用して、指定エリア外の部分を黒くする
        fg_mask = cv2.bitwise_and(fg_mask, mask)

        # 前景から輪郭の抽出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最も右側のX座標を更新
        max_right_x_local = 0
        # 1フレームでcontours(検出された輪郭の数)繰り返す########################
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 30 and w > 20:  # 条件に合う輪郭をフィルタリング
                right_x = x + w  # 左端＋幅で右端の座標
                if right_x > max_right_x_local:  # １つのフレームで同時に2つ以上の人間を検出した際、最も右の人に線を引くため
                    max_right_x_local = right_x
        ###################################################################
        # 変更があれば今回のフレームでのmax_right_xの更新、なければ更新しない
        if max_right_x_local > 0:
            max_right_x = max_right_x_local# 今回のフレームでのmax_right_xの更新
        ######確定線の更新(2秒)#############################
        # 位置履歴に現在の位置を追加
        position_history.append(max_right_x) #.appendで配列の末尾に追加
        if len(position_history) > 60:
            position_history.pop(0)

        # 確定線の更新(過去60フレームの数値が誤差+-5なら更新)
        if len(position_history) == 60 and max_right_x != frame_width:
            count = 0
            for i in range(1,60):
                if abs(position_history[i] - position_history[0]) <= 5:
                    count = count + 1
            if count == 59:
                last_max_right_x = max_right_x  # 赤線の位置を更新
        ##################################################
        # マスク範囲を緑色で描画
        # cv2.polylines(画像名、[すべての頂点]、isClosed=始点と終点を結ぶか、color=（色）、thickness=線の太さ)
        cv2.polylines(frame, [roi_area], isClosed=True, color=(0, 255, 0), thickness=2)

        # 検出結果を描画
        if max_right_x > 0:
            # cv2.line(画像名、（始点）、（終点）、（色）、線の太さ)
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame_height), (255, 0, 0), 2)
            cv2.line(frame, (last_max_right_x, 0), (last_max_right_x, frame_height), (0, 0, 255), 2)

        # 結果の表示(出力時の名前、画像名)
        cv2.imshow('Queue_Video', frame)

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break
    ###################################################################################################

    cap.release()
    cv2.destroyAllWindows()

#バスの時刻表の表示
def read_bus_schedule():
    with open('bus_schedule.json', 'r', encoding='utf-8') as file:
        schedule = json.load(file)
    return schedule

# 次のバス時刻を取得する関数
def get_next_bus_time(times):
    now = datetime.now().strftime('%H:%M')
    now_time = datetime.strptime(now, '%H:%M')

    # 時刻をdatetime型に変換し、現在時刻以降の最も早い時刻を取得
    times = [datetime.strptime(time, '%H:%M') for time in times]
    future_times = [time for time in times if time > now_time]

    if future_times:
        return future_times[0].strftime('%H:%M')
    else:
        return None  # 次のバス時刻がなければNoneを返す

# 縦線の位置をJSON形式で返す
@app.route('/get_bus_time')
def get_bus_time():
    bus_schedule = read_bus_schedule()
    next_buses = {}
    for bus_stop in bus_schedule["bus_stops"]:
        next_time = get_next_bus_time(bus_stop["times"])
        next_buses[bus_stop["stop"]] = next_time

    return jsonify({
        'line': last_max_right_x,
        'next_buses': next_buses
    })

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