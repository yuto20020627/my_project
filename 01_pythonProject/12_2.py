import cv2
import threading
from flask import Flask, render_template, jsonify
from collections import deque
import numpy as np

# Flaskアプリケーションのセットアップ
app = Flask(__name__)

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2435_1.MOV')

# 動画のフレームの幅(3)、高さ(4)を設定、出力
min_left_x = cap.get(3)
height = cap.get(4)
print('幅：' + str(min_left_x) + '高さ：' + str(height))

# グローバル変数
max_right_x = 0
final_max_right_x = None
max_values = deque(maxlen=300)  # ヒストグラム用に直近20フレームのデータを保持

def calculate_mode(values):
    """リストから最頻値を計算する"""
    if len(values) == 0:
        return None  # 空リストの場合は計算しない

    try:
        # 値をヒストグラム化（ビン幅10）
        bins = np.arange(min(values), max(values) + 10, 10)
        hist, bin_edges = np.histogram(values, bins=bins)

        # ヒストグラムが計算されないケースを処理
        if len(hist) == 0 or np.sum(hist) == 0:
            return None

        # 最頻値（頻度が最も高いビンの中心値を返す）
        max_bin_idx = np.argmax(hist)
        return (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) // 2
    except Exception as e:
        print(f"ヒストグラム計算中にエラーが発生しました: {e}")
        return None

def process_video():
    global max_right_x, final_max_right_x, max_values

    # 背景差分法のセットアップ、MOG2のセットアップ
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

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
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > w and h > 50 and w > 30:  # 条件に合う輪郭をフィルタリング
                right_x = x + w  # 左端＋幅で右端の座標
                if right_x > max_right_x_local:
                    max_right_x_local = right_x
        max_right_x = max_right_x_local

        # 最新のmax_right_xをヒストグラム用データに追加
        if max_right_x > 0:
            max_values.append(max_right_x)

        # ヒストグラムから最頻値を計算して確定線を更新
        mode_value = calculate_mode(list(max_values))
        if mode_value is not None:
            final_max_right_x = mode_value

        # 描画処理
        if max_right_x > 0:  # 動的な線
            cv2.line(frame, (max_right_x, 0), (max_right_x, frame.shape[0]), (255, 0, 0), 2)
        if final_max_right_x is not None:  # 確定線
            cv2.line(frame, (final_max_right_x, 0), (final_max_right_x, frame.shape[0]), (0, 255, 0), 2)

        # 結果の表示
        cv2.imshow('Original Video with Detection', frame)

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 縦線の位置をJSON形式で返す
@app.route('/get_bus_time')
def get_bus_time():
    # numpyデータ型の場合はintに変換
    if isinstance(final_max_right_x, np.integer):
        line_value = int(final_max_right_x)
    else:
        line_value = final_max_right_x if final_max_right_x is not None else 0

    return jsonify({'line': line_value})


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
