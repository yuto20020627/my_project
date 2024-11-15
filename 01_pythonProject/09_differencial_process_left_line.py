#動画からHTML表示、MOG2を使わない、元フレーム、前景マスクを表示、一番左の人に縦線、画像をバスの待ち列
import cv2
import numpy as np  # numpy　列の数値計算
from flask import Flask, render_template, jsonify  # render_template, jsonify の修正
import threading

# Flaskアプリケーションのセットアップ
app = Flask(__name__)

# 動画の読み込みまたはカメラを起動
cap = cv2.VideoCapture('video/IMG_2420.MOV')

# 最初のフレームを背景モデルとして取得
ret, background = cap.read()  # ret　正常に動画を読み込めたか background 動画を格納
if not ret:  # ret == False
    print("Failed to capture background")
    cap.release()
    exit()

# グレースケール化（差分計算を簡単にするため）
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# 設定
min_duration_frames = 30  # 1秒間（フレームレートが30FPSの場合）
min_contour_area = 10 * 10  # 最小の検出面積（10x10ピクセル）

# 検出したエリアを保持するための辞書
detected_areas = {}  # キーとして(x,y,w,h)を持ち、値として何フレーム検出したかを持つx,yは左上の座標、wは幅、hは高さ

# 動画出力の設定
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 出力動画のフォーマット
out = cv2.VideoWriter('output_with_line.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

#動画のフレームの幅(3)、高さ(4)を設定
min_left_x = cap.get(3)
height = cap.get(4)
print('幅：' + str(min_left_x) + '高さ：' + str(height))

# バス待ち時間ページの情報を定期的に更新
def process_video():

    global min_left_x
    while cap.isOpened():  # 動画が開かれているか
        ret, frame = cap.read()  # 動画からフレームを1つ読み込む関数
        if not ret:  # ret == False
            break

        # 現在のフレームをグレースケール化
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 背景と現在のフレームとの差分を計算
        diff = cv2.absdiff(background, gray_frame)

        # 差分画像を二値化（しきい値を設定して、前景と背景を区別）しきい値３０以上を白（２５５）、以下を黒（０）
        _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # ノイズ除去（小さいブロブを除去）モロフォジー変換
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 輪郭抽出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_left_x_local = cap.get(3) #フレーム幅を最小値
        # 各輪郭をループ処理
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # 最小の外接矩形を計算
            area = w * h

            # 面積が閾値以上の場合
            if area > min_contour_area:
                if (x, y, w, h) not in detected_areas:
                    # そのエリアが新たに検出された場合、新規に追加
                    detected_areas[(x, y, w, h)] = 1
                else:
                    # 既に存在する場合はフレーム数を増加
                    detected_areas[(x, y, w, h)] += 1

                # 継続フレーム数が閾値を超えた場合
                if detected_areas[(x, y, w, h)] >= min_duration_frames:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 緑色の矩形を描画

                # 左端のX座標を更新,縦線を引くため
                left_x = x
                if left_x < min_left_x_local:
                    min_left_x_local = left_x
        # min_lift_xの値を更新
        min_left_x = min_left_x_local

        # 最も左側のX座標に基づいて縦線を描画
        if min_left_x < cap.get(3):
            cv2.line(frame, (int(min_left_x), 0), (int(min_left_x), frame.shape[0]), (255, 0, 0), 2)  # 元のフレームに青線を描画
            cv2.line(fg_mask, (int(min_left_x), 0), (int(min_left_x), fg_mask.shape[0]), (255, 0, 0), 2)  # 前景マスクに青線を描画

        # 動画として出力
        out.write(frame)  # フレームを動画として保存

        # 結果の表示（オプションでfg_maskを表示する場合）
        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Original Video with Detection', frame)

        # ESCキーで終了
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # 動画キャプチャを解放し、OpenCVウィンドウを閉める
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# バス待ち時間の情報をJSON形式で返す
@app.route('/get_bus_time')
def get_bus_time():
    return jsonify({'line': min_left_x})

# HTMLページをレンダリング
@app.route('/')
def index():
    return render_template('Queue_display_left.html')

# 動画処理を別スレッドで実行
thread = threading.Thread(target=process_video)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    app.run(threaded=True)