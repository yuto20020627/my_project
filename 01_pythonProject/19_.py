from datetime import datetime
from flask import Flask, jsonify
import json

app = Flask(__name__)


# バスの時刻表を読み込む関数
def read_bus_schedule():
    with open('bus_schedule.json', 'r', encoding='utf-8') as file:
        schedule_data = json.load(file)
    return schedule_data
###, 'r', encoding='utf-8'この部分を省略したい

# 次のバス時刻を取得する関数
def get_next_bus_time(times):
    # 現在の時刻を文字型strでを出力し、#datetime型に変換
    now_str = datetime.now().strftime('%H:%M')
    now = datetime.strptime(now_str, '%H:%M')

    # #scheduleのtimeをdatetime型に変換し、現在時刻以降の時刻全てを配列として取得
    times = [datetime.strptime(time, '%H:%M') for time in times]
    future_times = [time for time in times if time > now]

    if future_times:
        # 配列の先頭(現在時刻以降の最も早い時刻)を返す
        return future_times[0].strftime('%H:%M')
    else:
        # 次のバス時刻がなければNoneを返す
        return None


# バスの次の時刻を取得して返すエンドポイント
@app.route('/get_bus_time')
def get_bus_time():
    bus_schedule = read_bus_schedule()
    for bus_stop_times in bus_schedule["bus_stops"]:
        next_bus_time = get_next_bus_time(bus_stop_times)
        if next_bus_time:
            break  # 最初に見つかった時刻でループを抜ける

    return jsonify({
        'line': last_max_right_x,
        'next_buses': next_bus_time

    })


if __name__ == '__main__':
    app.run(debug=True)