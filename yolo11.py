import cv2
from ultralytics import YOLO
import csv
import os

# YOLO11モデルをロード
model = YOLO("yolo11n.pt")

# 結果を保存するフォルダを作成
#output_folder = "output_frames"
#os.makedirs(output_folder, exist_ok=True)

# CSVファイルの準備
with open('detection_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # CSVヘッダーを書き込む
    #writer.writerow(["Frame Filename", "Label", "Confidence", "x1", "y1", "x2", "y2"])

    # Windowsのカメラ起動
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした。")
        exit()

    frame_count = 0  # フレーム番号

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        # YOLOで物体検出を行う
        results = model.predict(frame)

        # 結果をフレームに描画して表示
        annotated_frame = results[0].plot()

        # フレーム画像を保存
        #frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        #cv2.imwrite(frame_filename, annotated_frame)

        # 検出結果をCSVに保存
        for result in results[0].boxes:
            # バウンディングボックス座標を取得
            x1, y1, x2, y2 = map(int, result.xyxy[0])  
            confidence = result.conf[0]  # 信頼度
            class_id = int(result.cls[0])  # クラスID（数値）
            label = f"class_{class_id}"  # ラベル名（モデルに応じて修正可能）

            # CSVに書き込む
            #writer.writerow([frame_filename, label, confidence, x1, y1, x2, y2])

            # 描画
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # バウンディングボックス
            #cv2.putText(annotated_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 物体名と信頼度
            cv2.putText(annotated_frame, f"Coords: ({x1},{y1}) ({x2},{y2})", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 座標を表示

        # ウィンドウに描画した結果を表示
        cv2.imshow("YOLO Detection", annotated_frame)

        # フレームカウンタを更新
        frame_count += 1

        # 'q'を押すと終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
