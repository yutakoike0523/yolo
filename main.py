from ultralytics import YOLO

# YOLOモデルをロードまたは初期化
model = YOLO("yolo11.yaml")  # カスタムモデル構成ファイル

# 学習の実行
model.train(
    data="C:\\Users\\koikeyuta\\Desktop\\yolo\\data.yaml",  # data.yamlファイルの正確なパスを指定
    epochs=100,                     # エポック数
    batch=16,                       # バッチサイズ
    imgsz=640,                      # 画像サイズ
    device=0                        # GPUを指定 (CPUのみの場合はdevice='cpu')
)
