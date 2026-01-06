# Lending Pool Score

Mô tả dự án

Đây là một dự án đánh giá điểm tín dụng/score cho lending pool, chứa model, xử lý dữ liệu, và API để phục vụ kết quả. Thư mục chính gồm:

- `model/`: mã nguồn và notebook huấn luyện/đánh giá model.

Hướng dẫn chạy code

1. Chuyển vào thư mục model:

```
cd model/
```

2. Đồng bộ (run) với `uv`:

```
uv sync
```

Nếu chưa cài `uv`, cài bằng pip:

```
pip install uv
```

3. Mở và chạy notebook:

Mở `RF_model_regression_v2.ipynb` trong `model/` bằng Jupyter Notebook hoặc JupyterLab và chạy các cell:

```
jupyter lab RF_model_regression_v2.ipynb
```

Hoặc khởi động Jupyter và mở file từ giao diện web.

Ghi chú

- Nếu cần môi trường ảo, tạo và kích hoạt trước khi cài package:

```
source .venv/bin/activate
```

--
