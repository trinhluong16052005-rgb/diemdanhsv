# train_faces.py
import os
import pickle
from glob import glob
import numpy as np
import cv2
import csv  # Đã chuyển import này lên đầu
from insightface.app import FaceAnalysis  # <-- THAY ĐỔI: Import thư viện mới

# ================== CẤU HÌNH ==================
THU_MUC_ANH = "app/data/faces_raw"  # ảnh đầu vào theo mã SV
THU_MUC_EMB = "app/data/embeddings"  # nơi lưu file .pkl
os.makedirs(THU_MUC_EMB, exist_ok=True)

# Ngưỡng chất lượng ảnh (Giữ nguyên, code của bạn rất tốt)
MIN_FACE_SIZE = 120  # min(w, h) mặt (px)
MIN_SHARPNESS = 80.0  # var(Laplacian) - tăng nếu còn mờ
BRIGHT_LOW, BRIGHT_HIGH = 40, 220  # mean gray hợp lệ

SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ================== HÀM TIỆN ÍCH ==================

def is_good_quality(img_bgr: np.ndarray, face_area: dict) -> bool:
    x = int(face_area.get("x", 0))
    y = int(face_area.get("y", 0))
    w = int(face_area.get("w", 0))
    h = int(face_area.get("h", 0))
    if min(w, h) < MIN_FACE_SIZE:
        return False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    y1, y2 = max(y, 0), max(y + h, 0)
    x1, x2 = max(x, 0), max(x + w, 0)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    brightness = float(np.mean(roi))
    if brightness < BRIGHT_LOW or brightness > BRIGHT_HIGH:
        return False

    sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    if sharp < MIN_SHARPNESS:
        return False

    return True


def remove_outliers(embs: list[np.ndarray], z: float = 1.0) -> np.ndarray:
    """
    Loại embedding 'lạc loài' so với centroid bằng cosine. (Giữ nguyên)
    """
    if len(embs) <= 2:
        return np.stack(embs, axis=0)

    E = np.stack(embs, axis=0).astype("float32")  # [N, D]
    # --- THAY ĐỔI: Xóa dòng chuẩn hóa L2 vì embedding đã được chuẩn hóa ---
    # E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    centroid = np.mean(E, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)  # Vẫn chuẩn hóa centroid

    sims = E @ centroid
    thr = float(sims.mean() - z * sims.std())
    kept = E[sims >= thr]
    return kept if kept.size else E  # nếu lọc sạch thì trả lại E gốc


# --- THAY ĐỔI: Đã xóa hàm represent_with_fallback() ---


# ================== PIPELINE TRAIN ==================
# --- THAY ĐỔI: Hàm chính giờ nhận 'model' làm tham số ---
def tao_du_lieu_huan_luyen(model: FaceAnalysis):
    du_lieu_ma: list[str] = []
    du_lieu_vector: list[np.ndarray] = []

    if not os.path.isdir(THU_MUC_ANH):
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {THU_MUC_ANH}")

    # --- Đọc danh sách CSV (để hiển thị họ tên) ---
    csv_path = "app/data/danhsach.csv"
    ma_to_ten = {}
    if os.path.exists(csv_path):
        # --- THAY ĐỔI: Đã xóa 'import csv' vì đã chuyển lên đầu ---
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "ma_sv" in row and "ho_ten" in row:
                    ma_to_ten[row["ma_sv"].strip()] = row["ho_ten"].strip()

    # --- Duyệt thư mục ảnh (Giữ nguyên logic chọn thư mục) ---
    students = sorted([d for d in os.listdir(THU_MUC_ANH)
                       if os.path.isdir(os.path.join(THU_MUC_ANH, d))])

    if not students:
        raise RuntimeError(f"Thư mục '{THU_MUC_ANH}' rỗng. Hãy thêm ảnh trước.")

    print("\n📁 CÁC THƯ MỤC ẢNH ĐÃ PHÁT HIỆN:")
    for i, sv in enumerate(students, start=1):
        ten = ma_to_ten.get(sv, "❓ Không tìm thấy trong CSV")
        print(f"  {i:02d}. {sv} – {ten}")

    print("\n👉 Nhập số thứ tự thư mục bạn muốn huấn luyện (vd: 1,3,5) hoặc 'all' để chọn tất cả:")
    choice = input("→ Lựa chọn: ").strip()

    if choice.lower() != "all":
        try:
            selected_idx = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            students = [students[i - 1] for i in selected_idx if 1 <= i <= len(students)]
            print(f"✅ Đã chọn {len(students)} thư mục để huấn luyện: {', '.join(students)}")
        except Exception:
            print("⚠️ Lựa chọn không hợp lệ, mặc định sẽ huấn luyện toàn bộ.")

    for ma_sv in students:
        thu_muc_sv = os.path.join(THU_MUC_ANH, ma_sv)
        print(f"\n➡️  Đang xử lý: {ma_sv}")

        # Gom danh sách ảnh hợp lệ
        img_paths = []
        for ext in SUPPORTED_EXTS:
            img_paths.extend(glob(os.path.join(thu_muc_sv, ext)))
        img_paths = sorted(img_paths)

        if not img_paths:
            print(f"⚠️  Không tìm thấy ảnh trong: {thu_muc_sv}")
            continue

        embs_sv: list[np.ndarray] = []

        for path in img_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"⚠️  Không đọc được ảnh: {path}")
                    continue

                # --- THAY ĐỔI: Gọi model insightface ---
                # Trích xuất (model.get() trả về list các đối tượng Face)
                faces = model.get(img)

                for idx, face in enumerate(faces, start=1):
                    # --- THAY ĐỔI: Lấy 'area' từ 'bbox' của insightface ---
                    # insightface trả về bbox [x1, y1, x2, y2]
                    bbox = face.bbox.astype(int)
                    area = {
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2] - bbox[0],
                        "h": bbox[3] - bbox[1]
                    }

                    if not is_good_quality(img, area):
                        continue

                    # --- THAY ĐỔI: Lấy embedding trực tiếp ---
                    emb = face.embedding.astype("float32")
                    # (Không cần gọi l2norm)
                    embs_sv.append(emb)

                    # ===== Lưu khuôn mặt đã được training (Giữ nguyên) =====
                    try:
                        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                        crop = img[y:y + h, x:x + w]
                        preview_dir = os.path.join(THU_MUC_EMB, "trained_faces", ma_sv)
                        os.makedirs(preview_dir, exist_ok=True)
                        save_path = os.path.join(preview_dir,
                                                 f"{os.path.splitext(os.path.basename(path))[0]}_{idx:02d}.jpg")
                        cv2.imwrite(save_path, crop)
                    except Exception:
                        pass

            except Exception as e:
                print(f"⚠️  Lỗi với ảnh {os.path.basename(path)}: {e}")

        if not embs_sv:
            print(f"⚠️  Không có embedding hợp lệ cho {ma_sv} → bỏ qua SV này")
            continue

        # Loại outlier để tăng ổn định
        embs_sv = list(remove_outliers(embs_sv, z=1.0))  # (Đã sửa hàm remove_outliers)

        print(f"✔️  Ảnh hợp lệ sau lọc/outlier: {len(embs_sv)}")

        # Lưu toàn bộ embedding đã lọc (Giữ nguyên)
        for emb in embs_sv:
            du_lieu_ma.append(ma_sv)
            du_lieu_vector.append(emb.astype("float32"))

    if not du_lieu_vector:
        raise RuntimeError("❌ Không có embedding nào được tạo. Kiểm tra lại dữ liệu.")

    du_lieu = {"ma_sv": du_lieu_ma, "vector": du_lieu_vector}
    duong_dan_luu = os.path.join(THU_MUC_EMB, "du_lieu_khuon_mat.pkl")

    with open(duong_dan_luu, "wb") as f:
        pickle.dump(du_lieu, f)

    print(f"\n🎉 Đã huấn luyện xong! Dữ liệu lưu tại: {duong_dan_luu}")


if __name__ == "__main__":
    # --- THAY ĐỔI: Khởi tạo model insightface trước khi gọi hàm ---
    print("Đang tải mô hình InsightFace (ArcFace)...")
    print("Lần chạy đầu tiên sẽ tự động tải model, có thể mất vài phút.")

    # Dùng 'CPUExecutionProvider' để chạy trên CPU
    # Dùng 'CUDAExecutionProvider' nếu bạn có GPU NVIDIA
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 cho CPU

    print("Mô hình đã tải xong, bắt đầu huấn luyện...")
    tao_du_lieu_huan_luyen(app)