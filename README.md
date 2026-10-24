<div align="center">

# Early Warning for Low Mathematics Performance

**Một quy trình Machine Learning có thể tái lập để nghiên cứu bài toán cảnh báo sớm học sinh có nguy cơ đạt điểm Toán cuối kỳ dưới 10/20.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## Tổng quan

Dự án sử dụng bộ dữ liệu **Student Performance** từ UCI Machine Learning Repository để khảo sát khả năng nhận diện sớm học sinh có nguy cơ đạt điểm cuối kỳ môn Toán (`G3`) dưới ngưỡng 10/20. Mô hình chỉ sử dụng thông tin ngoài các cột điểm `G1`, `G2` và `G3`, nhằm mô phỏng tình huống cảnh báo trước khi biết điểm cuối kỳ.

Repository bao gồm một notebook trình bày toàn bộ quy trình nghiên cứu, các mô-đun Python tách theo chức năng, mã thu thập dữ liệu, dữ liệu minh họa để kiểm tra ngoại tuyến, kiểm thử và tài liệu về phương pháp.

> **Trạng thái dữ liệu:** `data/student-mat-demo.csv` chỉ là dữ liệu minh họa để kiểm tra khả năng thực thi của chương trình. Khi có Internet, notebook hoặc `python run_project.py --crawl` sẽ cố tải CSV chính thức từ UCI. Chỉ sử dụng kết quả sinh từ CSV chính thức khi báo cáo benchmark; nguồn dữ liệu được lưu cùng siêu dữ liệu để kiểm tra.

## Câu hỏi nghiên cứu

> Trong phạm vi bộ dữ liệu UCI, các đặc trưng học tập và bối cảnh không bao gồm điểm số có thể nhận diện nhóm học sinh có `G3 < 10` đến mức nào, và hiệu quả dự đoán khác nhau ra sao giữa lớp 0 và lớp 1?

### Định nghĩa nhãn

- `at_risk = 1`: `G3 < 10` — nhóm cần quan tâm theo quy ước của nghiên cứu.
- `at_risk = 0`: `G3 >= 10` — nhóm còn lại.

Ngưỡng 10/20 là quy ước được dùng trong project, không phải chuẩn phổ quát cho tất cả hệ thống giáo dục. `G1`, `G2`, `G3` không được sử dụng làm đặc trưng đầu vào; `G3` chỉ dùng để tạo nhãn.

## Quy trình nghiên cứu

1. **Data acquisition & provenance:** lấy trang dữ liệu công khai UCI, tìm tệp ZIP, trích CSV và lưu siêu dữ liệu nguồn.
2. **Dataset profiling:** kiểm tra kích thước dữ liệu, loại cột, dữ liệu thiếu, bản ghi trùng và phân bố nhãn.
3. **Exploratory Data Analysis (EDA):** mô tả phân bố nhãn và mối liên hệ quan sát được giữa đặc trưng và nhãn.
4. **Feature analysis:** dùng Pearson/point-biserial cho biến số hoặc thứ bậc, Cramér's V cho biến phân loại; chỉ tính trên tập train.
5. **Feature-set comparison:** so sánh nhóm 6 đặc trưng học tập cốt lõi với nhóm 30 đặc trưng không phải điểm.
6. **Cross-validation:** giữ riêng 20% làm test; trên 80% còn lại áp dụng `RepeatedStratifiedKFold` với 5 folds và 2 lần lặp.
7. **Model selection & thresholding:** lựa chọn mô hình theo Average Precision trung bình, dùng F1 lớp 1 và độ gọn của tập đặc trưng làm tiêu chí phụ; chọn ngưỡng từ dự đoán OOF của train.
8. **Final evaluation:** đánh giá trên test sau khi đã chốt các lựa chọn, phân tích chỉ số từng lớp, ma trận nhầm lẫn, False Positives (FP) và False Negatives (FN).
9. **Interpretation & artifacts:** lưu báo cáo, bảng kết quả, biểu đồ và pipeline đã huấn luyện.

## Các mô hình được so sánh

| Thuật toán | Vai trò | Vấn đề cần phân tích |
|---|---|---|
| **Dummy Classifier** | Baseline dựa trên lớp phổ biến nhất | Mức hiệu quả tối thiểu khi không học quan hệ giữa đặc trưng và nhãn |
| **Logistic Regression** | Mô hình phân loại tuyến tính, đầu ra có thể diễn giải như xác suất | Khả năng biểu diễn quan hệ phi tuyến còn hạn chế |
| **Decision Tree** | Học quy tắc phân chia dữ liệu và quan hệ phi tuyến | Nguy cơ overfitting và độ nhạy với mẫu dữ liệu |
| **Random Forest** | Tổng hợp nhiều cây để giảm phương sai của một cây đơn | Độ phức tạp giải thích cao hơn và không bảo đảm thắng trên dữ liệu nhỏ |

Các mô hình được đánh giá theo cùng thiết kế phân chia dữ liệu, bộ tiền xử lý và tập đặc trưng tương ứng. Việc chọn mô hình dựa trên kết quả thực nghiệm thay vì giả định một thuật toán luôn tốt nhất.

## Đánh giá và diễn giải kết quả

Dự án không sử dụng Accuracy làm tiêu chí duy nhất. Các thước đo bao gồm:

- **Precision lớp 1:** trong các học sinh được cảnh báo, bao nhiêu em thực sự thuộc nhóm `G3 < 10`.
- **Recall lớp 1:** trong các học sinh thuộc nhóm `G3 < 10`, mô hình phát hiện được bao nhiêu em.
- **F1 lớp 1:** cân bằng giữa Precision và Recall.
- **Balanced Accuracy:** trung bình Recall của hai lớp.
- **Average Precision:** đánh giá khả năng xếp hạng lớp 1 trên nhiều ngưỡng.
- **Confusion matrix:** cho biết số dự đoán đúng/sai theo từng lớp, bao gồm FP và FN.

Insight trọng tâm là kiểm tra xem hiệu quả ở lớp 0 và lớp 1 có chênh lệch hay không. Nếu các chỉ số lớp 0 cao nhưng Recall lớp 1 thấp, mô hình có thể bỏ sót nhiều học sinh cần hỗ trợ. Nếu Recall lớp 1 tăng nhưng Precision giảm mạnh, số cảnh báo nhầm có thể tăng. Kết luận cuối cùng phải được sinh từ lần chạy trên CSV UCI chính thức, không chép sẵn kết luận khi chưa xem số liệu.

## Cấu trúc repository

```text
student_learning_ml_github_repo_v2/
├── data/
│   ├── student-mat-demo.csv       # Dữ liệu minh họa, không dùng làm benchmark
│   ├── student-mat.csv            # CSV chính thức, được tạo khi crawl thành công
│   ├── README_DU_LIEU.md
│   └── raw/                       # HTML, ZIP, dữ liệu nguồn và metadata
├── notebooks/
│   └── 01_Du_an_ML_hoc_tap_hoan_chinh.ipynb
├── src/
│   ├── crawl_data.py              # Thu thập và lưu dữ liệu nguồn
│   ├── data_loader.py             # Nạp dữ liệu và xác định nguồn
│   ├── eda.py                     # Hồ sơ dữ liệu và EDA
│   ├── preprocessing.py           # Tiền xử lý
│   ├── train.py                   # CV, chọn mô hình, đánh giá và báo cáo
│   ├── predict.py                 # Suy luận với mô hình đã lưu
│   └── config.py                  # Tham số và nhóm đặc trưng
├── docs/                          # Phương pháp, thuật ngữ và tài liệu vấn đáp
├── tests/                         # Kiểm thử
├── outputs/                       # Bảng kết quả, JSON và biểu đồ được sinh
├── reports/                       # Báo cáo EDA và báo cáo mô hình
├── models/                        # Pipeline đã huấn luyện
├── requirements.txt
├── run_project.py
├── CITATION.cff
└── LICENSE
```

## Cài đặt và chạy

### Google Colab

1. Tải repository dưới dạng ZIP và tải lên Colab.
2. Giải nén repository; mở `notebooks/01_Du_an_ML_hoc_tap_hoan_chinh.ipynb`.
3. Chạy các ô theo thứ tự. Khi cần tải bộ dữ liệu chính thức, bật kết nối Internet.

Có thể chạy phiên bản dòng lệnh từ một ô Colab:

```python
%cd /content/student_learning_ml_github_repo_v2
%pip install -q -r requirements.txt
!python run_project.py --crawl
```

Để chỉ kiểm tra khả năng thực thi ngoại tuyến với dữ liệu minh họa:

```python
!python run_project.py --demo
```

### Máy tính cá nhân

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_project.py --crawl
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py --crawl
```

## Các sản phẩm đầu ra

| Tệp | Nội dung |
|---|---|
| `outputs/so_sanh_kiem_tra_cheo.csv` | Trung bình và độ lệch chuẩn các chỉ số qua folds |
| `outputs/so_sanh_mo_hinh_va_nhom_dac_trung.csv` | So sánh thuật toán và tập đặc trưng |
| `outputs/lien_he_bien_so_voi_nhan_train.csv` | Liên hệ biến số/thứ bậc với nhãn trên train |
| `outputs/lien_he_bien_phan_loai_voi_nhan_train.csv` | Cramér's V cho biến phân loại |
| `outputs/chi_so_theo_tung_lop.csv` | Precision, Recall, F1 theo từng lớp |
| `outputs/chi_so_test.json` | Chỉ số test và thông tin lựa chọn mô hình/ngưỡng |
| `outputs/so_sanh_nguong_train_oof.csv` | So sánh ngưỡng dựa trên OOF của train |
| `outputs/tam_quan_trong_hoan_vi_test_hau_kiem.csv` | Permutation importance hậu kiểm trên test |
| `reports/EDA_report.md` | Hồ sơ dữ liệu và EDA |
| `reports/model_report.md` | Tổng hợp CV, test, diễn giải và giới hạn |
| `models/mo_hinh_canh_bao_hoc_tap.joblib` | Pipeline, tập đặc trưng và ngưỡng đã chốt |

## Giới hạn và sử dụng có trách nhiệm

- Dataset gồm 395 học sinh từ hai trường ở Bồ Đào Nha; kết quả không mặc nhiên đại diện cho bối cảnh Việt Nam.
- Nhãn `G3 < 10` là một quy ước của project. Khác biệt về chương trình học, chính sách chấm điểm và bối cảnh trường học có thể làm giảm khả năng chuyển giao.
- Mối liên hệ thống kê không chứng minh quan hệ nhân quả; permutation importance cũng không chứng minh một biến gây ra kết quả.
- Kích thước mẫu nhỏ khiến ước lượng hiệu quả có thể biến động. Báo cáo kết quả trung bình và độ lệch chuẩn qua CV để thể hiện sự bất định.
- Đây là dự án học tập, không phải công cụ chẩn đoán hoặc quyết định về học sinh. Không dùng kết quả mô hình làm căn cứ duy nhất để dán nhãn, kỷ luật hay phân bổ cơ hội giáo dục.
- Kết quả từ dữ liệu demo chỉ dùng kiểm thử phần mềm, không được trình bày như benchmark chính thức.

## Nguồn dữ liệu và trích dẫn

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). License: CC BY 4.0. Chi tiết tại [`data/README_DU_LIEU.md`](data/README_DU_LIEU.md) và [`CITATION.cff`](CITATION.cff).
