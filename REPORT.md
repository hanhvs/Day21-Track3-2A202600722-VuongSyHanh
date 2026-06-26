# Lab 21 — Báo cáo Fine-tuning LoRA / QLoRA

**Học viên:** Vuong Sy Hanh — `2A202600722` — Track 3

## 1. Thiết lập thí nghiệm

Toàn bộ thí nghiệm được chạy trên Google Colab với GPU Tesla T4 16 GB. Các tham số cố định để đảm bảo so sánh công bằng giữa các rank:

- **Base model:** `unsloth/Qwen2.5-3B-bnb-4bit`.
- **Quantization:** QLoRA 4-bit (NF4) thông qua Unsloth.
- **LoRA target modules:** `q_proj`, `v_proj` (cấu hình lõi theo rubric).
- **Huấn luyện:** 3 epoch, learning rate `2e-4`, cosine scheduler, warmup ratio `0.10`, `optim="adamw_8bit"`, bật gradient checkpointing, `seed=42`.
- **Batching:** `per_device_train_batch_size=2`, `gradient_accumulation_steps=4` → effective batch size `8`.
- **Dataset nguồn:** `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, được chuẩn hóa về định dạng Alpaca.
- **`max_seq_length` đã chọn:** `1024` (làm tròn lên từ p95, vẫn an toàn với VRAM của T4).

Lưu ý cài đặt: trên Colab cần dùng `bash scripts/install_colab.sh` thay cho `pip install -r requirements.txt`, vì Unsloth + TRL phụ thuộc vào thứ tự cài đặt theo kiểu notebook.

## 2. Chuẩn bị dữ liệu

Script [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) thực hiện toàn bộ pipeline tiền xử lý:

- nạp dataset nguồn và chuẩn hóa về ba trường Alpaca `instruction`, `input`, `output`;
- lọc bỏ các dòng rỗng, trùng lặp hoặc output quá ngắn;
- dựng cột `text` theo prompt template huấn luyện;
- chia train/eval theo tỉ lệ `90/10` với seed cố định;
- tính thống kê độ dài token (`p90`, `p95`, …) và đề xuất `max_seq_length`;
- xuất `train.jsonl`, `eval.jsonl`, `dataset_full.jsonl` và `dataset_stats.json`.

Prompt template (khi có `input`):

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Khi `input` rỗng thì khối `### Input:` bị bỏ đi.

**Kết quả thực tế** (từ `data/processed/dataset_stats.json`):

| Chỉ số | Giá trị |
|---|---:|
| Mẫu lấy từ nguồn | 300 |
| Số dòng bị loại khi làm sạch | 16 |
| Còn lại sau chuẩn hóa | 284 |
| Train / Eval | 255 / 29 |

Thống kê độ dài token: min `28`, mean `244.04`, median `202`, p90 `528`, **p95 `563`**, p99 `708`, max `738`. Vì p95 là 563 nên `max_seq_length` được làm tròn lên `1024` để phủ gần như toàn bộ ví dụ mà vẫn an toàn cho T4.

## 3. Kết quả thí nghiệm theo Rank

Script [`scripts/train_lora_rank.py`](scripts/train_lora_rank.py) huấn luyện lần lượt từng rank, giữ nguyên mọi cấu hình khác; chỉ `rank` và `lora_alpha = 2 * rank` thay đổi:

| Cấu hình | Rank | Alpha |
|---|---:|---:|
| rank thấp | 8 | 16 |
| baseline | 16 | 32 |
| rank cao | 64 | 128 |

Tổng hợp từ `results/rank_experiment_summary.csv`:

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---|---:|---:|---:|---:|---:|
| 8 | 1,843,200 | 5.95 min | 10.27 GB | 1.5160 | 4.5542 |
| 16 | 3,686,400 | 6.00 min | 10.29 GB | 1.4691 | 4.3455 |
| 64 | 14,745,600 | 5.98 min | 10.43 GB | **1.4629** | **4.3185** |

**Nhận xét:**

- `r=64` cho eval loss và perplexity thấp nhất, nhưng khoảng cách so với `r=16` rất hẹp (perplexity `4.3185` so với `4.3455`).
- `r=8` nhẹ nhất về tham số nhưng cũng yếu nhất về perplexity.
- Thời gian huấn luyện gần như giống nhau ở cả ba lần (~6 phút) do dataset nhỏ; điểm khác biệt thực sự nằm ở số tham số huấn luyện và mức VRAM nhỉnh hơn một chút khi rank lớn.
- Từ `r=16` lên `r=64`, số tham số tăng gấp 4 lần nhưng perplexity chỉ cải thiện rất nhỏ.

*Lưu ý trung thực:* dòng đánh giá base model không được đưa vào bảng vì log Colab thu được không chứa kết quả eval của base model từ `evaluate_adapters.py`; tôi không bịa con số còn thiếu này.

## 4. Phân tích đường cong Loss

Script [`scripts/plot_results.py`](scripts/plot_results.py) gộp log của từng rank thành `results/loss_history.csv` và `results/loss_curve.png`.

Dựa trên log thực tế trên Colab (96 step mỗi lần chạy):

- Cả ba lần chạy đều cho training loss giảm mượt, không có dấu hiệu mất ổn định hay phân kỳ.
- `r=16` giảm từ vùng `1.65–1.73` ở đầu xuống `1.30–1.41` về cuối.
- `r=8` có hình dạng tương tự nhưng hội tụ về chỉ số đánh giá kém hơn một chút.
- `r=64` đạt training loss cuối thấp nhất và perplexity tốt nhất, nhưng mức vượt trội so với `r=16` chỉ ở mức khiêm tốn.

File ảnh `loss_curve.png` được sinh trên Colab; phần phân tích ở đây dựa trên log đã ghi lại thay vì xem trực tiếp ảnh trong workspace cục bộ.

## 5. So sánh định tính

Script [`scripts/evaluate_adapters.py`](scripts/evaluate_adapters.py) đánh giá base model trên cùng tập eval, nạp lại lần lượt adapter `r=8`, `r=16`, `r=64`, sinh output cho 10 prompt và lưu `results/qualitative_comparison.csv`.

Các prompt tiêu biểu dùng để so sánh:

1. `Giải thích QLoRA là gì theo cách dễ hiểu cho sinh viên năm nhất.`
2. `Viết hướng dẫn từng bước để chuẩn bị dataset Alpaca format cho fine-tuning.`
3. `So sánh ngắn gọn giữa prompt engineering, RAG và fine-tuning bằng tiếng Việt.`
4. `Trả lời theo định dạng gạch đầu dòng: khi nào nên dùng LoRA rank thấp?`
5. `Giải thích vì sao phải giữ experiment fair khi so sánh r=8, r=16, r=64.`

Nguyên tắc đánh giá: không chỉ chọn các ví dụ có lợi, giữ lại ít nhất một ví dụ mà fine-tuned model tương đương hoặc không rõ ràng tốt hơn base.

*Trạng thái:* log Colab xác nhận `results/qualitative_comparison.csv` đã được sinh thành công, nhưng phần văn bản output cụ thể không có trong log đang nằm trong workspace này, nên tôi không tóm tắt nội dung các câu trả lời khi chưa có dữ liệu gốc.

## 6. Kết luận: đánh đổi theo Rank

Về mặt định lượng, `r=64` thắng với eval loss `1.4629` và perplexity `4.3185`. Tuy nhiên lợi thế này rất mỏng: `r=16` đạt perplexity `4.3455` — gần như ngang ngửa — trong khi chỉ dùng 1/4 số tham số huấn luyện của `r=64`. Peak VRAM cũng chỉ tăng nhẹ từ `10.29 GB` lên `10.43 GB`, và thời gian huấn luyện gần như không đổi vì dataset nhỏ. Nói cách khác, đánh đổi chính ở đây **không phải runtime** mà là hiệu quả tham số và độ gọn của adapter.

- Nếu VRAM eo hẹp hoặc cần adapter thật nhỏ → `r=8` an toàn nhất, nhưng perplexity yếu nhất.
- Nếu chỉ ưu tiên chất lượng tuyệt đối → `r=64` tốt nhất theo số liệu.
- **Lựa chọn cân bằng nhất cho bài lab này là `r=16`:** mạnh hơn hẳn `r=8`, gần chạm `r=64`, và tránh được việc tăng gấp 4 tham số chỉ để đổi lấy cải thiện perplexity rất nhỏ.

## 7. Những điều rút ra

- Rank của LoRA quyết định "dung lượng" của adapter: rank cao tăng khả năng biểu diễn nhưng kéo theo nhiều tham số huấn luyện và chi phí bộ nhớ hơn.
- QLoRA giúp fine-tune được các model instruction lớn trên GPU hạn chế nhờ lượng tử hóa base model đã đóng băng xuống 4-bit.
- Perplexity hữu ích để so sánh nhưng chưa đủ; cần kiểm tra định tính để xác nhận khả năng tuân theo instruction và định dạng.

## 8. Hạn chế & hướng phát triển

- Môi trường cục bộ không có CUDA, nên mọi artifact huấn luyện đều phải sinh trên Colab rồi diễn giải lại từ log đã xuất.
- Cấu hình target modules hiện theo bản lõi (`q_proj`, `v_proj`) thay vì cấu hình mở rộng toàn bộ layer.
- Dòng eval của base model và toàn bộ output định tính được sinh trên Colab nhưng chưa được sao chép về workspace này, nên không trình bày lại khi thiếu nguồn.
- Hướng tiếp theo sau khi hoàn thành phần lõi:
  - đẩy adapter tốt nhất lên Hugging Face Hub;
  - thử LoRA trên toàn bộ target modules;
  - thêm W&B để theo dõi lịch sử thí nghiệm gọn gàng hơn.
