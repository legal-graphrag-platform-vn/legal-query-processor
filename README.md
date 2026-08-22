# Query Processing Fine-tuning

Thực hiện bởi sinh viên Học viện Công nghệ Bưu chính Viễn thông Cơ sở Hồ Chí Minh (PTIT HCM):

Đặng Xuân Lâm - N22DCCN047
Nguyễn Anh Kha - N22DCCN078
Phan Nhật Minh - N22DCC054
Giáo viên hướng dẫn: Nguyễn Thị Bích Nguyên

Mô-đun Query Processing chuẩn hóa câu hỏi trước retrieval, bao gồm resolve ngữ
cảnh hội thoại, viết lại câu hỏi độc lập, phân rã query, gán intent và biểu diễn
quan hệ giữa các subquery.

## Output contract

```text
conversation_history + current_query
-> status
-> standalone_query
-> plan_type
-> subqueries[{id, query, intent, depends_on}]
-> clarification_question
```

Giá trị hợp lệ:

- `status`: `ready`, `needs_clarification`.
- `plan_type`: `single`, `parallel`, `comparison`, `multi_hop`.
- `intent`: `factual`, `definition`, `validity`, `hierarchy`.

`depends_on` chỉ được tham chiếu đến subquery đứng trước. Khi không đủ ngữ cảnh,
`status` là `needs_clarification`, `standalone_query` và `plan_type` là `null`,
`subqueries` rỗng và `clarification_question` phải có giá trị.

## Dataset

Dataset gồm khoảng 7.000 mẫu, chia theo tỷ lệ 80/10/10. Các biến thể có cùng
query gốc hoặc cùng logic phân rã được giữ trong một split thông qua
`split_group_id`.

Hai biểu diễn được lưu song song:

- `final/`: dữ liệu chuẩn có input, output và metadata truy vết.
- `sft/`: dữ liệu hội thoại `messages` dùng cho supervised fine-tuning.

Lịch sử hội thoại được chuẩn hóa thành chuỗi bắt đầu bằng `user`, luân phiên role
và kết thúc bằng `assistant`; query hiện tại là lượt `user` cuối. Target là JSON
của assistant. `sample_id` chỉ dùng để truy vết và không tham gia tính loss.

Nguồn dữ liệu gồm mẫu non-multi-hop sinh bằng DeepSeek V4 Flash và mẫu
multi-hop sinh từ dependency blueprint kiểm soát. Các mẫu bị rule audit gắn cờ,
history không hợp lệ hoặc nhóm có label xung đột đã bị loại khỏi tập cuối. Dataset
hiện có trạng thái `auto_audited_not_human_gold`.

## Fine-tuning

- Base model: `Qwen/Qwen3-4B-Instruct-2507`.
- Phương pháp: supervised fine-tuning bằng QLoRA 4-bit.
- Loss: causal language modeling cross-entropy trên phần assistant.
- Số epoch: 2.
- Tổng optimizer step: 710.

System prompt vẫn được giữ khi inference để cung cấp contract đầu ra. Fine-tune
không đưa tri thức pháp luật vào model; mục tiêu là học hành vi xử lý query và
sinh JSON ổn định.

## Artifact

```text
final/    Dataset chuẩn theo split
sft/      Dataset SFT theo split
tests/    Kiểm tra contract dữ liệu
```

Raw synthetic runs, pilot/smoke artifacts, dữ liệu trung gian và prediction
smoke-test không được giữ trong workspace sau khi dataset cuối được chốt.
