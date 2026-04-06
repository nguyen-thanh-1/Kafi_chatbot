# 🔑 Cách Set Gemini API Key

## ⚠️ Lỗi hiện tại
```
❌ Gemini API key không tìm thấy. 
Vui lòng set GEMINI_API_KEY environment variable hoặc truyền api_key parameter.
```

## ✅ Giải pháp

### Option 1: Set Environment Variable (Khuyến nghị)

**Windows PowerShell:**
```powershell
# Set API key (thay your-api-key bằng key thật)
$env:GEMINI_API_KEY="AIzaSy..."

# Kiểm tra đã set chưa
echo $env:GEMINI_API_KEY

# Chạy lại
uv run .\pageindex_multiformat.py
```

**Lưu ý:** API key chỉ tồn tại trong session hiện tại. Nếu đóng PowerShell phải set lại.

**Để set vĩnh viễn (Windows):**
```powershell
# Set system environment variable
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIzaSy...', 'User')

# Sau đó restart PowerShell
```

---

### Option 2: Dùng file wrapper (Dễ hơn)

**Bước 1:** Mở file `pageindex_gemini.py`

**Bước 2:** Sửa dòng này:
```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # ← Thay bằng API key của bạn
```

Thành:
```python
GEMINI_API_KEY = "AIzaSy..."  # ← API key thật của bạn
```

**Bước 3:** Chạy:
```bash
uv run .\pageindex_gemini.py
```

---

## 🔑 Lấy Gemini API Key

1. Truy cập: https://aistudio.google.com/apikey
2. Đăng nhập Google account
3. Click **"Create API Key"**
4. Copy API key (dạng: `AIzaSy...`)
5. Dùng một trong 2 option trên

---

## 📝 So sánh 2 cách

| Cách | Ưu điểm | Nhược điểm |
|------|---------|------------|
| **Env Var** | An toàn hơn, không lưu trong code | Phải set mỗi lần mở PowerShell |
| **File wrapper** | Tiện lợi, chỉ set 1 lần | API key lưu trong file (ít an toàn) |

---

## ✅ Khuyến nghị

**Cho development:** Dùng **Option 2** (file wrapper) - Tiện hơn

**Cho production:** Dùng **Option 1** (env var) - An toàn hơn

---

## 🚀 Quick Start

```powershell
# Cách nhanh nhất:
# 1. Lấy API key từ https://aistudio.google.com/apikey
# 2. Set env var
$env:GEMINI_API_KEY="AIzaSy..."

# 3. Chạy
uv run .\pageindex_multiformat.py
```

Hoặc:

```powershell
# Dùng wrapper (dễ hơn)
# 1. Sửa GEMINI_API_KEY trong pageindex_gemini.py
# 2. Chạy
uv run .\pageindex_gemini.py
```

---

**Bây giờ hãy chọn 1 trong 2 cách và thử lại!** 🎯
