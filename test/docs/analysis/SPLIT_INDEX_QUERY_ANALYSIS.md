# Tách Index và Query - Phân tích Tối ưu

## 🎯 Ý tưởng: Tách 2 quá trình

### Hiện tại (rag_pro_v2.py):
```
1. Load Embedding (CPU)
2. Load Reranker (CPU)  
3. Load LLM (GPU) ← 12GB VRAM
4. Index (chỉ dùng Embedding CPU)
5. Query (dùng cả 3 models)
```

**Vấn đề:**
- ❌ Embedding chạy CPU (chậm 4x)
- ❌ LLM chiếm 12GB VRAM nhưng không dùng khi index
- ❌ Không thể chạy Embedding trên GPU (vì LLM đã chiếm hết VRAM)

---

## ✅ Giải pháp: Tách thành 2 scripts

### Script 1: `rag_index.py` (CHỈ INDEX)
```
1. Load Embedding (GPU) ← Dùng toàn bộ VRAM!
2. Index (Embedding GPU - NHANH 4x)
3. Save cache + FAISS index
4. Unload Embedding
```

**VRAM:**
- Embedding (GPU): ~3GB
- LLM: 0GB (chưa load)
- **Total: 3GB** ✅

**Tốc độ:**
- Embedding GPU: **4x nhanh hơn CPU**
- Index 800-page PDF: **~2-3 phút** (thay vì 8-10 phút)

---

### Script 2: `rag_query.py` (CHỈ QUERY)
```
1. Load index từ disk
2. Load Embedding (CPU) ← Nhẹ, chỉ embed query
3. Load Reranker (CPU)
4. Load LLM (GPU) ← Dùng toàn bộ VRAM!
5. Query
```

**VRAM:**
- Embedding (CPU): 0GB
- Reranker (CPU): 0GB
- LLM (GPU): ~12GB
- **Total: 12GB** ✅

**Tốc độ:**
- Query embedding: CPU OK (chỉ 1 query, không ảnh hưởng)
- LLM inference: GPU (nhanh)

---

## 📊 So sánh Performance

### Hiện tại (V2 - All in one):

| Operation | Device | Time | VRAM |
|-----------|--------|------|------|
| **Index** | Embedding CPU | 8-10 min | 12GB (LLM idle) |
| **Query** | All models | 7.5s | 12GB |

**Vấn đề:**
- ❌ Lãng phí 12GB VRAM khi index
- ❌ Embedding CPU chậm 4x

---

### Tách riêng (Optimized):

#### **Script 1: Index Only**
| Operation | Device | Time | VRAM |
|-----------|--------|------|------|
| **Index** | Embedding GPU | **2-3 min** | 3GB |

**Cải thiện:**
- ✅ Nhanh hơn **3-4x** (8-10 min → 2-3 min)
- ✅ Tiết kiệm 9GB VRAM
- ✅ Embedding chạy GPU (tối ưu)

#### **Script 2: Query Only**
| Operation | Device | Time | VRAM |
|-----------|--------|------|------|
| **Query** | Embedding CPU + LLM GPU | 7.5s | 12GB |

**Không đổi:**
- Query vẫn nhanh như cũ
- LLM vẫn chạy GPU

---

## 🚀 Lợi ích cụ thể

### 1. **Index nhanh hơn 3-4x**
```
Before: 8-10 phút (Embedding CPU)
After:  2-3 phút (Embedding GPU)
Speedup: 3-4x
```

### 2. **Tiết kiệm VRAM khi index**
```
Before: 12GB (LLM idle)
After:  3GB (chỉ Embedding)
Saved:  9GB VRAM
```

### 3. **Linh hoạt hơn**
- Index nhiều lần không cần load LLM
- Query nhiều lần không cần re-index
- Có thể chạy index trên máy khác (không cần LLM)

### 4. **Dễ maintain**
- Index script đơn giản hơn
- Query script tập trung vào inference
- Dễ debug từng phần

---

## 📝 Workflow mới

### Lần đầu tiên:
```bash
# Bước 1: Index (Embedding GPU)
uv run rag_index.py --force
# → 2-3 phút, tạo FAISS index + cache

# Bước 2: Query (LLM GPU)
uv run rag_query.py
# → Load index từ disk, sẵn sàng query
```

### Thêm tài liệu mới:
```bash
# Chỉ cần re-index
uv run rag_index.py
# → Chỉ index file mới (cache hit cao)
# → Không cần load LLM
```

### Query nhiều lần:
```bash
# Chỉ cần query
uv run rag_query.py
# → Load index 1 lần, query nhiều lần
# → Không cần re-index
```

---

## 🎯 Kết luận

### ✅ Nên tách vì:
1. **Index nhanh hơn 3-4x** (Embedding GPU)
2. **Tiết kiệm 9GB VRAM** khi index
3. **Linh hoạt hơn** (index/query riêng)
4. **Dễ maintain** (code đơn giản hơn)

### ⚠️ Trade-off:
- Phải chạy 2 scripts riêng
- Không thể index + query trong 1 lần chạy

### 💡 Khuyến nghị:
**TÁCH RA!** Lợi ích lớn hơn nhiều so với bất tiện.

---

## 📊 Performance Summary

| Metric | V2 (All-in-one) | Tách riêng | Improvement |
|--------|-----------------|------------|-------------|
| **Index time** | 8-10 min | **2-3 min** | **3-4x faster** |
| **Index VRAM** | 12GB | **3GB** | **75% less** |
| **Query time** | 7.5s | 7.5s | Same |
| **Query VRAM** | 12GB | 12GB | Same |
| **Flexibility** | Low | **High** | Better |

**Tổng kết: Tách ra TỐI ƯU HƠN RẤT NHIỀU!** 🚀
