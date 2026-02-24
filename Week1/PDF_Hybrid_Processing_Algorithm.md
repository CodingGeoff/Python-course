# 混合型PDF处理算法详解（文本+扫描混合）

本文档详细说明如何处理包含**可提取文本页**和**扫描/图片页**的混合型PDF文件。

**适用场景**：
- 学术论文中部分页面是扫描的图片（如图表、老文献）
- PDF中混合了可复制的文字页和图片页
- 扫描版PDF中部分页面文字无法直接提取

**历史版本位置**：`history/07_pdf_translation.ipynb`, `history/08_attention_is_all_you_need.ipynb`, `history/10_terminology_extraction.ipynb`

---

## 目录

1. [架构差异对比](#架构差异对比)
2. [核心算法流程](#核心算法流程)
3. [扫描件检测机制](#扫描件检测机制)
4. [PDF转图像与OCR](#pdf转图像与ocr)
5. [完整代码实现](#完整代码实现)
6. [性能与精度优化](#性能与精度优化)
7. [错误处理策略](#错误处理策略)
8. [与纯文本PDF处理的对比](#与纯文本pdf处理的对比)

---

## 架构差异对比

### 纯文本PDF处理（02/03 notebook当前实现）

```
PDF文件
    │
    ▼
┌─────────────────┐
│ fitz.open()    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ page.get_text() │ ← 直接提取文本
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 文本清洗        │
│ (连字符修复等)  │
└────────┬────────┘
         │
         ▼
    结构化文本输出
```

### 混合型PDF处理（history版本）

```
PDF文件
    │
    ▼
┌─────────────────┐
│ fitz.open()    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ page.get_text() │ ← 尝试直接提取
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────────────────┐
│文本>50?│ │ 文本<50字符             │
│ 是     │ │ 判定为扫描件/图片        │
└───┬────┘ └───────────┬─────────────┘
    │                  │
    │                  ▼
    │        ┌─────────────────┐
    │        │ get_pixmap()    │ ← PDF渲染为图片
    │        │ (2倍分辨率)     │
    │        └────────┬────────┘
    │                 │
    │                 ▼
    │        ┌─────────────────┐
    │        │ PIL.Image       │ ← 格式转换
    │        └────────┬────────┘
    │                 │
    │                 ▼
    │        ┌─────────────────┐
    │        │ pytesseract     │ ← OCR识别
    │        │ image_to_string │
    │        └────────┬────────┘
    │                 │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ 合并所有页文本   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ 文本清洗        │
    └────────┬────────┘
             │
             ▼
    结构化文本输出
```

---

## 核心算法流程

### 智能PDF解析器 (`smart_pdf_parser` / `smart_extract_text_from_pdf`)

```python
import fitz           # PyMuPDF
import pytesseract    # OCR引擎
from PIL import Image # 图像处理
import io             # 字节流转换

def smart_pdf_parser(pdf_path):
    """
    能够同时处理文本版和扫描版 PDF 的智能解析器
    
    核心逻辑:
    1. 逐页尝试直接文本提取
    2. 检测文本量判断是否为扫描页
    3. 对扫描页启用OCR
    4. 合并所有页面文本
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for i, page in enumerate(doc):
        # Step 1: 尝试直接提取文本
        text = page.get_text()
        
        # Step 2: 扫描件检测
        if len(text.strip()) < 50:  # 阈值: 50字符
            print(f"  ⚠️ 第 {i+1} 页疑似扫描/图表，启用 OCR...")
            
            # Step 3: PDF页面 → 高分辨率图片
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Step 4: PyMuPDF图像 → PIL图像
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Step 5: OCR识别
            text = pytesseract.image_to_string(img, lang='eng')
        
        full_text.append(text)
    
    return "\n".join(full_text)
```

---

## 扫描件检测机制

### 1. 基于字符数的启发式检测

```python
# 检测逻辑: 如果一页提取的字符数少于阈值，判定为扫描件/图片
THRESHOLD = 50  # 字符数阈值

text = page.get_text()
if len(text.strip()) < THRESHOLD:
    # 触发OCR流程
    enable_ocr = True
```

**阈值选择依据**：
| 页面类型 | 典型字符数 | 处理方式 |
|----------|-----------|----------|
| 纯文字页 | > 500字符 | 直接提取 |
| 图表页 | 0-20字符 | OCR识别 |
| 扫描页 | 0-30字符 | OCR识别 |
| 混合页 | 30-100字符 | 视情况而定 |

### 2. 更精确的检测策略（可扩展）

```python
def is_scanned_page(page, text):
    """
    多维度扫描页检测
    
    检测维度:
    1. 字符数
    2. 可识别单词比例
    3. 图片元素检测
    """
    # 维度1: 基础字符数
    if len(text.strip()) < 50:
        return True
    
    # 维度2: 有效单词比例检测
    words = text.split()
    meaningful_words = [w for w in words if len(w) > 2]
    if len(words) > 0 and len(meaningful_words) / len(words) < 0.3:
        return True
    
    # 维度3: 检测页面图片元素数量
    image_list = page.get_images()
    if len(image_list) > 0 and len(text.strip()) < 100:
        return True
    
    return False
```

---

## PDF转图像与OCR

### 1. 高分辨率渲染 (`get_pixmap`)

```python
# 基础渲染 (72 DPI)
pix = page.get_pixmap()

# 高分辨率渲染 (144 DPI, 2倍缩放)
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

# 超高分辨率渲染 (288 DPI, 4倍缩放)
pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
```

**矩阵参数说明**：
```python
fitz.Matrix(zoom_x, zoom_y)
# zoom=2: 2倍清晰度，提高OCR识别率
# zoom=4: 4倍清晰度，适合低质量扫描件，但内存消耗大
```

### 2. 图像格式转换

```python
# PyMuPDF Pixmap → PNG字节流
img_data = pix.tobytes("png")

# PNG字节流 → PIL Image
img = Image.open(io.BytesIO(img_data))

# 可选: 图像预处理（提高OCR精度）
# 转换为灰度图
img_gray = img.convert('L')

# 二值化处理
from PIL import ImageOps
img_binary = ImageOps.invert(img_gray.point(lambda x: 0 if x < 128 else 255))
```

### 3. OCR识别配置

```python
# 基础英文识别
text = pytesseract.image_to_string(img, lang='eng')

# 中文识别
text = pytesseract.image_to_string(img, lang='chi_sim')

# 中英文混合识别
text = pytesseract.image_to_string(img, lang='eng+chi_sim')

# 带配置选项（提高精度）
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(
    img, 
    lang='eng',
    config=custom_config
)
```

**Tesseract配置参数**：
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--oem` | OCR引擎模式 | 3 (默认+LSTM) |
| `--psm` | 页面分割模式 | 6 (统一文本块) |
| `--psm 3` | 自动分页 | 适合多栏文档 |
| `--psm 6` | 统一文本块 | 适合普通文本 |

---

## 完整代码实现

### 实现1: 基础混合型解析器

```python
import fitz
import pytesseract
from PIL import Image
import io
import os

# Windows Tesseract路径配置
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class HybridPDFProcessor:
    """混合型PDF处理器（支持文本页+扫描页）"""
    
    SCAN_THRESHOLD = 50  # 扫描检测阈值（字符数）
    OCR_DPI = 2          # OCR渲染倍数
    
    @classmethod
    def parse_pdf(cls, pdf_path):
        """
        解析混合型PDF
        
        Returns:
            tuple: (full_text, pages_list, ocr_pages_info)
                - full_text: 合并的全文
                - pages_list: 每页文本列表
                - ocr_pages_info: 哪些页使用了OCR
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF不存在: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pages_content = []
        ocr_pages = []
        
        print(f"🚀 开始解析 PDF (共 {len(doc)} 页)...")
        
        for i, page in enumerate(doc):
            # 尝试直接提取
            text = page.get_text()
            
            # 检测是否为扫描页
            if len(text.strip()) < cls.SCAN_THRESHOLD:
                print(f"  ⚠️ 第 {i+1} 页疑似扫描件，启用 OCR...")
                
                try:
                    # PDF → 图片
                    pix = page.get_pixmap(matrix=fitz.Matrix(cls.OCR_DPI, cls.OCR_DPI))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    
                    # OCR识别
                    text = pytesseract.image_to_string(img, lang='eng')
                    ocr_pages.append(i + 1)
                    
                except Exception as e:
                    print(f"  ❌ 第 {i+1} 页 OCR 失败: {e}")
                    text = ""
            
            # 基础清洗
            text = text.replace('\n', ' ').strip()
            pages_content.append(text)
        
        full_text = "\n".join(pages_content)
        
        print(f"✅ 解析完成，共 {len(pages_content)} 页")
        if ocr_pages:
            print(f"   OCR处理页: {ocr_pages}")
        
        return full_text, pages_content, ocr_pages
```

### 实现2: 带缓存的高级解析器

```python
import hashlib
import pickle

class CachedHybridProcessor(HybridPDFProcessor):
    """带缓存的混合型PDF处理器"""
    
    CACHE_DIR = "./pdf_cache"
    
    @classmethod
    def parse_with_cache(cls, pdf_path):
        """带缓存的解析（避免重复OCR）"""
        # 生成文件指纹
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_file = os.path.join(cls.CACHE_DIR, f"{file_hash}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            print("📂 使用缓存结果...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 执行解析
        result = cls.parse_pdf(pdf_path)
        
        # 保存缓存
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

---

## 性能与精度优化

### 1. 分辨率与速度的平衡

```python
# 快速模式（适合高质量扫描件）
FAST_MODE = fitz.Matrix(1.5, 1.5)  # 108 DPI

# 标准模式（推荐）
STANDARD_MODE = fitz.Matrix(2, 2)  # 144 DPI

# 高质量模式（适合低质量/老旧扫描件）
HIGH_QUALITY_MODE = fitz.Matrix(3, 3)  # 216 DPI
```

### 2. 并行OCR处理

```python
from concurrent.futures import ThreadPoolExecutor

def ocr_page_batch(page_info):
    """批量OCR处理函数"""
    page_num, page = page_info
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang='eng')
    return page_num, text

def parallel_parse_pdf(pdf_path, max_workers=4):
    """并行解析PDF（适合多页扫描文档）"""
    doc = fitz.open(pdf_path)
    
    # 收集需要OCR的页面
    ocr_tasks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 50:
            ocr_tasks.append((i, page))
    
    # 并行OCR
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(ocr_page_batch, ocr_tasks))
    
    # 合并结果
    # ... (合并逻辑)
```

### 3. OCR精度优化技巧

```python
def preprocess_for_ocr(img):
    """OCR前图像预处理"""
    # 转换为灰度
    img = img.convert('L')
    
    # 自动对比度增强
    from PIL import ImageOps
    img = ImageOps.autocontrast(img)
    
    # 去噪（可选）
    # img = img.filter(ImageFilter.MedianFilter())
    
    return img
```

---

## 错误处理策略

### 1. Tesseract未安装处理

```python
def safe_ocr(img, lang='eng'):
    """安全的OCR调用"""
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract未安装，请访问: https://github.com/UB-Mannheim/tesseract/wiki")
        return ""
    except Exception as e:
        print(f"❌ OCR失败: {e}")
        return ""
```

### 2. 多路径Tesseract配置

```python
def setup_tesseract():
    """自动查找Tesseract安装路径"""
    potential_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        r'D:\Tesseract-OCR\tesseract.exe',
        'tesseract'  # Linux/Mac PATH
    ]
    
    for path in potential_paths:
        if os.path.exists(path) or path == 'tesseract':
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    
    return False
```

### 3. 部分页面失败处理

```python
def robust_parse_pdf(pdf_path):
    """健壮性PDF解析（单页失败不影响整体）"""
    doc = fitz.open(pdf_path)
    pages_content = []
    failed_pages = []
    
    for i, page in enumerate(doc):
        try:
            text = page.get_text()
            
            # OCR分支
            if len(text.strip()) < 50:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang='eng')
                except Exception as ocr_err:
                    print(f"  ⚠️ 第 {i+1} 页 OCR 失败，跳过: {ocr_err}")
                    failed_pages.append(i + 1)
                    text = ""
            
            pages_content.append(text)
            
        except Exception as e:
            print(f"  ❌ 第 {i+1} 页处理失败: {e}")
            failed_pages.append(i + 1)
            pages_content.append("")
    
    if failed_pages:
        print(f"⚠️ 共 {len(failed_pages)} 页处理失败: {failed_pages}")
    
    return "\n".join(pages_content), pages_content, failed_pages
```

---

## 与纯文本PDF处理的对比

| 特性 | 纯文本处理 (02/03) | 混合型处理 (history) |
|------|-------------------|---------------------|
| **适用场景** | 纯文本PDF | 文本+扫描混合PDF |
| **核心依赖** | PyMuPDF | PyMuPDF + Tesseract |
| **处理速度** | 快 (10-50页/秒) | 慢 (1-5页/秒，含OCR) |
| **准确率** | 100%（文本层） | 依赖OCR质量 (90-99%) |
| **内存占用** | 低 | 高（需存储图像） |
| **扫描页检测** | 无 | 有（字符数阈值） |
| **配置复杂度** | 简单 | 中等（需安装Tesseract） |

### 使用建议

```python
# 根据PDF类型选择处理器
def auto_select_processor(pdf_path):
    """自动选择合适的处理器"""
    doc = fitz.open(pdf_path)
    
    # 采样检测前3页
    scan_count = 0
    for i in range(min(3, len(doc))):
        text = doc[i].get_text()
        if len(text.strip()) < 50:
            scan_count += 1
    
    # 如果超过一半采样页是扫描件，使用混合型处理器
    if scan_count >= 2:
        print("📄 检测到扫描版PDF，启用混合型处理器")
        return HybridPDFProcessor.parse_pdf(pdf_path)
    else:
        print("📄 检测到文本版PDF，使用标准处理器")
        return DocumentProcessor.parse_pdf(pdf_path)
```

---

## 实际输出示例

### 输入
- 25页学术论文PDF
- 前18页：可提取文本
- 后7页：扫描图片（参考文献）

### 处理日志
```
🚀 开始解析 PDF (共 25 页)...
  ⚠️ 第 19 页疑似扫描件，启用 OCR...
  ⚠️ 第 20 页疑似扫描件，启用 OCR...
  ⚠️ 第 21 页疑似扫描件，启用 OCR...
  ⚠️ 第 22 页疑似扫描件，启用 OCR...
  ⚠️ 第 23 页疑似扫描件，启用 OCR...
  ⚠️ 第 24 页疑似扫描件，启用 OCR...
  ⚠️ 第 25 页疑似扫描件，启用 OCR...
✅ 解析完成，共 25 页
   OCR处理页: [19, 20, 21, 22, 23, 24, 25]
```

### 输出内容
```
（前18页直接提取的文本）
...
（第19-25页OCR识别的文本）
References:
[1] Smith, J. et al. (2024). Advanced NLP Techniques...
[2] ...
```

---

## 总结

混合型PDF处理算法的核心价值在于：

1. **智能检测**：通过字符数阈值自动识别扫描页
2. **无缝切换**：文本页直接提取，扫描页自动OCR
3. **统一输出**：合并所有页面为一致的文本格式
4. **错误容忍**：单页失败不影响整体处理

**推荐升级路径**：
- 当前02/03 notebook使用纯文本处理（适合现代电子PDF）
- 需要处理扫描/老文献时，可采用history版本中的混合型处理方案

---

*文档版本: 1.0*  
*最后更新: 2026-02-24*  
*关联文档: PDF_Processing_Algorithm.md*
