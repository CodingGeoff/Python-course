# AGENTS.md - Python NLP 课程项目上下文

## 项目概述

这是一个面向翻译学和语言学专业的 **Python 自然语言处理 (NLP) 教学课程项目** (Week 1)，专注于学术文本处理、双语对齐和术语提取。

**项目目标**：通过实践案例学习 Python 在翻译研究、学术文本分析和自然语言处理中的应用。

---

## 核心技术栈

| 类别 | 技术/库 | 用途 |
|------|---------|------|
| **NLP核心** | NLTK, SpaCy, TextBlob, jieba | 分词、词性标注、命名实体识别、情感分析 |
| **数据处理** | pandas, numpy, scikit-learn | 数据表格处理、TF-IDF算法、机器学习 |
| **PDF处理** | PyMuPDF (fitz) | PDF文本提取与解析 |
| **文档生成** | python-docx, openpyxl | Word/Excel 文件生成 |
| **可视化** | matplotlib, seaborn, wordcloud | 图表、热力图、词云 |
| **Web应用** | Streamlit | 交互式Web演示界面 |
| **爬虫** | BeautifulSoup, requests | 网页数据抓取 |
| **OCR** | pytesseract, PIL | 图片文字识别 |

---

## 项目结构

```
Week1/
├── 📓 Jupyter Notebooks (核心教学文件)
│   ├── 01 bilingual-text-alignment.ipynb      # 双语文本对齐工具
│   ├── 02 en_thesis_terminology_extraction.ipynb  # 英文论文术语提取
│   └── 03 zh_thesis_terminology_extraction.ipynb  # 中文论文术语提取
│
├── 🌐 Web应用
│   └── app.py                                  # Streamlit 交互式Web应用
│
├── 📂 数据与资产
│   ├── assets/                                 # 生成的术语表、对齐文件
│   │   ├── Academic_Glossary.xlsx             # 学术术语表
│   │   ├── aligned_*.xlsx                     # 双语对齐结果
│   │   └── Layout_*.docx                      # Word格式对齐文档
│   ├── essays/                                # 学术论文PDF样本
│   │   ├── paper.pdf, paper2.pdf, paper3.pdf  # 英文论文
│   │   └── transformer_paper.pdf              # Transformer论文
│   └── CGTN-bilingual-news.txt                # 双语新闻文本示例
│
├── 📜 历史版本
│   └── history/                               # 历史迭代代码备份
│
└── 🐍 虚拟环境
    └── venv/                                  # Python虚拟环境
```

---

## 核心功能模块

### 1. 双语文本对齐 (01 bilingual-text-alignment.ipynb)

**功能**：将中英双语交替排列的文本自动对齐，生成Excel和Word格式的对照表。

**工作流程**：
1. 自动检测包含 `bilingual` 关键字的文本文件
2. 根据中文字符识别中文行，其余为英文行
3. 生成对齐的DataFrame并导出Excel
4. 生成格式化的Word文档（Times New Roman + 宋体）

**输入**：`CGTN-bilingual-news.txt` (中英文交替)
**输出**：
- `aligned_YYYYMMDD_HHMMSS.xlsx` - Excel对齐文件
- `Layout_YYYYMMDD_HHMMSS.docx` - Word格式文档

---

### 2. 英文论文术语提取 (02 en_thesis_terminology_extraction.ipynb)

**功能**：使用TF-IDF算法和SpaCy从英文学术论文中提取专业术语。

**核心算法**：
- **候选词提取**：基于SpaCy Matcher语法模式匹配名词短语
  - 模式示例：`ADJ + NOUN`, `NOUN + NOUN`, `ADJ + NOUN + NOUN`
- **过滤规则**：
  - 黑名单过滤（停用词、通用学术词汇、出版相关词汇）
  - 长度限制（4-50字符，1-3个词）
  - 词形还原去重（单复数归并）
  - 子串冗余消除
- **TF-IDF评分**：计算术语在文档中的重要性

**输出字段**：
- `Term` - 术语原文
- `Score` - 综合评分 (Freq × TF-IDF)
- `Freq` - 出现频率
- `TF-IDF` - TF-IDF权重
- `Context` - 上下文示例

---

### 3. 中文论文术语提取 (03 zh_thesis_terminology_extraction.ipynb)

**功能**：针对中文学术论文的术语提取系统。

**中文适配特性**：
- 使用SpaCy中文模型 (`zh_core_web_sm`)
- jieba辅助分词
- 中文PDF换行修复（中文字符间换行合并）
- 中文语义黑名单（"摘要"、"研究"、"分析"等学术通用词）
- 坏前缀/后缀过滤（"大学"、"学院"、"报告"等）

---

### 4. Streamlit Web应用 (app.py)

**功能**：提供一个交互式的NLP工具箱界面。

**模块**：
1. **🏠 资源加载** - NLP环境初始化
2. **🧠 NLTK & SpaCy** - 分词、词性标注、NER、句法依存
3. **😊 情感与词云** - TextBlob情感分析、WordCloud可视化
4. **📊 TF-IDF与术语挖掘** - 交互式TF-IDF矩阵热力图
5. **🔤 翻译质量评估** - BLEU分数计算、Diff差异对比
6. **🕷️ 爬虫与数据采集** - 实时网页抓取
7. **📂 文件自动化处理** - Excel/CSV数据清洗
8. **👁️ OCR智能识别** - Tesseract OCR文字提取
9. **💻 Python交互沙盒** - Jupyter风格的代码执行环境

**启动命令**：
```bash
streamlit run app.py
```

---

## 开发与运行指南

### 环境准备

```bash
# 1. 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载SpaCy模型
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

# 4. 下载NLTK数据 (应用首次运行时会自动下载)
```

### 运行Jupyter Notebook

```bash
jupyter notebook
# 或
jupyter lab
```

### 运行Streamlit应用

```bash
streamlit run app.py
```

---

## 关键依赖 (requirements.txt)

```
beautifulsoup4==4.14.3    # 网页解析
jieba==0.42.1              # 中文分词
matplotlib==3.10.8         # 可视化
nltk==3.9.2                # NLP基础库
numpy==2.4.1               # 数值计算
pandas==2.3.3              # 数据处理
pillow==12.1.0             # 图像处理
pytesseract==0.3.13        # OCR
python-docx==1.2.0         # Word文档
requests==2.32.5           # HTTP请求
scikit-learn==1.8.0        # 机器学习/TF-IDF
seaborn==0.13.2            # 统计可视化
spacy==3.8.11              # 高级NLP
streamlit==1.53.0          # Web应用框架
textblob==0.19.0           # 情感分析
tqdm==4.67.1               # 进度条
wordcloud==1.9.6           # 词云生成
```

---

## 数据文件约定

### 输入文件
- **双语文本**：`*bilingual*.txt` (中英文交替行)
- **学术论文**：`essays/*.pdf` (PDF格式)

### 输出文件命名规则
- **对齐文件**：`aligned_YYYYMMDD_HHMMSS.xlsx`
- **Word布局**：`Layout_YYYYMMDD_HHMMSS.docx`
- **术语表**：`Glossary_*.xlsx` 或 `Academic_Glossary.xlsx`

---

## 开发约定

### 代码风格
- 使用 **中文注释** 和中文文档字符串（教学目的）
- Notebook采用 `Cell 1`, `Cell 2` 等标记进行逻辑分区
- 类名使用 PascalCase (如 `TerminologyExtractor`)
- 方法名使用 snake_case (如 `extract_candidates`)

### 错误处理
- 使用 try-except 包裹文件I/O操作
- 提供中文错误提示信息
- 自动降级策略（如本地文件不存在时自动下载备用文件）

### 性能考虑
- SpaCy模型全局加载，避免重复初始化
- TF-IDF向量化使用 `ngram_range=(1, 3)` 限制内存使用
- 文本清洗截断至150万字符 (`clean_text[:1500000]`)

---

## 常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| NLTK数据缺失 | 应用会自动下载 `punkt`, `stopwords` 等必要资源 |
| SpaCy模型未找到 | 运行 `python -m spacy download en_core_web_sm` |
| Tesseract OCR不可用 | 需单独安装Tesseract-OCR并配置路径 |
| 中文字体显示问题 | Word生成使用 `SimSun` (宋体) + XML字体设置 |

---

## 学术引用示例

项目中使用的测试论文：
- **Transformer论文**: "Attention Is All You Need" (Vaswani et al., 2017)
- **翻译研究论文**: 关于中国翻译家群体的系列研究 (许钧、金其斌等)

---

## 教学目的说明

本项目专为以下教学设计：
- **翻译技术**：CAT工具原理、术语管理、双语对齐
- **计算语言学**：NLP基础、文本挖掘、语料库构建
- **Python编程**：数据处理、文件自动化、Web开发

---

*Generated by iFlow CLI on 2026-02-24*
