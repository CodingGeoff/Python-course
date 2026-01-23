
# %% [Cell 2]

# === Cell 1: 环境配置与依赖检查 ===
import sys
import os

# 1. 定义依赖库
packages = [
    "pymupdf",       # PDF 解析
    "spacy",         # 高级 NLP
    "scikit-learn",  # TF-IDF 算法
    "pandas",        # 数据表格
    "openpyxl",      # Excel 导出
    "requests",      # 下载测试文件
    "nltk"           # 辅助停用词库
]

print("🛠️ 正在检查环境...")
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"  ⬇️ 正在安装缺失库: {pkg}...")
#         !{sys.executable} -m pip install {pkg} -q  # [Magic Command]

# 2. 下载 spaCy 模型与 NLTK 数据
import spacy
import nltk

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("  ⬇️ 下载 spaCy 英文模型...")
#     !{sys.executable} -m spacy download en_core_web_sm  # [Magic Command]
    nlp = spacy.load("en_core_web_md")

nltk.download('stopwords', quiet=True)
print("✅ 所有系统依赖已就绪！")



# %% [Cell 3]

# === Cell 2: 文本清洗与 PDF 解析模块 ===
import fitz  # PyMuPDF
import re
import requests

class DocumentProcessor:
    @staticmethod
    def normalize_text(text):
        """
        【核心清洗函数】
        必须确保提取阶段和 TF-IDF 阶段使用完全相同的清洗逻辑，
        否则会导致匹配失败（分数归零）。
        """
        if not text: return ""
        
        # 1. 修复 PDF 换行连字符 (ex- ample -> example)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 2. 移除引用标记 [1], [12-14]
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        
        # 3. 移除特殊字符，只保留字母和空格
        # 这一步非常重要，它去除了 "30%" 中的 %，防止 "30%" 被识别为名词
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 4. 压缩多余空格并转小写
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    @staticmethod
    def parse_pdf(pdf_path):
        """解析 PDF，返回 (全文, 页面列表)"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ 文件未找到: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        full_text = []
        pages_corpus = []
        
        print(f"📖 正在解析 {os.path.basename(pdf_path)} (共 {len(doc)} 页)...")
        for page in doc:
            text = page.get_text()
            # 简单过滤掉太短的页面（通常是图片或只有页码）
            if len(text) > 100:
                # 注意：这里我们保留原始文本结构给 TF-IDF，
                # 但在内部计算时会调用 normalize_text
                pages_corpus.append(text) 
                full_text.append(text)
                
        return " ".join(full_text), pages_corpus

    @staticmethod
    def download_sample(url, filename="sample.pdf"):
        if not os.path.exists(filename):
            print(f"⬇️ 正在下载测试文献: {url}...")
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
        return filename

# 准备测试数据
pdf_url = "https://arxiv.org/pdf/1706.03762.pdf" # Transformer Paper
local_pdf = DocumentProcessor.download_sample(pdf_url, "paper.pdf")
raw_text, raw_pages = DocumentProcessor.parse_pdf(local_pdf)
print("✅ 文本预处理模块就绪。")



# %% [Cell 4]

# === Cell 3: 生产级核心算法引擎 (The Ultimate Edition) ===
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import pandas as pd
from collections import Counter

class TerminologyExtractor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.matcher = Matcher(self.nlp.vocab)
        self.rejection_log = [] 
        
        # --- 1. 严格语法模式 ---
        # 仅保留名词短语结构
        self.matcher.add("TERM", [
            [{"POS": "ADJ"}, {"POS": "NOUN"}],
            [{"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}]
        ])
        
        # --- 2. 语义狙击黑名单 (你提供的完整列表) ---
        self.blacklist_tokens = {
            # 1. 元数据与引用
            "et", "al", "figure", "fig", "table", "tbl", "doi", "http", "https", 
            "www", "url", "pdf", "html", "xml", "volume", "vol", "issue", "iss", "page", 
            "pp", "p", "author", "authors", "press", "publisher", "copyright", "copyrighted", 
            "reserved", "permission", "license", "licence", "creative", "commons",
            "cc", "publication", "published", "print", "online", "abstract", "keywords", "reference", 
            "references", "citation", "citations", "bibliography", "ref", "refs",
            
            # 2. 图表与视觉
            "line", "solid", "dashed", "dotted", "circle", "circles", 
            "triangle", "triangles", "square", "squares", "diamond", "diamonds", 
            "axis", "axes", "plot", "plots", "curve", "curves", "graph", "graphs", 
            "color", "colour", "red", "blue", "green", "yellow", "black", "white", 
            "grey", "gray", "purple", "orange", "pink", "brown", "shown", 
            "represented", "representation", "illustrated", "illustration", "bar", 
            "bars", "histogram", "heatmap", "scatter", "boxplot", "legend",
            
            # 3. 学术通用废话
            "study", "studies", "data", "dataset", "datasets", "result", "results", 
            "analysis", "analyses", "method", "methods", "methodology", "conclusion", 
            "conclusions", "discussion", "introduction", "background", "objective", 
            "objectives", "aim", "aims", "purpose", "purposes", "finding", "findings", 
            "observation", "observations", "note", "notes", "remark", "remarks", 
            "section", "sections", "part", "parts", "chapter", "chapters", "article", 
            "paper", "manuscript", "text", "paragraph", "para", "supplementary", "suppl",
            "appendix", "appendices", "supplemental", "extended",
            
            # 4. 时间/统计/计量
            "time", "times", "month", "months", "week", "weeks", "year", "years", 
            "day", "days", "hour", "hours", "minute", "minutes", "second", "seconds",
            "baseline", "follow", "followup", "period", "periods", 
            "duration", "interval", "intervals", "total", "average", "avg", "mean", 
            "median", "mode", "range", "number", "numbers", "score", "scores", 
            "rate", "rates", "ratio", "ratios", "level", "levels", "value", "values", 
            "difference", "differences", "comparison", "comparisons", "sample", 
            "samples", "size", "sizes", "frequency", "frequencies", "percentage", 
            "percent", "proportion", "proportions", "count", "counts",
            
            # 5. 研究对象/分组
            "participant", "participants", "subject", "subjects", "patient", "patients",
            "population", "populations", "cohort", "cohorts", "case", "cases", 
            "control", "controls", "group", "groups", "arm", "arms", "trial", "trials",
            "type", "types", "category", "categories", "class", "classes",
            
            # 6. 统计方法论
            "model", "models", "regression", "logistic", "linear", "mixed", 
            "random", "randomized", "randomization", "intercept", "intercepts", 
            "variable", "variables", "factor", "factors", "effect", "effects", 
            "outcome", "outcomes", "measure", "measures", "metric", "metrics", 
            "test", "tests", "anova", "chi", "square", "p", "value", 
            "significance", "significant", "ns", "confidence", "interval", "ci", 
            "or", "rr", "hr", "hazard", "adjusted", "unadjusted", "estimate", "estimates",
            
            # 7. 期刊/出版
            "nature", "health", "science", "cell", "nejm", "lancet", "bmj", "jama", 
            "plos", "one", "elsevier", "springer", "wiley", "taylor", "francis", "sage", 
            "oxford", "cambridge", "university",
            
            # 8. 项目/专属
            "playsmart", "gameplay", "videogame", "videogames",
            
            # 9. 缩写/OCR
            "subst", "abus", "prev", "treat", "dept", "univ", "inst", "conf", 
            "proc", "nat", "int", "ext", "vs", "etc", "viz", "cf", "seq", "ed", 
            "eds", "edn", "rev", "trans", "tr", "vols", "no", "nos", 
            "figs", "tabs", "spp", "ms", "msn", "phd", "md", "dr", "prof"
        }
        
        # --- 3. 形容词前缀黑名单 ---
        self.bad_starters = {
            "great", "good", "bad", "high", "low", "big", "small", "large",
            "significant", "total", "mean", "average", "various", "several",
            "different", "similar", "other", "new", "old", "positive", "negative",
            "primary", "secondary", "main", "major", "minor", "severe", "mild",
            "moderate", "general", "specific", "common", "rare", "frequent",
            "infrequent", "little", "chief", "key", "central", "core", "overall",
            "particular", "same", "another", "young", "oldest", "youngest",
            "neutral", "uncommon", "single", "double", "short", "long"
        }

        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def robust_clean(text):
        if not text: return ""
        # 1. 修复连字符换行
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # 2. 仅保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 3. 强力去重 (fix: months months)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        # 4. 压缩空格
        return re.sub(r'\s+', ' ', text).strip().lower()

    def validate_term(self, term):
        words = term.split()
        
        # A. 长度/结构限制
        if len(term) < 4: return False, "Length (Too Short)"
        if len(words) > 3: return False, "Length (Too Long)"
        
        # B. 重复词检测 (Fix: pad pad)
        if len(set(words)) == 1 and len(words) > 1:
            return False, "Repetitive Words (e.g. pad pad)"
        
        # C. 黑名单检查
        for w in words:
            if w in self.blacklist_tokens:
                return False, f"Blacklist Token ({w})"
        
        # D. 形容词前缀检查
        if words[0] in self.bad_starters:
            return False, f"Bad Starter ({words[0]})"
        
        # E. 停用词边界
        if words[0] in self.stop_words: return False, "Stopword Start"
        if words[-1] in self.stop_words: return False, "Stopword End"
        
        # F. 垃圾词检测
        for w in words:
            if len(w) < 2: return False, "Single Char Noise"
            if not re.search(r'[aeiouy]', w): return False, "No Vowels (Garbage)"
            
        return True, "Valid"

    def extract_candidates(self, text):
        self.rejection_log = []
        clean_text = self.robust_clean(text)
        doc = self.nlp(clean_text[:1500000]) 
        matches = self.matcher(doc)
        
        candidates = set()
        
        for _, start, end in matches:
            span = doc[start:end]
            term = span.text.strip()
            
            is_valid, reason = self.validate_term(term)
            
            if is_valid:
                candidates.add(term)
            else:
                self.rejection_log.append({
                    "Rejected Term": term,
                    "Reason": reason
                })
                
        return list(candidates)

    def process_deduplication(self, df):
        """
        【新增功能】: 后处理去重
        1. 词形还原归并 (sub layer vs sub layers)
        2. 子串冗余消除 (term memory vs short term memory)
        """
        if df.empty: return df
        
        # --- 1. 单复数/词形还原归并 ---
        # 策略：将所有词还原为 lemma，保留得分最高的那个原文形式
        df['Lemma'] = df['Term'].apply(lambda x: " ".join([token.lemma_ for token in self.nlp(x)]))
        # 按 Lemma 分组，保留 Score 最高的行
        df_dedup_lemma = df.sort_values('Score', ascending=False).drop_duplicates(subset=['Lemma'])
        
        # --- 2. 子串包含去重 ---
        # 策略：如果 'A' 是 'B' 的子串，且 'B' 也是有效术语，
        # 通常保留长词(B)更具体，或者保留短词(A)更通用？
        # 在学术提取中，通常去除被长词包含的短词，除非短词分数极高。
        # 这里采用安全策略：如果 A 是 B 的子串，且 A 的分数没有比 B 高出 50%，则删除 A。
        
        terms = df_dedup_lemma.to_dict('records')
        to_drop_indices = set()
        
        for i, small in enumerate(terms):
            for j, big in enumerate(terms):
                if i == j: continue
                
                # 检查子串 (使用 set 判断单词集合是否包含，防止 'ear' in 'tear' 误判)
                small_tokens = set(small['Term'].split())
                big_tokens = set(big['Term'].split())
                
                if small_tokens.issubset(big_tokens) and len(small_tokens) < len(big_tokens):
                    # 如果短词分数没有显著高于长词 (例如高 1.5 倍)，则认为它是长词的冗余部分
                    if small['Score'] < (big['Score'] * 1.5):
                        to_drop_indices.add(i)
                        break
        
        # 根据索引过滤
        final_terms = [t for i, t in enumerate(terms) if i not in to_drop_indices]
        
        # 清理临时列
        df_final = pd.DataFrame(final_terms).drop(columns=['Lemma'], errors='ignore')
        print(f"✂️ 自动归并去重：{len(df)} -> {len(df_final)} (剔除冗余/单复数)")
        return df_final

    def compute_tfidf(self, pages_list, vocabulary):
        if not vocabulary: return {}
        print("📊 正在计算 TF-IDF 矩阵...")
        vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            preprocessor=self.robust_clean,
            ngram_range=(1, 3),
            norm='l2'
        )
        try:
            X = vectorizer.fit_transform(pages_list)
            scores = X.sum(axis=0).A1
            return dict(zip(vectorizer.get_feature_names_out(), scores))
        except Exception as e:
            print(f"⚠️ TF-IDF Error: {e}")
            return {}

# 初始化
extractor = TerminologyExtractor(nlp)
print("✅ 核心算法引擎已构建 (Audit + Deduplication Edition)")



# %% [Cell 5]

# === Cell 3: 生产级核心算法引擎 (最终修复版) ===
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import pandas as pd
from collections import Counter

class TerminologyExtractor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.matcher = Matcher(self.nlp.vocab)
        self.rejection_log = [] 
        
        # --- 1. 语法模式 ---
        # 仅保留名词短语结构
        self.matcher.add("TERM", [
            [{"POS": "ADJ"}, {"POS": "NOUN"}],
            [{"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}]
        ])
        
        # --- 2. 语义狙击黑名单 (完整版) ---
        self.blacklist_tokens = {
            # 1. 元数据与引用
            "et", "al", "figure", "fig", "table", "tbl", "doi", "http", "https", 
            "www", "url", "pdf", "html", "xml", "volume", "vol", "issue", "iss", "page", 
            "pp", "p", "author", "authors", "press", "publisher", "copyright", "copyrighted", 
            "reserved", "permission", "license", "licence", "creative", "commons",
            "cc", "publication", "published", "print", "online", "abstract", "keywords", "reference", 
            "references", "citation", "citations", "bibliography", "ref", "refs",
            
            # 2. 图表与视觉
            "line", "solid", "dashed", "dotted", "circle", "circles", 
            "triangle", "triangles", "square", "squares", "diamond", "diamonds", 
            "axis", "axes", "plot", "plots", "curve", "curves", "graph", "graphs", 
            "color", "colour", "red", "blue", "green", "yellow", "black", "white", 
            "grey", "gray", "purple", "orange", "pink", "brown", "shown", 
            "represented", "representation", "illustrated", "illustration", "bar", 
            "bars", "histogram", "heatmap", "scatter", "boxplot", "legend",
            
            # 3. 学术通用废话
            "study", "studies", "data", "dataset", "datasets", "result", "results", 
            "analysis", "analyses", "method", "methods", "methodology", "conclusion", 
            "conclusions", "discussion", "introduction", "background", "objective", 
            "objectives", "aim", "aims", "purpose", "purposes", "finding", "findings", 
            "observation", "observations", "note", "notes", "remark", "remarks", 
            "section", "sections", "part", "parts", "chapter", "chapters", "article", 
            "paper", "manuscript", "text", "paragraph", "para", "supplementary", "suppl",
            "appendix", "appendices", "supplemental", "extended",
            
            # 4. 时间/统计/计量
            "time", "times", "month", "months", "week", "weeks", "year", "years", 
            "day", "days", "hour", "hours", "minute", "minutes", "second", "seconds",
            "baseline", "follow", "followup", "period", "periods", 
            "duration", "interval", "intervals", "total", "average", "avg", "mean", 
            "median", "mode", "range", "number", "numbers", "score", "scores", 
            "rate", "rates", "ratio", "ratios", "level", "levels", "value", "values", 
            "difference", "differences", "comparison", "comparisons", "sample", 
            "samples", "size", "sizes", "frequency", "frequencies", "percentage", 
            "percent", "proportion", "proportions", "count", "counts",
            
            # 5. 研究对象/分组
            "participant", "participants", "subject", "subjects", "patient", "patients",
            "population", "populations", "cohort", "cohorts", "case", "cases", 
            "control", "controls", "group", "groups", "arm", "arms", "trial", "trials",
            "type", "types", "category", "categories", "class", "classes",
            
            # 6. 统计方法论
            "model", "models", "regression", "logistic", "linear", "mixed", 
            "random", "randomized", "randomization", "intercept", "intercepts", 
            "variable", "variables", "factor", "factors", "effect", "effects", 
            "outcome", "outcomes", "measure", "measures", "metric", "metrics", 
            "test", "tests", "anova", "chi", "square", "p", "value", 
            "significance", "significant", "ns", "confidence", "interval", "ci", 
            "or", "rr", "hr", "hazard", "adjusted", "unadjusted", "estimate", "estimates",
            
            # 7. 期刊/出版
            "nature", "health", "science", "cell", "nejm", "lancet", "bmj", "jama", 
            "plos", "one", "elsevier", "springer", "wiley", "taylor", "francis", "sage", 
            "oxford", "cambridge", "university",
            
            # 8. 项目/专属
            "playsmart", "gameplay", "videogame", "videogames",
            
            # 9. 缩写/OCR
            "subst", "abus", "prev", "treat", "dept", "univ", "inst", "conf", 
            "proc", "nat", "int", "ext", "vs", "etc", "viz", "cf", "seq", "ed", 
            "eds", "edn", "rev", "trans", "tr", "vols", "no", "nos", 
            "figs", "tabs", "spp", "ms", "msn", "phd", "md", "dr", "prof"
        }
        
        # --- 3. 形容词前缀黑名单 ---
        self.bad_starters = {
            "great", "good", "bad", "high", "low", "big", "small", "large",
            "significant", "total", "mean", "average", "various", "several",
            "different", "similar", "other", "new", "old", "positive", "negative",
            "primary", "secondary", "main", "major", "minor", "severe", "mild",
            "moderate", "general", "specific", "common", "rare", "frequent",
            "infrequent", "little", "chief", "key", "central", "core", "overall",
            "particular", "same", "another", "young", "oldest", "youngest",
            "neutral", "uncommon", "single", "double", "short", "long"
        }

        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def robust_clean(text):
        if not text: return ""
        # 1. 修复连字符换行
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # 2. 仅保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 3. 强力去重 (fix: months months)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        # 4. 压缩空格
        return re.sub(r'\s+', ' ', text).strip().lower()

    def validate_term(self, term):
        words = term.split()
        
        # A. 长度/结构限制
        if len(term) < 4: return False, "Length (Too Short)"
        if len(words) > 3: return False, "Length (Too Long)"
        
        # B. 重复词检测 (Fix: pad pad)
        if len(set(words)) == 1 and len(words) > 1:
            return False, "Repetitive Words (e.g. pad pad)"
        
        # C. 黑名单检查
        for w in words:
            if w in self.blacklist_tokens:
                return False, f"Blacklist Token ({w})"
        
        # D. 形容词前缀检查
        if words[0] in self.bad_starters:
            return False, f"Bad Starter ({words[0]})"
        
        # E. 停用词边界
        if words[0] in self.stop_words: return False, "Stopword Start"
        if words[-1] in self.stop_words: return False, "Stopword End"
        
        # F. 垃圾词检测
        for w in words:
            if len(w) < 2: return False, "Single Char Noise"
            if not re.search(r'[aeiouy]', w): return False, "No Vowels (Garbage)"
            
        return True, "Valid"

    def extract_candidates(self, text):
        self.rejection_log = []
        clean_text = self.robust_clean(text)
        doc = self.nlp(clean_text[:1500000]) 
        matches = self.matcher(doc)
        
        candidates = set()
        
        for _, start, end in matches:
            span = doc[start:end]
            term = span.text.strip()
            
            is_valid, reason = self.validate_term(term)
            
            if is_valid:
                candidates.add(term)
            else:
                self.rejection_log.append({
                    "Rejected Term": term,
                    "Reason": reason
                })
                
        return list(candidates)

    def process_deduplication(self, df):
        """
        【新增功能】: 后处理去重 - 修复 AttributeError 的关键部分
        1. 词形还原归并 (sub layer vs sub layers)
        2. 子串冗余消除 (term memory vs short term memory)
        """
        if df.empty: return df
        
        # --- 1. 单复数/词形还原归并 ---
        # 策略：将所有词还原为 lemma，保留得分最高的那个原文形式
        df['Lemma'] = df['Term'].apply(lambda x: " ".join([token.lemma_ for token in self.nlp(x)]))
        # 按 Lemma 分组，保留 Score 最高的行
        df_dedup_lemma = df.sort_values('Score', ascending=False).drop_duplicates(subset=['Lemma'])
        
        # --- 2. 子串包含去重 ---
        # 策略：安全去除完全被长词包含且分数较低的短词
        terms = df_dedup_lemma.to_dict('records')
        to_drop_indices = set()
        
        for i, small in enumerate(terms):
            for j, big in enumerate(terms):
                if i == j: continue
                
                small_tokens = set(small['Term'].split())
                big_tokens = set(big['Term'].split())
                
                # 如果短词集合是长词集合的子集，且长度确实更短
                if small_tokens.issubset(big_tokens) and len(small_tokens) < len(big_tokens):
                    # 如果短词分数没有显著高于长词 (例如高 1.5 倍)，则认为它是长词的冗余部分
                    if small['Score'] < (big['Score'] * 1.5):
                        to_drop_indices.add(i)
                        break
        
        # 根据索引过滤
        final_terms = [t for i, t in enumerate(terms) if i not in to_drop_indices]
        
        # 清理临时列
        df_final = pd.DataFrame(final_terms).drop(columns=['Lemma'], errors='ignore')
        print(f"✂️ 自动归并去重：{len(df)} -> {len(df_final)} (剔除冗余/单复数)")
        return df_final

    def compute_tfidf(self, pages_list, vocabulary):
        if not vocabulary: return {}
        print("📊 正在计算 TF-IDF 矩阵...")
        vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            preprocessor=self.robust_clean,
            ngram_range=(1, 3),
            norm='l2'
        )
        try:
            X = vectorizer.fit_transform(pages_list)
            scores = X.sum(axis=0).A1
            return dict(zip(vectorizer.get_feature_names_out(), scores))
        except Exception as e:
            print(f"⚠️ TF-IDF Error: {e}")
            return {}

# 重新初始化引擎
extractor = TerminologyExtractor(nlp)
print("✅ 核心算法引擎已构建 (包含 process_deduplication)")



# %% [Cell 6]

# === Cell 3.5: 术语排除原因深度分析 ===
import pandas as pd

def analyze_rejections(extractor_instance):
    if not extractor_instance.rejection_log:
        print("⚠️ 审计日志为空，请先运行提取流程。")
        return
    
    df_log = pd.DataFrame(extractor_instance.rejection_log)
    total_rejected = len(df_log)
    unique_rejected = df_log['Rejected Term'].nunique()
    print(f"🛑 共拦截 {total_rejected} 次，涉及 {unique_rejected} 个唯一短语。")
    print("-" * 60)
    
    # 统计主要原因
    reason_counts = df_log['Reason'].value_counts().head(10)
    print("📊 拦截原因 Top 10:")
    print(reason_counts)
    
    return df_log

# 这里只是定义，实际运行在 Cell 4 后
print("✅ 审计工具已就绪")



# %% [Cell 7]

# === Cell 4: 综合评分与 Pipeline 执行 (集成去重) ===
import pandas as pd

def run_full_pipeline(full_text, pages_corpus, config):
    print("🚀 Step 1: 正在提取候选术语...")
    candidates = extractor.extract_candidates(full_text)
    print(f"   -> 初步找到 {len(candidates)} 个候选词")
    
    print("🚀 Step 2: 计算 TF-IDF 权重...")
    tfidf_scores = extractor.compute_tfidf(pages_corpus, candidates)
    
    print("🚀 Step 3: 统计频率与评分...")
    clean_full_text = DocumentProcessor.normalize_text(full_text)
    
    results = []
    
    for term in candidates:
        freq = clean_full_text.count(term)
        if freq < config['min_freq']: continue
        
        tfidf_val = tfidf_scores.get(term, 0)
        
        # 词长奖励
        len_bonus = 1.2 if len(term.split()) > 1 else 1.0
        
        final_score = (freq * config['w_freq'] + tfidf_val * config['w_tfidf']) * len_bonus
        
        # 查找语境
        start_idx = full_text.lower().find(term)
        context = ""
        if start_idx != -1:
            ctx_start = max(0, start_idx - 30)
            ctx_end = min(len(full_text), start_idx + len(term) + 30)
            context = "..." + full_text[ctx_start:ctx_end].replace('\n', ' ') + "..."
            
        results.append({
            "Term": term,
            "Score": round(final_score, 4),
            "Freq": freq,
            "TF-IDF": round(tfidf_val, 4),
            "Context": context
        })
        
    df = pd.DataFrame(results)
    
    # === 关键步骤：调用去重逻辑 ===
    print("🚀 Step 4: 执行智能去重 (单复数归并/子串消除)...")
    df_clean = extractor.process_deduplication(df)
    
    return df_clean.sort_values("Score", ascending=False)

# =======================
# 🎛️ 权重配置
# =======================
config = {
    "min_freq": 3,       # 最小频率
    "w_freq": 0.5,       # 频率权重
    "w_tfidf": 12.0      # TF-IDF 权重 (进一步拉高，强调特异性)
}

# 执行检查
if 'raw_text' in locals():
    # 运行提取
    df_final = run_full_pipeline(raw_text, raw_pages, config)
    
    # 运行审计
    print("\n" + "="*40)
    _ = analyze_rejections(extractor)
    print("="*40 + "\n")
    
    # 显示结果
    print(f"📊 最终提取了 {len(df_final)} 个高质量术语")
    display(df_final.head(2000).style.background_gradient(subset=['Score'], cmap='Greens'))
else:
    print("⚠️ 请先运行 Cell 2 下载并解析 PDF")



# %% [Cell 8]

# === Cell 5: 结果导出 ===
output_file = "Academic_Glossary.xlsx"

if 'df_final' in locals() and not df_final.empty:
    # 格式化导出
    export_df = df_final.rename(columns={
        "Term": "英文术语",
        "Score": "推荐分",
        "Freq": "词频",
        "TF-IDF": "专业度",
        "Context": "原文语境"
    })
    
    # 插入空白翻译列
    export_df.insert(1, "中文翻译", "")
    
    try:
        export_df.to_excel(output_file, index=False)
        print(f"🎉 导出成功！文件位置: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ 导出失败 (请关闭 Excel 文件后重试): {e}")
else:
    print("⚠️ 没有数据可导出")

