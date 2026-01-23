
# %% [Cell 2]

# === Cell 1: 环境配置 (中文适配版) ===
import sys
import os

# 1. 安装必要的库 (增加 jieba 用于辅助处理，虽然主要用 spaCy)
packages = [
    "pymupdf",       # PDF 解析
    "spacy",         # NLP 核心
    "scikit-learn",  # TF-IDF
    "pandas",        # 数据处理
    "openpyxl",      # Excel 导出
    "requests",      # 下载示例
    "jieba"          # 中文辅助分词
]

print("🛠️ 正在检查环境依赖...")
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"  ⬇️ 正在安装: {pkg}...")
#         !{sys.executable} -m pip install {pkg} -q  # [Magic Command]

# 2. 下载 spaCy 中文模型
import spacy
try:
    # 尝试加载中文模型
    nlp = spacy.load("zh_core_web_sm")
    print("✅ 中文 NLP 模型 (zh_core_web_sm) 已加载")
except OSError:
    print("⬇️ 未检测到中文模型，正在下载 (约 15MB)...")
#     !{sys.executable} -m spacy download zh_core_web_sm  # [Magic Command]
    nlp = spacy.load("zh_core_web_sm")
    print("✅ 中文 NLP 模型下载并加载完成")

print("🚀 环境准备就绪！")



# %% [Cell 3]

# === Cell 2: PDF 解析与中文排版修复 (自动搜寻版) ===
import fitz  # PyMuPDF
import re
import requests
import os
import glob  # 新增：用于查找文件

class ChinesePDFProcessor:
    @staticmethod
    def clean_text_structure(text):
        """
        专门修复中文 PDF 的排版问题
        """
        if not text: return ""
        
        # 1. 修复中文换行 (关键)：如果前一个字是中文，后一个字也是中文，中间的换行和空格删掉
        text = re.sub(r'([\u4e00-\u9fa5])\s*\n\s*([\u4e00-\u9fa5])', r'\1\2', text)
        
        # 2. 修复英文连字符 (Word- breaking) -> Wordbreaking
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 3. 移除引用标记 [1], [1-3]
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        
        # 4. 统一标点符号
        return text.strip()

    @staticmethod
    def parse_pdf(pdf_path):
        """解析 PDF，返回 (全文, 页面列表, 标题)"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ 文件未找到: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        full_text = []
        pages_corpus = []
        
        # 尝试获取标题
        title = doc.metadata.get('title', '')
        if not title or "Untitled" in title or not title.strip():
            title = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 清洗标题中的非法字符
        title = re.sub(r'[\\/*?:"<>|]', '_', title)
        
        print(f"📖 正在解析: 《{title}》 (共 {len(doc)} 页)...")
        
        for page in doc:
            raw_text = page.get_text()
            cleaned_page = ChinesePDFProcessor.clean_text_structure(raw_text)
            
            if len(cleaned_page) > 50: # 忽略太短的页
                pages_corpus.append(cleaned_page)
                full_text.append(cleaned_page)
                
        return "".join(full_text), pages_corpus, title

# === 自动执行部分 (Auto-Run) ===

# 1. 自动查找当前目录下的所有 PDF
pdf_files = glob.glob("chinesepaper.pdf")

if not pdf_files:
    print("❌ 错误：当前目录下没找到任何 PDF 文件！")
    print("💡 请在左侧文件栏上传你的论文 PDF，然后重新运行此 Cell。")
    # 为了防止后续报错，设置空变量
    raw_text, raw_pages, doc_title = "", [], ""
else:
    # 2. 默认取第一个找到的 PDF
    target_pdf = pdf_files[0]
    print(f"✅ 检测到文件: {target_pdf}")
    
    try:
        # 3. 真正开始执行解析
        raw_text, raw_pages, doc_title = ChinesePDFProcessor.parse_pdf(target_pdf)
        print(f"🎉 解析成功！文本长度: {len(raw_text)} 字符")
        print("👉 请继续运行 Cell 3 和 Cell 4")
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        raw_text, raw_pages, doc_title = "", [], ""



# %% [Cell 4]

# === Cell 3: 中文术语提取核心引擎 (Suffix Filter Edition) ===
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import jieba

class ChineseTermExtractor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.matcher = Matcher(self.nlp.vocab)
        self.rejection_log = []
        
        # --- 1. 中文语法模式 ---
        self.matcher.add("TERM", [
            [{"POS": "NOUN"}, {"POS": "NOUN"}], 
            [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "ADJ"}, {"POS": "NOUN"}],
            [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
            [{"POS": "PROPN"}, {"POS": "NOUN"}],
            [{"POS": "NOUN"}, {"POS": "PROPN"}] 
        ])
        
        # --- 2. 语义黑名单 (针对你反馈的“改革开放”、“研究成果”等优化) ---
        self.blacklist = {
            "摘要", "引言", "绪论", "目录", "参考文献", "致谢", "附录", 
            "图表", "公式", "代码", "关键词", "数据", "实验",
            "研究", "分析", "讨论", "结论", "方法", "结果", "背景",
            "改革开放", "新时代", "一带一路", "十三五", "十二五", # 历史/政治背景词
            "中国", "我国", "国内", "国外", "国际", # 过于宽泛的地域
            "时间", "年份", "年度", "百分比", "平均值", "总数",
            "问题", "对策", "建议", "影响", "意义", "价值", "作用",
            "现状", "趋势", "特点", "特征", "机制", "路径", "模式"
        }
        
        # --- 3. 坏后缀 (Bad Suffixes) - 核心修复点 ---
        # 只要词语以这些结尾，且不是黑名单里的词，大概率是机构或通用废话
        self.bad_suffixes = (
            "大学", "学院", "学校", "分校", "校区", # 机构：浙江大学
            "协会", "学会", "委员会", "理事会", "组织", "机构", "中心", "基地", # 组织：行业协会
            "成果", "成效", "成绩", "成就", # 结果：研究成果
            "活动", "会议", "论坛", "研讨会", "讲座", # 事件：交流活动
            "人员", "人才", "队伍", "群体", "学者", "专家", # 人员：研究人员
            "情况", "状况", "态势", "形势", "局面", # 状态
            "报告", "论文", "文章", "刊物", "期刊", "杂志", # 文献
            "阶段", "时期", "时代", "年代", "世纪", # 时间
            "水平", "能力", "素质", "素养" # 程度
        )
        
        # 坏前缀
        self.bad_prefixes = (
            "各种", "大量", "许多", "不同", "主要", "重要", "核心", "关键", "基本",
            "相关", "有关", "上述", "如下", "该", "本", "某", "高", "低", "大", "小",
            "加强", "促进", "推动", "主要"
        )

    @staticmethod
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def validate_term(self, term):
        clean_t = term.strip()
        
        # 1. 基础长度
        if len(clean_t) < 2: return False, "太短"
        if len(clean_t) > 10: return False, "太长"
        
        # 2. 黑名单 (精确匹配)
        if clean_t in self.blacklist: return False, f"黑名单词汇 ({clean_t})"
        
        # 3. 坏后缀检测 (Fix: 浙江大学, 研究成果, 交流活动)
        if clean_t.endswith(self.bad_suffixes):
            return False, f"无效后缀 ({clean_t[-2:]})"
            
        # 4. 坏前缀检测
        if clean_t.startswith(self.bad_prefixes):
            return False, f"无效前缀 ({clean_t[:2]})"
            
        # 5. 纯数字/符号混杂
        if re.match(r'^[0-9a-zA-Z\s]+$', clean_t) and len(clean_t) < 4:
            return False, "纯英文/数字过短"
            
        return True, "有效"

    def extract_candidates(self, full_text):
        self.rejection_log = []
        print("🔍 正在进行 NLP 模式匹配 (Suffix Filter Enabled)...")
        
        doc = self.nlp(full_text[:1000000])
        matches = self.matcher(doc)
        
        candidates = set()
        for _, start, end in matches:
            span = doc[start:end]
            term = span.text.strip()
            # 中文去空格
            if re.search(r'[\u4e00-\u9fa5]', term):
                term = term.replace(" ", "")
            
            is_valid, reason = self.validate_term(term)
            
            if is_valid:
                candidates.add(term)
            else:
                self.rejection_log.append({"Term": term, "Reason": reason})
                
        return list(candidates)

    def process_deduplication(self, df):
        if df.empty: return df
        print("✂️ 执行智能子串去重...")
        
        terms = df.sort_values("Score", ascending=False).to_dict('records')
        to_drop = set()
        
        for i, short in enumerate(terms):
            for j, long in enumerate(terms):
                if i == j: continue
                s_txt = short['Term']
                l_txt = long['Term']
                
                # 只有当短词是长词的子串时
                if s_txt in l_txt:
                    # 如果短词得分低于长词得分的 300% (放宽标准，倾向于保留长词)
                    # 例如：保留 '翻译学科建设'，删除 '学科建设' (因为学科建设太通用)
                    if short['Score'] < (long['Score'] * 3.0):
                        to_drop.add(s_txt)
                        
        df_clean = df[~df['Term'].isin(to_drop)]
        print(f"   去除冗余词汇: {len(df) - len(df_clean)} 个")
        return df_clean

    def compute_tfidf(self, pages, vocab):
        print("📊 计算 TF-IDF 权重...")
        
        def chinese_tokenizer(text):
            return jieba.lcut(text)

        vectorizer = TfidfVectorizer(
            vocabulary=vocab,
            tokenizer=chinese_tokenizer, 
            ngram_range=(1, 3),
            norm='l2'
        )
        try:
            X = vectorizer.fit_transform(pages)
            scores = X.sum(axis=0).A1
            return dict(zip(vectorizer.get_feature_names_out(), scores))
        except Exception as e:
            print(f"⚠️ TF-IDF 计算异常 (可能词表为空): {e}")
            return {}

# 初始化
extractor = ChineseTermExtractor(nlp)
print("✅ 中文核心提取引擎升级完成 (包含坏后缀过滤)")



# %% [Cell 5]

# === Cell 4: 运行管道 (智能对接版) ===
import pandas as pd

def run_full_pipeline(text_data, pages_data, config):
    # 1. 提取
    if not hasattr(extractor, 'extract_candidates'):
        raise AttributeError("请重新运行 Cell 3 更新提取器代码")
        
    candidates = extractor.extract_candidates(text_data)
    print(f"✅ 初步识别候选词: {len(candidates)} 个")
    
    # 2. 权重
    tfidf_map = extractor.compute_tfidf(pages_data, candidates)
    
    # 3. 评分
    results = []
    print("⚖️ 正在综合评分...")
    
    for term in candidates:
        freq = text_data.count(term)
        if freq < config['min_freq']: continue
        
        tfidf = tfidf_map.get(term, 0)
        
        # 中文词长奖励 (更倾向于 4字 或 3字词，2字词容易太泛)
        len_bonus = 1.0
        if len(term) == 2: len_bonus = 0.8  # 惩罚2字词 (e.g. 研究)
        if len(term) >= 4: len_bonus = 1.3  # 奖励4字成语/术语
        
        score = (freq * config['w_freq'] + tfidf * config['w_tfidf']) * len_bonus
        
        # 语境
        idx = text_data.find(term)
        ctx = text_data[idx-20:idx+len(term)+20].replace('\n', '') if idx > -1 else ""
        
        results.append({
            "Term": term,
            "Score": round(score, 2),
            "Freq": freq,
            "TF-IDF": round(tfidf, 2),
            "Context": "..." + ctx + "..."
        })
        
    df = pd.DataFrame(results)
    
    # 4. 去重
    df_final = extractor.process_deduplication(df)
    
    return df_final.sort_values("Score", ascending=False)

# 权重配置 (针对中文优化)
config = {
    "min_freq": 2,      
    "w_freq": 0.3,      # 进一步降低词频权重，防止高频废话
    "w_tfidf": 20.0     # 极度依赖 TF-IDF 区分度
}

# === 自动对接 Cell 2 变量 ===
target_text = None
target_pages = None

if 'raw_text' in locals() and raw_text:
    print("🔗 使用自动搜寻的 PDF 数据 (raw_text)")
    target_text = raw_text
    target_pages = raw_pages
elif 'full_text' in locals() and full_text:
    print("🔗 使用旧版 PDF 数据 (full_text)")
    target_text = full_text
    target_pages = pages_corpus
else:
    print("❌ 无法找到 PDF 数据，请先运行 Cell 2")

if target_text:
    try:
        df_result = run_full_pipeline(target_text, target_pages, config)
        
        # 打印审计日志摘要
        print("\n🛑 过滤日志 Top 5:")
        log_df = pd.DataFrame(extractor.rejection_log)
        if not log_df.empty:
            print(log_df['Reason'].value_counts().head(5))
            
        print(f"\n🏆 最终提取术语: {len(df_result)} 个")
        display(df_result.head(2000).style.background_gradient(subset=['Score'], cmap='Oranges'))
    except Exception as e:
        print(f"❌ 运行错误: {e}")



# %% [Cell 6]

# === Cell 5: 导出与审计 ===
# 1. 查看拦截日志 (可选)
def show_audit_log(extractor):
    log = pd.DataFrame(extractor.rejection_log)
    if not log.empty:
        print(f"🛑 审计日志: 共拦截 {len(log)} 个无效词")
        print(log['Reason'].value_counts().head())
        # display(log.head(5))

show_audit_log(extractor)

# 2. 导出结果
if 'df_result' in locals():
    # 自动文件名
    safe_title = re.sub(r'[\\/*?:"<>|]', '_', title) if 'title' in locals() else "Result"
    filename = f"术语表_{safe_title}.xlsx"
    
    # 整理列名
    export_df = df_result.rename(columns={
        "Term": "术语 (中文/英文)",
        "Score": "推荐分",
        "Freq": "词频",
        "Context": "原文语境"
    })
    export_df.insert(1, "人工翻译/备注", "") # 插入空列方便填空
    
    try:
        export_df.to_excel(filename, index=False)
        print(f"\n🎉 成功导出文件: {filename}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

