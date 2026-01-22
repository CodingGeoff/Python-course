import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
import spacy
from nltk.translate.bleu_score import sentence_bleu
from wordcloud import WordCloud
import io
import sys
import os
from contextlib import contextmanager
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib

# --- 页面全局配置 ---
st.set_page_config(
    page_title="NLP Magic Box: Python 自然语言处理",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS 美化 ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #333; margin-top: 2rem;}
    .stAlert {border-radius: 10px;}
    div.stButton > button:first-child {background-color: #4F8BF9; color: white; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# --- 核心修复：自动检测并下载 NLTK 资源 ---
@st.cache_resource
def init_nlp_environment():
    """
    初始化环境，自动解决 LookupError 和 MissingCorpusError
    """
    status_container = st.status("正在初始化 NLP 核心组件...", expanded=True)
    

    
    # 保留用户主目录路径作为备选
    user_nltk_dir = os.path.expanduser('~/nltk_data')
    if user_nltk_dir not in nltk.data.path:
        nltk.data.path.append(user_nltk_dir)
    
    # 2. 定义必须的包列表
    required_packages = [
        'punkt', 
        'punkt_tab',    # 解决 tokenizer 报错的关键
        'averaged_perceptron_tagger', 
        'wordnet', 
        'stopwords',
        'omw-1.4',
        'brown',        # TextBlob 名词提取依赖
        'conll2000'     # TextBlob 组块分析依赖
    ]
    
    try:
        for pkg in required_packages:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{pkg}')
                except LookupError:
                    status_container.write(f"正在下载资源: {pkg} ...")
                    nltk.download(pkg, quiet=True)
        
        status_container.write("正在加载 SpaCy 模型...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            status_container.write("正在下载 SpaCy 模型 (en_core_web_sm)...")
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            
        status_container.update(label="✅ 环境初始化完成！", state="complete", expanded=False)
        return nlp
        
    except Exception as e:
        status_container.update(label="❌ 初始化失败", state="error")
        st.error(f"严重错误: {str(e)}")
        st.stop()

# 加载模型
nlp = init_nlp_environment()

# --- 工具函数：重定向输出 ---
@contextmanager
def st_capture(output_func):
    with io.StringIO() as stdout, io.StringIO() as stderr:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        try:
            yield stdout
        except Exception as e:
            output_func(f"❌ 运行时错误: {e}")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            output_func(stdout.getvalue())

# --- 侧边栏 ---
with st.sidebar:
    st.image("./assets/Python-logo-notext.svg", width=100)
    st.title("课程导航")
    
    menu = {
        "dashboard": "🏠 资源加载",
        "nlp_core": "🧠 NLTK & SpaCy ",
        "sentiment": "😊 情感与词云可视化",
        "advanced_text": "📊 TF-IDF 与术语挖掘",
        "translation": "🔤 翻译质量评估 (BLEU/Diff)",
        "web_data": "🕷️ 爬虫与数据采集",
        "files": "📂 文件自动化处理",
        "ocr": "👁️ OCR 智能识别",
        "sandbox": "💻 Python 交互沙盒"
    }
    
    selection = st.radio("", list(menu.keys()), format_func=lambda x: menu[x])
    
    st.info("💡 提示：所有模块均支持代码实时修改")
    st.progress(100)

# ================= 模块内容 =================

if selection == "dashboard":
    st.markdown('<div class="main-header">Python NLP & Data Science</div>', unsafe_allow_html=True)
    # st.markdown("### 👋 欢迎来到数据科学课堂")
    
    # col1, col2, col3 = st.columns(3)
    # col1.metric("已加载模块", "8 个")
    # col2.metric("NLTK 状态", "🟢 Ready")
    # col3.metric("SpaCy 状态", "🟢 Ready")
    
    # st.divider()
    # st.markdown("""
    # **本系统专为教学设计，包含以下高阶功能：**
    # * ✨ **自动依赖修复**：解决 `punkt_tab` 和 `MissingCorpus` 问题。
    # * 📊 **高级可视化**：集成了 Seaborn 热力图和 SpaCy 句法树。
    # * 🛠️ **实战案例**：从爬虫到 OCR，覆盖完整数据链路。
    # """)

elif selection == "nlp_core":
    st.header("🧠 NLP 核心：分词、词性与实体")
    
    text = st.text_area("输入文本 (Text)", 
        "Apple is looking at buying U.K. startup for $1 billion. Python represents the future of AI.", height=100)
    
    tab1, tab2, tab3 = st.tabs(["词性标注 (POS)", "实体识别 (NER)", "句法依存 (Dependency)"])
    
    with tab1:
        if st.button("NLTK 分析", key="btn_pos"):
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            
            # 使用带颜色的 DataFrame
            df_pos = pd.DataFrame(tags, columns=["单词 (Token)", "词性 (Tag)"])
            
            # 高亮动词和名词
            def color_pos(val):
                if val.startswith('V'): return 'color: red; font-weight: bold'
                if val.startswith('N'): return 'color: blue; font-weight: bold'
                return ''
            
            st.dataframe(df_pos.style.applymap(color_pos, subset=['词性 (Tag)']), use_container_width=True)
            st.caption("🔴 红色=动词, 🔵 蓝色=名词")

            st.code("""
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
    
# 使用带颜色的 DataFrame
df_pos = pd.DataFrame(tags, columns=["单词 (Token)", "词性 (Tag)"])

# 高亮动词和名词
def color_pos(val):
    if val.startswith('V'): return 'color: red; font-weight: bold'
    if val.startswith('N'): return 'color: blue; font-weight: bold'
    return ''
            """)

    with tab2:
        if st.button("SpaCy NER 分析", key="btn_ner"):
            doc = nlp(text)
            from spacy import displacy
            html = displacy.render(doc, style="ent", jupyter=False)
            st.components.v1.html(html, height=150, scrolling=True)
            
            data = [{"实体": ent.text, "类型": ent.label_, "解释": spacy.explain(ent.label_)} for ent in doc.ents]
            st.table(data)

    with tab3:
        st.markdown("##### 句法依存树 (展示单词间的语法关系)")
        if st.button("生成依存关系", key="btn_dep"):
            doc = nlp(text)
            from spacy import displacy
            # 这里设置较小的距离以适应屏幕
            options = {"compact": True, "color": "#4F8BF9", "bg": "#ffffff", "font": "Source Sans Pro"}
            html = displacy.render(doc, style="dep", options=options, jupyter=False)
            st.components.v1.html(html, height=400, scrolling=True)

elif selection == "sentiment":
    st.header("😊 情感分析与词云")
    
    default_text = "I love Python! It's amazing and super fast. But sometimes libraries version conflict is annoying and terrible."
    text = st.text_area("分析文本", default_text)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 情感仪表盘")
        blob = TextBlob(text)
        
        # 极性可视化
        polarity = blob.sentiment.polarity
        st.slider("情感极性 (Polarity)", min_value=-1.0, max_value=1.0, value=polarity, disabled=True)
        
        if polarity > 0.5: st.success("情感判定: 非常积极 😍")
        elif polarity > 0: st.info("情感判定: 稍微积极 🙂")
        elif polarity < -0.5: st.error("情感判定: 非常消极 😡")
        elif polarity < 0: st.warning("情感判定: 稍微消极 🙁")
        else: st.write("情感判定: 中性 😐")
        
        # 句子级分析
        with st.expander("查看逐句情感分析"):
            for sent in blob.sentences:
                st.write(f"📝 *{sent}* -> Score: {sent.sentiment.polarity:.2f}")

    with col2:
        st.subheader("☁️ 动态词云")
        if st.button("生成词云"):
            wc = WordCloud(background_color='white', colormap='viridis', width=800, height=400).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

elif selection == "advanced_text":
    st.header("📊 高级文本挖掘: TF-IDF & 关键词")
    st.info("TF-IDF (Term Frequency-Inverse Document Frequency) 是比单纯词频更科学的关键词提取方法。")
    
    corpus_txt = st.text_area("输入语料库 (每行代表一个文档/句子):", 
        "Machine learning is fascinating.\nDeep learning is a subset of machine learning.\nData science uses python heavily.\nPython is great for backend too.")
    
    if st.button("计算 TF-IDF 矩阵"):
        corpus = [line for line in corpus_txt.split('\n') if line.strip()]
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=[f"Doc {i+1}" for i in range(len(corpus))])
        
        st.write("### TF-IDF 热力图 (颜色越深，词越重要)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df_tfidf, annot=True, cmap="YlGnBu", ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        st.write("### 原始数据表")
        st.dataframe(df_tfidf)

elif selection == "translation":
    st.header("🔤 翻译质量评估 (BLEU & Diff)")
    
    col1, col2 = st.columns(2)
    ref = col1.text_input("标准参考译文:", "The quick brown fox jumps over the lazy dog")
    cand = col2.text_input("机器/学生译文:", "The fast brown fox jumps over the lazy dog")
    
    if st.button("开始评估"):
        # BLEU Score
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        cand_tokens = nltk.word_tokenize(cand.lower())
        score = sentence_bleu(ref_tokens, cand_tokens)
        
        st.metric("BLEU Score (0-1)", f"{score:.4f}", delta="越高越好")
        
        # Diff Viewer
        st.subheader("🔍 差异对比")
        diff = difflib.HtmlDiff().make_file([ref], [cand], context=True, numlines=1)
        # 清洗一下HTML样式以适应Streamlit
        st.components.v1.html(diff, height=200, scrolling=True)
        st.caption("左侧为参考，右侧为输入。颜色高亮显示差异。")

elif selection == "web_data":
    st.header("🕷️ 实时网页爬虫")
    
    url = st.text_input("目标 URL:", "https://www.python.org")
    
    if st.button("抓取数据"):
        with st.spinner("正在发送 HTTP 请求..."):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                r = requests.get(url, headers=headers, timeout=5)
                r.encoding = r.apparent_encoding
                
                st.success(f"请求成功! 状态码: {r.status_code}")
                
                soup = BeautifulSoup(r.text, 'html.parser')
                
                tab1, tab2 = st.tabs(["链接分析", "文本内容"])
                
                with tab1:
                    links = [{"Text": a.get_text(strip=True), "HREF": a.get('href')} for a in soup.find_all('a', href=True)]
                    df_links = pd.DataFrame(links)
                    st.dataframe(df_links, use_container_width=True)
                    st.caption(f"共发现 {len(links)} 个链接")
                    
                with tab2:
                    # 提取所有段落
                    paras = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
                    st.write(paras[:5]) # 只显示前5段
                    
            except Exception as e:
                st.error(f"爬取失败: {e}")

elif selection == "files":
    st.header("📂 自动化文件处理")
    
    demo_type = st.radio("选择演示类型", ["Excel 数据清洗", "XML 解析"])
    
    if demo_type == "Excel 数据清洗":
        uploaded = st.file_uploader("上传 Excel (.xlsx)", type="xlsx")
        if uploaded:
            df = pd.read_excel(uploaded)
            st.write("原始数据预览:", df.head())
            
            st.markdown("#### 🛠️ 快速操作")
            col1, col2 = st.columns(2)
            if col1.button("填充缺失值 (用 0)"):
                df_filled = df.fillna(0)
                st.dataframe(df_filled)
            if col2.button("生成统计描述"):
                st.write(df.describe())
    
    else:
        st.info("XML 解析演示：将层级结构转换为 DataFrame")
        uploaded = st.file_uploader("上传 XML", type="xml")
        if uploaded:
            tree = ET.parse(uploaded)
            root = tree.getroot()
            st.code(ET.tostring(root, encoding='unicode')[:500] + "...", language="xml")

elif selection == "ocr":
    st.header("👁️ OCR 光学字符识别")
    st.markdown("将图片转换为可编辑文本。**注意：需要安装 Tesseract 软件。**")
    
    img_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
    
    if img_file:
        st.image(img_file, width=300)
        if st.button("提取文字"):
            try:
                import pytesseract
                from PIL import Image
                
                # 尝试常见的安装路径 (Windows)
                potential_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'D:\Program Files\Tesseract-OCR\tesseract.exe'
                ]
                
                # 检查系统PATH
                tesseract_cmd = 'tesseract'
                for p in potential_paths:
                    if os.path.exists(p):
                        tesseract_cmd = p
                        break
                
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                
                image = Image.open(img_file)
                text = pytesseract.image_to_string(image, lang='eng')
                
                st.success("识别成功！")
                st.text_area("识别结果", text, height=200)
                
            except Exception as e:
                st.error("OCR 引擎调用失败")
                st.warning(f"错误详情: {e}")
                st.info("请确保已安装 Tesseract-OCR 并配置了路径。")

elif selection == "sandbox":
    st.header("💻 Python 交互式沙盒 (Jupyter Notebook 风格)")
    
    # ==================== 初始化 Session State ====================
    if "code_blocks" not in st.session_state:
        # 初始化代码块列表：每个块包含id、content、output、expanded
        st.session_state.code_blocks = [
            {
                "id": 1,
                "content": "# 欢迎使用 Python 沙盒！\n# 尝试运行下面的代码，或选择模板快速开始\nimport pandas as pd\nimport numpy as np\n\n# 创建示例数据框\ndf = pd.DataFrame({\n    'Name': ['Alice', 'Bob', 'Charlie'],\n    'Age': [25, 30, 35],\n    'City': ['New York', 'London', 'Paris']\n})\nprint('数据框创建成功：')\nprint(df)\n\n# 计算年龄均值\nprint(f\"\\n年龄均值：{df['Age'].mean():.1f}\")",
                "output": "",
                "expanded": True
            }
        ]
    if "next_block_id" not in st.session_state:
        st.session_state.next_block_id = 2
    
    # ==================== 自定义 CSS 美化 (Jupyter 风格) ====================
    st.markdown("""
    <style>
        /* 代码块容器样式 */
        .code-block-container {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
            background-color: #ffffff;
        }
        /* 代码块头部（按钮区） */
        .code-block-header {
            background-color: #f8f9fa;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        /* 代码编辑区 */
        .stTextArea[data-testid=\"stTextArea\"] textarea {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
            border: none;
            box-shadow: none;
        }
        /* 输出区域样式 */
        .code-output {
            padding: 1rem;
            background-color: #fafafa;
            border-top: 1px solid #e0e0e0;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        /* 按钮组样式 */
        .btn-group {
            display: flex;
            gap: 0.5rem;
        }
        /* 模板选择框样式 */
        .template-selector {
            margin-bottom: 1rem;
            padding: 0.5rem;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== 丰富的代码模板库 ====================
    TEMPLATES = {
        "📝 基础 - Pandas 创建数据": """import pandas as pd
import numpy as np

# 创建带随机数据的DataFrame
data = {
    'Product': ['Laptop', 'Phone', 'Tablet', 'Headphones'],
    'Price': np.random.randint(100, 1000, size=4),
    'Sales': np.random.randint(50, 500, size=4),
    'Rating': np.round(np.random.uniform(3.0, 5.0, size=4), 1)
}
df = pd.DataFrame(data)

# 基本数据探索
print(\"=== 产品销售数据 ===\")
print(df)
print(\"\\n=== 数据基本统计 ===\")
print(df.describe())
print(f\"\\n总销售额：${(df['Price'] * df['Sales']).sum():,}\")""",
        
        "📊 可视化 - Matplotlib 绘图": """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文环境
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文环境
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 绘制正弦曲线
ax1.plot(x, y1, color='#4F8BF9', linewidth=2, label='sin(x)')
ax1.set_title('Sine Wave', fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()
ax1.grid(alpha=0.3)

# 绘制柱状图（随机数据）
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 50, size=5)
ax2.bar(categories, values, color='#2ECC71', alpha=0.8)
ax2.set_title('Bar Chart', fontsize=14)
ax2.set_xlabel('Category')
ax2.set_ylabel('Value')

plt.tight_layout()
st.pyplot(fig)  # Streamlit 显示图表""",
        
        "🧠 NLP - NLTK 分词 & 词性标注": """import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 确保已加载NLTK资源（沙盒外已初始化）
text = \"Apple is looking at buying U.K. startup for $1 billion. Python is awesome!\"

# 分词
tokens = word_tokenize(text)
print(\"=== 分词结果 ===\")
print(tokens)

# 词性标注
tags = pos_tag(tokens)
print(\"\\n=== 词性标注结果 ===\")
for token, tag in tags[:10]:  # 显示前10个
    print(f\"{token:<10} -> {tag} ({nltk.help.upenn_tagset(tag) if tag in nltk.data.load('help/tagsets/upenn_tagset.pickle') else '未知标签'})\")""",
        
        "🔍 NLP - SpaCy 命名实体识别": """import spacy

# 加载预训练模型（沙盒外已初始化）
nlp = spacy.load(\"en_core_web_sm\")

# 待分析文本
text = \"Elon Musk founded Tesla in 2003 and SpaceX in 2002. He was born in South Africa.\"
doc = nlp(text)

# 提取命名实体
print(\"=== 命名实体识别结果 ===\")
for ent in doc.ents:
    print(f\"实体: {ent.text:<15} 类型: {ent.label_:<8} 解释: {spacy.explain(ent.label_)}\")

# 可视化实体（简化版）
ent_text = \" | \".join([f\"[{ent.text}: {ent.label_}]\" for ent in doc.ents])
print(f\"\\n实体汇总: {ent_text}\")""",
        
        "📈 文本挖掘 - TF-IDF 关键词提取": """from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 示例语料库
corpus = [
    \"Machine learning is the study of computer algorithms that improve automatically through experience.\",
    \"Deep learning is a subset of machine learning that uses neural networks with many layers.\",
    \"Natural language processing is a field of AI that focuses on the interaction between computers and humans using natural language.\",
    \"Python is a popular programming language for machine learning and data science tasks.\"
]

# 初始化TF-IDF向量化器（过滤停用词）
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# 转换为DataFrame便于查看
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f\"文档 {i+1}\" for i in range(len(corpus))]
)

# 显示TF-IDF矩阵（保留3位小数）
print(\"=== TF-IDF 矩阵 ===\")
print(df_tfidf.round(3))

# 提取每个文档的Top3关键词
print(\"\\n=== 各文档Top3关键词 ===\")
for i, doc in enumerate(corpus):
    top_words = df_tfidf.iloc[i].sort_values(ascending=False).head(3)
    print(f\"文档 {i+1}: {', '.join(top_words.index)}\")""",
        
        "☁️ 可视化 - 词云生成": """import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string

# 示例文本
text = \"Python is a powerful programming language for data science machine learning and artificial intelligence. Python is easy to learn and has a large ecosystem of libraries like pandas numpy matplotlib scikit-learn.\"

# 清理文本（移除标点）
text_clean = text.translate(str.maketrans('', '', string.punctuation)).lower()

# 生成词云
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    colormap='viridis',
    max_words=50,
    contour_width=1,
    contour_color='#4F8BF9'
).generate(text_clean)

# 显示词云
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
plt.tight_layout()
st.pyplot(fig)

# 输出词频Top5
words = text_clean.split()
word_freq = pd.Series(words).value_counts().head(5)
print(\"=== 词频Top5 ===\")
print(word_freq)""",
        
        "🌐 网络爬虫 - 基础网页抓取": """import requests
from bs4 import BeautifulSoup
import pandas as pd

# 目标URL（示例：Python官网首页）
url = \"https://www.python.org\"

# 发送请求（模拟浏览器）
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=5)
    response.encoding = response.apparent_encoding  # 自动识别编码
    
    print(f\"请求成功！状态码: {response.status_code}\")
    
    # 解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 提取页面标题
    title = soup.find('title').get_text(strip=True)
    print(f\"\\n页面标题: {title}\")
    
    # 提取前5个链接
    links = []
    for a in soup.find_all('a', href=True)[:5]:
        links.append({
            '文本': a.get_text(strip=True)[:50],  # 截断长文本
            '链接': a['href']
        })
    
    print(\"\\n=== 前5个链接 ===\")
    print(pd.DataFrame(links))
    
except Exception as e:
    print(f\"请求失败: {str(e)}\")""",
        
        "😊 NLP - 情感分析": """from textblob import TextBlob

# 示例文本（混合情感）
texts = [
    \"I love Python! It's the best programming language ever.\",
    \"The library version conflict is so frustrating and annoying.\",
    \"Data science is interesting but sometimes challenging.\",
    \"Streamlit makes building data apps easy and fun!\"
]

# 逐句分析情感
print(\"=== 情感分析结果 ===\")
results = []
for i, text in enumerate(texts):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # 极性：-1(消极) ~ 1(积极)
    subjectivity = blob.sentiment.subjectivity  # 主观性：0(客观) ~ 1(主观)
    
    # 情感判定
    if polarity > 0.5:
        sentiment = \"非常积极 😍\"
    elif polarity > 0:
        sentiment = \"稍微积极 🙂\"
    elif polarity < -0.5:
        sentiment = \"非常消极 😡\"
    elif polarity < 0:
        sentiment = \"稍微消极 🙁\"
    else:
        sentiment = \"中性 😐\"
    
    results.append({
        '文本': text[:40] + \"...\" if len(text) > 40 else text,
        '极性': round(polarity, 3),
        '主观性': round(subjectivity, 3),
        '情感': sentiment
    })

# 显示结果
print(pd.DataFrame(results).to_string(index=False))"""
    }
    
    # ==================== 模板选择器 ====================
    st.markdown('<div class="template-selector">', unsafe_allow_html=True)
    selected_template = st.selectbox(
        "📋 选择代码模板（替换第一个代码块）",
        options=list(TEMPLATES.keys()),
        index=0,
        key="template_selector"
    )
    
    col_template, col_add = st.columns([1, 1])
    with col_template:
        if st.button("📥 应用模板到第一个代码块", use_container_width=True):
            st.session_state.code_blocks[0]["content"] = TEMPLATES[selected_template]
            st.session_state.code_blocks[0]["output"] = ""
            st.rerun()  # 刷新页面
    
    with col_add:
        if st.button("➕ 添加新的空白代码块", use_container_width=True):
            st.session_state.code_blocks.append({
                "id": st.session_state.next_block_id,
                "content": f"# 新代码块 #{st.session_state.next_block_id}\\n# 在这里编写你的代码...\\nprint('Hello from Block #{st.session_state.next_block_id}!')",
                "output": "",
                "expanded": True
            })
            st.session_state.next_block_id += 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== 执行代码的核心函数 ====================
    def execute_code(code):
        """执行代码并捕获输出（包括print和图表）"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        # 准备执行环境（注入常用库）
        exec_env = {
            'st': st,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'nltk': nltk,
            'spacy': spacy,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'TextBlob': TextBlob,
            'WordCloud': WordCloud,
            'TfidfVectorizer': TfidfVectorizer,
            'nlp': nlp  # 复用已加载的SpaCy模型
        }
        
        # 捕获标准输出/错误
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # 执行代码
                exec(code, exec_env)
            
            # 获取输出
            stdout = output_buffer.getvalue()
            stderr = error_buffer.getvalue()
            
            if stderr:
                return f"❌ 执行错误：\\n{stderr}"
            elif stdout:
                return stdout
            else:
                return "✅ 代码执行成功（无输出）"
        
        except Exception as e:
            return f"❌ 运行时异常：\\n{type(e).__name__}: {str(e)}"
    
    # ==================== 渲染所有代码块 ====================
    for idx, block in enumerate(st.session_state.code_blocks):
        st.markdown(f'<div class="code-block-container">', unsafe_allow_html=True)
        
        # 代码块头部（按钮区）
        st.markdown('<div class="code-block-header">', unsafe_allow_html=True)
        col1, col2 = st.columns([8, 2])
        with col1:
            st.markdown(f"**代码块 #{block['id']}**", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="btn-group">', unsafe_allow_html=True)
            # 运行按钮
            if st.button(f"▶️ 运行", key=f"run_{block['id']}", use_container_width=False):
                block["output"] = execute_code(block["content"])
                st.rerun()
            # 删除按钮（至少保留1个代码块）
            if len(st.session_state.code_blocks) > 1:
                if st.button(f"🗑️ 删除", key=f"delete_{block['id']}", use_container_width=False):
                    st.session_state.code_blocks.pop(idx)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 代码编辑区
        block["content"] = st.text_area(
            label=f"Code Block {block['id']}",
            value=block["content"],
            height=500,
            key=f"code_{block['id']}",
            label_visibility="collapsed"
        )
        
        # 输出区域
        st.markdown(f'<div class="code-output">{block["output"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== 辅助功能 ====================
    st.divider()
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("🧹 清空所有代码块输出", type="secondary"):
            for block in st.session_state.code_blocks:
                block["output"] = ""
            st.rerun()
    with col_reset:
        if st.button("🔄 重置沙盒（恢复初始状态）", type="secondary"):
            del st.session_state.code_blocks
            del st.session_state.next_block_id
            st.rerun()
    
    # 提示信息
    st.info("💡 提示：沙盒已预加载所有课程相关库（NLTK/SpaCy/Scikit-learn等），可直接使用；图表会自动显示在代码块输出区。")

st.markdown("---")
st.caption("© 2026 NLP Course Demo")