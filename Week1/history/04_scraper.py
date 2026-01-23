# 完整可运行版本（兼容普通Python/Jupyter环境）
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# ========== 1. 基础配置 ==========
# 设置pandas显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 目标URL和请求头
URL = "https://www.python.org"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br'
}

# 初始化response变量（解决作用域问题）
response = None

# ========== 2. 发送HTTP请求 ==========
try:
    start_time = time.time()
    response = requests.get(URL, headers=headers, timeout=10)
    response.raise_for_status()  # 触发4xx/5xx错误
    print(f"✅ 请求成功! 状态码: {response.status_code}")
    print(f"⏱️ 耗时: {time.time() - start_time:.4f} 秒")
except requests.exceptions.RequestException as e:
    print(f"❌ 请求失败: {e}")
    exit()  # 请求失败则终止程序

# ========== 3. 解析页面内容 ==========
soup = BeautifulSoup(response.text, 'html.parser')
data_list = []

# 3.1 提取导航链接（适配当前页面结构）
# 修正导航选择器：匹配python.org的主导航
nav_links = soup.select('div.top-bar > nav > ul > li > a')
for link in nav_links:
    text = link.get_text(strip=True)
    href = link.get('href')
    if href and href.startswith('/'):
        href = URL + href
    if text:
        # 统一列结构：Category/Text/URL/Date
        data_list.append({
            'Category': 'Nav',
            'Text': text,
            'URL': href,
            'Date': None
        })

# 3.2 提取最新新闻（修正选择器，适配当前页面）
# python.org的Latest News区块选择器：.list-recent-events > li
news_items = soup.select('.list-recent-events > li')
for item in news_items:
    try:
        # 提取日期（time标签的datetime属性更稳定）
        date_elem = item.find('time')
        date = date_elem.get('datetime').split('T')[0] if date_elem else None
        
        # 提取新闻标题
        title_elem = item.find('h3', class_='event-title') or item.find('a')
        title = title_elem.get_text(strip=True) if title_elem else None
        
        # 提取新闻链接
        news_href = title_elem.get('href') if title_elem else None
        if news_href and news_href.startswith('/'):
            news_href = URL + news_href
        
        if title and date:
            data_list.append({
                'Category': 'News',
                'Text': title,
                'URL': news_href,
                'Date': date
            })
    except AttributeError as e:
        print(f"⚠️ 单条新闻解析失败: {e}")
        continue

# ========== 4. 数据处理与展示 ==========
# 创建DataFrame（统一列结构）
df = pd.DataFrame(data_list)
print(f"\n=== 抓取结果 ({len(df)} 条) ===")

# 兼容Jupyter/普通Python环境的展示方式
def show_data(data):
    try:
        # Jupyter环境优先用display
        display(data)
    except NameError:
        # 普通环境用print
        print(data)

show_data(df.head(10))

# 过滤新闻数据
news_df = df[df['Category'] == 'News'].dropna(subset=['Date'])
print("\n=== 最新新闻 ===")
show_data(news_df)

# ========== 5. 保存数据 ==========
df.to_csv('python_org_data.csv', index=False)
print("\n✅ 数据已保存到 python_org_data.csv")