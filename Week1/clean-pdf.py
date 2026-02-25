import os
import io
import re
import uuid
import threading
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ---------------------------------------------------------
# 1. Tesseract 引擎自动寻路机制
# ---------------------------------------------------------
def setup_tesseract():
    if os.name == 'nt':
        potential_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'D:\Program Files\Tesseract-OCR\tesseract.exe'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
    return True if os.name != 'nt' else False

TESSERACT_AVAILABLE = setup_tesseract()

def generate_safe_filename(original_path, output_dir):
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    return os.path.join(output_dir, f"{name_without_ext}_{timestamp}_{short_uuid}.txt")

# ---------------------------------------------------------
# 2. ★ 终极升级：多维 NLP 与空间感知清洗引擎 ★
# ---------------------------------------------------------
class UltimateTextCleaner:
    @staticmethod
    def is_noise_block(text, y0, y1, page_height):
        """基于空间坐标和NLP规则判定是否为干扰块 (页眉/页脚/署名/日期)"""
        text = text.strip()
        if not text:
            return True

        # 1. 空间坐标判定：位于页面极高或极低处的短文本，大概率为页眉页脚
        is_top = y0 < (page_height * 0.12)
        is_bottom = y1 > (page_height * 0.88)
        word_count = len(text.split())

        if (is_top or is_bottom) and word_count < 15:
            return True

        # 2. 正则模式识别：匹配孤立的日期 (如: 25 November 2025, Nov 25 2025)
        if re.fullmatch(r'^(?:[0-3]?\d\s+)?[A-Z][a-z]{2,8}\s+\d{4}$', text):
            return True

        # 3. 正则模式识别：匹配包含罗马数字的前缀或孤立页码 (如: x Series Editor’s Introduction)
        if re.match(r'^([xvi]+|\d+)\s+([A-Z].*)?$', text, re.IGNORECASE) and word_count < 8:
            return True
            
        # 4. 匹配学术文章特有的下载水印戳
        if re.search(r'Downloaded from http', text, re.IGNORECASE):
            return True

        # 5. NLP 句法试探：判定短署名或书名 (词数极少，无标点结尾，且首字母大写密集)
        if word_count < 6 and not text[-1] in ".?!\"'":
            # 计算大写字母开头的单词比例
            words = text.split()
            title_case_words = sum(1 for w in words if w.istitle())
            if title_case_words / len(words) > 0.6:  # 如果大部分词首字母大写，多半是人名/书名
                return True

        return False

    @staticmethod
    def heal_text(text):
        """修复文本内的连字符和多余换行"""
        # 修复连字符换行断词: "misin-\nformation" -> "misinformation"
        text = re.sub(r'([a-zA-Z]+)[-\xad]\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
        # 将段内剩余换行转为空格
        text = text.replace('\n', ' ')
        # 压缩多余空格
        return re.sub(r'\s{2,}', ' ', text).strip()

# ---------------------------------------------------------
# 3. 稳健型核心处理 Worker (支持跨页缝合)
# ---------------------------------------------------------
class PDFProcessorWorker:
    def __init__(self, pdf_paths, output_dir, scan_threshold, ocr_lang, gui_callback, log_callback, finish_callback):
        self.pdf_paths = pdf_paths
        self.output_dir = output_dir
        self.scan_threshold = scan_threshold
        self.ocr_lang = ocr_lang
        self.gui_callback = gui_callback
        self.log_callback = log_callback
        self.finish_callback = finish_callback
        self.is_cancelled = False

    def run(self):
        total_files = len(self.pdf_paths)
        for file_idx, pdf_path in enumerate(self.pdf_paths):
            if self.is_cancelled:
                break
            
            self.log_callback(f"\n[{file_idx+1}/{total_files}] 🚀 开始提取与深度清洗: {os.path.basename(pdf_path)}")
            output_path = generate_safe_filename(pdf_path, self.output_dir)
            
            try:
                self._process_single_pdf(pdf_path, output_path, file_idx, total_files)
            except Exception as e:
                self.log_callback(f"❌ 错误: {str(e)}")
                continue 

        self.finish_callback()

    def _process_single_pdf(self, pdf_path, output_path, file_idx, total_files):
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        final_document_text = ""
        previous_text_ends_incomplete = False # 用于跨页无缝缝合的标记

        for i, page in enumerate(doc):
            if self.is_cancelled: break
                
            try:
                raw_text = page.get_text()
                page_height = page.rect.height
                page_blocks_text = []
                
                # 扫描件判定机制保持不变...
                if len(raw_text.strip()) < self.scan_threshold:
                    self.log_callback(f"  🔍 第 {i+1} 页启用 OCR ({self.ocr_lang})...")
                    if TESSERACT_AVAILABLE:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                        blocks = ocr_text.split('\n\n')
                        cleaned_blocks = [UltimateTextCleaner.heal_text(b) for b in blocks if b.strip()]
                        page_blocks_text = cleaned_blocks
                else:
                    self.log_callback(f"  📄 第 {i+1} 页空间结构解析中...")
                    blocks = page.get_text("blocks")
                    
                    # 按 Y 轴坐标排序，确保阅读顺序
                    blocks.sort(key=lambda b: (b[1], b[0])) 
                    
                    for b in blocks:
                        if b[6] == 0:  # 类型0为纯文本块
                            x0, y0, x1, y1, block_text = b[0], b[1], b[2], b[3], b[4]
                            
                            # ★ 核心：空间域与规则联合过滤噪音 ★
                            if UltimateTextCleaner.is_noise_block(block_text, y0, y1, page_height):
                                continue
                                
                            cleaned = UltimateTextCleaner.heal_text(block_text)
                            if cleaned:
                                page_blocks_text.append(cleaned)

                # ★ 核心：跨页跨块的自然语言缝合逻辑 ★
                for block_idx, text_chunk in enumerate(page_blocks_text):
                    if not text_chunk: continue
                    
                    # 判断当前块的开头是否为小写字母
                    starts_with_lower = text_chunk[0].islower() if text_chunk else False
                    
                    if previous_text_ends_incomplete and (starts_with_lower or text_chunk[0] in ",;:'\""):
                        # 如果上一块没结束，且这一块是小写开头，说明是一句话被切断了，直接空格缝合
                        final_document_text += " " + text_chunk
                    else:
                        # 否则作为新段落换行拼接
                        if final_document_text:
                            final_document_text += "\n\n"
                        final_document_text += text_chunk
                    
                    # 更新状态变量：判断这一块是不是“未完待续”
                    if text_chunk[-1] not in ".?!\"'":
                        previous_text_ends_incomplete = True
                    else:
                        previous_text_ends_incomplete = False

            except Exception as page_error:
                self.log_callback(f"  ❌ 第 {i+1} 页解析异常: {str(page_error)}")

            self.gui_callback((file_idx + ((i + 1) / total_pages)) / total_files)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_document_text.strip())
        self.log_callback(f"✅ 提取完成！已安全导出至: \n{output_path}")

# ---------------------------------------------------------
# 4. GUI 面板 (保持极简与美观)
# ---------------------------------------------------------
class ModernPDFApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("✨ 智能PDF文本解析引擎 V3.0 (终极纯净版)")
        self.geometry("900x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.pdf_files = []
        self.setup_ui()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="PDF Core UI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.btn_add_files = ctk.CTkButton(self.sidebar_frame, text="📁 导入 PDF 文件", command=self.add_files, height=40)
        self.btn_add_files.grid(row=1, column=0, padx=20, pady=10)

        self.btn_clear_files = ctk.CTkButton(self.sidebar_frame, text="🗑️ 清空列表", command=self.clear_files, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_clear_files.grid(row=2, column=0, padx=20, pady=10)

        self.label_lang = ctk.CTkLabel(self.sidebar_frame, text="OCR 识别语言:")
        self.label_lang.grid(row=3, column=0, padx=20, pady=(20, 0), sticky="w")
        self.lang_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["eng", "chi_sim", "eng+chi_sim"])
        self.lang_option.set("eng+chi_sim")
        self.lang_option.grid(row=4, column=0, padx=20, pady=10)

        self.label_threshold = ctk.CTkLabel(self.sidebar_frame, text="扫描件触发阈值 (字符数):")
        self.label_threshold.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.threshold_entry = ctk.CTkEntry(self.sidebar_frame)
        self.threshold_entry.insert(0, "50")
        self.threshold_entry.grid(row=6, column=0, padx=20, pady=10, sticky="n")

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="🚀 启动深度净化与导出", command=self.start_processing, height=50, fg_color="#2FA572", hover_color="#106A43")
        self.btn_start.grid(row=7, column=0, padx=20, pady=(10, 30))

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.main_frame, text="等待导入文件...", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.console_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.log_to_console("初始化完成。多维空间与 NLP 深度过滤系统已激活。")
        if TESSERACT_AVAILABLE:
            self.log_to_console("✅ 检测到 Tesseract，自动图文识别处于就绪状态。")

        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progress_bar.set(0)

    def log_to_console(self, msg):
        self.console_textbox.insert("end", msg + "\n")
        self.console_textbox.see("end")

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            for f in self.pdf_files[:]:
                pass
            self.pdf_files.extend([f for f in files if f not in self.pdf_files])
            self.status_label.configure(text=f"已导入 {len(self.pdf_files)} 个 PDF 文件准备处理")
            self.log_to_console(f"📁 新增导入了 {len(files)} 个文件。")

    def clear_files(self):
        self.pdf_files.clear()
        self.status_label.configure(text="等待导入文件...")
        self.log_to_console("🗑️ 任务列表已清空。")
        self.progress_bar.set(0)

    def start_processing(self):
        if not self.pdf_files:
            messagebox.showwarning("警告", "请先导入至少一个 PDF 文件！")
            return
        output_dir = filedialog.askdirectory(title="选择纯文本导出文件夹")
        if not output_dir:
            return

        self.btn_start.configure(state="disabled", text="⚙️ 净化处理中...")
        self.btn_add_files.configure(state="disabled")
        self.progress_bar.set(0)
        self.console_textbox.delete("1.0", "end")
        
        self.processor = PDFProcessorWorker(
            self.pdf_files, output_dir, int(self.threshold_entry.get()), self.lang_option.get(),
            lambda v: self.after(0, self.progress_bar.set, v),
            lambda msg: self.after(0, self.log_to_console, msg),
            lambda: self.after(0, self.process_finished)
        )
        threading.Thread(target=self.processor.run, daemon=True).start()

    def process_finished(self):
        self.btn_start.configure(state="normal", text="🚀 启动深度净化与导出")
        self.btn_add_files.configure(state="normal")
        self.status_label.configure(text="🎉 所有任务净化处理完毕！")
        self.log_to_console("\n============== 任务结束 ==============")

if __name__ == "__main__":
    app = ModernPDFApp()
    app.mainloop()