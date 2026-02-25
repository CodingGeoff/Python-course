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
            r'C:/Software/Tesseract-OCR/tesseract.exe',
            r'C:/Program Files/Tesseract-OCR/tesseract.exe',
            r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe',
            r'D:/Program Files/Tesseract-OCR/tesseract.exe'
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
# 2. ★ 升级版：带学术引用处理与透明拦截的清洗引擎 ★
# ---------------------------------------------------------
class UltimateTextCleaner:
    @staticmethod
    def inspect_block(text, y0, y1, page_height, safe_mode):
        """审查文本块，决定是保留还是拦截，并返回拦截原因"""
        text = text.strip()
        if not text:
            return True, "空白符"

        # 1. 绝对垃圾信息过滤
        if re.search(r'Downloaded from http', text, re.IGNORECASE):
            return True, "学术下载水印"

        if re.fullmatch(r'^(?:[0-3]?\d\s+)?[A-Z][a-z]{2,8}\s+\d{4}$', text):
            return True, "孤立日期"

        if re.match(r'^([xvi]+|\d+)\s*$', text, re.IGNORECASE):
            return True, "孤立页码"

        # 2. 大写标题免死金牌 (即使在边缘也不拦截)
        # 例如 "HOW DOES NATIVE ADVERTISING AFFECT SOCIETY AND DEMOCRACY?"
        if text.isupper() and len(text) > 5:
            return False, ""

        # 3. 空间位置过滤 (顶部 8% 或 底部 8% 的极短文本)
        is_top = y0 < (page_height * 0.08)
        is_bottom = y1 > (page_height * 0.08)  # 修正：y1 > page_height * 0.92, 这里稍作冗余判定
        is_bottom = y1 > (page_height * 0.92)
        word_count = len(text.split())

        if (is_top or is_bottom) and word_count < 10:
            return True, "边缘页眉/页脚"

        # 4. 严苛模式下的句法过滤 (安全模式下关闭，防止误杀短标题)
        if not safe_mode:
            if word_count < 6 and not text[-1] in ".?!\"'":
                words = text.split()
                title_case_words = sum(1 for w in words if w.istitle())
                if words and (title_case_words / len(words) > 0.6):
                    return True, "无标点首字母大写(疑似署名)"

        return False, ""

    @staticmethod
    def format_citations(text):
        """
        智能处理学术引用数字。
        将单词后紧跟标点和数字的格式 (例如 industry.67)
        转化为标准纯文本带括号格式 (例如 industry. [67])
        """
        # 匹配: 至少2个字母 + 标点(.,!?"') + 1到3位数字 + (空格或行尾)
        text = re.sub(r'([a-zA-Z]{2,}[\.\,\?!\'"]+)(\d{1,3})(?=\s|$)', r'\1 [\2]', text)
        return text

    @staticmethod
    def heal_text(text):
        """修复文本内的连字符、多余换行，并格式化引用"""
        # 如果是全大写标题，直接空格缝合所有行
        if text.isupper():
            text = text.replace('\n', ' ')
        else:
            # 修复连字符换行断词
            text = re.sub(r'([a-zA-Z]+)[-\xad]\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
            # 将段内剩余换行转为空格
            text = text.replace('\n', ' ')
        
        # 处理文内引用数字
        text = UltimateTextCleaner.format_citations(text)
        
        # 压缩多余空格
        return re.sub(r'\s{2,}', ' ', text).strip()

# ---------------------------------------------------------
# 3. 稳健型核心处理 Worker
# ---------------------------------------------------------
class PDFProcessorWorker:
    def __init__(self, pdf_paths, output_dir, scan_threshold, ocr_lang, safe_mode, gui_callback, log_callback, finish_callback):
        self.pdf_paths = pdf_paths
        self.output_dir = output_dir
        self.scan_threshold = scan_threshold
        self.ocr_lang = ocr_lang
        self.safe_mode = safe_mode
        self.gui_callback = gui_callback
        self.log_callback = log_callback
        self.finish_callback = finish_callback
        self.is_cancelled = False

    def run(self):
        total_files = len(self.pdf_paths)
        for file_idx, pdf_path in enumerate(self.pdf_paths):
            if self.is_cancelled: break
            
            self.log_callback(f"\n[{file_idx+1}/{total_files}] 🚀 开始提取: {os.path.basename(pdf_path)}")
            output_path = generate_safe_filename(pdf_path, self.output_dir)
            
            try:
                self._process_single_pdf(pdf_path, output_path, file_idx, total_files)
            except Exception as e:
                self.log_callback(f"❌ 严重错误跳过: {str(e)}")
                continue 

        self.finish_callback()

    def _process_single_pdf(self, pdf_path, output_path, file_idx, total_files):
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        final_document_text = ""
        previous_text_ends_incomplete = False

        for i, page in enumerate(doc):
            if self.is_cancelled: break
                
            try:
                raw_text = page.get_text()
                page_height = page.rect.height
                page_blocks_text = []
                
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
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0])) 
                    
                    for b in blocks:
                        if b[6] == 0:
                            x0, y0, x1, y1, block_text = b[0], b[1], b[2], b[3], b[4]
                            
                            # 进行审查并获取原因
                            is_noise, reason = UltimateTextCleaner.inspect_block(block_text, y0, y1, page_height, self.safe_mode)
                            
                            if is_noise:
                                # 核心要求：明确告知用户过滤了什么
                                preview_text = block_text.replace('\n', ' ').strip()[:30]
                                if preview_text:
                                    self.log_callback(f"    🗑️ 拦截 [{reason}]: {preview_text}...")
                                continue
                                
                            cleaned = UltimateTextCleaner.heal_text(block_text)
                            if cleaned:
                                page_blocks_text.append(cleaned)

                # 跨页缝合与标题排版逻辑
                for text_chunk in page_blocks_text:
                    if not text_chunk: continue
                    
                    is_heading = text_chunk.isupper() and len(text_chunk) > 5
                    
                    if is_heading:
                        # 如果是标题，强制独立段落
                        final_document_text += f"\n\n{text_chunk}\n\n"
                        previous_text_ends_incomplete = False
                    else:
                        starts_with_lower = text_chunk[0].islower()
                        
                        if previous_text_ends_incomplete and (starts_with_lower or text_chunk[0] in ",;:'\""):
                            # 缝合上一句
                            final_document_text += " " + text_chunk
                        else:
                            # 新起一段
                            final_document_text += ("\n\n" if final_document_text else "") + text_chunk
                        
                        # 判定结尾
                        previous_text_ends_incomplete = text_chunk[-1] not in ".?!\"'"

            except Exception as page_error:
                self.log_callback(f"  ❌ 第 {i+1} 页解析异常: {str(page_error)}")

            self.gui_callback((file_idx + ((i + 1) / total_pages)) / total_files)

        with open(output_path, 'w', encoding='utf-8') as f:
            # 清理多余的连续换行
            cleaned_final_text = re.sub(r'\n{3,}', '\n\n', final_document_text.strip())
            f.write(cleaned_final_text)
        self.log_callback(f"✅ 提取完成！已导出至: \n{output_path}")

# ---------------------------------------------------------
# 4. GUI 面板 (新增安全模式切换)
# ---------------------------------------------------------
class ModernPDFApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("✨ 智能PDF文本解析引擎 V4.0 (防误杀与引用保留版)")
        self.geometry("950x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.pdf_files = []
        self.setup_ui()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="PDF Core UI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.btn_add_files = ctk.CTkButton(self.sidebar_frame, text="📁 导入 PDF 文件", command=self.add_files, height=40)
        self.btn_add_files.grid(row=1, column=0, padx=20, pady=10)

        self.btn_clear_files = ctk.CTkButton(self.sidebar_frame, text="🗑️ 清空列表", command=self.clear_files, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_clear_files.grid(row=2, column=0, padx=20, pady=10)

        self.label_lang = ctk.CTkLabel(self.sidebar_frame, text="OCR 识别语言:")
        self.label_lang.grid(row=3, column=0, padx=20, pady=(15, 0), sticky="w")
        self.lang_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["eng", "chi_sim", "eng+chi_sim"])
        self.lang_option.set("eng+chi_sim")
        self.lang_option.grid(row=4, column=0, padx=20, pady=10)

        self.label_threshold = ctk.CTkLabel(self.sidebar_frame, text="扫描件判定阈值:")
        self.label_threshold.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.threshold_entry = ctk.CTkEntry(self.sidebar_frame)
        self.threshold_entry.insert(0, "50")
        self.threshold_entry.grid(row=6, column=0, padx=20, pady=5, sticky="n")

        # 新增：安全模式复选框
        self.safe_mode_var = ctk.BooleanVar(value=True)
        self.safe_mode_checkbox = ctk.CTkCheckBox(self.sidebar_frame, text="安全模式 (保留短标题/防误删)", variable=self.safe_mode_var)
        self.safe_mode_checkbox.grid(row=7, column=0, padx=20, pady=15, sticky="w")

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="🚀 启动透明化解析", command=self.start_processing, height=50, fg_color="#2FA572", hover_color="#106A43")
        self.btn_start.grid(row=8, column=0, padx=20, pady=(10, 30), sticky="s")

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.main_frame, text="等待导入文件...", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.console_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.log_to_console("初始化完成。拦截动作将在控制台透明化输出。")
        self.log_to_console("✅ 学术文内引用数字 (如 industry.67) 智能转换已就绪。")
        self.log_to_console("✅ 大写标题免死金牌机制已生效。")

        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progress_bar.set(0)

    def log_to_console(self, msg):
        self.console_textbox.insert("end", msg + "\n")
        self.console_textbox.see("end")

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            for f in self.pdf_files[:]: pass
            self.pdf_files.extend([f for f in files if f not in self.pdf_files])
            self.status_label.configure(text=f"已导入 {len(self.pdf_files)} 个 PDF 文件准备处理")
            self.log_to_console(f"📁 新增导入 {len(files)} 个文件。")

    def clear_files(self):
        self.pdf_files.clear()
        self.status_label.configure(text="等待导入文件...")
        self.log_to_console("🗑️ 列表已清空。")
        self.progress_bar.set(0)

    def start_processing(self):
        if not self.pdf_files:
            messagebox.showwarning("警告", "请先导入至少一个 PDF 文件！")
            return
        output_dir = filedialog.askdirectory(title="选择纯文本导出文件夹")
        if not output_dir:
            return

        self.btn_start.configure(state="disabled", text="⚙️ 处理中...")
        self.btn_add_files.configure(state="disabled")
        self.progress_bar.set(0)
        self.console_textbox.delete("1.0", "end")
        
        self.processor = PDFProcessorWorker(
            pdf_paths=self.pdf_files, 
            output_dir=output_dir, 
            scan_threshold=int(self.threshold_entry.get()), 
            ocr_lang=self.lang_option.get(),
            safe_mode=self.safe_mode_var.get(),
            gui_callback=lambda v: self.after(0, self.progress_bar.set, v),
            log_callback=lambda msg: self.after(0, self.log_to_console, msg),
            finish_callback=lambda: self.after(0, self.process_finished)
        )
        threading.Thread(target=self.processor.run, daemon=True).start()

    def process_finished(self):
        self.btn_start.configure(state="normal", text="🚀 启动透明化解析")
        self.btn_add_files.configure(state="normal")
        self.status_label.configure(text="🎉 所有任务处理完毕！")
        self.log_to_console("\n============== 任务结束 ==============")

if __name__ == "__main__":
    app = ModernPDFApp()
    app.mainloop()