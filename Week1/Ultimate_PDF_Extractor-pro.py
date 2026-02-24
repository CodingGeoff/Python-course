import os
import io
import re
import time
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
    else:
        return True
    return False

TESSERACT_AVAILABLE = setup_tesseract()

def generate_safe_filename(original_path, output_dir):
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    return os.path.join(output_dir, f"{name_without_ext}_{timestamp}_{short_uuid}.txt")

# ---------------------------------------------------------
# 2. ★ 新增：高级 NLP 文本清洗引擎 ★
# ---------------------------------------------------------
class SmartTextCleaner:
    @staticmethod
    def clean_block(text):
        """对单个逻辑段落进行深度清洗"""
        if not text or not text.strip():
            return ""

        # 规则 1: 清理特定学术下载水印/页眉页脚 (针对你提供的样本)
        text = re.sub(r'Downloaded from http\S+\s+by guest on.*?\n?', '', text, flags=re.IGNORECASE)
        
        # 规则 2: 修复被连字符打断的英文单词 (例如 "misin-\nformation" -> "misinformation")
        # \xad 是软连字符(soft hyphen)，也需要考虑进去
        text = re.sub(r'([a-zA-Z]+)[-\xad]\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
        
        # 规则 3: 将段落内的剩余单换行符替换为空格（重组自然段）
        text = text.replace('\n', ' ')
        
        # 规则 4: 清除因替换产生的多余连续空格，保留单个空格
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()

# ---------------------------------------------------------
# 3. 稳健型核心处理 Worker (已升级 Block 提取)
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
                self.log_callback("\n⚠️ 任务被用户强行终止！")
                break
                
            self.log_callback(f"\n[{file_idx+1}/{total_files}] 🚀 开始处理: {os.path.basename(pdf_path)}")
            output_path = generate_safe_filename(pdf_path, self.output_dir)
            
            try:
                self._process_single_pdf(pdf_path, output_path, file_idx, total_files)
            except Exception as e:
                self.log_callback(f"❌ 严重错误！文件 {os.path.basename(pdf_path)} 处理失败: {str(e)}")
                continue 

        self.finish_callback()

    def _process_single_pdf(self, pdf_path, output_path, file_idx, total_files):
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        full_text_list = []

        for i, page in enumerate(doc):
            if self.is_cancelled:
                break
                
            try:
                # 首先使用普通文本提取来判断是否为扫描件
                raw_text = page.get_text()
                page_text_output = ""
                
                # 扫描件判定
                if len(raw_text.strip()) < self.scan_threshold:
                    if not TESSERACT_AVAILABLE:
                        self.log_callback(f"  ⚠️ 第 {i+1} 页疑似扫描件，但未配置 Tesseract，提取空白。")
                    else:
                        self.log_callback(f"  🔍 第 {i+1} 页疑似扫描/图表，启动 OCR ({self.ocr_lang})...")
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                        
                        # OCR 出来的文本通常用 \n\n 分隔段落，我们以此为界切分为假定的 block 进行清洗
                        pseudo_blocks = ocr_text.split('\n\n')
                        cleaned_blocks = [SmartTextCleaner.clean_block(b) for b in pseudo_blocks if b.strip()]
                        page_text_output = "\n\n".join(cleaned_blocks)
                else:
                    self.log_callback(f"  📄 第 {i+1} 页提取为结构化纯文本。")
                    # ★ 核心升级：使用 get_text("blocks") 获取物理文本块 ★
                    blocks = page.get_text("blocks")
                    cleaned_blocks = []
                    
                    for b in blocks:
                        # b[6] == 0 代表这是一个文本块（排除图片等）
                        if b[6] == 0:
                            block_text = b[4]
                            # 过滤掉孤立的页码（如单独的一行 "x" 或 "12"）
                            if re.fullmatch(r'^(x|v|i+|\d+)\s*$', block_text.strip(), re.IGNORECASE):
                                continue
                                
                            cleaned = SmartTextCleaner.clean_block(block_text)
                            if cleaned:
                                cleaned_blocks.append(cleaned)
                                
                    # 组合当前页所有段落，段落之间保留两个换行符
                    page_text_output = "\n\n".join(cleaned_blocks)

                full_text_list.append(f"--- Page {i+1} ---\n{page_text_output}\n")
                
            except Exception as page_error:
                self.log_callback(f"  ❌ 第 {i+1} 页解析异常: {str(page_error)}")
                full_text_list.append(f"--- Page {i+1} [EXTRACTION FAILED] ---\n")

            self.gui_callback((file_idx + ((i + 1) / total_pages)) / total_files)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(full_text_list))
            self.log_callback(f"✅ 完成！已安全导出至: \n{output_path}")
        except Exception as io_err:
            self.log_callback(f"❌ 文件保存失败: {str(io_err)}")


# ---------------------------------------------------------
# 4. 极致美观的现代化 GUI 面板 (保持不变)
# ---------------------------------------------------------
class ModernPDFApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("✨ 智能混合型 PDF 文本提取引擎 V2.0 (NLP增强版)")
        self.geometry("900x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.pdf_files = []
        self.worker_thread = None
        self.processor = None

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

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="🚀 开始提取并导出", command=self.start_processing, height=50, fg_color="#2FA572", hover_color="#106A43")
        self.btn_start.grid(row=7, column=0, padx=20, pady=(10, 30))

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.main_frame, text="等待导入文件...", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.console_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.log_to_console("初始化完成。NLP 启发式段落重组已激活。")
        if TESSERACT_AVAILABLE:
            self.log_to_console("✅ 系统检测到 Tesseract OCR 引擎可用。")
        else:
            self.log_to_console("⚠️ 未在标准路径检测到 Tesseract，扫描件提取将被跳过。")

        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progress_bar.set(0)

    def log_to_console(self, msg):
        self.console_textbox.insert("end", msg + "\n")
        self.console_textbox.see("end")

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            for f in files:
                if f not in self.pdf_files:
                    self.pdf_files.append(f)
            self.status_label.configure(text=f"已导入 {len(self.pdf_files)} 个 PDF 文件准备处理")
            self.log_to_console(f"📁 新增导入了 {len(files)} 个文件。")

    def clear_files(self):
        self.pdf_files.clear()
        self.status_label.configure(text="等待导入文件...")
        self.log_to_console("🗑️ 任务列表已清空。")
        self.progress_bar.set(0)

    def update_progress(self, value):
        self.progress_bar.set(value)

    def process_finished(self):
        self.btn_start.configure(state="normal", text="🚀 开始提取并导出")
        self.btn_add_files.configure(state="normal")
        self.status_label.configure(text="🎉 所有任务处理完毕！")
        self.log_to_console("\n============== 任务结束 ==============")
        messagebox.showinfo("成功", "所有 PDF 处理完毕！文本已进行段落重组并保存。")

    def start_processing(self):
        if not self.pdf_files:
            messagebox.showwarning("警告", "请先导入至少一个 PDF 文件！")
            return

        try:
            threshold = int(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("错误", "阈值必须是整数！")
            return

        output_dir = filedialog.askdirectory(title="选择纯文本导出文件夹")
        if not output_dir:
            return

        self.btn_start.configure(state="disabled", text="⚙️ 处理中...")
        self.btn_add_files.configure(state="disabled")
        self.progress_bar.set(0)
        self.console_textbox.delete("1.0", "end")
        self.log_to_console("🚀 引擎启动！开始混合处理与 NLP 清洗流程...")

        ocr_lang = self.lang_option.get()

        self.processor = PDFProcessorWorker(
            pdf_paths=self.pdf_files,
            output_dir=output_dir,
            scan_threshold=threshold,
            ocr_lang=ocr_lang,
            gui_callback=lambda v: self.after(0, self.update_progress, v),
            log_callback=lambda msg: self.after(0, self.log_to_console, msg),
            finish_callback=lambda: self.after(0, self.process_finished)
        )
        
        self.worker_thread = threading.Thread(target=self.processor.run, daemon=True)
        self.worker_thread.start()

if __name__ == "__main__":
    app = ModernPDFApp()
    app.mainloop()