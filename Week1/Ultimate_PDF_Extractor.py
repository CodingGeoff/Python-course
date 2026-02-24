import os
import io
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
# 1. Tesseract 引擎自动寻路机制 (防报错核心)
# ---------------------------------------------------------
def setup_tesseract():
    # 尝试自动寻找 Tesseract，如果找不到，程序也不会崩溃，而是优雅降级
    if os.name == 'nt':  # Windows 自动探测
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
    else: # Mac/Linux 默认在环境变量中
        return True
    return False

TESSERACT_AVAILABLE = setup_tesseract()

# ---------------------------------------------------------
# 2. 绝对防重名文件生成器
# ---------------------------------------------------------
def generate_safe_filename(original_path, output_dir):
    """生成绝对不会重名的输出文件路径"""
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 时间戳 (YYYYMMDD_HHMMSS) + 短 UUID (4位)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    
    safe_name = f"{name_without_ext}_{timestamp}_{short_uuid}.txt"
    return os.path.join(output_dir, safe_name)

# ---------------------------------------------------------
# 3. 稳健型核心处理 Worker
# ---------------------------------------------------------
class PDFProcessorWorker:
    def __init__(self, pdf_paths, output_dir, scan_threshold, ocr_lang, gui_callback, log_callback, finish_callback):
        self.pdf_paths = pdf_paths
        self.output_dir = output_dir
        self.scan_threshold = scan_threshold
        self.ocr_lang = ocr_lang
        self.gui_callback = gui_callback  # 更新进度条的回调
        self.log_callback = log_callback  # 输出日志的回调
        self.finish_callback = finish_callback # 完成时的回调
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
                continue # 就算整个文件崩溃，也绝不影响下一个文件

        self.finish_callback()

    def _process_single_pdf(self, pdf_path, output_path, file_idx, total_files):
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        full_text_list = []

        for i, page in enumerate(doc):
            if self.is_cancelled:
                break
                
            try:
                # 步骤 1: 尝试直接提取
                text = page.get_text()
                
                # 步骤 2: 智能判定扫描页
                if len(text.strip()) < self.scan_threshold:
                    if not TESSERACT_AVAILABLE:
                        self.log_callback(f"  ⚠️ 第 {i+1} 页疑似扫描件，但未配置 Tesseract，提取空白。")
                    else:
                        self.log_callback(f"  🔍 第 {i+1} 页疑似扫描/图表，启动 OCR ({self.ocr_lang})...")
                        # 2倍超清渲染
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        # 静默处理 OCR，防止报错
                        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                else:
                    self.log_callback(f"  📄 第 {i+1} 页提取为纯文本。")

                full_text_list.append(f"--- Page {i+1} ---\n{text.strip()}\n")
                
            except Exception as page_error:
                self.log_callback(f"  ❌ 第 {i+1} 页解析异常: {str(page_error)}")
                full_text_list.append(f"--- Page {i+1} [EXTRACTION FAILED] ---\n")

            # 更新整体进度
            overall_progress = (file_idx + ((i + 1) / total_pages)) / total_files
            self.gui_callback(overall_progress)

        # 写入绝对安全的文件路径
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(full_text_list))
            self.log_callback(f"✅ 完成！已安全导出至: \n{output_path}")
        except Exception as io_err:
            self.log_callback(f"❌ 文件保存失败: {str(io_err)}")


# ---------------------------------------------------------
# 4. 极致美观的现代化 GUI 面板
# ---------------------------------------------------------
class ModernPDFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 主题配置
        ctk.set_appearance_mode("Dark")  # 深色护眼模式
        ctk.set_default_color_theme("blue")

        self.title("✨ 智能混合型 PDF 文本提取引擎 V1.0")
        self.geometry("900x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.pdf_files = []
        self.worker_thread = None
        self.processor = None

        self.setup_ui()

    def setup_ui(self):
        # 左侧控制面板
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="PDF Core UI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.btn_add_files = ctk.CTkButton(self.sidebar_frame, text="📁 导入 PDF 文件", command=self.add_files, height=40)
        self.btn_add_files.grid(row=1, column=0, padx=20, pady=10)

        self.btn_clear_files = ctk.CTkButton(self.sidebar_frame, text="🗑️ 清空列表", command=self.clear_files, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_clear_files.grid(row=2, column=0, padx=20, pady=10)

        # 参数设置区域
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

        # 右侧内容面板
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 状态面板
        self.status_label = ctk.CTkLabel(self.main_frame, text="等待导入文件...", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # 终端风格日志输出
        self.console_textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # 初始系统日志
        self.log_to_console("初始化完成。等待就绪。")
        if TESSERACT_AVAILABLE:
            self.log_to_console("✅ 系统检测到 Tesseract OCR 引擎可用。")
        else:
            self.log_to_console("⚠️ 未在标准路径检测到 Tesseract，扫描件提取将被跳过。请确保已安装。")

        # 进度条
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, height=15)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.progress_bar.set(0)

    def log_to_console(self, msg):
        """线程安全的日志更新"""
        self.console_textbox.insert("end", msg + "\n")
        self.console_textbox.see("end") # 自动滚动到底部

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
        """线程安全的进度条更新"""
        self.progress_bar.set(value)

    def process_finished(self):
        """线程安全的完成状态更新"""
        self.btn_start.configure(state="normal", text="🚀 开始提取并导出")
        self.btn_add_files.configure(state="normal")
        self.status_label.configure(text="🎉 所有任务处理完毕！")
        self.log_to_console("\n============== 任务结束 ==============")
        messagebox.showinfo("成功", "所有 PDF 处理完毕！提取的文本已保存。")

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

        # 锁定 UI 防止重复点击
        self.btn_start.configure(state="disabled", text="⚙️ 处理中...")
        self.btn_add_files.configure(state="disabled")
        self.progress_bar.set(0)
        self.console_textbox.delete("1.0", "end")
        self.log_to_console("🚀 引擎启动！开始混合处理流程...")

        # 提取参数
        ocr_lang = self.lang_option.get()

        # 核心：将耗时操作扔进后台线程，绝不卡死界面
        self.processor = PDFProcessorWorker(
            pdf_paths=self.pdf_files,
            output_dir=output_dir,
            scan_threshold=threshold,
            ocr_lang=ocr_lang,
            gui_callback=lambda v: self.after(0, self.update_progress, v), # 使用 after 确保线程安全
            log_callback=lambda msg: self.after(0, self.log_to_console, msg),
            finish_callback=lambda: self.after(0, self.process_finished)
        )
        
        self.worker_thread = threading.Thread(target=self.processor.run, daemon=True)
        self.worker_thread.start()

if __name__ == "__main__":
    app = ModernPDFApp()
    app.mainloop()