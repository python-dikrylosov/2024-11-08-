import tkinter as tk
from tkinter import ttk
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import base64
import io
from fpdf import FPDF
import messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python Детектирование нарушений правил дорожного движения")
        self.geometry("1500x600")
        # Создаем контейнер для вкладок
        tab_control = ttk.Notebook(self)
        # Добавляем страницы
        self.tab1 = ttk.Frame(tab_control)
        self.tab2 = ttk.Frame(tab_control)
        self.tab3 = ttk.Frame(tab_control)
        self.tab4 = ttk.Frame(tab_control)
        tab_control.add(self.tab1, text='Главная')
        tab_control.add(self.tab2, text='Загрузка Видео')
        tab_control.add(self.tab3, text='Результаты Анализа')
        tab_control.add(self.tab4, text='География')
        tab_control.pack(expand=True, fill='both')
        # Главная страница
        label = tk.Label(
            self.tab1,
            text="Добро пожаловать в программу анализа видео!\n"
                 "ОАО «РЖД» владеет одним из самых крупных \nкорпоративных парков транспортных средств в России. \n"
                 "В процессе эксплуатации транспортных средств \nвыявляются случаи не соблюдения водителями правил дорожного движения, \n"
                 "что влечет за собой учащение аварийных ситуаций \nи расходы на оплату административных штрафов.\n"
                 "Создание прототипа программного обеспечения, \nпозволяющего с помощью искусственного интеллекта \n"
                 "анализировать видеозаписи с видеорегистратора \n"
                 "установленного в автотранспортном средстве \n"
                 "для выявления нарушений правил дорожного движения.",
            font=("Arial", 16),
            fg="red",bg=None
        )
        label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        # Страница загрузки видео
        self.file_path = None
        self.video_frame = None
        self.cap = None
        self.video_loaded = False
        load_button = tk.Button(
            self.tab2,
            text="Выберите файл видео",
            command=self.load_video
        )
        load_button.pack(pady=10)
        self.progress_label = tk.Label(self.tab2, text="", font=("Arial", 12), fg="green")
        self.progress_label.pack(pady=10)
        # Индикатор выполнения
        self.progress_bar = ttk.Progressbar(self.tab2, length=200, mode='determinate')
        self.progress_bar.pack(pady=10)
        # Страница результатов анализа
        self.table_frame = tk.Frame(self.tab3)
        self.graph_frame = tk.Frame(self.tab3)
        self.table_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.graph_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        # Переменные для хранения результатов анализа
        self.analysis_data = None
        self.analysis_complete = False
    def load_video(self):
        file_types = [
            ("Video files", "*.mp4"),
            ("All files", "*.*")
        ]
        self.file_path = tk.filedialog.askopenfilename(filetypes=file_types)
        if self.file_path:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                print("Ошибка при открытии файла.")
                return
            self.video_loaded = True
            self.progress_label.config(text=f"Видеофайл успешно загружен: {self.file_path}")
            self.start_analysis()  # Запуск анализа в отдельном потоке
    def start_analysis(self):
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Анализ видеофайла...")
        thread = threading.Thread(target=self.analyze_video)
        thread.start()
    def update_progress(self, progress):
        self.progress_bar["value"] = progress
        self.progress_label.config(text=f"Прогресс анализа: {progress}%")
    def analyze_video(self):
        if self.video_loaded:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            duration = frame_count / fps
            data = {
                'Frame': list(range(frame_count)),
                'Time (sec)': [(i / fps) for i in range(frame_count)]
            }
            df = pd.DataFrame(data)
            # Симуляция длительного анализа
            for i in range(frame_count):
                self.update_progress(int((i + 1) / frame_count * 100))  # Обновление прогресса
                self.after(10)  # Задержка для имитации обработки
            # Сохранение результатов анализа
            self.analysis_data = df
            self.analysis_complete = True
            self.progress_label.config(text="Анализ завершён!")
            self.show_results()  # Показать результаты на странице "Результаты"
    # Отображение результатов анализа
    def show_results(self):
        if self.analysis_complete:
            # Очистить фреймы от предыдущих виджетов
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            # Таблица данных
            table = ttk.Treeview(self.table_frame)
            table['columns'] = ('Frame', 'Time (sec)')
            table.column('#0', width=0, stretch=False)
            table.heading('#0', text='', anchor=tk.CENTER)
            table.column('Frame', anchor=tk.CENTER, width=80)
            table.column('Time (sec)', anchor=tk.CENTER, width=80)
            table.heading('Frame', text='Кадр', anchor=tk.CENTER)
            table.heading('Time (sec)', text='Время (сек)', anchor=tk.CENTER)
            index = 0
            for row in self.analysis_data.itertuples(index=False):
                table.insert(parent='', index='end', iid=index,
                             values=(row[0], round(row[1], 2)))#(row.Frame, round(row['Time (sec)'], 2)))  # Исправлено!
                index += 1

            table.pack(pady=20)
            # График зависимости времени от кадра
            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            ax.plot(self.analysis_data['Frame'], self.analysis_data['Time (sec)'])
            ax.set_xlabel('Кадры')
            ax.set_ylabel('Время (с)')
            ax.set_title('Зависимость времени от номера кадра')
            canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # Функция для сохранения отчета в PDF
    def save_as_pdf(self):

        # Создание PDF-документа
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', size=14)
        # Заголовок документа
        title = "Отчет по анализу видео"
        pdf.cell(200, 10, txt=title, ln=1, align='C')
        # Подзаголовки
        subheading_1 = "Таблица данных"
        subheading_2 = "График зависимости времени от кадра"
        pdf.ln(10)
        pdf.write(8, subheading_1)
        pdf.line(10, 30, 190, 30)
        column_width = 90
        cell_height = 7
        for col in ['Frame', 'Time (sec)']:
            pdf.cell(column_width, cell_height, txt=col, border=1)
        pdf.ln(cell_height)
        index = 0
        for row in self.analysis_data.itertuples(index=False):
            pdf.cell(column_width, cell_height, str(row.Frame), border=1)
            pdf.cell(column_width, cell_height, f"{round(row['Time (sec)'], 2)} сек", border=1)
            if index % 2 != 0:
                pdf.ln(cell_height)
            index += 1
        if index % 2 == 0:
            pdf.ln(cell_height)
        # Добавление графика
        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(self.analysis_data['Frame'], self.analysis_data['Time (sec)'])
        ax.set_xlabel('Кадры')
        ax.set_ylabel('Время (с)')
        ax.set_title('Зависимость времени от номера кадра')
        filename = "graph.png"
        figure.savefig(filename)
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        imgdata = base64.decodebytes(encoded_string)
        tempFile = io.BytesIO(imgdata)
        pdf.image(tempFile, x=None, y=None, w=150, h=100, type='', link='')
        output_filename = "video_analysis_report.pdf"
        pdf.output(output_filename)
        messagebox.showinfo("Готово!", f"ПDF отчет сохранен как {output_filename}.")
if __name__ == "__main__":
    app = App()
    app.mainloop()