import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
# Загрузка данных из Excel файла
df = pd.read_excel('markup_df_train.xlsx')
gt = df
gt.to_csv('grounded_true.csv', index=False)
# Уникальные значения столбца 'наименование нарушения'
gt['наименование нарушения'].unique()


# Функция для генерации симулированного набора данных
def simul_sub(df):
    violation_list = df['наименование нарушения'].unique()
    print(violation_list)
    len_base_df = len(df)
    # Доля строк для удаления
    fraction_to_remove = 0.4
    num_to_remove = int(len_base_df * fraction_to_remove)
    # Доля строк для дублирования
    fraction_to_duplicate = 0.3
    num_to_duplicate = int(len_base_df * fraction_to_duplicate)
    # Удаление случайных строк
    rows_to_remove = df.sample(num_to_remove).index
    data_dropped = df.drop(rows_to_remove)
    # Дублирование случайных строк
    rows_to_duplicate = data_dropped.sample(num_to_duplicate)
    data_final = pd.concat([data_dropped, rows_to_duplicate])
    data_final = data_final.reset_index(drop=True)
    # Изменение части строк
    fraction_to_change = 0.5
    rows_to_change = data_final.sample(frac=fraction_to_change).index
    data_final.loc[rows_to_change, 'наименование нарушения'] = np.random.choice(violation_list,
                                                                                size=len(rows_to_change))
    # Добавление шума ко времени нарушения
    noise_level = 10
    noise = np.random.randint(-noise_level, noise_level + 1, size=data_final.shape[0])
    data_final['время нарушения (в секундах)'] = data_final['время нарушения (в секундах)'] + noise
    return data_final
# Генерация симулированного набора данных
sub = simul_sub(gt.copy())
sub.head()
sub.to_csv('submission.csv', index=False)
# Проверка близости предсказанных значений времени нарушения
for i, r_sub in sub.iterrows():
    video_gt = gt[
        (gt['номер видео'] == r_sub['номер видео']) & (gt['наименование нарушения'] == r_sub['наименование нарушения'])]
    if len(video_gt) == 0: continue
    pred_sec = r_sub['время нарушения (в секундах)']
    closest_number = min(video_gt['время нарушения (в секундах)'].values, key=lambda x: abs(x - pred_sec))
    print(closest_number, pred_sec)

# Функция для проверки корректности правил
def correct_check(gt, sub):
    for rule in sub['наименование нарушения'].unique():
        if rule not in gt['наименование нарушения'].unique():
            raise Exception(
                f"Правила '{rule}' нет в разметке. Проверьте корректность его написания или убирите его. Должны остаться только те правила, которые есть в grounded true разметке")
# Выполнение проверки корректности правил
correct_check(gt, sub)
# Функция для предварительного расчета баллов
def pre_calc_score(gt, sub):
    pred_seconds = []
    correct_predictions = []
    AE_count_rules_FP = []
    AE_count_rules_FN = []
    for i, r_gt in gt.iterrows():
        video_sub = sub[(sub['номер видео'] == r_gt['номер видео']) & (
                    sub['наименование нарушения'] == r_gt['наименование нарушения'])]
        video_gt = gt[(gt['номер видео'] == r_gt['номер видео']) & (
                    gt['наименование нарушения'] == r_gt['наименование нарушения'])]
        if len(video_sub) == 0:
            pred_seconds.append(np.NaN)
            correct_predictions.append(False)
            FP = max(0 - len(video_gt), 0)  # Всегда будет 0
            FN = abs(min(0 - len(video_gt), 0))
            AE_count_rules_FP.append(FP)
            AE_count_rules_FN.append(FN)
            continue
        true_sec = r_gt['время нарушения (в секундах)']
        pred_sec = min(video_sub['время нарушения (в секундах)'].values, key=lambda x: abs(x - true_sec))
        pred_seconds.append(pred_sec)
        correct_prediction = np.abs(pred_sec - true_sec) < 5
        correct_predictions.append(correct_prediction)
        FP = max(len(video_sub) - len(video_gt), 0)
        FN = abs(min(len(video_sub) - len(video_gt), 0))
        AE_count_rules_FP.append(FP)
        AE_count_rules_FN.append(FN)
    gt['pred_seconds'] = pred_seconds
    gt['Корректность предсказания'] = correct_predictions
    gt['В предсказании было больше нарушений на кол-во'] = AE_count_rules_FP
    gt['В предсказании было меньше нарушений на кол-во'] = AE_count_rules_FN
    return gt
# Предварительный расчет баллов
result_table = pre_calc_score(gt.copy(), sub.copy())
# Расчет итогового балла
score = max(0, (result_table['Корректность предсказания'].sum() - (
            result_table['В предсказании было больше нарушений на кол-во'].sum() * 0.5))) / len(result_table)
# Вывод результата
print(f"Итоговый балл: {score:.4f}")

def img_detect_color(image, show=False):
    color_select = np.copy(image)
    red_threshold = 130
    green_threshold = 130
    blue_threshold = 120
    thresholds = (image[:, :, 0] < red_threshold) | \
                 (image[:, :, 1] < green_threshold) | \
                 (image[:, :, 2] < blue_threshold)
    color_select[thresholds] = [0, 0, 0]
    if show:
        plt.imshow(color_select)
        plt.title("Выделение +- белого цвета")
        plt.show()
    return color_select
def mask_area_on_image(image, show=False):
    mask = np.zeros_like(image)
    height, width, _ = mask.shape
    polygon = np.array([[
        (int(width * 0.4), height),
        (int(width * 0.6), height),
        (int(width * 0.6), int(height * 0.7)),
        (int(width * 0.4), int(height * 0.7))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    if show:
        image_with_border = masked_image.copy()
        cv2.polylines(image_with_border, [polygon], isClosed=True, color=(255, 0, 0), thickness=1)
        plt.imshow(image_with_border)
        plt.title("Выделение региона дороги на изображении")
        plt.show()
    return masked_image
def lines_detect(image, show=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    if show:
        plt.imshow(edges, cmap='gray')
        plt.title("Выделение линий")
        plt.show()
    return edges
def detect_road_marking(base_image, image, show=False):
    threshold_value = 65
    min_line_length = 60
    max_line_gap = 50
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180,
                            threshold=threshold_value,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if show:
        line_image = np.zeros_like(base_image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        combined_image = cv2.addWeighted(base_image, 0.8, line_image, 1, 0)
        plt.imshow(combined_image)
        plt.title("Выделение дорожной разметки")
        plt.show()
    if lines is not None:
        return lines
    else:
        return []
def line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def does_line_intersect_zone(x1, y1, x2, y2, zone_start, zone_end, height):
    if (zone_start <= x1 <= zone_end) or (zone_start <= x2 <= zone_end):
        return True
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    y_start = (-A * zone_start - C) / B if B != 0 else None
    y_end = (-A * zone_end - C) / B if B != 0 else None
    if y_start is not None and (0 <= y_start <= height):
        return True
    if y_end is not None and (0 <= y_end <= height):
        return True
    return False

def does_center_intersect_line_center(x1, y1, x2, y2, image_center_x):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    line_len = line_length(x1, y1, x2, y2)
    offset = 0.2 * line_len
    center_start_x = mid_x - offset
    center_end_x = mid_x + offset
    return center_start_x <= image_center_x <= center_end_x

def line_crossing_check(lines, image, min_len_line=60, ignore_horizontal=True, verbose=False):
    height, width, _ = image.shape
    zone_width = width * 0.1
    zone_start = (width / 0.2) - (zone_width / 2)
    zone_end = (width / 0.2) + (zone_width / 2)
    image_center_x = width / 2
    if len(lines) == 0:
        return False
    if len(lines) > 20:
        return False
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_len = line_length(x1, y1, x2, y2)
        intersects_zone = does_line_intersect_zone(x1, y1, x2, y2, zone_start, zone_end, height)
        center_intersects_line_center = does_center_intersect_line_center(x1, y1, x2, y2, image_center_x)
        if (y1 > y2) or (y1 == y2 and x1 < x2):
            left_x, left_y = x1, y1  # Если первая точка ниже, она считается левой нижней
        else:
            left_x, left_y = x2, y2  # Иначе считается, что второй точка - левая нижняя
        if ignore_horizontal:
            if abs(y2 - y1) < height * 0.2:  # Если линия почти горизонтальная
                continue  # Переходим к следующей линии
        if center_intersects_line_center:
            if verbose:
                # Вывод информации о пересекающей линии, если verbose=True
                print(f"Line with length {int(line_len)} intersects the 10% center zone, "
                     f"the lower-left point is right of the center, and "
                     f"the 20% center section of the line intersects the image center.")
            return True  # Если условия выполнены, возвращаем True
    return False  # Если не найдено ни одной удовлетворительной линии, возвращаем False

def process_frame(frame, show=False):
    image = img_detect_color(frame, show)
    image = mask_area_on_image(image, show)
    image = lines_detect(image, show)
    lines = detect_road_marking(frame, image, show)
    violation = line_crossing_check(lines, frame, min_len_line=60)
    return violation

def main_analyse(video_path, frames_to_take=50, show=False, debug_sec=[]):
    result_analise = []
    violation_frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count_frame // fps
    freq = max(1, count_frame // frames_to_take)
    print(f"Частота выборки кадров: {freq}")
    success, frame = cap.read()
    count = 0
    with tqdm(total=count_frame // freq, desc="Processing frames"):
        if count % freq == 0:
            time_sec = count // fps
            if time_sec in debug_sec:
                show = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGB)
            if show:
                plt.imshow(frame)
                plt.title("Current Frame")
                plt.show()
            violation = process_frame(frame, show=show)
            result_analise.append([violation, time_sec])
            if violation:
                violation_frames.append(frame)
            if time_sec in debug_sec:
                show = False
                print(f'\n\nОбработка кадра на {time_sec} секунде')
                print('Правило нарушено' if violation else 'Правило не нарушено')
                print('----------------------------')
            success, frame = cap.read()
            count += 1
    cap.release()
    cv2.destroyAllWindows()
    return [result_analise, violation_frames]

def pre_calc_score(gt, sub):
    pred_seconds = []
    correct_predictions = []
    AE_count_rules_FP = []
    AE_count_rules_FN = []
    for i, r_gt in gt.iterrows():
        video_sub = sub[(sub['номер видео'] == r_gt['номер видео']) & (sub['наименование нарушения'] == r_gt['наименование нарушения'])]
        video_gt = gt[(gt['номер видео'] == r_gt['номер видео']) & (gt['наименование нарушения'] == r_gt['наименование нарушения'])]
        if len(video_sub) == 0:
            pred_seconds.append(np.NaN)
            correct_predictions.append(False)
            FP = max(0 - len(video_gt), 0)  # Всегда будет 0
            FN = abs(min(0 - len(video_gt), 0))
            AE_count_rules_FP.append(FP)
            AE_count_rules_FN.append(FN)
            continue
        true_sec = r_gt['время нарушения (в секундах)']
        pred_sec = min(video_sub['время нарушения (в секундах)'].values, key=lambda x: abs(x - true_sec))
        pred_seconds.append(pred_sec)
        correct_prediction = np.abs(pred_sec - true_sec) < 5
        correct_predictions.append(correct_prediction)
        FP = max(len(video_sub) - len(video_gt), 0)
        FN = abs(min(len(video_sub) - len(video_gt), 0))
        AE_count_rules_FP.append(FP)
        AE_count_rules_FN.append(FN)
    gt['pred_seconds'] = pred_seconds
    gt['Корректность предсказания'] = correct_predictions
    gt['В предсказании было больше нарушений на кол-во'] = AE_count_rules_FP
    gt['В предсказании было меньше нарушений на кол-во'] = AE_count_rules_FN
    return gt

def correct_check(gt, sub):
    for rule in sub['наименование нарушения'].unique():
        if rule not in gt['наименование нарушения'].unique():
            raise Exception(f"Правила '{rule}' нет в разметке. Проверьте корректность его написания или убирите его. Должны остаться только те правила, которые есть в grounded true разметке")

correct_check(gt, sub)

result_table = pre_calc_score(gt.copy(), sub.copy())

score = max(0,(result_table['Корректность предсказания'].sum() - (result_table['В предсказании было больше нарушений на кол-во'].sum() * 0.5))/len(result_table))

print(f"Итоговый балл: {score:.4f}")


def save_violations(images, output_folder):
    for idx, img in enumerate(images):
        file_name = f"{idx}.png"
        full_file_path = os.path.join(output_folder, file_name)
        plt.savefig(full_file_path)
        print(f"Сохранено изображение с индексом {idx} в папке {output_folder}")

def main():
    result_analise, violation_frames = main_analyse('/videos/AKN00077.mp4', frames_to_take=500, show=False, debug_sec=[])
    print(result_analise)


