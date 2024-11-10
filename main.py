import datetime
import io
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyedflib
import seaborn as sns
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.responses import FileResponse
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import load_model
from feature_generating import generate_features
from load_data import get_dataset

app = FastAPI()


# Модель данных для ответа
class AnalysisResponse(BaseModel):
    signal_labels: List[str]
    n_signals: int
    percentage_marked: float


# Функция для обработки EDF файла
def visualize_absense(file_path):
    def read_dataset(file_path):
        edf_file = pyedflib.EdfReader(file_path)
        n_signals = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        signals = [edf_file.readSignal(i) for i in range(n_signals)]
        edf_file.close()
        return signal_labels, signals

    # Чтение данных из EDF файла
    signal_labels, signals = read_dataset(file_path)
    signals = np.array(signals)
    data = pd.DataFrame(signals).T.rename(columns={i: signal_labels[i] for i in range(len(signal_labels))})

    # Добавляем пример разметки для демонстрации
    data['target'] = np.where(np.arange(len(data)) % 2 == 0, 'ds', 'is')

    # Рассчитываем процент размеченных данных
    percentage_marked = data['target'].notna().mean()
    return data, percentage_marked

def prepare_sequences_for_prediction(new_data, seq_length=200):
    sequences = []
    for i in range(len(new_data) - seq_length + 1):
        sequences.append(new_data[i:i + seq_length])
    return np.array(sequences)


def get_predict(dataset):
    model = load_model('cnnlstm_model.keras')
    # dataset = generate_features(dataset)
    prepare_sequences_for_prediction(dataset)
    predictions = model.predict(dataset)
    return predictions


def summirize_predictions(predictions, frequency=400):
    periods = []
    start_index = 0

    for i in range(1, len(predictions)):
        if predictions[i] != predictions[start_index]:
            start_time = datetime.timedelta(seconds=start_index/frequency)
            end_time = datetime.timedelta(seconds=i/frequency)
            periods.append((str(start_time), str(end_time), predictions[start_index]))
            start_index = i

    # Добавить последний период
    start_time = datetime.timedelta(seconds=start_index/frequency)
    end_time = datetime.timedelta(seconds=len(predictions)/frequency)
    periods.append((str(start_time), str(end_time), predictions[start_index]))
    return periods


def convert_periods_to_txt(periods):
    class_mapping = {0: 'ds', 1: 'is', 2: 'swd'}
    txt_lines = []

    for start_time, end_time, cls in periods:
        class_label = class_mapping[cls]
        txt_lines.append(f"{class_label}1 {start_time}")
        txt_lines.append(f"{class_label}2 {end_time}")

    return "\n".join(txt_lines)


def get_file(input_path: str, output_path: str):
    dataset = get_dataset(input_path)
    print(dataset)
    predictions = get_predict(dataset)
    print(predictions)
    markers = summirize_predictions(predictions)
    print(markers)
    txt_content = convert_periods_to_txt(markers)
    print(txt_content)
    with open(output_path, "w") as txt_file:
        txt_file.write(txt_content)

# Эндпоинт для анализа загруженного EDF файла
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith('.edf'):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы EDF")

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Обработка файла и выполнение анализа
        dataset, percentage_marked = visualize_absense(file_location)
        signal_labels = dataset.columns[:-1].tolist()  # Исключаем колонку 'target'
        n_signals = len(signal_labels)

        os.remove(file_location)  # Удаление временного файла после обработки

        # Возвращаем результат анализа
        return AnalysisResponse(
            signal_labels=signal_labels,
            n_signals=n_signals,
            percentage_marked=percentage_marked
        )

    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


# Эндпоинт для анализа данных с визуализацией графиков
@app.post("/plot")
async def plot_graphs(file: UploadFile = File(...)):
    if not file.filename.endswith('.edf'):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы EDF")

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Загрузка и разметка данных
        dataset, _ = visualize_absense(file_location)

        os.remove(file_location)  # Удаление временного файла после обработки

        fig, axs = plt.subplots(6, 1, figsize=(15, 30), constrained_layout=True,
                                gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})

        # 1. Разброс значений для 'FrL', 'FrR', 'OcR'
        scatter_data = dataset[['FrL', 'FrR', 'OcR']].copy()
        scatter_data.plot(marker='.', alpha=0.5, linestyle='None', subplots=True, ax=axs[0:3])
        for i, ax in enumerate(axs[0:3]):
            ax.set_ylabel('Value')
            ax.set_title(f"Scatter for {scatter_data.columns[i]}")

        # 2. Boxplots для 'FrL', 'FrR', 'OcR'
        for i, name in enumerate(['FrL', 'FrR', 'OcR'], start=3):
            sns.boxplot(data=dataset[-12000:], x='target', y=name, ax=axs[i])
            axs[i].set_ylabel('Value')
            axs[i].set_title(f"Boxplot for {name}")

        # Сохранение графиков в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # Возврат изображения в ответе
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.post("/decomposition")
async def decomposition(file: UploadFile = File(...)):
    if not file.filename.endswith('.edf'):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы EDF")

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        dataset, _ = visualize_absense(file_location)
        os.remove(file_location)

        fig, axs = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True)
        df = dataset['FrR']  # Убедитесь, что 'FrR' есть в данных
        decomposition = seasonal_decompose(df[-40000:-5000], model='additive', period=400)

        decomposition.observed.plot(ax=axs[0])
        axs[0].set_ylabel('Observed')
        axs[0].set_title('Seasonal Decomposition of FrR - Observed')

        decomposition.trend.plot(ax=axs[1])
        axs[1].set_ylabel('Trend')
        axs[1].set_title('Trend')

        decomposition.seasonal.plot(ax=axs[2])
        axs[2].set_ylabel('Seasonal')
        axs[2].set_title('Seasonal')

        decomposition.resid.plot(ax=axs[3])
        axs[3].set_ylabel('Residual')
        axs[3].set_title('Residual')

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.edf'):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы EDF")
    file_location = rf"{os.getcwd()}/temp_{file.filename}"
    output_txt = rf"{os.getcwd()}/{file_location}.txt"
    print(file_location, output_txt)
    try:
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        get_file(file_location, output_txt)
        return FileResponse(output_txt, media_type='text/plain', filename=f"{file.filename}.txt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
        if os.path.exists(output_txt):
            os.remove(output_txt)
