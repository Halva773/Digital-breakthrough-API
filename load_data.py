import numpy as np
import pandas as pd
import pyedflib
import os

from feature_generating import generate_features

def read_dataset(file_path):
    edf_file = pyedflib.EdfReader(file_path)
    try:
        n_signals = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        signals = [edf_file.readSignal(i) for i in range(n_signals)]
        annotations = edf_file.readAnnotations() if len(edf_file.readAnnotations()) > 0 else None
    finally:
        edf_file.close()
    return signal_labels, signals, annotations


def read_annotations(annotations):
    return [(i + 1, int(annot * 400), annotations[2][i]) for i, annot in enumerate(annotations[0])]


def read_txt_markers(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels


def convert_to_sec(time: str):
    s = list(map(int, time.split(':')))
    return s[0] * 3600 + s[1] * 60 + s[2]


def get_dataset(data_file_path):
    signal_labels, signals, annotations = read_dataset(data_file_path)
    signals = np.array(signals)
    data = pd.DataFrame(signals).T.rename(columns={i: signal_labels[i] for i in range(len(signal_labels))})
    return data


def load_marked_dataset(file, folder='ECoG_fully_marked_(4+2 files, 6 h each)',
                        base_path=r"C:/Users/Артём/PycharmProjects/International_hack/data"):
    dataset_file_path = fr"{base_path}/{folder}/{file}.edf"
    markers_file_path = fr"{base_path}/{folder}/{file}.txt"
    dataset = get_dataset(dataset_file_path, markers_file_path)
    return dataset


# Функция для сохранения датасета в CSV файлами блоками
def save_in_blocks(dataframe, num_blocks, output_dir="output_blocks", base_filename="dataset_block"):
    # Создаем директорию для сохранения блоков, если она еще не существует
    import os
    os.makedirs(output_dir, exist_ok=True)

    block_size = len(dataframe) // num_blocks
    for i in range(num_blocks):
        start_row = i * block_size
        end_row = start_row + block_size if i < num_blocks - 1 else len(dataframe)
        block = dataframe.iloc[start_row:end_row]
        filename = os.path.join(output_dir, f"{base_filename}_{i + 1}.csv")
        block.to_csv(filename, index=False)
        print(f"Сохранен блок {i + 1}: {filename}")



def extract_sequential_samples(df, target_col, sample_size=5000, exclude_value=None):
    df[target_col] = df[target_col].replace({exclude_value: pd.NA})
    df_filtered = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Инициализация переменных
    result_df = pd.DataFrame()
    grouped_samples = {value: 0 for value in df_filtered[target_col].unique()}

    current_group = []
    current_value = df_filtered[target_col].iloc[0]

    # Проход по строкам для группировки последовательностей
    for index, value in enumerate(df_filtered[target_col]):
        if value == current_value:
            current_group.append(index)
        else:
            # Проверка, не превышен ли лимит строк для текущего класса
            if grouped_samples[current_value] < sample_size:
                selected_rows = current_group[:sample_size - grouped_samples[current_value]]
                result_df = pd.concat([result_df, df_filtered.loc[selected_rows]])
                grouped_samples[current_value] += len(selected_rows)

            # Обновляем текущую группу
            current_group = [index]
            current_value = value

    # Обработка последней группы
    if len(current_group) > 0 and grouped_samples[current_value] < sample_size:
        selected_rows = current_group[:sample_size - grouped_samples[current_value]]
        result_df = pd.concat([result_df, df_filtered.loc[selected_rows]])
        grouped_samples[current_value] += len(selected_rows)

    return result_df.reset_index(drop=True)


def get_fully_marked_files(folder_path: str) -> list:
    # Получаем список всех файлов и папок в указанной папке
    all_files = os.listdir(folder_path)
    # Отбираем файлы, которые заканчиваются на '_fully_marked'
    fully_marked_files = [file for file in all_files if file.endswith('_fully_marked.edf')]
    # Создаем полный путь к файлам
    full_paths = [file for file in fully_marked_files]
    return full_paths


if __name__ == '__main__':
    directory = r"data/ECoG_fully_marked_(4+2 files, 6 h each)"
    files = get_fully_marked_files(directory)
    whole_dataset = []
    for dataset_name in files:
        print(dataset_name)
        dataset = load_marked_dataset(dataset_name[:-4],
                                      folder="ECoG_fully_marked_(4+2 files, 6 h each)",
                                      base_path=r'C:/Users/blago/PycharmProjects/animated-garbanzo/data')
        dataset = generate_features(dataset)
        dataset = extract_sequential_samples(dataset[10000:], target_col='target', sample_size=12000, exclude_value=None)
        dataset = dataset[dataset['target'].notna()]
        dataset['target'] = dataset['target'].replace({'ds': 0, 'is': 1, 'swd': 2})
        whole_dataset.append(dataset)
    data = pd.concat(whole_dataset, ignore_index=True)
    print(data.shape)
    print(data.head())
    data.to_csv('to_train_data.csv')