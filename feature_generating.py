import pandas as pd
import numpy as np



def stochastic_oscillator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет стохастический осциллятор для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления осциллятора (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями стохастического осциллятора.
    """
    low_min = series.rolling(window=window, min_periods=1).min()
    high_max = series.rolling(window=window, min_periods=1).max()
    stochastic = ((series - low_min) / (high_max - low_min)) * 100
    return stochastic


def relative_strength_index(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Вычисляет индекс относительной силы (RSI) для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления RSI (по умолчанию 7).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями RSI.
    """
    # Вычисляем изменения между последовательными значениями
    delta = series.diff()
    # Вычисляем приросты (gains) и потери (losses)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Средние значения приростов и потерь
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    # Вычисляем RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detrended_price_oscillator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет осциллятор цены без тренда (DPO).

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для сдвига (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями DPO.
    """
    # Сдвигаем временной ряд на заданное количество периодов (окно)
    shifted_series = series.shift(window)
    # Разница между текущей ценой и сдвинутой ценой
    dpo = series - shifted_series
    return dpo

def momentum_indicator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет индикатор импульса (Momentum Indicator).

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления индикатора импульса (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд с значениями индикатора импульса.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Входные данные должны быть pd.Series")

    # Индикатор импульса: разница между текущей ценой и ценой за n периодов назад
    momentum = series - series.shift(window)
    return momentum

def generate_features(dataset, windows=[100], columns=['FrL', 'FrR', 'OcR']):
    for column in columns:
        for window in windows:
            dataset[f"{column}_stachostic_{window}"] = stochastic_oscillator(dataset[column], window=window)
            dataset[f"{column}_RSI_{window}"] = relative_strength_index(dataset[column], window=window)
            dataset[f"{column}_DPO_{window}"] = detrended_price_oscillator(dataset[column], window=window)
            dataset[f"{column}_momentum_{window}"] = momentum_indicator(dataset[column], window=window)
    return dataset


def add_base_features(data, window_sizes=[400]):
    for window in window_sizes:
        data[f'mean_window_{window}_FrL'] = data['FrL'].rolling(window=window).mean()
        data[f'mean_window_{window}_FrR'] = data['FrR'].rolling(window=window).mean()
        data[f'mean_window_{window}_OcR'] = data['OcR'].rolling(window=window).mean()
        data[f'min_window_{window}_FrL'] = data['FrL'].rolling(window=window).min()
        data[f'max_window_{window}_FrL'] = data['FrL'].rolling(window=window).max()
        data[f'min_window_{window}_FrR'] = data['FrR'].rolling(window=window).min()
        data[f'max_window_{window}_FrR'] = data['FrR'].rolling(window=window).max()
        data[f'min_window_{window}_OcR'] = data['OcR'].rolling(window=window).min()
        data[f'max_window_{window}_OcR'] = data['OcR'].rolling(window=window).max()
        data[f'corr_window_{window}_FrL_FrR'] = data['FrL'].rolling(window=window).corr(data['FrR'])
        data[f'corr_window_{window}_FrL_OcR'] = data['FrL'].rolling(window=window).corr(data['OcR'])
        data[f'corr_window_{window}_FrR_OcR'] = data['FrR'].rolling(window=window).corr(data['OcR'])
    return data

if __name__ == '__main__':
    data = pd.read_csv('marked_dataset.csv', index_col=0)
    print(data.shape)
    fish_genered_data = generate_features(data)[10000:-10000]
    print(fish_genered_data.shape)
    print(fish_genered_data.head())