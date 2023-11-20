import numpy as np
import pandas as pd
from numpy.fft import fft
import matplotlib.pyplot as plt

class FiberOpticDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.num_rows = 0
        self.num_cols = 0
        self.fs = 10240 / 600  # 采样率
        self.freq_axis = None
        self.fft_result = None
        self.fft_result_db = None
        self.processed_data = pd.DataFrame(columns=['Length', 'Max_Frequency', 'Max_Gain'])

    def load_data(self):
        # 从CSV文件加载数据
        self.data = pd.read_csv(self.file_path, header=None)
        # 获取数据的行数和列数
        self.num_rows, self.num_cols = self.data.shape
        print(f'Number of rows: {self.num_rows}')
        print(f'Number of columns: {self.num_cols}')
        # 取data的第一列
        signal = self.data.iloc[:, 0].values

        # 获取频率轴
        freq_axis = np.fft.fftfreq(len(signal), 1 / self.fs)
        self.freq_axis = freq_axis[1:len(signal) // 2]  # 这里的“1”是去掉第一个点

        # 打印频率轴freq_axis
        # print(self.freq_axis.shape[0])
        self.fft_result = np.zeros((self.freq_axis.shape[0], self.num_cols))
        self.fft_result_db = np.zeros((self.freq_axis.shape[0],self.num_cols))
        # print(self.fft_result.shape)
        # print(self.fft_result_db.shape)

        if self.data is None:
            print("数据加载失败")
        else:
            print("数据加载成功")

    def fft_data(self):
        for col in range(self.num_cols):
            # 对每一列的数据进行FFT变换
            signal = self.data.iloc[:, col].values
            # 去掉均值
            signal = signal - np.mean(signal)
            # FFT变换
            fft_result = fft(signal)

            # 将双边谱转换成单边谱
            half_len = len(signal) // 2

            fft_result = np.abs(fft_result[1:half_len]) / self.num_rows * 2
            # print(fft_result.shape)

            # 将坐标转化为dB形式
            magnitude = np.abs(fft_result)
            magnitude_db = 10 * np.log10(magnitude)

            # 将新计算的值更新到矩阵中
            self.fft_result[:, col] = fft_result
            self.fft_result_db[:, col] = magnitude_db

    def save_data(self,output_file_path):
        # 将fft_result_db保存到CSV文件
        np.savetxt(output_file_path, self.fft_result_db, delimiter=',')

    def plot_data(self,start_index,end_index):
        for col in range(start_index,end_index+1):
            # 画出谱线
            # plt.figure(figsize=(10, 6))
            plt.plot(self.freq_axis, self.fft_result_db[:, col])
            plt.title(f'Single-Sided Amplitude Spectrum x = {col}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.show()


if __name__ == "__main__":

    file_path = "C:\\Users\\liu-i\\Desktop\\FFT\\data\\2023_11_05-15_35_48--271332.csv"
    output_file_path = "C:\\Users\\liu-i\\Desktop\\FFT\\data\\my_output_file.csv"

    processor = FiberOpticDataProcessor(file_path)
    processor.load_data()
    processor.fft_data()
    processor.save_data(output_file_path)
    processor.plot_data(100,110)