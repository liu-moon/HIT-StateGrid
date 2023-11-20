import numpy as np
import pandas as pd
from numpy.fft import fft
import matplotlib.pyplot as plt

class FiberOpticDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.fft_result = np.array([]).reshape(0, 0)
        self.fft_result_db = np.array([]).reshape(0, 0)
        self.num_rows = 0
        self.num_cols = 0
        self.fs = 10240 / 600  # 采样率
        self.freq_axis = None
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
        print(self.freq_axis.shape[0])

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
            fft_result = np.fft.fft(signal)
            # # 获取频率轴
            # freq_axis = np.fft.fftfreq(len(signal), 1 / self.fs)

            # 将双边谱转换成单边谱
            half_len = len(signal) // 2
            # freq_axis = freq_axis[1:half_len]   # 这里的“1”是去掉第一个点
            fft_result = np.abs(fft_result[1:half_len]) / self.num_rows * 2

            # 将新计算的值追加到矩阵中
            # self.fft_result = np.vstack([self.fft_result, fft_result])
            self.fft_result = np.append(self.fft_result, fft_result)

            # 将坐标转化为dB形式
            magnitude = np.abs(fft_result)
            magnitude_db = 10 * np.log10(magnitude)

            # 将新计算的值追加到矩阵中
            # self.fft_result_db = np.vstack([self.fft_result_db, magnitude_db])
            self.fft_result_db = np.append(self.fft_result_db, magnitude_db)

        print(self.fft_result)
        print(self.fft_result_db)
        print(self.fft_result.shape)
        print(self.fft_result_db.shape)

            # 画出谱线
            # plt.figure(figsize=(10, 6))
            # plt.plot(freq_axis, magnitude_db)
            # plt.title('Single-Sided Amplitude Spectrum')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude (dB)')
            # plt.grid(True)
            # plt.show()

    def process_data(self, output_file_path):
        if self.data is None:
            print("请先加载数据")
            return

        # 获取数据的行数和列数
        num_rows, num_cols = self.data.shape
        # 打印数据的行数和列数
        print(f'Number of rows: {num_rows}')
        print(f'Number of columns: {num_cols}')

        # 创建一个新的DataFrame来保存处理后的数据
        processed_data = pd.DataFrame(columns=['Point', 'Max_Frequency', 'Max_Gain'])

        for col in range(10):
            # 对每一列的数据进行FFT变换
            time_series = self.data.iloc[:, col].values
            fft_result = fft(time_series)

            # 去除直流成分
            fft_result[0] = 0

            # 找到FFT最大增益点及其对应的频率
            max_frequency_index = np.argmax(np.abs(fft_result))
            max_frequency = max_frequency_index / num_rows
            max_gain = np.abs(fft_result[max_frequency_index])

            # 将结果添加到新的DataFrame中
            # processed_data = processed_data.append({
            #     'Point': f'Point_{col + 1}',
            #     'Max_Frequency': max_frequency,
            #     'Max_Gain': max_gain
            # }, ignore_index=True)

            # append已经弃用，使用concat代替
            # processed_data = pd.DataFrame([],columns=['Point', 'Max_Frequency', 'Max_Gain'])
            # processed_data = pd.concat([processed_data if not processed_data.empty else None,
            #                             pd.DataFrame([[f'Point_{col + 1}', max_frequency, max_gain]])
            #                             ],ignore_index=True)
            # processed_data = pd.concat([processed_data,
            #                             pd.DataFrame([[f'Point_{col + 1}', max_frequency, max_gain]])
            #                             ],ignore_index=True)
            # processed_data = pd.concat(
            #     [processed_data, pd.DataFrame([[f'Point_{col + 1}', max_frequency, max_gain]],
            #                                             columns=['Point', 'Max_Frequency', 'Max_Gain'])],
            #                                             ignore_index=True)
            processed_data = pd.concat([processed_data, pd.DataFrame([[f'Point_{col + 1}', max_frequency, max_gain]], columns=['Point', 'Max_Frequency', 'Max_Gain'])], ignore_index=True)


        # 保存处理后的数据到CSV文件
        processed_data.to_csv(output_file_path, index=False)

    def plot_fft(self):
        if self.data is None:
            print("请先加载数据")
            return

        # 获取数据的行数和列数
        num_rows, num_cols = self.data.shape

        

        # 设置绘图的样式
        plt.style.use('ggplot')

        # 创建一个新的Figure对象
        fig, axs = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols), sharex=True)

        for col in range(num_cols):
            # 对每一列的数据进行FFT变换
            time_series = self.data.iloc[:, col].values
            fft_result = fft(time_series)

            # 去除直流成分
            fft_result[0] = 0

            # 绘制FFT图像
            frequency = np.fft.fftfreq(num_rows)
            axs[col].plot(frequency, np.abs(fft_result))
            axs[col].set_title(f'Point {col + 1}')
            axs[col].set_xlabel('Frequency')
            axs[col].set_ylabel('Amplitude')

        # 调整布局
        plt.tight_layout()

        # 显示图像
        plt.show()

    def test_fft(self):
        if self.data is None:
            print("请先加载数据")
            return

        # 获取数据的行数和列数
        num_rows, num_cols = self.data.shape

        # sampling rate
        sr = 10240 / 600
        # # sampling interval
        ts = 1.0 / sr

        # 打印数据的行数和列数
        print(f'Number of rows: {num_rows}')
        print(f'Number of columns: {num_cols}')

        # 取出数据的全部行600列
        x = self.data.iloc[:, 600-1].values
        N = len(x)

        # 去直流
        x = x - np.mean(x)
        # fft
        xdft = fft(x)
        xdft = xdft[:N // 2 + 1]
        psdx = (1 / (sr * N)) * np.abs(xdft) ** 2
        psdx[1:-1] = 2 * psdx[1:-1]

        # 转化为db
        psdx_db = 10 * np.log10(psdx)

        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)

        # plt.stem(freq, np.abs(X), 'b', \
        #          markerfmt=" ", basefmt="-b")
        plt.plot(psdx_db)

        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.show()
        # plt.xlim(0, 10)

        # plt.subplot(122)
        # plt.plot(t, ifft(X), 'r')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()

    def test_fft_gpt(self):
        if self.data is None:
            print("请先加载数据")
            return

        fs = 10240 / 600  # 采样率
        L = 10240


        # 取出数据的全部行600列
        signal = self.data.iloc[:, 600 - 1].values
        print(signal)

        # 去掉均值
        signal = signal - np.mean(signal)
        print(signal)

        # FFT变换
        fft_result = np.fft.fft(signal)
        print(fft_result)
        print(np.abs(fft_result))

        # 获取频率轴
        freq_axis = np.fft.fftfreq(len(signal), 1 / fs)

        # 将双边谱转换成单边谱
        half_len = len(signal) // 2
        freq_axis = freq_axis[1:half_len]
        fft_result = np.abs(fft_result[1:half_len]) / L * 2

        # 将坐标转化为dB形式
        magnitude = np.abs(fft_result)
        magnitude_db = 10 * np.log10(magnitude)
        print(magnitude_db)
        print(np.size(magnitude_db))

        # 画出谱线
        plt.figure(figsize=(10, 6))
        plt.plot(freq_axis, magnitude_db)
        plt.title('Single-Sided Amplitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # 示例用法
    file_path = "2023_11_05-15_35_48--271332.csv"
    output_file_path = "your_output_file.csv"

    processor = FiberOpticDataProcessor(file_path)
    processor.load_data()
    # processor.fft_data()
    # processor.plot_fft()
    # processor.process_data(output_file_path)
    # processor.test_fft()
    # processor.test_fft_gpt()