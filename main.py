import numpy as np
import pandas as pd
from numpy.fft import fft
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class FiberOpticDataProcessor:

    def __init__(self, file_path):
        """
        类成员初始化函数
        :param file_path: 要处理的文件路径
        """
        # 初始化
        # file_path: 要处理的文件路径
        self.file_path = file_path
        # 加载的数据
        self.data = None
        # 数据的行数和列数
        self.num_rows = 0
        self.num_cols = 0
        # 采样率
        # self.fs = 10240 / 600
        self.fs = 10240 / 271.332
        # 频率轴
        self.freq_axis = None
        # 频率轴的公差
        self.freq_axis_step = None
        # FFT变换的结果
        self.fft_result = None
        # FFT变换的结果的dB形式
        self.fft_result_db = None
        # 文件夹中所有文件的名称list
        self.file_names = []
        # 处理后的数据 暂时未用到
        self.processed_data = pd.DataFrame(columns=['Length', 'Max_Frequency', 'Max_Gain'])

    def load_data(self):
        """
        加载数据，其中计算了频率轴以及重置了fft_result、fft_result_db的size
        """
        # 从CSV文件加载数据
        self.data = pd.read_csv(self.file_path, header=None)
        # 获取数据的行数和列数
        self.num_rows, self.num_cols = self.data.shape
        print(f'Number of rows: {self.num_rows}')
        print(f'Number of columns: {self.num_cols}')
        # 取data的第一列
        signal = self.data.iloc[:, 0].values

        # 计算频率轴的公差
        # 打印num_cols，并提示
        # print(f'Number of columns: {self.num_cols}')
        # 打印num_rows，并提示
        # print(f'Number of rows: {self.num_rows}')

        self.freq_axis_step = 1 / ((self.num_rows) * (1/ self.fs))
        print(f'freq_axis_step: {self.freq_axis_step}')

        # 获取频率轴
        freq_axis = np.fft.fftfreq(len(signal), 1 / self.fs).reshape(-1, 1)

        # freq_axis 转置
        # freq_axis = freq_axis.reshape(-1, 1)
        # self.freq_axis = np.transpose(freq_axis[0:len(signal) // 2])
        self.freq_axis = freq_axis[0:len(signal) // 2]
        # print(self.freq_axis)
        # self.freq_axis = np.transpose(self.freq_axis)
        # print(self.freq_axis)
        # self.freq_axis = freq_axis[1:len(signal) // 2]  # 这里的“1”是去掉第一个点

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
        """
        对数据进行FFT变换
        :return: fft_result, fft_result_db
        """
        for col in tqdm(range(self.num_cols),desc="FFT",unit="cols"):
            # 对每一列的数据进行FFT变换
            signal = self.data.iloc[:, col].values
            # 去掉均值
            signal = signal - np.mean(signal)
            # FFT变换
            fft_result = fft(signal)

            # 将双边谱转换成单边谱
            half_len = len(signal) // 2

            fft_result = np.abs(fft_result[0:half_len]) / self.num_rows * 2
            # print(fft_result.shape)

            # 将坐标转化为dB形式
            magnitude = np.abs(fft_result)
            with np.errstate(divide='ignore'):
                magnitude_db = 10 * np.log10(magnitude)

            # 将新计算的值更新到矩阵中
            self.fft_result[:, col] = fft_result
            self.fft_result_db[:, col] = magnitude_db

    def save_data(self,output_file_path):
        """
        将fft_result_db保存到CSV文件
        :param output_file_path: 保存的文件路径
        """
        print("saving to file")
        # 将fft_result_db保存到CSV文件
        # np.savetxt(output_file_path, np.transpose(self.freq_axis), delimiter=',')
        # self.append_tofile(output_file_path, self.freq_axis, delimiter=',', chunk_size=10000)
        self.append_tofile(output_file_path, self.fft_result_db, delimiter=',', chunk_size=10000)
        print("saved")



    def plot_data(self,start_index,end_index):
        """
        画出FFT的结果
        :param start_index: 光缆的开始索引
        :param end_index: 光缆的结束索引
        """
        for col in range(start_index,end_index+1):
            # 画出谱线
            # plt.figure(figsize=(10, 6))
            # db形式
            # plt.plot(self.freq_axis, self.fft_result_db[:, col])
            # 非db形式
            plt.plot(self.freq_axis, self.fft_result[:, col])
            plt.title(f'Single-Sided Amplitude Spectrum x = {col}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.show()

    def get_all_files_in_directory(self,directory):
        """
        获取目录下的所有文件的名称
        """
        # 使用 os.listdir 获取目录下的所有文件和子目录
        all_files = os.listdir(directory)

        # 使用列表推导式过滤出所有文件
        files = [file for file in all_files if os.path.isfile(os.path.join(directory, file))]

        # 保存到file_names中
        self.file_names = files

        # 打印文件名
        for file in files:
            print(file)

    def process_folder(self,folder_path,output_folder):
        """
        处理文件夹中的所有文件
        :param folder_path: 文件夹路径
        :param output_folder: 输出文件夹路径
        """
        # 获取文件夹中的所有文件
        self.get_all_files_in_directory(folder_path)

        # i = 0
        # 遍历所有文件
        for file_name in self.file_names:
            # i=i+1
            # if(i>2):
            #     break
            # 构造完整的文件路径
            file_path = os.path.join(folder_path,file_name)
            print(file_path)

            # 创建一个FiberOpticDataProcessor对象
            processor = FiberOpticDataProcessor(file_path)

            # 加载数据
            processor.load_data()

            # 对数据进行FFT变换
            processor.fft_data()

            # 构造输出文件路径
            output_file_path = os.path.join(output_folder,f"fft_result_{file_name}")

            # 保存数据
            processor.save_data(output_file_path)

    def append_tofile(self,file_path, new_data, delimiter=',', chunk_size=10000):
        """
        将新数据追加到文件中
        :param file_path: 文件路径
        :param new_data: 追加的数据
        :param delimiter: 分隔符
        :param chunk_size: 一次处理的大小
        :return:
        """
        # 1. 打开文件，如果不存在则创建
        with open(file_path, 'a' if os.path.exists(file_path) else 'w') as file:
            # 2. 将新数据分块写入文件
            for i in range(0, len(new_data), chunk_size):
                chunk = new_data[i:i + chunk_size]
                np.savetxt(file, chunk, delimiter=delimiter)

    def get_max_frequency_range_intensity(self, start_freq, end_freq):
        """
        获取特定频率范围内的最大频率值以及其对应的强度
        :param start_freq: 起始频率
        :param end_freq: 结束频率
        :return: max_frequencies, max_intensities
        """
        # 获取起始频率和结束频率的索引
        # 获取起始频率并向下取整
        start_index = int(np.floor(start_freq / self.freq_axis_step))
        # 获取结束频率并向上取整
        end_index = int(np.ceil(end_freq / self.freq_axis_step))
        print(f'start_index: {start_index}')
        print(f'end_index: {end_index}')

        # 在指定范围内找到最大强度及其索引
        # max_intensities: 每一列中最大强度的值
        max_intensities_index = np.argmax(self.fft_result_db[start_index:end_index+1,:], axis=0)
        # print(f'max_intensities_index: {max_intensities_index}')

        max_intensities = self.fft_result_db[start_index:end_index+1,:][max_intensities_index, np.arange(self.num_cols)]
        # print(f'max_intensities: {max_intensities}')

        # 获取最大强度对应的频率
        # 频率索引要添加一个偏移量
        max_frequencies = self.freq_axis[max_intensities_index+start_index]
        return max_frequencies, max_intensities
        # print(f'max_frequencies: {max_frequencies}')

if __name__ == "__main__":

    # 读取的文件路径
    file_path = "C:\\Users\\liu-i\\Desktop\\FFT\\data\\2023_11_05-15_35_48--271332.csv"
    # 保存的文件路径
    output_file_path = "C:\\Users\\liu-i\\Desktop\\FFT\\data\\my_output.csv"

    # 10240 / 271.332
    # 频率轴的计算方法
    # print(1 / ((10240)*(271.332 / 10240)))
    # for i in range(10):
        # print(i / ((10240)*(271.332 / 10240)))

    # # 创建一个FiberOpticDataProcessor对象
    processor = FiberOpticDataProcessor(file_path)
    # # 加载数据
    processor.load_data()
    # # 对数据进行FFT变换
    processor.fft_data()
    # # 保存数据
    processor.save_data(output_file_path)
    # # 画图
    processor.plot_data(100,110)

    start_freq = 0.1
    end_freq = 0.4
    # processor.get_max_frequency_range_intensity(start_freq, end_freq)
    max_frequencies, max_intensities = processor.get_max_frequency_range_intensity(start_freq, end_freq)
    print(f"max_frequencies: {max_frequencies}")
    print(f"max_intensities: {max_intensities}")
    # print(f"max_frequency: {max_frequency}")
    # print(f"max_intensity: {max_intensity}")

    # 测试get_all_files_in_directory函数
    # directory_path = "D:\\永安变电站\\20231103铁岭永安变.part01\\20231103铁岭永安变\\永安变测试数据_20231112\\振动设备\\anpu2x10"
    # processor.get_all_files_in_directory(directory_path)

    # 测试process_folder函数
    # folder_path = "D:\\永安变电站\\20231103铁岭永安变.part01\\20231103铁岭永安变\\永安变测试数据_20231112\\振动设备\\anpu2x10"
    # output_folder = "C:\\Users\\liu-i\\Desktop\\FFT\\data"
    # processor_none = FiberOpticDataProcessor(None)
    # processor_none.process_folder(folder_path,output_folder)