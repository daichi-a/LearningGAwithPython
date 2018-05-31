# -*- coding: utf-8 -*-
from GABaseLib import Population
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

class PopulationCUDA(Population, object):
    def __init__(self, population_size, chromosome_length, random_seed):
        super(PopulationCUDA, self).__init__(population_size, chromosome_length, random_seed)
        
        # Evaluate FunctionをCUDAで記述する
        self.__evaluate_module = SourceModule("""
__global__ void evaluate(int* score, int* chromosomeLength, int* chromosome){
        const int index = threadIdx.x;
        score[index] = 0;
        for(int i=0; i<chromosomeLength[0]; i++){
            score[index] += chromosome[chromosomeLength[0] * index + i];
        }
}
""")
        # Evaluate Functionをコンパイルして，
        # 関数オブジェクトにして，
        # インスタンス変数として保持する
        self.__evaluate_function = \
            self.__evaluate_module.get_function("evaluate")

        # CUDAにはnumpy配列しか渡せないので，必要な値のnumpy配列を生成しておく
        self.chromosome_length_in_array = \
            np.empty(1, dtype=np.int32)
        self.chromosome_length_in_array[0] = self.chromosome_length

        return None

    def evaluate(self):
        # CUDAに渡すために全ての染色体を一つの配列に入れる
        chromosome_array_for_cuda = \
            np.empty(self.population_size * self.chromosome_length, \
                     dtype=np.int32)
        for (i, chromosome) in enumerate(self.chromosome_list):
            for j in np.arange(self.chromosome_length):
                chromosome_array_for_cuda[self.chromosome_length * i + j] = \
                    chromosome[j]
        # pycuda.driver.InとOutを使って，実行する
        # 1024は同時に計算できるスレッド数(x, y, z)で，ここでは集団サイズ
        # x * y * zの値がGPUの規定以下(ここでは1024)/になっていなければならない
        # x * y * zの値が規定以下なら，x, y, zの割り振りは自由
        # gridはスレッドをさらにまとめた(x, y)もので今回は1つで十分
        self.__evaluate_function(drv.Out(self.score_array), \
                                 drv.In(self.chromosome_length_in_array), \
                                 drv.In(chromosome_array_for_cuda),
                                 block=(1024, 1, 1),
                                 grid=(1, 1))
        print(self.score_array)
