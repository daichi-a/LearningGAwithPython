# coding: utf-8
import numpy as np

class Population:
    def __init__(self, population_size, chromosome_length, random_seed):
        # コンストラクタ
        # 継承するのでメンバ変数はprivateにはできない
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.score_array = \
            np.zeros([self.population_size], dtype=np.int32)
        np.random.seed(random_seed)
        
    def initialize(self):
        # Populationの初期化を行うインスタンス関数
        # ランダムに染色体ビット配列を生成する
        self.chromosome_list = list([])
        for i in range(self.population_size):
            new_chromosome = \
                np.zeros(self.chromosome_length, dtype=np.int32)
            for j in range(self.chromosome_length):
                if np.random.random() >= 0.5:
                    new_chromosome[j] = 1
                else:
                    new_chromosome[j] = 0
            self.chromosome_list.append(new_chromosome)

    def evaluate(self):
        # 一つ一つの染色体ビット配列に対して，評価(点数付け)を行うインスタンス関数
        # このプログラムは，全てのビットを1にする→つまり合計を最大化する最適化を行う
        # したがって，染色体ビット配列の合計値が大きい方が評価が高い
        for (i, chromosome) in enumerate(self.chromosome_list):
            chromosome_sum = 0
            for gene in chromosome:
                chromosome_sum += gene
            self.score_array[i] = chromosome_sum

    def onepoint_crossover(self, tournament_unit_size):
        # 1点交叉を行うインスタンス関数
        # 新しい世代を格納するリストを新たに作って，ポインタをキープしておく
        new_chromosome_list = list([])

        for i in range(int(np.floor(self.population_size / 2))):
            # 1回の交叉で2つできるので，1/2回数でよい
            # ただし集団サイズが2で割り切れなければならない

            parent_list = list([])
            # 親二つを選別する
            for j in range(2):
                # トーナメント方式
                # 集団の中からランダムにtournament_unit_size分選んで，
                # その中で一番点数が高い個体(染色体)を親に選ぶ
                tournament_unit_index_array = \
                    np.zeros(tournament_unit_size, dtype=np.int32)
                tournament_unit_score_array = \
                    np.zeros(tournament_unit_size, dtype=np.int32)
                for k in range(tournament_unit_size):
                    random_selected_index = int(np.floor(np.random.random() * \
                                                     self.population_size))
                    tournament_unit_index_array[k] = random_selected_index
                    tournament_unit_score_array[k] = \
                        self.score_array[random_selected_index]
                    
                best_chromosome_in_tournament_unit = \
                    tournament_unit_index_array[np.argmax(tournament_unit_score_array)]
                parent_list.append(self.chromosome_list[best_chromosome_in_tournament_unit])

            # 2つ親が得られたら，ランダムに交叉する1点を選ぶ
            crossover_point = \
                np.floor(np.random.random() * self.chromosome_length)
            # 0だったら交叉されないので修正
            if crossover_point == 1:
                crossover_point = 1
            new_chromosome0 = np.empty(self.chromosome_length, dtype=np.int32)
            new_chromosome1 = np.empty(self.chromosome_length, dtype=np.int32)
            parent0 = parent_list[0]
            parent1 = parent_list[1]
            for current_point in range(self.chromosome_length):
                if current_point <= crossover_point:
                    # 交叉点まで
                    new_chromosome0[current_point] = \
                        parent0[current_point]
                    new_chromosome1[current_point] = \
                        parent1[current_point]
                else:
                    # 交叉点のあと
                    new_chromosome0[current_point] = \
                        parent_list[1][current_point]
                    new_chromosome1[current_point] = \
                        parent_list[0][current_point]
            new_chromosome_list.append(new_chromosome0)
            new_chromosome_list.append(new_chromosome1)

        # population size分子供を作り終わったら，
        # 保持しているポインタを新しいもので置き換える
        self.chromosome_list = new_chromosome_list

    def mutation(self, mutation_rate):
        # 突然変異を行うインスタンス関数
        # ランダムで生成した値がmutarion_rate以下なら，ビットをひっくり返す
        for chromosome in self.chromosome_list:
            for current_position in np.arange(self.chromosome_length):
                if np.random.random() <= mutation_rate:
                    chromosome[current_position] = \
                        chromosome[current_position] * -1 + 1

    def get_best_individual(self):
        best_score_index = np.argmax(self.score_array)
        return self.chromosome_list[best_score_index]

    def get_best_score(self):
        chromosome_sum = 0
        best_individual = self.get_best_individual()
        for gene in best_individual:
            chromosome_sum += gene
        return chromosome_sum

    def get_individual(self, index):
        return self.chromosome_list[index]
