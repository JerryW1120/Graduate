'''改进：
    1.Pearson系数在小数据集上可以出结果了
    2.在调用recommend_with_predict函数时，避免对top10_simliar函数的重复调用，减少运行时间
    3.Laplace值通过库函数添加(np.random.laplace)
    4.预测算法改正，之前参数有错误
    5.此处的预测分数未进行均值中心化
    6.均值中心化之后还是一个评分为5的占优势，需要多个用户共同参与预测了
    7.我tm的，终于搞定了多个用户，看样子结果还蛮不错的，应该可以使劲跑了
    8.将参与预测评分的用户增加到了40个
    大数据集：283228个用户，58098部电影（已替换）
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import *



# 最开始的部分，读取数据
movies = pd.read_csv("/home/jerryw/Desktop/graduate/data/ml-latest/movies.csv")
ratings = pd.read_csv("/home/jerryw/Desktop/graduate/data/ml-latest/ratings.csv")
data = pd.merge(movies, ratings, on='movieId')  # 通过两数据框之间的movieId连接
data = data[['userId', 'rating', 'movieId', 'title']].sort_values('userId')  # 未排序的是timestamp, genres
data = data.to_csv('/home/jerryw/Desktop/graduate/data/ml-latest/data.csv', index=False)

file = open("/home/jerryw/Desktop/graduate/data/ml-latest/data.csv", 'r', encoding='UTF-8')
# 读取data.csv中每行中除了名字的数据
data = {}  # 存放每位用户评论的电影和评分
for line in file.readlines()[1:27753445]: # 大数据集的data总共27753445行
    # 注意这里不是readline()
    line = line.strip().split(',')
    # 如果字典中没有某位用户，则使用用户ID来创建这位用户
    if not line[0] in data.keys():
        data[line[0]] = {line[3]: line[1]}
    # 否则直接添加以该用户ID为key字典中
    else:
        data[line[0]][line[3]] = line[1]
file.close()
# print(data)

# for userId in data:
#     print (userId)
#     for movieId in data[userId]:
#         print ("   ", movieId, ':', data[userId][movieId])


"""计算任何两位用户之间的相似度，由于每位用户评论的电影不完全一样，所以兽先要找到两位用户共同评论过的电影
"""


def Euclidean(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    # 找到两位用户都评论过的电影，并计算欧式距离
    for key in user1_data.keys():
        if key in user2_data.keys():
            # distance越大表示两者越相似
            distance += pow(float(user1_data[key])-float(user2_data[key]), 2)

    return 1/(1+sqrt(distance))  # 这里返回值越小，相似度越大


def cosine(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    common = {}

    # 找到两位用户都评论过的电影
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # 如果没有共同评论过的电影，则返回0
    n = len(common)  # 共同电影数目
    # print(n,common)

    # 计算评分和
    sum1 = sum([float(user1_data[movie]) for movie in common])
    sum2 = sum([float(user2_data[movie]) for movie in common])

    # 计算评分平方和
    sum1Sq = sum([pow(float(user1_data[movie]), 2) for movie in user1_data])
    sum2Sq = sum([pow(float(user2_data[movie]), 2) for movie in user2_data])

    # 计算乘积和
    PSum = sum([float(user1_data[it])*float(user2_data[it]) for it in common])

    # 计算相关系数
    den = sqrt((sum1Sq)*(sum2Sq))
    if den == 0:
        return 0
    result = PSum/den
    return result


def pearson_sim(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    common = {}

    # 找到两位用户都评论过的电影
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # 如果没有共同评论过的电影，则返回0
    n = len(common)  # 共同电影数目
    # print(n,common)

    # 计算评分和
    sum1 = sum([float(user1_data[movie]) for movie in common])
    sum2 = sum([float(user2_data[movie]) for movie in common])

    # 计算评分平方和
    sum1Sq = sum([pow(float(user1_data[movie]), 2) for movie in common])
    sum2Sq = sum([pow(float(user2_data[movie]), 2) for movie in common])

    # 计算乘积和
    PSum = sum([float(user1_data[it]) * float(user2_data[it]) for it in common])

    # 计算相关系数
    num = n * PSum - (sum1 * sum2)
    den = sqrt((n * sum1Sq - pow(sum1, 2)) * (n * sum2Sq - pow(sum2, 2)))
    if den == 0:
        return 0
    r = num / den
    return r


def pearson_sim_pro(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    common = {} # 以字典形式包括目标用户和参考用户共同看过的电影

    # 找到两位用户都评论过的电影
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # 如果没有共同评论过的电影，则返回0
    n = len(common) 
    # if user2 == '366' or user2 == '417':
    #     print(common)
    #     for movie in common.keys():
    #         print(user1, movie, user1_data[movie])
    #         print(user2, movie, user2_data[movie])
    
    # 计算分子,9742是电影数目
    sum_of_product_of_xy = 58098 * sum(float(user1_data[movie]) * float(user2_data[movie]) for movie in common)
    product_of_sum_of_x_and_y = sum(float(user1_data[movie]) for movie in common) * sum(float(user2_data[movie]) for movie in common)
    num = sum_of_product_of_xy - product_of_sum_of_x_and_y
    

    # 计算分母参数以及分母
    sum1 = pow(sum([float(user1_data[movie]) for movie in user1_data]), 2)
    sum2 = pow(sum([float(user2_data[movie]) for movie in user2_data]), 2)
    sum1Sq = 58098 * sum([pow(float(user1_data[movie]), 2) for movie in user1_data])
    sum2Sq = 58098 * sum([pow(float(user2_data[movie]), 2) for movie in user2_data])
    den1 = sum1Sq - sum1
    den2 = sum2Sq - sum2
    den = sqrt(den1 * den2)

    if den == 0:
        return 0
    r = num / den
    return r


# 计算某个用户与其他用户的相似度
def top_simliar(userID, choice):
    # 空列表，有东西再往进加
    res = []
    for userid in data.keys(): #这里要做循环，把目标用户与所有其他用户的相似度都计算出来再排序
        # 排除与自己计算相似度
        if not userid == userID:
            if(choice == '1'):
                simliar = cosine(userID, userid)
            elif(choice == '2'):
                simliar = pearson_sim(userID, userid)
            else:
                simliar = pearson_sim_pro(userID, userid)
            res.append((userid, simliar))
    res.sort(key = lambda val: val[1])
    res.reverse()
    return res

def recommend_with_predict(user):
    # 相似度最高的用户
    print("1.cosine\n2.preason_sim\n3.preason_sim_pro")
    choice = input("用哪个方法？")
    # 打印相关度最高的前十个用户
    RES = top_simliar(user, choice)
    # print("相关度最高的前二十名用户：", RES)
    
    # 得到用户名和相似值，为下一步计算做准备
    top_sim_user_data = [] # 存放最相似用户看过的电影和相应的评分
    weight = [] # 存放最相似用户的相似度
   
    count = 0
    while count < 40:
        top_sim_user_data.append(data[RES[count][0]])
        weight.append(float(RES[count][1]))
        # print("最相似用户", count + 1, "的名字和相似度：\n", RES[count][0], "\n", RES[count][1], "\n")        
        count += 1
        
    user_data = data[user]
    # 计算评分平均数
    ave = []
    count = 0
    while count < 40:
        ave.append(sum(float(top_sim_user_data[count][movie]) for movie in top_sim_user_data[count].keys()) / len(top_sim_user_data[count]))
        count += 1

    # 被推荐人的平均分
    ave_predicted = sum(float(user_data[movie]) for movie in user_data.keys()) / len(user_data) 
    
    # 读取电影名字
    file = open("/home/jerryw/Desktop/graduate/data/ml-latest/movies.csv", 'r', encoding='UTF-8')
    movies = []
    for line in file.readlines()[1:58099]:
        line = line.strip().split(',')
        movies.append(line[1])
    file.close()
    
    movies = sorted(movies)

    # 用一个字典存储各个用户的电影及其分数，如果没有看过，则分数为0
    user_scores = {}
    count = 0
    while count < 40:
        user_scores[count] = {}
        # 模仿之前的data导入
        for movie in movies:            
            if movie not in top_sim_user_data[count].keys():
                user_scores[count][movie] =  0.0
            else:
                user_scores[count][movie] = float(top_sim_user_data[count][movie])
        count += 1

    predict_movies = []
    # 做出预测
    choice_of_laplace = input("要不要加差分隐私？\n“y”代表确认")
    if choice_of_laplace == 'y':
        sensitivety = float(input("输入敏感度："))
        epsilon = float(input("输入差分隐私参数："))
        beta = sensitivety / epsilon

    # 根据前几名最相似的用户，将他们对各自电影的评分乘以权重，得到预测分数
    for movie in movies:
        count = 0
        sum_of_weight = 0 # 权重求和
        predict_score = 0
        while count < 40:
            # 评分为0，即该用户没有看过这个电影
            if user_scores[count][movie] == 0.0: 
                weight_temp =  0.0 # weight_temp是循环到每个用户的权重值，算加权总分的时候可以用
            else:
                weight_temp = weight[count]
                predict_score += float(user_scores[count][movie] - ave[count]) * weight_temp
            sum_of_weight += weight_temp           
            count += 1

        # 若权重和不为0则计算原始预测值，反之，原始预测值为被推荐用户对其所有观看电影的平均分
        if sum_of_weight != 0: 
            predict_score = predict_score / sum_of_weight + ave_predicted
        else:
            predict_score = ave_predicted

        # 判断要不要加拉普拉斯噪声
        if choice_of_laplace == 'y': 
            predict_score += np.random.laplace(0, beta, 1)

        predict_movies.append((movie, float(predict_score)))
    
    # 去重
    predict_movies = list(set(predict_movies))

    # 按照评分从高到低排序
    predict_movies.sort(key = lambda val: val[1])
    predict_movies.reverse()
    return predict_movies[:10]


if __name__ == '__main__':
    user = input("输入要推荐的人的编号：")
    Recommendations = recommend_with_predict(user)
    print(Recommendations)
    print("\n")
