'''改进：
    1.Pearson系数在小数据集上可以出结果了
    2.在调用recommend_with_predict函数时，避免对top10_simliar函数的重复调用，减少运行时间
    3.Laplace值通过库函数添加（line 284）
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import *

'''拉普拉斯的实现'''
# def laplace_function(numbers, beta):
#     result = (1 / (2 * beta)) * np.e**(-1 * (np.abs(numbers) / beta))
#     return result
# #在-5到5之间等间隔的取10000个数
# numbers = np.linspace(-5,5,10000)
# y1 = [laplace_function(number_picked, 0.5) for number_picked in numbers]
# y2 = [laplace_function(number_picked, 1) for number_picked in numbers]
# y3 = [laplace_function(number_picked, 2) for number_picked in numbers]


# plt.plot(numbers,y1,color='r',label='beta:0.5')
# plt.plot(numbers,y2,color='g',label='beta:1')
# plt.plot(numbers,y3,color='b',label='beta:2')
# plt.title("Laplace distribution")
# plt.legend()
# plt.show()


# 最开始的部分，读取数据
movies = pd.read_csv("/Users/jerryw/Desktop/ml-latest-small/movies.csv")
ratings = pd.read_csv("/Users/jerryw/Desktop/ml-latest-small/ratings.csv")
data = pd.merge(movies, ratings, on='movieId')  # 通过两数据框之间的movieId连接
data = data[['userId', 'rating', 'movieId', 'title']
            ].sort_values('userId')  # 未排序的是timestamp, genres
data = data.to_csv(
    '/Users/jerryw/Desktop/ml-latest-small/data.csv', index=False)

file = open("/Users/jerryw/Desktop/ml-latest-small/data.csv",
            'r', encoding='UTF-8')
# 读取data.csv中每行中除了名字的数据
data = {}  # 存放每位用户评论的电影和评分
for line in file.readlines()[1:100837]:
    # 注意这里不是readline()
    line = line.strip().split(',')
    # 如果字典中没有某位用户，则使用用户ID来创建这位用户
    if not line[0] in data.keys():
        data[line[0]] = {line[3]: line[1]}
    # 否则直接添加以该用户ID为key字典中
    else:
        data[line[0]][line[3]] = line[1]

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
    # print(user1, user2)
    # print(common)

    
    # 计算分子
    sum_of_product_of_xy = 9743 * sum(float(user1_data[movie]) * float(user2_data[movie]) for movie in common)
    product_of_sum_of_x_and_y = sum(float(user1_data[movie]) for movie in common) * sum(float(user2_data[movie]) for movie in common)
    num = sum_of_product_of_xy - product_of_sum_of_x_and_y
    

    # 计算分母参数以及分母
    sum1 = pow(sum([float(user1_data[movie]) for movie in user1_data]), 2)
    sum2 = pow(sum([float(user2_data[movie]) for movie in user2_data]), 2)
    sum1Sq = 9743 * sum([pow(float(user1_data[movie]), 2) for movie in user1_data])
    sum2Sq = 9743 * sum([pow(float(user2_data[movie]), 2) for movie in user2_data])
    den1 = sum1Sq - sum1
    den2 = sum2Sq - sum2
    # print(den1)
    # print(den2)
    den = sqrt(den1 * den2)

    if den == 0:
        return 0
    r = num / den
    return r


# 计算某个用户与其他用户的相似度
def top10_simliar(userID, choice):
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
    return res[:4]


# 用随机数加噪声
def noisy_count(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1.-u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value


# 加入随机噪声使得数据查询结果不确定
def laplace_mech(data, sensitivety, epsilon):
    data += noisy_count(sensitivety, epsilon)
    return data


# 根据用户推荐电影给其他人
def recommend(user):
    # 相似度最高的用户
    print("1.cosine\n2.preason_sim\n3.preason_sim_pro")
    choice = input("用哪个方法？")
    # 打印相关度最高的前四个用户
    RES = top10_simliar('1', choice)
    print(RES)
    top_sim_user = top10_simliar(user, choice)[0][0]
    print("\n", top_sim_user, "\n")
    # 相似度最高的用户的观影记录
    items = data[top_sim_user]
    recommendations = []
    # 筛选出该用户未观看的电影并添加到列表中
    for item in items.keys():
        if item not in data[user].keys():
            recommendations.append((item, items[item]))
    recommendations.sort(key=lambda val: val[1], reverse=True)  # 按照评分排序
    # 返回评分最高的10部电影
    return recommendations[:10]


def recommend_with_predict(user):
    # 相似度最高的用户
    print("1.cosine\n2.preason_sim\n3.preason_sim_pro")
    choice = input("用哪个方法？")
    # 打印相关度最高的前四个用户
    RES = top10_simliar(user, choice)
    print(RES)
    # 得到用户名和相似值，为下一步计算做准备
    message1 = RES[0]
    top_sim_user_1 = message1[0]
    # 计算预测值的权重
    weight1 = message1[1]
    print("最相似用户1的名字和相似度：\n", top_sim_user_1, "\n", weight1, "\n")
    message2 = RES[1]
    top_sim_user_2 = message2[0]
    weight2 = message2[1]
    print("最相似用户2的名字和相似度：\n", top_sim_user_2, "\n", weight1, "\n")

    top_sim_user_1_data = data[top_sim_user_1]
    top_sim_user_2_data = data[top_sim_user_2]
    predict_movies = []

    # 做出预测
    choice_of_laplace = input("要不要加差分隐私？\n“y”代表确认")
    if choice_of_laplace == 'y':
        sensitivety = float(input("输入敏感度："))
        epsilon = float(input("输入差分隐私参数："))
        beta = sensitivety / epsilon

        for movie in top_sim_user_1_data.keys():
            if movie not in top_sim_user_2_data.keys():
                # 遍历用户1的时候，如果用户2未看过该电影，预测评分直接为用户1的评分
                predict_score = float(top_sim_user_1[1]) + np.random.laplace(0, beta, 1)
                predict_movies.append((movie, predict_score))
            else:
                # 如果用户2看过该电影，预测评分为两名用户的评分分别乘以各自的权重（相似度）
                predict_score = (float(top_sim_user_1[1]) * weight1 + float(top_sim_user_2[1]) * weight2) / (weight1 + weight2)
                predict_score = predict_score + np.random.laplace(0, beta, 1)
                predict_movies.append((movie, predict_score))

        for movie in top_sim_user_2_data:
            # 遍历一遍用户2，把用户2看过但是用户1没看过的电影再照上面的办法生成预测评分
            if movie in top_sim_user_1_data:
                continue
            else:
                predict_score = float(top_sim_user_2[1]) + np.random.laplace(0, beta, 1)
                predict_movies.append((movie, predict_score))
    else:
        for movie in top_sim_user_1_data.keys():
            if movie not in top_sim_user_2_data.keys():
                predict_movies.append((movie, float(top_sim_user_1[1])))
            else:
                # print(type(top_sim_user_1[1]))
                predict_score = (float(top_sim_user_1[1]) * weight1 + float(top_sim_user_2[1]) * weight2) / (weight1 + weight2)
                predict_movies.append((movie, predict_score))
        for movie in top_sim_user_2_data:
            if movie in top_sim_user_1_data:
                continue
            else:
                predict_movies.append((movie, float(top_sim_user_2[1])))


    # 按照评分从高到低排序
    predict_movies.sort(key = lambda val: val[1])
    predict_movies.reverse()
    return predict_movies[:10]


if __name__ == '__main__':
    user = input("输入要推荐的人的编号：")
    way_to_recommend = input("用哪种推荐方法？\n1.找一个相似度最高的人然后拉取其列表\n2.找两个相似度最高的人然后预测这个人的分数\n")
    if way_to_recommend == '1':
        Recommendations = recommend(user)
    elif way_to_recommend == '2':
        Recommendations = recommend_with_predict(user)
    print(Recommendations)
    print("\n")
