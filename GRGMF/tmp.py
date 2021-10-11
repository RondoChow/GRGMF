import numpy as np
import pandas as pd

sequence1 = 'TGTTACGG'
sequence2 = 'GGTTGACTA'
s1 = ''
s2 = ''
gap = -2
score_matrix = pd.read_excel('score.xlsx')  # 匹配得分
print(score_matrix)
best_matrix = np.empty(shape=(len(sequence2) + 1, len(sequence1) + 1), dtype=int)


def get_match_score(s1, s2):
    score = score_matrix[s1][s2]
    return score


def get_matrix_max(matrix):  # 得到最大分数下标
    Max = matrix.max()
    for i in range(len(sequence2) + 1):
        for j in range(len(sequence1) + 1):
            if matrix[i][j] == Max:
                return (i, j)


for i in range(len(sequence2) + 1):
    for j in range(len(sequence1) + 1):
        if i == 0 or j == 0:
            best_matrix[i][j] = 0
        else:
            match = get_match_score(sequence2[i - 1], sequence1[j - 1])
            gap1_score = best_matrix[i - 1][j] + gap
            gap2_score = best_matrix[i][j - 1] + gap
            match_score = best_matrix[i - 1][j - 1] + match
            score = max(gap1_score, gap2_score, match_score)
            if score > 0:
                best_matrix[i][j] = score
            else:
                best_matrix[i][j] = 0
print(best_matrix)

# traceback
i, j = get_matrix_max(best_matrix)
while (best_matrix[i][j] != 0):
    match = get_match_score(sequence2[i - 1], sequence1[j - 1])
    if i > 0 and j > 0 and best_matrix[i][j] == best_matrix[i - 1][j - 1] + match:
        s1 += sequence1[j - 1]
        s2 += sequence2[i - 1]
        i -= 1;
        j -= 1
    elif i > 0 and best_matrix[i, j] == best_matrix[i - 1, j] + gap:
        s1 += '-'
        s2 += sequence2[i - 1]
        i -= 1
    else:
        s1 += sequence1[j - 1]
        s2 += '-'
        j -= 1
print((s1[::-1] + '\n' + s2[::-1]))