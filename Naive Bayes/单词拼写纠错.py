import re
import collections as cls
# 函数说明： 把word里面的单词全部小写
def words(text): return re.findall('[a-z]+',text.lower())

# 函数说明：创建词频列表，通过语料库放出各个单词的概率，语料库越大训练的准确率就越高，求先验概率
def train(features):
    model = cls.defaultdict(lambda: 1)
    for f in features:
        model[f]+=1
    return model

# 建立词汇表
NWORDS = train(words(open("mywordslib.txt").read()))
# 建立字母表
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# 函数说明：把一个单词可能的集中变换放入到一个集合中，或者说是编辑距离相差为1的，求编辑距离
def edits1(word):
    n = len(word)
    return set([word[0:i]+word[i+1:] for i in range(n)]+[word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]+
              [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet]+[word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet])

# 函数说明：求编辑距离等于2的，可能性会很多，所以只返回候选词，选出正确的单词即在预料库中出现过的
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

# 函数说明：判断单词是否出现过
def known(words): return set(w for w in words if w in NWORDS)

# 函数说明：candidates就是一个集合，or的作用是体现优先级，先算前面的，再算后面编辑距离为1的，最后是返回编辑距离为2的
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates,key = lambda w: NWORDS[w])
print(correct('morw'))
