from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

def cut_word(sentence):
    return list(jieba.cut(sentence, cut_all=False))

prediction = "他发生了什么事？"
reference = "他发生了什么事？"

predictions = cut_word(prediction)
references = [cut_word(reference)]

print("分词结果：")
print("predictions:", predictions)
print("references:", references)

# 你的原始写法
if len(predictions) == 1:
    weights = (1.0,)
elif len(predictions) == 2:
    weights = (0.5, 0.5)
elif len(predictions) == 3:
    weights = (1/3, 1/3, 1/3)
else:
    weights = (0.25, 0.25, 0.25, 0.25)
bleu1 = sentence_bleu(references, predictions, weights=weights)
print(f"你的写法 BLEU: {bleu1}")

# 推荐写法
max_order = min(4, len(predictions), len(references[0]))
if max_order == 0:
    bleu2 = 0.0
else:
    weights2 = tuple([1.0 / max_order] * max_order)
    bleu2 = sentence_bleu(references, predictions, weights=weights2, smoothing_function=SmoothingFunction().method4)
print(f"推荐写法 BLEU: {bleu2}")
