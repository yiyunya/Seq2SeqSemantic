def load_out_text(path):
    f = open(path, 'r', encoding="utf-8")
    lines = f.readlines()
    stopwords = ['(', ')', ',']
    sentences = []
    for line in lines:
        tmp = line.split('answer')
        if len(tmp) > 0:
            tmp = tmp[1][:-3]
#            print(tmp)
            word = ''
            ans = []
            for letter in tmp:
                if letter in stopwords:
                    if word != '':
                        ans.append(word)
                        word = ''
                    ans.append(letter)
                else:
                    word = word + letter
            sentences.append(ans)
    return(sentences)


def write_out(path,type = 'train'):
    sentences = load_out_text(path)
    f = open('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/'+type+'2.out', 'w')
    for sentence in sentences:
        for token in sentence:
            print(token, end=' ', file=f)
        print('', file=f)

write_out('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d2_train_out.txt')
write_out('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d2_valid_out.txt', type = 'val')
