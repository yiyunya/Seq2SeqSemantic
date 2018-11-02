def load_in_text(path):
    f = open(path, 'r')
    lines = f.readlines()
    sentences = []
    for line in lines:
        tmp = line.split(',')
        if len(tmp) > 0:
            tmp[0] = tmp[0].split('[')[1]
            tmp[-1] = tmp[-1].split(']')[0]
        sentences.append(tmp)
    return sentences

def write_in(path, type = 'train'):
    sentences = load_in_text(path)
    f = open('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/'+type+'2.in','w')
    for sentence in sentences:
        for token in sentence:
            print(token, end = ' ',file = f)
        print('', file = f)

write_in('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d2_train_in.txt')
write_in('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d2_valid_in.txt', type = 'val')
write_in('/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d2_test_in.txt', type = 'test')