import numpy as np
import Project4_code.Porter_Stemmer_Python as PorterStemmer
from beautifultable import BeautifulTable
import re


def get_sentences():
    file = open("Project4_sentences/sentences.txt", "r", encoding="utf8")
    result = re.sub(r"[^A-z \n]", "", file.read().lower()).split('\n')
    file.close()
    return result


def get_stop():
    file = open("Project4_sentences/stop_words.txt", 'r', encoding="utf8")
    result = file.read().split('\n')
    file.close()
    return result


def tokenize(sent):
    return list(map(lambda sentence: sentence.split(), sent))


def delete_stop(sent):
    stop_words = get_stop()

    def remove_stop_words_from_one_sentence(sentence):
        return list(filter(lambda word: not stop_words.__contains__(word), sentence))

    return list(map(remove_stop_words_from_one_sentence, sent))


def stemm(sent):
    porter = PorterStemmer.PorterStemmer()

    def delete_stop_one(sentence):
        return list(map(lambda word: porter.stem(word, 0, len(word) - 1), sentence))

    return list(map(delete_stop_one, sent))


def clean_text():
    return stemm(delete_stop(tokenize(get_sentences())))


def get_words(min=0):
    if min > 0:
        def filter_minimally_used_words(tuple):
            return tuple[1] > min

        def tuple_to_value(tuple):
            return tuple[0]

        return list(map(tuple_to_value, filter(filter_minimally_used_words, count_words().items())))
    return list(count_words())


def count_words():
    new_list = clean_text()

    words = {}

    for sentence in new_list:
        for word in sentence:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

    return words


def feat_vector(min=0):
    new_list = clean_text()

    words = get_words(min=min)

    def sentence_row(sentence):
        vector = len(words) * [0]

        for word in sentence:
            # If we only want the words with `n` min,
            # the word may not exist in `encountered_words`
            if words.__contains__(word):
                vector[words.index(word)] += 1

        return vector

    return list(map(sentence_row, new_list))


def print_TDM_table(table):
    table[0].insert(0, "Keyword set")
    table[0].insert(int(len(table)/2), "Keyword set")
    for y in range(1, len(table)):
        table[y].insert(0, y)

    for z in range(1, len(table)):
        table[z].insert(int(len(table)/2), z)

    print_table_1 = BeautifulTable(max_width=400)
    print_table_1.column_headers = table[0][0:int(len(table[0])/2)]
    for x in range(1, int(len(table))):
        print_table_1.append_row(table[x][0:int(len(table)/2)])
    print(print_table_1)

    print_table_2 = BeautifulTable(max_width=400)
    print_table_2.column_headers = table[0][int(len(table[0])/2):int(len(table[0]))]
    for x in range(1, int(len(table))):
        print_table_2.append_row(table[x][int(len(table)/2):int(len(table))])
    print(print_table_2)


def nearest_cluster(cluster_weights, current_pattern):
    distance = np.zeros(len(cluster_weights))

    # Euclidean distance:

    # For each cluster
    for i in range(len(cluster_weights)):
        distance[i] = 0

        # For each word
        for j in range(len(current_pattern)):
            val = pow((current_pattern[j] - cluster_weights[i][j]), 2)
            distance[i] = distance[i] + val

    min_index = 0

    for i in range(len(distance)):
        if distance[i] < distance[min_index]:
            min_index = i

    return min_index


def learn(objects, iterations, cluster=1):
    weight = np.random.rand(cluster, len(objects[0]))

    for q in range(iterations):
        for obj in objects:
            index = nearest_cluster(weight, obj)
            change = 0.05 * (obj - weight[index])
            weight[index] = weight[index] + change

    return weight


def split(weights, obj):
    sent = []

    for _ in weights:
        sent.append([])

    file = open("Project4_sentences/sentences.txt", 'r', encoding="utf8")
    sentences = file.read().split('\n')

    for x in range(len(obj)):
        index = nearest_cluster(weights, obj[x])
        sent[index].append((x, sentences[x]))

    file.close()
    return sent


def normalize(vect):
    count = len(vect[0])
    max_count = np.zeros(count)

    for x in range(count):
        for vector in vect:
            if vector[x] > max_count[x]:
                max_count[x] = vector[x]

    for x in range(len(vect)):
        for i in range(count):
            vect[x][i] = 1 if vect[x][i] else 0

    return vect


def print_feature_vector(words):
    print_table_1 = BeautifulTable(max_width=400)
    print_table_1.column_headers = ["Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count", "Word", "Count"]
    tempList = []
    for key, value in count_words().items():
        tempList.append(key)
        tempList.append(value)
        if len(tempList) == 18:
            print_table_1.append_row(tempList)
            del tempList[:]

    print(print_table_1)


def main():
    min = 2
    vector = feat_vector(min=min)

    table = [get_words(min=min)] + vector

    print_TDM_table(table)

    norm_vect = normalize(vector)

    result = learn(norm_vect, 500, cluster=20)

    sent = split(result, norm_vect)

    sent = list(filter(lambda x: x, sent))

    print_feature_vector(count_words().items())

    occured_twice_or_more = list(({k: v for k, v in count_words().items() if v > 1}).items())
    print("Number of words with 2 or more occurrences:", len(occured_twice_or_more))

    occured_three_or_more = list(({k: v for k, v in count_words().items() if v > 2}).items())
    print("Number of words with 3 or more occurrences:", len(occured_three_or_more))

    # for a in range(len(occured_twice_or_more)):
    #     for b in range(len(occured_twice_or_more[a])):
    #         print(occured_twice_or_more[a][b])



    def to_string(tuple):
        return str(tuple[0]) + ") " + tuple[1]

    combine_sent = list(
        map(lambda cluster: list(map(to_string, cluster)), sent))

    num_of_clusters = len(combine_sent)

    for a in range(len(combine_sent)):
        print("Cluster ", str(a + 1) + ":")
        for b in range(len(combine_sent[a])):
            print(combine_sent[a][b])

    print("Completed", num_of_clusters)



if __name__ == "__main__":
    main()
