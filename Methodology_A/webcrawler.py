
# import urllib.request

# url="http://www.amazon.com/s?i=alexa-skills&bbn=13727921011&rh=n%3A13727921011%2Cp_n_date%3A14284927011&page=2&qid=1601731757&ref=sr_pg_3";
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# uf = urllib.request.urlopen(url);
# html = uf.read();

# print (html);



# import requests
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'};
# page = requests.get("https://www.amazon.com/s?i=alexa-skills&bbn=13727921011&rh=n%3A13727921011%2Cp_n_date%3A14284927011&page=2&qid=1601731757&ref=sr_pg_3",headers)
# html_contents = page.text
# print(html_contents)


# import urllib
# opener = urllib.request.urlopen();
# opener.addheaders = [('User-agent', 'Mozilla/81.0.1')]
# response = opener.open('https://www.amazon.com/s?i=alexa-skills&bbn=13727921011&rh=n%3A13727921011%2Cp_n_date%3A14284927011&page=2&qid=1601731757&ref=sr_pg_3')
# html_contents = response.read()

import tensorflow_hub as hub
import tensorflow as tf
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt


def GUSE(item):
    # return embed([item]).numpy().tolist(); 
    return embed([item]).numpy();

def embed(newSteps):
    tempEmbeeding=[];
    for item in newSteps:
        item=item.lower();
        tempembed=GUSE(item);
        tempEmbeeding.append(tempembed);




        # tempembed=GUSE(item);
        # tempEmbeeding.append(tempembed);
        # trueEmbeedingArrayBig=np.append(trueEmbeedingArrayBig,tempembed);

    return tempEmbeeding;

# def compare(input_From_User_Embedding,trueEmbeedingArrayBig):
#     size=len(trueEmbeedingArrayBig)/512;
#     resizedEmbedding=np.reshape(trueEmbeedingArrayBig,(int(size),512));
#     tree = BallTree(resizedEmbedding, leaf_size=20)              # doctest: +SKIP
#     dist, ind = tree.query(input_From_User_Embedding, k=5)
#     print(dist);

def compare(input_From_User_Embedding,trueEmbeedingArrayBig):
    size=len(trueEmbeedingArrayBig);
    # print(size);
    # resizedEmbedding=np.reshape(trueEmbeedingArrayBig,(int(size),512));

    # print(trueEmbeedingArrayBig);
    resizedEmbedding=np.reshape(trueEmbeedingArrayBig,(int(size),512));
    # print(len(resizedEmbedding));

    tree = BallTree(resizedEmbedding, leaf_size=20)              # doctest: +SKIP
    dist, ind = tree.query(input_From_User_Embedding, k=10)

    indexs=[];
    for i in range(len(dist[0])):
        if dist[0][i]>0.8:
            indexs.append(ind[0][i]);

    # print(dist);
    # print(ind);
    # print(indexs)
    return indexs;




def bert():
    print('1');
    max_seq_length = 128  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2",trainable=True)
    print('2');
    bert_layer = hub.KerasLayer("/Users/senqiao/Desktop/533Project/bert_en_uncased_L-24_H-1024_A-16_2",trainable=True)
    print('3');

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    print(pooled_output);





Name_List=[];
Rating_List=[];
Avaiability_List=[];
Language_List=[];
Description_List=[];
# 100 page of featured
filename="ListofSkills.txt"
with open(filename, newline='') as Listfile:
    count=0;
    for line in Listfile:
        if line !='\n' and line!= 'Free Download\n':
            if count==0:
                Name_List.append(line);
            elif count==2 and 'Available' not in line:
                Rating_List.append(line);
            elif count==2 and 'Available' in line:
                Rating_List.append('');
                Avaiability_List.append(line);
                count=count+1;
            elif count==3:
                Avaiability_List.append(line);
            elif count==4:
                Language_List.append(line);
            elif count==5:
                Description_List.append(line);
                count=-1;

            count=count+1;




####################################################
# compare names
unique={};
unique_array=[];

for i in Name_List:
    # if 'Jeopardy' in i:
    #     print (i);
    if i.replace('\n','').replace('!','').replace('.','').replace('(','').replace(')','') not in unique_array:
        unique_array.append(i.replace('\n','').replace('!','').replace('.','').replace('(','').replace(')',''))
        unique[i]=1;
    else:
        unique[i]=unique[i]+1;

print({k: v for k, v in sorted(unique.items(), key=lambda item: item[1])});



#####################################Graph bar

# fig = plt.figure()

# count = [];
# names = [];

# for k, v in unique.items():
#     count.append(k);
#     names.append(int(v));


# ax = fig.add_axes([0,0,1,1])
# ax.set_ylabel('Count');

# ax.set_xlabel('Unique Skill Names');

# ax.set_ylim(0, 25)

# ax.bar(count,names)
# plt.show()

#####################################################
# Google Universeral Encoder
# embed = hub.load("/universal-sentence-encoder_4/");
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/1");
embed_array=embed(unique_array);


covert_embed=[];
for i in embed_array:
    covert_embed.append(i.numpy());

for j in range(len(covert_embed)):
    # print("ssssss1");
    print('Orginal Skill Name:');
    print(unique_array[j]);
    # unique_array
    appendme=[];
    appendme.append(covert_embed[j]);
    index=compare(appendme,covert_embed);
    Similar_array=[];
    for i in index:
        # print(i[1]);
        Similar_array.append(unique_array[i]);
    print('Similar Skills:');
    print(Similar_array);
################################################
# bert();

# unique_array=["Jeopardy Unofficial","Jeopardy"];
# embed = hub.load("/Users/senqiao/Desktop/Recomandation/universal-sentence-encoder_4/");
# embed_array=embed(unique_array);
# covert_embed=[];
# for i in embed_array:
#     covert_embed.append(i.numpy());

# for j in range(len(covert_embed)):
#     # print("ssssss1");
#     print('Orginal:');
#     print(unique_array[j]);
#     # unique_array
#     appendme=[];
#     appendme.append(covert_embed[j]);
#     index=compare(appendme,covert_embed);
#     Similar_array=[];
#     for i in index:
#         # print(i[1]);
#         Similar_array.append(unique_array[i]);
#     print('Similar:');
#     print(Similar_array);

