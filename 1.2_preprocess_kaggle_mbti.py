import pandas as pd

#read dataset
mbti=pd.read_csv('corpora/kaggle_mbti/mbti_1.csv')

#clean dataset
mbti['posts']=mbti['posts'].str.replace('\|\|\|',' ')

#group by personality type
types=mbti.groupby('type')['posts'].apply(lambda x: '\n'.join(x)).reset_index()

#write a file for each personality type
for row in types.iterrows():
    print(row[1]['type'])
    with open(f"corpora/kaggle_mbit/output/{row[1]['type']}.txt", 'w') as file:
        file.write(row[1]['posts'])




