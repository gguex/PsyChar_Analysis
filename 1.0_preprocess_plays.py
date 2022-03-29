import re
import pandas as pd

# Input/output paths
input_corpus_path = "corpora/Hamlet.txt"
output_tsv_path = "corpora/Hamlet.tsv"

# Loading the corpus
with open(input_corpus_path) as corpus_file:
    corpus_txt = corpus_file.read()

# Splitting into acts
corpus_act_list = corpus_txt.split("ACT")
# Removing the beggining
corpus_act_list = corpus_act_list[6:]

# Splitting into scenes
corpus_actscene_llist = []
for corpus_act in corpus_act_list:
    corpus_scene_list = corpus_act.split("SCENE")
    corpus_actscene_llist.append(corpus_scene_list[1:])

# Making a function for the treatment of sentences
def sentence_treatment(sent):
    # Remove "appartées"
    processed_sent = re.sub("\[.+]", " ", sent)
    # Remove EOL
    processed_sent = re.sub("\n", " ", processed_sent)
    # Remove extra spaces
    processed_sent = re.sub(" +", " ", processed_sent).strip()
    # Return result
    return processed_sent


# Making the dataframe for the corpus
corpus_df = pd.DataFrame(columns=["act", "scene", "char_from", "char_to", "sentence"])

# Looping of act and scene
for id_act, corpus_act in enumerate(corpus_actscene_llist):
    for id_scene, scene in enumerate(corpus_act):
        # Removing scene number
        if re.search("I\.", scene) is not None:
            scene = scene[re.search("I\.", scene).end():]
        elif re.search("V\.", scene) is not None:
            scene = scene[re.search("V\.", scene).end():]
        last_end = None
        char_list, sent_list = [], []
        # Looping on character
        for expr in re.finditer("[A-Z `'’]{3,}\.", scene):
            # Adding character name
            char_list.append(expr.group(0)[:-1].strip())
            if last_end is not None:
                sent = scene[last_end:expr.start()]
                # Adding the sentence
                sent_list.append(sentence_treatment(sent))
            last_end = expr.end()
        sent = scene[last_end:]
        # Adding the last sentence
        sent_list.append(sentence_treatment(sent))

        # Making the "char_to" vector
        char_to = []
        for id_char, char_from in enumerate(char_list):
            rev_list = [char for char in char_list[id_char::-1] if char != char_from]
            if len(rev_list) == 0:
                char_to.append([char for char in char_list if char != char_from][0])
            else:
                char_to.append(rev_list[0])

        # Making the scene dataframe
        scene_df = pd.DataFrame({"act": [id_act+1]*len(sent_list), "scene": [id_scene+1]*len(sent_list),
                                 "char_from": char_list, "char_to": char_to, "sentence": sent_list})
        corpus_df = pd.concat([corpus_df, scene_df], ignore_index=True)

corpus_df.to_csv(output_tsv_path, sep="\t")