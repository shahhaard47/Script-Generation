"""
Check for fast_train_<max_len>.csv and fast_val_<max_len>.csv files
If they don't exist, ask to create them

MUST BE EXECUTED FROM HOME DIR (Script-Generation/)
"""

import pandas as pd
import syntok.segmenter as segmenter
import re
import os


def getTrainValSplit(min_freq=50, split_train=0.9, data_file='data/train.csv', verbose=True):
    """
    create train-val split from data from `data_file`
    """
    df = pd.read_csv(data_file, delimiter=',')
    if verbose:
        print('all columns:', df.columns)
    # drop cols < freq
    stats = df.iloc[:, 4:].sum(axis=0)
    cols = list(stats[stats < min_freq].index)
    for col in cols:
        df.drop(col, axis=1, inplace=True)
    if verbose:
        print("new genre totals:\n", df.iloc[:, 4:].sum())
    # drop rows that don't belong to any genres now
    onehotlabels = df.iloc[:, 4:]
    rows = onehotlabels.sum(axis=1)
    if verbose:
        print('total rows before: ', len(rows))
    # remove rows with sum 0 ; no labels
    rows = list(rows[rows == 0].index)
    for row in rows:
        df.drop(row, axis=0, inplace=True)
    if verbose:
        print('total rows after: ', len(df))

    msk = np.random.rand(len(df)) < split_train
    train = df[msk]
    val = df[~msk]
    if verbose:
        print('train:', len(train), '; val:', len(val), '; all:', len(df))
    split_stats = pd.concat([train.iloc[:, 4:].sum(), val.iloc[:, 4:].sum(
    ), train.iloc[:, 4:].sum()/df.iloc[:, 4:].sum()], axis=1)
    if verbose:
        print(split_stats)

    # label->int and int->label mapping
    if verbose:
        print('creating label str<->int mappings')
    cols = df.columns
    idx = 0
    genre_to_int = {}
    int_to_genre = {}
    for c in cols[4:]:
        int_to_genre[idx] = c
        genre_to_int[c] = idx
        idx += 1
    if verbose:
        print(genre_to_int)
    if verbose:
        print(int_to_genre)

    return train, val, genre_to_int, int_to_genre


def chunkset(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def __extractValueFromToken__(tokenLst):
    return [t.value for t in tokenLst]


def chunkScript(movie, max_len=300):
    """
    add `[SEP]` token at the end of each sentence in script
    """
    outscript = ['']
    # movie = re.sub('\s+',' ', movie)
    prev_len = 0
    try:
        segmentation = list(segmenter.process(movie))

        for paragraph in segmentation:
            for s in paragraph:
                sent = ' '.join([t.value for t in s]) + ' [SEP]'
                if (prev_len + len(s) < max_len-2):
                    outscript[-1] += sent
                    prev_len += len(s)
                elif (len(s) < max_len-2):
                    outscript.append(sent)
                    prev_len = len(s)
                else:
                    # new sentence is longer than max_len
                    # print(len(outscript))
                    outscript += [' '.join(__extractValueFromToken__(s2))
                                  for s2 in chunkset(s, max_len - 1)]
                    # print(len(outscript))
                    prev_len = len(outscript[-1])
    except:
        print("segmentatin error for movie:", movie)
        return []

    return outscript


def chunkDataframe(dframe, max_len=300):
    new_frame = pd.DataFrame().reindex_like(dframe)
    new_frame.drop(new_frame.index, axis=0, inplace=True)
    for i in range(0, len(dframe)):
        if i % 20 == 0:
            print('processing', i, 'of', len(dframe),
                  ' -- title:', dframe.iloc[i].title)
        thisrow = dframe.iloc[i]
        chunked_script = chunkScript(thisrow.script, max_len=max_len)
        thisrow.at['script'] = ''
        for c in chunked_script:
            thisrow.at['script'] = c
            new_frame = new_frame.append(thisrow, ignore_index=True)
    return new_frame


def create_fastBertData(max_len, train_file, val_file, verbose=True):
    train, val, genre_to_int, int_to_genre = getTrainValSplit(verbose=verbose)
    # NOTE: can't get genre<->mappings to FastBert code or anywhere on colab yet
    # NOTE: if min_freq is passed to getTrainValSplit(), fast_label.csv and labels list passed to BertDataBunch will need to be modified
    # FINAL STEP (execute only once / or to generate of with different max_len)
    new_val = chunkDataframe(val, max_len=max_len)
    new_val.to_csv(val_file)
    if verbose:
        print('chunked val:', len(new_val), '; original_val', len(val))
    if verbose:
        print('created file:', val_file)

    new_train = chunkDataframe(train, max_len=max_len)
    new_train.to_csv(train_file)
    if verbose:
        print('chunked train (smaller rows):', len(
            new_train), '; original_train:', len(train))
    if verbose:
        print('created file:', train_file)

    print("---------- saved train and test CSV ------------")


def check_fastBert_data(max_len, verbose=True):
    train_file = 'data/fast_train_' + str(max_len) + '.csv'
    val_file = 'data/fast_val_' + str(max_len) + '.csv'

    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        if verbose:
            print("train val files exist! good to go!")
        return True  # both exist!

    decision = input(train_file + ' or ', val_file,
                     "don't exist.\nCreate them? (y|n): ")
    if decision == 'y':
        create_fastBertData(max_len, train_file, val_file, verbose=verbose)
    else:
        return False

    return True
