import argparse
import os
import random
import shutil
import glob
import torchaudio

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from python_speech_features import mfcc, delta
from pydub import AudioSegment


def UciPreprocess():
    infile_path = os.path.join("data", "uci", "accent-mfcc-data-1.csv")
    outfile_path = os.path.join("data", "uci", "accent-mfcc-data.csv")
    col = "language"

    df = pd.read_csv(infile_path)
    accented_df = df[df[col].notnull()]
    label_list = accented_df[col].unique()
    label_list.sort()
    label_dict = dict(zip(label_list, [_ for _ in range(len(label_list))]))
    print(label_dict)

    accented_df["label"] = accented_df[col].apply(lambda x: label_dict[x])
    accented_df.to_csv(outfile_path)


def splitAudio(input_path, output_path, length=5):
    """ 
    Split audio to $length seconds files. The sample lost is less than $length second to keep the feature in the same shape

    file_name(str): path of the audio file
    output_path(str): path of the output audio file
    length(float): maximum length of clips in seconds
    """

    file_name, file_ext = (
        os.path.split(input_path)[1].split(".")[0],
        os.path.split(input_path)[1].split(".")[1],
    )
    audio = AudioSegment.from_file(input_path, format=file_ext)
    total_segments = int(audio.duration_seconds / length)

    for i in range(total_segments):
        audio[i * 1000 : (i + length) * 1000].export(
            os.path.join(output_path, file_name + "_" + str(i) + "." + file_ext),
            format="wav",
        )

    return True


def kagglePreprocess(split_audio=False, test_size=0.3, save_mfcc=False):
    base_path = os.path.join("data", "kaggle", "clips")
    if split_audio:
        shutil.rmtree(base_path)
        path = os.path.join("data", "kaggle")
        directory_list = list(
            filter(lambda x: len(x.split(".")) == 1 and x != "clips", os.listdir(path))
        )
        for directory in directory_list:
            output_path = os.path.join(base_path, directory)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            input_path = os.path.join(path, directory)
            file_list = [os.path.join(input_path, i) for i in os.listdir(input_path)]
            # just for map function
            output_path_list = [output_path for i in file_list]
            call_lazy_map = list(map(splitAudio, file_list, output_path_list))

    labels = ["china-mandarin", "english", "french", "hindi"]
    labels.sort()
    columns = ["path", "native_language"]
    col = "native_language"
    records = []

    label_dict = dict(zip(labels, [_ for _ in range(len(labels))]))
    print(label_dict)

    for label in labels:
        print("Processing ", label)
        folder_path = os.path.join(base_path, label)
        # Remove .npy files
        if save_mfcc:
            for _file in glob.glob(os.path.join(folder_path, "*.npy"), recursive=True):
                os.remove(_file)
        file_list = os.listdir(folder_path)
        for file in file_list:
            if save_mfcc:
                # Save mfcc feature in .npz files
                file_name = os.path.join(folder_path, file)
                waveform, samplerate = torchaudio.load(file_name)
                feature = mfcc(
                    waveform, samplerate=samplerate, winlen=0.0025, appendEnergy=False
                )
                delta_mfcc = delta(feature, 1)
                delta_delta_mfcc = delta(delta_mfcc, 1)
                mfccs = np.concatenate((feature, delta_mfcc, delta_delta_mfcc), axis=1)
                feature = np.expand_dims(mfccs.T, 0)
                save_file_name = os.path.splitext(file_name)[0] + ".npy"
                np.save(save_file_name, feature)
                record = [save_file_name, label]

            else:
                record = [os.path.join(folder_path, file), label]
            records.append(record)

    df = DataFrame(records, columns=columns)
    num_data = df[col].value_counts().to_frame().min(axis=0)[col]

    out_df = df[df[col] == "english"].sample(n=num_data)
    for label in labels:
        if label == "english":
            continue
        resampled_df = df[df[col] == label].sample(n=num_data)
        out_df = pd.concat([out_df, resampled_df])

    out_df["label"] = out_df["native_language"].apply(lambda x: label_dict[x])
    out_df.to_csv(os.path.join("data", "kaggle", "dev.csv"))
    test_df = out_df.sample(frac=test_size)
    train_df = out_df.drop(test_df.index)
    test_df.to_csv(os.path.join("data", "kaggle", "test.csv"))
    train_df.to_csv(os.path.join("data", "kaggle", "train.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    if str(args.dataset) == "uci":
        UciPreprocess()

    elif str(args.dataset) == "kaggle":
        kagglePreprocess(split_audio=False, save_mfcc=True)

    else:
        available_files = ["dev", "train", "test"]
        labels = [
            "us",
            "indian",
            "england",
            "australia",
            "african",
            "philippines",
            "ireland",
        ]
        if str(args.dataset) in available_files:
            file_name = args.dataset
            file = os.path.join("data", file_name + ".tsv")
            df = pd.read_csv(file, sep="\t")
            col = "accent"
            # Rows with accent labels
            accented_df = df[
                (df[col].notnull()) & (df[col] != "NAN") & (df[col].isin(labels))
            ]
            label_list = accented_df[col].unique()
            num_data = accented_df[col].value_counts().to_frame().min(axis=0)[col]
            # print(accented_df[col].value_counts().to_frame().min(axis=0)["accent"])
            # To get 1:1 data, take the sample of other accents greater than min
            out_df = accented_df[accented_df[col] == "us"].sample(n=num_data)
            for label in labels:
                if label == "us":
                    continue
                resampled_df = accented_df[accented_df[col] == label].sample(n=num_data)
                out_df = pd.concat([out_df, resampled_df])
            print(out_df)
            label_list.sort()
            label_dict = dict(zip(label_list, [_ for _ in range(len(label_list))]))
            print(label_dict)
            out_df = out_df.reset_index(drop=True)
            out_df["label"] = out_df[col].apply(lambda x: label_dict[x])
            # Save data dictionary with accent labels
            out_df.to_csv(os.path.join("data", file_name + "_filtered.tsv"), sep="\t")
            # Move files to data/clips
            dst_path = os.path.join("data", "clips")
            src_path = os.path.join(
                "cv-corpus-7.0-2021-07-21-en.tar.gz",
                "cv-corpus-7.0-2021-07-21",
                "en",
                "clips",
            )
            file_list = list(out_df["path"])
            for file in file_list:
                src = os.path.join(src_path, file)
                dst = os.path.join(dst_path, file)
                # shutil.copyfile(src, dst)
        else:
            print("Wrong argument.\n Usage: python preprocess $dataset")
