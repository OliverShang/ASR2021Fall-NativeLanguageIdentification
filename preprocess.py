import pandas as pd
import shutil
import os
import random
import argparse

from pandas.core.frame import DataFrame


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


def kagglePreprocess():
    base_path = os.path.join("data", "kaggle")
    labels = ["hindi", "china-mandarin", "french", "english", "korean"]
    labels.sort()
    columns = ["path", "native_language"]

    records = []

    label_dict = dict(zip(labels, [_ for _ in range(len(labels))]))
    print(label_dict)

    for label in labels:
        folder_path = os.path.join(base_path, label)
        file_list = os.listdir(folder_path)
        for file in file_list:
            record = [os.path.join(folder_path, file), label]
            records.append(record)

    df = DataFrame(records, columns=columns)
    # num_data = (
    #     df["native_language"].value_counts().to_frame().min(axis=0)["native_language"]
    # )
    df["label"] = df["native_language"].apply(lambda x: label_dict[x])
    df.to_csv(os.path.join(base_path, "dev.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    if str(args.dataset) == "uci":
        UciPreprocess()

    elif str(args.dataset) == "kaggle":
        kagglePreprocess()

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
            print("Wrong argument.\n Usage: python preprocess dev")
