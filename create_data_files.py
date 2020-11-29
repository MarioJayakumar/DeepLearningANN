import sys
import os
import glob
import pandas as pd
import numpy as np

DATA_PATH = "./data/MoviePosters/SampleMoviePosters/SampleMoviePosters"
TRAIN_DATA = 'train_img'
TEST_DATA = 'test_img'
VAL_DATA = 'val_img'
TRAIN_IMG_FILE = 'train_img.txt'
TEST_IMG_FILE = 'test_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE = 'test_label.txt'


path = "./data/MoviePosters/SampleMoviePosters/SampleMoviePosters"
train_glob = glob.glob(path + "/train_img/" + "*.jpg")
test_glob = glob.glob(path + "/test_img/" + "*.jpg")
val_glob = glob.glob(path + "/val_img/" + "*.jpg")

def get_id(filename):
    start_index = filename.rfind("/") + 1
    end_index = filename.rfind(".jpg")
    return filename[start_index:end_index]

train_names = []
for fp in train_glob:
    train_names.append(get_id(fp))    
test_names = []
for fp in test_glob:
    test_names.append(get_id(fp))    
val_names = []
for fp in val_glob:
    val_names.append(get_id(fp))    

print(len(train_names))
print(len(test_names))
print(len(val_names))

df = pd.read_csv("./data/MoviePosters/MovieGenre.csv", encoding="ISO-8859-1")
genres = []
length = len(df)
for n in range(len(df)):
    g = str(df.loc[n]["Genre"])
    genres += g.split("|")

classes = list(set(genres))
classes.sort()
num_classes = len(classes)
print(classes)

def get_classes_from_movie(movie_id):
    match = df["imdbId"] == np.int64(int(movie_id))
    row = df.loc[df["imdbId"] == np.int64(int(movie_id))]
    genres = str(row["Genre"].values[0]).split("|")
    y = np.zeros(num_classes)
    for g in genres:
        y[classes.index(g)] = 1
    return y


with open(path + "/class_labels.txt", "w") as class_file:
    for c in classes:
        class_file.write(c + "\n")

with open(path + "/train_img.txt", "w") as train_file:
    for f in train_names:
        train_file.write(f + ".jpg\n")

with open(path + "/test_img.txt", "w") as test_file:
    for f in test_names:
        test_file.write(f + ".jpg\n")

with open(path + "/val_img.txt", "w") as val_file:
    for f in val_names:
        val_file.write(f + ".jpg\n")

with open(path + "/train_label.txt", "w") as train_file:
    for name in train_names:
        for value in get_classes_from_movie(name):
            if value == 0:
                train_file.write("0 ")
            else:
                train_file.write("1 ")
        train_file.write("\n")

with open(path + "/test_label.txt", "w") as test_file:
    for name in test_names:
        for value in get_classes_from_movie(name):
            if value == 0:
                test_file.write("0 ")
            else:
                test_file.write("1 ")
        test_file.write("\n")

with open(path + "/val_label.txt", "w") as val_file:
    for name in val_names:
        for value in get_classes_from_movie(name):
            if value == 0:
                val_file.write("0 ")
            else:
                val_file.write("1 ")
        val_file.write("\n")

