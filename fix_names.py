import pandas as pd
import os
import glob

df = pd.read_csv("./data/MoviePosters/MovieGenre.csv", encoding="ISO-8859-1")


path = "./data/MoviePosters/SampleMoviePosters/All"
img_glob = glob.glob(path + "/train_img/" + "*.jpg")
img_glob += glob.glob(path + "/test_img/" + "*.jpg")
img_glob += glob.glob(path + "/val_img/" + "*.jpg")
print(len(img_glob))

def get_imagepath(fp):
    return "https://images-na.ssl-images-amazon.com/images/M/" + fp.split("/")[-1]

for fp in img_glob:
    #print(fp)
    row = df.loc[df["Poster"] == get_imagepath(fp)]
    imdbId = row["imdbId"].values[0]
    current_name = fp
    prefix = "/".join(fp.split("/")[:-1])
    new_name = prefix + "/" + str(imdbId) + ".jpg"
    os.rename(current_name, new_name)


