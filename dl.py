import glob
import pandas as pd

df = pd.read_csv("./data/MoviePosters/MovieGenre.csv", encoding="ISO-8859-1")

with open("links.txt", "w") as f:
    for n in range(len(df)):
        link = str(df.loc[n]["Poster"])
        f.write(link + "\n")
    
