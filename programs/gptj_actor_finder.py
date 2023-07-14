import gpt4all
import pandas as pd

answers = []

imdb_movies = pd.read_csv('../data/imdb_movies.csv', index_col="names")
gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")

for index, row in imdb_movies.iterrows():
    crew = row['crew']
    name = row['orig_title']
    response = gptj.generate("Given the crew of a movie, identify only the most famous actor, meaning the actor "
                             "that is the most identifiable and recognizable to any given person in the public. "
                             "Output ONLY the actor's name and nothing else."
                             "Example:"
                             "Movie Name: The Super Mario Bros. Movie"
                             "Crew: Chris Pratt, Mario (voice), Anya Taylor-Joy, Princess Peach (voice), Charlie Day, "
                             "Luigi (voice),"
                             "Jack Black, Bowser (voice), Keegan-Michael Key, Toad (voice), Seth Rogen, Donkey Kong ("
                             "voice), Fred Armisen, Cranky Kong (voice), Kevin Michael Richardson, Kamek (voice), "
                             "Sebastian Maniscalco, Spike (voice)"
                             "Output: The most famous actor is Chis Pratt"
                             "\n\n\n\n"
                             "Here is your input:"
                             f"Movie Name: {name}"
                             f"Crew: {crew}")

    answers.append(response)

    print(answers)
