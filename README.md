# Finding Smilar Movie Summaries
You will source a collection of movie summaries, such as those found on IMDB. You will then use a variety of existing text similarity metrics to find the most similar movie descriptions. Which metrics work best for this task? You will also compile a collection of known twin films - that is movies that were made around the same time and have a very similar plot line, although their setting and characters may be different. This will effectively form our gold standard for assessing how well the different metrics work. We can also compare the summaries of books that were made into films – how well do our metrics work on these?

### Libraries to Install
`pip install transformers torch matplotlib numpy`

Also ensure that the interpreter used to run this project is Python Version 3.10.x, there may be issues if versions 3.12.x or later are used, especially on Mac.

### How to Run
Run each ModernBERT_(filename).py code depending on what is being tested.
There may be issues on first run, so run it again and it should display the similarity scores for comparing:
1. summaries of two randomly selected movies (compareRandomMovies)
2. sentences in different languages (compareText)
3. summaries of known twin films (compareTwinMovies)
Each code will save the notched boxplot results in the specified folder, instead of displaying as a pop-up.
