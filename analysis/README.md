# Data Analysis
This folder contains code for our analysis of the manipulative tactics in the from fields and subject lines of the emails.

## Python setup
Create a virtual environment with the packages listed in the `requirements.txt` file.

    pip install -r requirements.txt

## Folder structure and files
* `data/learning` contains the labelled sample used for training and evaluation.
* `src` contains several jupyter notebooks.
    * These notebooks correspond to each of the tactics that we documented.
        * `Sample.ipynb` and `Featurize samples.ipynb` together generate features that are subsequently digested by the other notebooks.
        * `Deceptive Analysis.ipynb` contains our analysis of the `obscured name`, `re:/fwd:`, and `ongoing thread` dark patterns.
        * `Forward Referencing Analysis.ipynb` contains our analysis of `forward referencing` clickbait.
        * `Sensationalism Analysis.ipynb` contains our analysis of `sensationalism` clickbait.
        * `Urgency Analysis.ipynb` contains our analysis of `urgency` clickbait.
        * `Aggregate Analyses.ipynb` aggregates all the results from the previous notebooks across senders.
    * `Email Address Disclosures.ipynb` contains our email address sharing analysis.
