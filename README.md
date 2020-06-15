# CandidateMatch

A standalone script for scraping a job ad from `glassdoor.com`, extracting key requirements and then analyzing a list of given CVs (pdf) and finding each candidate score (%) in terms of string similarity.

Metrics used: 
Edit based --> Jaro Winkler, Levenshtein

Token based --> Jaccard, Cosine Similarity

Sequence based --> longest common substring similarity, Ratcliff-Obershelp similarity


Each extracted skill from CV is compared with each extracted requirement from the ad. All of the 6 above metrics are extracted for each such combination (skill - requirement). A final match per skill/requirement is computed by taking the mean (Sum(metrics)/6). Finally, all calculated scores are averaged with respect to the total skills and requirements.
