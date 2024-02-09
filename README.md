# SETUP
1. Install python dependencies
2. Setup Weaviate docker container
   - instructions can be found [here](https://weaviate.io/developers/weaviate/installation/docker-compose), but the `docker-compose.yml` in this repo should work just fine
3. Run the Weaviate docker container
```bash
docker compose up -d
```
4. Place all the source PDFs in the `data_files` directory
5. Define your `OPENAI_API_KEY` in the `.env` file

# RUN
- either in Jupyter Notebook or in a Python script
```python
# import the text analyzer
from vector_db import VectorDBTextAnalyzer

# define the path to the PDFs
pdf_path = 'data_files'
db_name = 'data_files_all'

# create and populate the vector database
with VectorDBTextAnalyzer(pdf_path, db_name) as vector_db_searcher:
    vector_db_searcher.create_db(cleanup=True)

# search for a concept/text/question
# if using jupyter notebook, put this into a new cell, as creating the database takes a lot of time (in that case, a new context manager is needed)
with VectorDBTextAnalyzer(pdf_path, db_name) as vector_db_searcher:
    text_to_match = 'effect of climate on species survival'
    response = vector_db_searcher.search_pages(text_to_match)

    # pretty-print the search results
    vector_db_searcher.print_search_results(text_to_match, response)
