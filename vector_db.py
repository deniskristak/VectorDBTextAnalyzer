import weaviate
import weaviate.classes as wvc
import pdfplumber
import os
from dotenv import load_dotenv


class VectorDBTextAnalyzer:
    def __init__(self, data_folder_path, db_name):
        load_dotenv()
        self.data_folder_path = data_folder_path
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.db_name = db_name
        self.client = None

    def __enter__(self):
        self.client = weaviate.connect_to_local(
            headers={"X-OpenAI-Api-Key": self.api_key}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def extract_text_from_data_files(self):
        pages = []
        for pdf_file in os.listdir(self.data_folder_path):
            if not pdf_file.endswith(".pdf"):
                print(f"Skipping {pdf_file} as it is not a PDF file")
            with pdfplumber.open(os.path.join(self.data_folder_path, pdf_file)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text from each page
                    page_text = page.extract_text()
                    # Add to dictionary; page numbers start at 1
                    pages.append({
                        "filename": pdf_file,
                        "page_number": i + 1,
                        "text": page_text
                    })
        return pages

    def create_db(self, cleanup=False):
        if cleanup:
            self.client.collections.delete(self.db_name)
        self.client.collections.create(
            name=self.db_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            generative_config=wvc.config.Configure.Generative.openai()
        )
        self._populate_db()

    def _populate_db(self):
        pages = self.extract_text_from_data_files()
        # this vectorises the chunks and saves them
        chunk_objs = list()
        for page in pages:
            chunk_objs.append({
                "filename": page["filename"],
                "page_number": int(page["page_number"]),
                "text": page["text"],
            })

        text_chunks = self.client.collections.get(self.db_name)
        text_chunks.data.insert_many(chunk_objs)

    def search_pages(self, query):
        text_chunks = self.client.collections.get(self.db_name)
        response = text_chunks.query.near_text(
            query=query,
            limit=3,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        # sort pages by distance
        return response

    @staticmethod
    def print_search_results(query, response):
        print("_______QUERY_______")
        print(query)
        print("-------------------")
        print("_______RESULTS_______")
        response_objects = sorted(response.objects, key=lambda x: x.metadata.distance)
        for i, res_obj in enumerate(response_objects):
            print("-------------------")
            print(f"File: {res_obj.properties['filename']}")
            print(f"Page: {res_obj.properties['page_number']}")
            print(f"Distance: {res_obj.metadata.distance}")
        print("\n")

    def close_db_conn(self):
        self.client.close()

