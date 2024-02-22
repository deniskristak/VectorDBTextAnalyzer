import weaviate
import weaviate.classes as wvc
import pdfplumber
import os
from dotenv import load_dotenv


class VectorDBTextAnalyzerBase:
    def __init__(self, data_folder_path, db_name):
        """
        Initialize the VectorDBTextAnalyzerBase class
        :param data_folder_path: The path to the folder containing the data files
        :param db_name: The name of the vector database to be created
        """
        load_dotenv()
        self.data_folder_path = data_folder_path
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.db_name = db_name
        self.client = None

    def __enter__(self):
        """
        Connect to the Weaviate client when entering the context
        """
        self.client = weaviate.connect_to_local(
            headers={"X-OpenAI-Api-Key": self.api_key}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the Weaviate client when exiting the context
        """
        if self.client:
            self.client.close()

    def extract_text_from_data_files(self):
        """
        This method should be implemented by the child class to extract text from the data files
        :return: A list of dictionaries, where each dictionary contains the following:
            {
                "filename": <source file name>,
                "chunk_number": <unique identifier for the chunk (unique for the file)>,
                "text": <text from the chunk>
            }
        """
        raise NotImplementedError

    def create_db(self, cleanup=False):
        """
        Create the vector database. If cleanup is True, the database is deleted before creating it.
        :param cleanup: If True, delete the database before creating it
        """
        if cleanup:
            self.client.collections.delete(self.db_name)
        self.client.collections.create(
            name=self.db_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            generative_config=wvc.config.Configure.Generative.openai()
        )
        self._populate_db()

    def _populate_db(self):
        """
        Populate the database with the text chunks extracted from the data files.
        """
        chunks = self.extract_text_from_data_files()
        # this vectorises the chunks and saves them
        chunk_objs = list()
        for chunk in chunks:
            chunk_objs.append({
                "filename": chunk["filename"],
                "chunk_number": int(chunk["chunk_number"]),
                "text": chunk["text"],
            })

        text_chunks = self.client.collections.get(self.db_name)
        text_chunks.data.insert_many(chunk_objs)

    def search_pages(self, query):
        """
        Search for pages in the database that contextually result from the query.
        :param query: The query to search for
        :return: The search response
        """
        text_chunks = self.client.collections.get(self.db_name)
        response = text_chunks.query.near_text(
            query=query,
            limit=3,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        # sort pages by distance
        return response

    def search_pages_generative(self, query, generative_task):
        """
        Search for pages in the database that contextually result from the query combining the resulted
        retrieved documents with a generative task.
        :param query: The query to search for
        :param generative_task: The generative task (e.g. "Summarize the document")
        :return: The search response
        """
        text_chunks = self.client.collections.get(self.db_name)
        response = text_chunks.generate.near_text(
            query=query,
            limit=3,
            grouped_task=generative_task,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        # sort pages by distance
        return response

    @staticmethod
    def print_search_results(query, response):
        """
        Print the search results
        :param query: The query
        :param response: The search response
        """
        print("_______QUERY_______")
        print(query)
        print("-------------------")
        print("_______RESULTS_______")
        response_objects = sorted(response.objects, key=lambda x: x.metadata.distance)
        for i, res_obj in enumerate(response_objects):
            print("-------------------")
            print(f"File: {res_obj.properties['filename']}")
            print(f"Chunk: {res_obj.properties['chunk_number']}")
            print(f"Distance: {res_obj.metadata.distance}")
        print("\n")

    def print_generated_response(self, query, generative_task, response):
        """
        Print the generated response
        :param query: The query
        :param generative_task: The generative task
        :param response: The generated response
        :return:
        """
        print("_______QUERY_______")
        print(query)
        print("_______TASK_______")
        print(generative_task)
        print("_______GENERATED RESULT________")
        print(response.generated)

    def close_db_conn(self):
        """
        Close the Weaviate client
        :return:
        """
        self.client.close()


class VectorDBTextAnalyzerPDF(VectorDBTextAnalyzerBase):
    def extract_text_from_data_files(self):
        """
        Extract text from PDF files. Each page is treated as a separate chunk.
        :return: A list of dictionaries, where each dictionary contains the mentioned fields (see parent method).
        """
        pages = []
        for filename in os.listdir(self.data_folder_path):
            if not filename.endswith(".pdf"):
                print(f"Skipping {filename} as it is not a PDF file")
            with pdfplumber.open(os.path.join(self.data_folder_path, filename)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text from each page
                    page_text = page.extract_text()
                    # Add to dictionary; page numbers start at 1
                    pages.append({
                        "filename": filename,
                        "chunk_number": i + 1,
                        "text": page_text
                    })
        return pages
