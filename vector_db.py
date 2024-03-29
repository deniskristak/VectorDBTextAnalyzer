import weaviate
import weaviate.classes as wvc
import os
from dotenv import load_dotenv


class VectorDBTextAnalyzerBase:
    def __init__(self, db_name):
        """
        Initialize the VectorDBTextAnalyzerBase class
        :param db_name: The name of the vector database to be created
        """
        load_dotenv()
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

    def populate_db(self, chunks):
        """
        Populate the database with the text chunks extracted from the data files.
        :param chunks: A list of dictionaries, where each dictionary contains the following:
            {
                "unique_identifier": <unique identifier for the chunk> (e.g. concatenation of filename and page number),
                "text": <text from the chunk>
            }
        """
        # this vectorises the chunks and saves them
        chunk_objs = list()
        for chunk in chunks:
            chunk_objs.append({
                "unique_identifier": chunk["unique_identifier"],
                "text": chunk["text"],
            })

        text_chunks = self.client.collections.get(self.db_name)
        text_chunks.data.insert_many(chunk_objs)

    def search(self, query, limit=3):
        """
        Search for chunks in the database that contextually result from the query.
        :param query: The query to search for
        :param limit: The maximum number of results to return
        :return: The search response
        """
        text_chunks = self.client.collections.get(self.db_name)
        response = text_chunks.query.near_text(
            query=query,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        return response

    def search_and_answer(self, query, generative_task, limit=3):
        """
        Search for pages in the database that contextually result from the query combining the resulted
        retrieved documents with a generative task.
        :param query: The query to search for
        :param generative_task: The generative task (e.g. "Summarize the document")
        :param limit: The maximum number of results to return
        :return: The search response
        """
        text_chunks = self.client.collections.get(self.db_name)
        response = text_chunks.generate.near_text(
            query=query,
            limit=limit,
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
            print(f"unique_identifier: {res_obj.properties['unique_identifier']}")
            print(f"text: {res_obj.properties['text']}")
            print(f"Distance: {res_obj.metadata.distance}")
        print("\n")

    @staticmethod
    def print_generated_response(query, generative_task, response):
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


def helolololo():
    """
    This is a test function
    :return: None
    """
    print("hello")
    print("hello")
    print("hello")
