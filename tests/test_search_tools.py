from custom_agent.search_tools import google_search_arxiv_id
import unittest


class TestGoogleSearchArxivId(unittest.TestCase):
    def test_google_search_arxiv_id(self):
        query = "Transformer"
        arxiv_id_list = google_search_arxiv_id(query)
        self.assertIsInstance(arxiv_id_list, list)
        print(arxiv_id_list)
