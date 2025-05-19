import re
import json
import requests
from datetime import datetime


def google_search_arxiv_id(query, num=10, end_date=None):
    url = "https://google.serper.dev/search"

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"

    payload = json.dumps(
        {
            "q": search_query,
            "num": num,
            "page": 1,
        }
    )

    # headers = {"X-API-KEY": "google serper api key", "Content-Type": "application/json"}
    headers = {
        "X-API-KEY": "0d8265613c091abbfd920f52e8b4937dece72d0b",
        "Content-Type": "application/json",
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code == 200:
            results = json.loads(response.text)

            arxiv_id_list, details = [], {}
            for paper in results["organic"]:
                if "snippet" in paper:
                    cited_by = (
                        re.search(r"Cited by (\d+)", paper["snippet"]).group(0)
                        if re.search(r"Cited by (\d+)", paper["snippet"])
                        else None
                    )
                arxiv_id = (
                    re.search(
                        r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", paper["link"]
                    ).group(1)
                    if re.search(
                        r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", paper["link"]
                    )
                    else None
                )

                if arxiv_id:
                    arxiv_id_list.append(arxiv_id)
                    details[arxiv_id] = {
                        "arxiv_id": arxiv_id,
                        "google_search_position": paper["position"],
                        "cited_by": cited_by,
                    }
            return list(set(arxiv_id_list))

        else:
            print(
                f"Failed to request google serper. Status code: {response.status_code}"
            )
            return None

    except requests.RequestException as e:
        print(f"An error occurred in serper: {e}")
        return None
