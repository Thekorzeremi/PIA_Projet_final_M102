import requests 

import requests

class GeocodeService:
    BASE_URL = "https://nominatim.openstreetmap.org/search"

    @staticmethod
    def geocode(adresse: str):
        params = {
            "q": adresse,
            "format": "json",
            "limit": 1
        }

        headers = {
            "User-Agent": "mon-app-python"
        }

        response = requests.get(
            GeocodeService.BASE_URL,
            params=params,
            headers=headers
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        return {
            "latitude": float(data[0]["lat"]),
            "longitude": float(data[0]["lon"])
        }