import requests
import geocoder

# WeatherAPI Key and URL
api_key = "c79b7657eeab46fc85b115422241110"
base_url = "http://api.weatherapi.com/v1/current.json"

def get_weather_info(api_key=api_key, base_url=base_url):
    """
    Get the current location using IP and fetch real-time weather data.
    
    Returns:
    str: A text prompt containing the weather information.
    """
    # Get the current location using IP
    g = geocoder.ip('me')

    if g.ok:
        city = g.city
        print(f"Detected City: {city}")

        # Fetch weather data for the detected city
        def get_real_time_weather(location, api_key=api_key, base_url=base_url):
            """
            Fetch real-time weather data for the specified location.

            Args:
            location (str): The location for which to fetch weather data.
            api_key (str): The API key for the WeatherAPI service.
            base_url (str): The base URL for the WeatherAPI service.

            Returns:
            tuple: A tuple containing the weather data (dict) or an error message (str).
            """
            try:
                response = requests.get(f"{base_url}?key={api_key}&q={location}")
                data = response.json()
                if "error" in data:
                    return None, "Location not found. Please try again."
                return data, None
            except Exception as e:
                return None, f"Error fetching data: {e}"

        data, error = get_real_time_weather(city)

        if data:
            # Extract relevant details from the data
            location_info = data['location']
            current_weather = data['current']

            city_name = location_info['name']
            country = location_info['country']
            condition = current_weather['condition']['text']
            temp_celsius = current_weather['temp_c']

            # Create a formatted text prompt
            weather_info = (f"The current weather in {city_name}, {country}:\n"
                            f"Condition: {condition}\n"
                            f"Temperature: {temp_celsius} °C")
            return weather_info
        else:
            return f"Error: {error}"
    else:
        return "Could not detect location."


# Example of how to call the function and get the weather info as a text prompt

