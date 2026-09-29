from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Your Streamlit app URL
APP_URL = "https://inr-values-prediction.streamlit.app/"

def keep_website_awake():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Browser won't open on screen, will run in the background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Visiting {APP_URL}...")
        driver.get(APP_URL)
        
        # Wait for 15 seconds to allow the page to load fully and Streamlit's WebSocket to connect
        time.sleep(15)
        print("Successfully visited and kept the app awake!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    keep_website_awake()
