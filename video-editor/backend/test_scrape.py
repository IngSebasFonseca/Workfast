import re
import requests
from bs4 import BeautifulSoup
import http.cookiejar as cookielib

cookie_file = "assets/library/facebook_cookies.txt"
cj = cookielib.MozillaCookieJar(cookie_file)
cj.load(ignore_discard=True, ignore_expires=True)

url = "https://www.facebook.com/malejarestrepooficial/reels/"

session = requests.Session()
session.cookies = cj
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

response = session.get(url, headers=headers)
html = response.text

print(f"Status Code: {response.status_code}")
print(f"HTML length: {len(html)}")

# Find any fb.watch or /reel/ or /videos/ links in the raw HTML
reels = set(re.findall(r'(/reel/\d+/?|/videos/\d+/?)', html))
print(f"Found reels/videos paths: {reels}")

video_ids = set(re.findall(r'"video_id":"(\d+)"', html))
print(f"Found video_ids: {video_ids}")

