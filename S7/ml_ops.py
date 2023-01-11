import requests
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
print(response.content)
print(response.json())

response = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'},
)

import requests
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
print(response.content)

with open(r'img.png','wb') as f:
   f.write(response.content)

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

print(response.content)