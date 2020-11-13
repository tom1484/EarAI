import requests as r


my_data = {'26': 'off'}
a = r.get('http://192.168.43.251', data=my_data)
