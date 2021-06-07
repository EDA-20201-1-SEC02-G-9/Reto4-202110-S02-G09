import urllib.request, json

def get_ip_data(ip):
    url = "https://ipinfo.io/{}/json".format(ip)
    with urllib.request.urlopen(url) as webpage:
        data = json.loads(webpage.read().decode())
        return data

def get_location(ip):
    data = get_ip_data(ip)
    loc = data['loc']
    latitude, longitude = tuple(loc.split(","))
    return float(latitude), float(longitude)