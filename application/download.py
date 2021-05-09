import threading
import requests

def download(link, filelocation):
    r = requests.get(link, stream=True)
    with open(filelocation, 'wb') as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)

def createNewDownloadThread(link, filelocation):
    download_thread = threading.Thread(target=download, args=(link,filelocation))
    download_thread.start()

for i in range(0,5):
    files = "C:\\test" + str(i) + ".png"
    print (files)
    createNewDownloadThread("http://stackoverflow.com/users/flair/2374517.png", files)