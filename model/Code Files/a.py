import requests
url = 'https://media.geeksforgeeks.org/wp-content/uploads/20210318103632/gfg-300x300.png'

# Send a GET request to the URL to download the image content
response = requests.get(url)

# Check that the request was successful (status code 200)
if response.status_code == 200:
    # Save the image content to a file
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
        print('Image saved successfully!')
else:
    print('Error: Failed to download image')