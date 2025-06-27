from boxsdk import OAuth2, Client
import os

developer_token = input("Developer Token: ")
folder_id = "325323078385" #this doesn't work unless permission on folder is greater than viewer...

oauth = OAuth2(
    client_id = "",
    client_secret = "",
    access_token = developer_token
)

client = Client(oauth)

folder = client.folder(folder_id = folder_id).get()
items = folder.get_items()

print("File owned by:", folder.owned_by)
print("Shared link?", folder.shared_link)

if not os.path.exists("./box_images"):
    os.makedirs("./box_images")

for item in items:
    if item.type == "file" and item.name.lower().endswith("dcm"):
        with open("./box_images/" + item.name, "wb") as f:
            item.download_to(f)