# your link goes here
link = "https://github.com/RedWinter101/React-App-Stable-Diffusion/blob/main/api/Mountain-10.png"

# note: this will break if a repo/organization or subfolder is named "blob" -- would be ideal to use a fancy regex
# to be more precise here
print(link.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/"))

# example output link:
# https://raw.githubusercontent.com/RedWinter101/React-App-Stable-Diffusion/main/api/Mountain-10.png
