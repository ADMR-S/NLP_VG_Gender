from urllib.request import urlopen

page = "https://finalfantasy.fandom.com/wiki/Final_Fantasy_VI_SNES_script"
html = urlopen(page).read().decode('utf-8')
o = open("raw/page01.html",'w')
o.write(html)
o.close()
