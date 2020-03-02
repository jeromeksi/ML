import socket


#Hôte	irc.root-me.org
#Protocole	IRC
#Port	6667
#Canal IRC	#root-me_challenge
#Bot	candy

irc = socket.socket()
irc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
irc.connect(('irc.root-me.org', 6667))
