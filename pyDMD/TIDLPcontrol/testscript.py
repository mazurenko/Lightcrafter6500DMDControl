__author__ = 'anton'

import Pyro4

ns = Pyro4.locateNS()
print ns
ns.ping()
nameserver_names = ns.list()

for idx, key in enumerate(nameserver_names.keys()):
    print "{} : {}".format(idx, key)

Pyro4.locateNS()