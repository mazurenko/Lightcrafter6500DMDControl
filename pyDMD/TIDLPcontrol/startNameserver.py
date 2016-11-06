import Pyro4
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = ['serpent', 'pickle']

if __name__ == "__main__":
    location = 'lilab'

    nameserver_ip_dict = {'woodshop': '192.168.0.102',
                          'lilab': '172.23.6.81'}

    host = nameserver_ip_dict[location]
    # Pyro4.naming.startNSloop(host=host, port=9090)
    Pyro4.naming.startNSloop(host='172.23.6.55', port=9090)