from flask import Flask
from flask import request, Response
from flask_cors import CORS, cross_origin
import socket
from requests import get, post
import re

app = Flask(__name__)
CORS(app)


iP_addresses = {}


@app.route("/init", methods=['POST'])
def init_network():
    node_name = request.args.get('node')
    port = request.args.get('port')
    ip = request.remote_addr
    iP_addresses[node_name] = f'{ip}:{port}'
    return f'{ip}:{port}'

@app.route("/get-ip", methods=['GET'])
def get_ip_for_node():
    node_name = request.args.get('node')
    if node_name not in iP_addresses.keys():
        return {"status": False}
    return {"status": True, "ip": iP_addresses[node_name]}


@app.route("/")
@cross_origin()
def test():
    return "Hello World!"


@app.route('/train-server/<path:path>', defaults={'path': ''}, methods=['GET', 'POST'])
def proxy_ml_server(path):
    path = request.path
    path_split = path.split('/')
    node = path_split[1]
    path = ''
    for i in range(2, len(path_split)):
        path += f'{path_split[i]}/'
    path = path[:-1]
    if 'train-server' not in iP_addresses.keys():
        return 'Train Server DOWN'
    train_server = iP_addresses['train-server']
    query_vars = request.args.to_dict()
    query_string = ''
    for key, val in query_vars.items():
        query_string += f'{key}={val}&'
    query_string = query_string[:-1]
    if request.method == 'GET':
        return get(f'http://{train_server}/{path}?{query_string}').content
    elif request.method == 'POST':
        return post(f'http://{train_server}/{path}?{query_string}').content
    


@app.route('/broadcast-to-nodes/<path:path>', defaults={'path': ''}, methods=['GET', 'POST'])
def broadcast_to_nodes(path):
    path = request.path
    path_split = path.split('/')
    node = path_split[1]
    path = ''
    for i in range(2, len(path_split)):
        path += f'{path_split[i]}/'
    path = path[:-1]
    query_var = request.args.to_dict()
    query_string = ''
    for key, val in query_var.items():
        query_string += f'{key}={val}&'
    query_string = query_string[:-1]
    for key in iP_addresses.keys():
        if re.search('\d', key):
            ip = iP_addresses[key]
        else:
            continue
        if request.method == 'GET':
            return get(f'http://{ip}/{path}?{query_string}').content
        elif request.method == 'POST':
            return post(f'http://{ip}/{path}?{query_string}').content
    return "No Suitable Methods in PROXY"



@app.route('/backend-server/<path:path>', defaults={'path': ''}, methods=['GET', 'POST'])
def backend_server(path):
    path = request.path
    path_split = path.split('/')
    node = path_split[1]
    path = ''
    for i in range(2, len(path_split)):
        path += f'{path_split[i]}/'
    path = path[:-1]
    query_var = request.args.to_dict()
    query_string = ''
    for key, val in query_var.items():
        query_string += f'{key}={val}&'
    query_string = query_string[:-1]
    if 'backend-server' not in iP_addresses.keys():
        return 'BACKEND SERVER DOWN'
    else:
        ip = iP_addresses['backend-server']
        if request.method == 'GET':
            return get(f'http://{ip}/{path}?{query_string}').content
        elif request.method == 'POST':
            return post(f'http://{ip}/{path}?{query_string}').content
    return "No Suitable Methods in PROXY"



app.run(host='0.0.0.0', port=5000, debug=True)