export async function load() {
    const model_path = "./r-94-tfjs/model.json"
    const model_config = await fetch(model_path);
    model_config_obj = await model_config.json();
    model = await tf.loadGraphModel(model_path);
    console.log(
        "Model loaded.",
    );
}

const model_path = "./r-94-tfjs/model.json"
var model_config_obj = null;
var model = null;

export const NODE_TYPE = {
    'Conv2D': 'Conv2D',
    'LINEAR': 'MatMul', // for fully connected layer
    'Const': 'Const',
}

const default_wanted_nodes = ['Conv2D', 'MatMul'];

var const_nodeList = {}
var conv2d_nodeList = {}
var matmul_nodeList = {}


export async function mapout(wanted_node) {
    while (model_config_obj == null) {
        await new Promise(r => setTimeout(r, 100));
    }
    console.log(model_config_obj);
    if (wanted_node == null) {
        wanted_node = default_wanted_nodes;
    }
    const nodes = model_config_obj['modelTopology']['node'];
    let ret = [];
    for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < wanted_node.length; j++) {
            if (nodes[i].op == wanted_node[j]) {
                console.log(nodes[i].name, nodes[i].op);
                ret.push(nodes[i]);
            }
            if (nodes[i].op == NODE_TYPE['Const']) {
                const_nodeList[nodes[i].name] = nodes[i];
            } else if (nodes[i].op == NODE_TYPE['Conv2D']) {
                conv2d_nodeList[nodes[i].name] = nodes[i];
            } else if (nodes[i].op == NODE_TYPE['LINEAR']) {
                matmul_nodeList[nodes[i].name] = nodes[i];
            }
        }
    }
    return ret;
}

export function mapoutSync(wanted_node) {
    if (model_config_obj == null || model == null) {
        console.log("mapoutSync: model_config_obj is null");
    }
    if (wanted_node == null) {
        wanted_node = default_wanted_nodes;
    }
    const nodes = model_config_obj['modelTopology']['node'];
    let ret = [];
    for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < wanted_node.length; j++) {
            if (nodes[i].op == wanted_node[j]) {
                console.log(nodes[i].name, nodes[i].op);
                ret.push(nodes[i]);
            }
            if (nodes[i].op == NODE_TYPE['Const']) {
                const_nodeList[nodes[i].name] = nodes[i];
            } else if (nodes[i].op == NODE_TYPE['Conv2D']) {
                conv2d_nodeList[nodes[i].name] = nodes[i];
            } else if (nodes[i].op == NODE_TYPE['LINEAR']) {
                matmul_nodeList[nodes[i].name] = nodes[i];
            }
        }
    }
    return ret;
}

function printNodes() {
    const nodes = model_config_obj['modelTopology']['node'];
    for (let i = 0; i < nodes.length; i++) {
        console.log(nodes[i].name);
    }
}

export function getSubnodeCount(node) {
    if (node.op == NODE_TYPE['Conv2D']) {
        let param_name = node.input[1];
        let param_node = const_nodeList[param_name];
        let ret = param_node.attr.value.tensor.tensorShape.dim[3].size;
        ret = parseInt(ret);
        return ret;
    } else if (node.op == NODE_TYPE['LINEAR']) {
        let param_name = node.input[1];
        let param_node = const_nodeList[param_name];
        let ret = param_node.attr.value.tensor.tensorShape.dim[1].size;
        ret = parseInt(ret);
        return ret;
    }
    console.error("getSubnodeCount(node): Not supported node type: " + node.op);
    return 0;
}

export async function predict(x, trackList_layerOutput) {
    while (model == null) {
        await new Promise(r => setTimeout(r, 100));
    }
    if (trackList_layerOutput == null) {
        trackList_layerOutput = ["output_1"];
    } else {
        trackList_layerOutput.push("output_1");
    }
    console.log(x.shape)
    let output_tensor_list = await model.executeAsync({ 'input_1': x }, trackList_layerOutput);
    console.log('finish predict')
    let y = output_tensor_list.pop();
    trackList_layerOutput.pop();
    let pred = y.argMax(1).dataSync()[0];
    console.log("Predicted: " + pred);
    return [pred, output_tensor_list];
}