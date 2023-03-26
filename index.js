//import tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"

import * as model from "./model.js"
import { VerticalLayerContainer, Layer_visual } from "./visual.js"
import * as dataset from "./dataset.js"

window.onload = function() {
    graphs = new GraphsManager();
    let init_p = Promise.all([model.load(), dataset.load()]);
    init_p.then(() => {
        remove_wait_cover();
        console.log("tf backend: ", tf.getBackend());
        graphs.addGraph();
        graphs.addGraph();
        graphs.addGraph();
        graphs.addGraph();
        graphs.addGraph();
        graphs.addGraph();
        graphs.addGraph();
        let img_tensor1 = dataset.getImagesByLabel(1)[3];
        let img_tensor2 = dataset.getImagesByLabel(9)[4];
        let img_tensor3 = dataset.getImagesByLabel(7)[9];
        let img_tensor4 = dataset.getImagesByLabel(1)[4];
        graphs.graphs[0].addImg_pred(img_tensor1, 1);
        graphs.graphs[1].addImg_pred(img_tensor2, 9);
        graphs.graphs[2].addImg_pred(img_tensor3, 7);

        for (let i = 0; i < 10; i++) {
            let img_tensor = dataset.getImagesByLabel(1)[i];
            graphs.graphs[4].addImg_pred(img_tensor, 1);
            graphs.graphs[3].addImg_pred(img_tensor, 1);
        }
        for (let i = 0; i < 10; i++) {
            let img_tensor = dataset.getImagesByLabel(9)[i];
            graphs.graphs[5].addImg_pred(img_tensor, 9);
            graphs.graphs[3].addImg_pred(img_tensor, 9);
        }
        for (let i = 0; i < 10; i++) {
            let img_tensor = dataset.getImagesByLabel(7)[i];
            graphs.graphs[6].addImg_pred(img_tensor, 7);
            graphs.graphs[3].addImg_pred(img_tensor, 7);
        }
    });
}

function remove_wait_cover() {
    let cover = document.getElementById("wait-cover");
    if (performance.now() < 1200) {
        setTimeout(remove_wait_cover, 500);
    } else {
        cover.remove();
    }
}

var graphs = null;


export class GraphsManager {
    html = document.getElementById("graphs");

    graphs = [];

    constructor() {
        this.html = document.getElementById("graphs");
    }

    addHTMLElement(element) {
        this.html.appendChild(element);
    }

    addGraph() {
        let graph = new Graph();
        this.graphs.push(graph);
    }
}


const classMap_label2index = {
    1: 0, // car
    9: 1, // truck
    7: 2 // horse
}
const classMap_index2label = {
    0: 1,
    1: 9,
    2: 7
}


class Graph {
    tracklist = [];
    tracklist_type = [];
    updaters = [];

    constructor() {
        let nodes = model.mapoutSync();
        for (let i = 0; i < nodes.length; i++) {
            this.tracklist.push(nodes[i].name);
            this.tracklist_type.push(nodes[i].op);
        }

        let container = new VerticalLayerContainer();
        container.attachTo(graphs);
        for (let i = 0; i < nodes.length; i++) {
            let layer = new Layer_visual(nodes[i].name, model.getSubnodeCount(nodes[i]));
            container.addLayer(layer);
            console.log(nodes[i].name, model.getSubnodeCount(nodes[i]));
            let updater = null;
            if (nodes[i].op == model.NODE_TYPE['Conv2D']) {
                updater = (tensor, gt, img_id) => {
                    let [c, alpha] = Conv2D_tensor_to_rgba(tensor);
                    let zeros = tf.zeros([c.shape[0]]);
                    if (gt == classMap_index2label[0]) {
                        // only keep the first channel
                        let rbga_tensor = tf.stack([c, zeros, zeros, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else if (gt == classMap_index2label[1]) {
                        // only keep the second channel
                        let rbga_tensor = tf.stack([zeros, c, zeros, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else if (gt == classMap_index2label[2]) {
                        // only keep the third channel
                        let rbga_tensor = tf.stack([zeros, zeros, c, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else {
                        console.error("invalid ground truth label: ", gt);
                    }
                }
            } else if (nodes[i].op == model.NODE_TYPE['LINEAR']) {
                updater = (tensor, gt, img_id) => {
                    let [c, alpha] = MatMul_tensor_to_rgba(tensor);
                    let zeros = tf.zeros([c.shape[0]]);
                    if (gt == classMap_index2label[0]) {
                        // only keep the first channel
                        let rbga_tensor = tf.stack([c, zeros, zeros, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else if (gt == classMap_index2label[1]) {
                        // only keep the second channel
                        let rbga_tensor = tf.stack([zeros, c, zeros, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else if (gt == classMap_index2label[2]) {
                        // only keep the third channel
                        let rbga_tensor = tf.stack([zeros, zeros, c, alpha], 1);
                        layer.update(rbga_tensor.arraySync(), img_id);
                    } else {
                        console.error("invalid ground truth label: ", gt);
                    }
                }
            } else {
                console.error("invalid node type: ", nodes[i].op);
            }

            this.updaters.push(updater);
        }
    }

    addImg_pred(tensor, gt) {
        console.log(this.tracklist)
        model.predict(tensor, this.tracklist).then((pred_output) => {
            console.log(pred_output[1][0].shape);
            let pred = pred_output[0];
            let img_id = get_imgID();
            this.update(pred_output[1], gt, img_id);
        });
    }

    // async TEST() {
    //     console.log("TEST");
    //     let [x, y] = await dataset.TEST_get_one();
    //     console.log(x, y);
    //     let gt = y.arraySync()[0];
    //     let pred = await model.predict(x, this.tracklist);
    //     console.log(pred[0]);
    //     this.update(pred[1], gt);
    // }



    update(tensors, gt, img_id) {
        console.log("update", this.updaters);
        for (let i = 0; i < this.updaters.length; i++) {
            console.log("start update", i);
            console.log(tensors);
            this.updaters[i](tensors[i], gt, img_id);
            console.log("finish update", i);
        }
    }
}


function sd(array) {
    let avg = array.reduce((a, b) => a + b) / array.length;
    let sum = array.map(x => (x - avg) ** 2).reduce((a, b) => a + b);
    return Math.sqrt(sum / array.length);
}

function unit_normalize(array) { // it is not the z-score, becuase all negative values are set to 0
    let avg = array.reduce((a, b) => a + b) / array.length;
    let std = sd(array);
    return (array.map(x => (x - avg) / std)).map(x => x > 0.5 ? x : 0);
}

function Conv2D_tensor_to_rgba(tensor) {
    // find the average of each feature map
    tensor = tf.reshape(tensor, [tensor.shape[1], tensor.shape[2], tensor.shape[3]]);
    tensor = tf.transpose(tensor, [2, 0, 1]);
    let avg_featureMap = tf.mean(tensor, [1, 2]);
    let avg_featureMap_array = avg_featureMap.arraySync();
    let avg_featureMap_array_unit = unit_normalize(avg_featureMap_array);
    avg_featureMap = tf.tensor(avg_featureMap_array_unit);

    // normalize the average to [0, 1]
    let max = tf.max(avg_featureMap);
    let min = tf.min(avg_featureMap);
    avg_featureMap = tf.sub(avg_featureMap, min);
    avg_featureMap = tf.div(avg_featureMap, tf.sub(max, min));

    // convert to r, g, or b
    let alpha = tf.ones([avg_featureMap.shape[0]]);
    let c = tf.mul(avg_featureMap, 255);
    return [c, alpha]
}

function MatMul_tensor_to_rgba(tensor, gt) {
    // find the average of each feature map
    tensor = tf.reshape(tensor, [tensor.shape[1]]);
    let tensor_array = tensor.arraySync();
    let tensor_array_unit = unit_normalize(tensor_array);
    tensor = tf.tensor(tensor_array_unit);

    // normalize the average to [0, 1]
    let max = tf.max(tensor);
    let min = tf.min(tensor);
    tensor = tf.sub(tensor, min);
    tensor = tf.div(tensor, tf.sub(max, min));

    // convert to r, g, or b
    let alpha = tf.ones([tensor.shape[0]]);
    let c = tf.mul(tensor, 255);
    return [c, alpha]
}



const get_imgID = (function() {
    let id = 0;
    return function() {
        return id++;
    }
})();