//import tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"

import * as model from "./model.js"
import { VerticalLayerContainer, Layer_visual, Add_placeholder } from "./visual.js"
import * as dataset from "./dataset.js"

window.onload = function() {
    graphs = new GraphsManager();
    let init_p = Promise.all([model.load(), dataset.load()]);
    init_p.then(() => {
        list_img();
        remove_wait_cover();
        console.log("tf backend: ", tf.getBackend());
        graphs.addGraph();
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



function list_img() {
    let list = document.getElementById("Title-img");
    console.log(list);
    let i1 = dataset.getImagesByLabelAndy(1);
    let i2 = dataset.getImagesByLabelAndy(9);
    let i3 = dataset.getImagesByLabelAndy(7);
    let photo_tensors = [...i1, ...i2, ...i3]
    console.log(photo_tensors);
    photo_tensors.sort((a, b) => { return Math.random() - 0.5 });
    for (let i = 0; i < 15; i++) {
        const canvas = document.createElement('canvas');
        canvas.className = 'list-img';
        canvas.draggable = true;
        canvas.width = 32;
        canvas.height = 32;
        canvas.style.width = '72px';
        canvas.style.height = '72px';
        let tensor_img = photo_tensors[i][0];
        console.log(tensor_img);
        let new_img = { y: photo_tensors[i][1], x: tensor_img, canvas: canvas };
        canvas.ondragstart = (e) => {
            e.dataTransfer.clearData();
            e.dataTransfer.setData('i', i);
            console.log(tensor_img);
            console.log("drag start");
        }
        let tensor = tf.reshape(photo_tensors[i][0], [32, 32, 3]);
        tensor = tf.div(tensor, 2);
        tensor = tf.add(tensor, 0.5);
        tf.browser.toPixels(tensor, canvas);
        list.appendChild(canvas);
        imgs.push(new_img);
    }
}

var graphs = null;
var imgs = [];


export class GraphsManager {
    html = document.getElementById("graphs");
    add_placeholder = null;

    graphs = [];

    constructor() {
        this.html = document.getElementById("graphs");
        this.add_placeholder = new Add_placeholder();
        this.add_placeholder.attachTo(this);
        this.add_placeholder.add_callback = () => {
            this.addGraph();
        }
    }

    addHTMLElement(element) {
        this.html.appendChild(element);
    }

    addGraph() {
        let graph = new Graph();
        this.graphs.push(graph);
        this.add_placeholder.remove();
        graph.attachTo(this);
        this.add_placeholder.attachTo(this);
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

    html = document.createElement("div");
    dropzone = document.createElement("div");

    constructor() {
        this.html.className = "graph";
        let nodes = model.mapoutSync();
        for (let i = 0; i < nodes.length; i++) {
            this.tracklist.push(nodes[i].name);
            this.tracklist_type.push(nodes[i].op);
        }
        this.dropzone.innerText = "Drop image here";
        this.dropzone.style.width = "85%";
        this.dropzone.style.height = "72px";
        this.dropzone.style.border = "5px solid #000";
        this.dropzone.style.borderRadius = "4px";
        this.dropzone.style.textAlign = "center";
        this.dropzone.style.lineHeight = "36px";
        this.dropzone.style.whiteSpace = "nowrap";
        this.dropzone.style.overflowY = "hidden";
        this.dropzone.style.overflowX = "auto";
        this.dropzone.style.display = "flex";
        this.dropzone.style.flexDirection = "row";
        // horizontal scroll only
        this.dropzone.addEventListener("drop", (e) => {
            if (this.dropzone.textContent != "")
                this.dropzone.innerText = "";
            let i = e.dataTransfer.getData('i');
            let x = imgs[i].x;
            let y = imgs[i].y;
            // copy the image from previous canvas
            let canvas = document.createElement('canvas');
            canvas.width = 32;
            canvas.height = 32;
            canvas.style.width = '64px';
            canvas.style.height = '64px';
            this.dropzone.appendChild(canvas);
            canvas.getContext('2d').drawImage(imgs[i].canvas, 0, 0);

            this.addImg_pred(x, y);
        });
        this.dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
        });
        this.html.appendChild(this.dropzone);

        let container = new VerticalLayerContainer();
        container.attachTo(this);
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
        console.log(gt)
        model.predict(tensor, this.tracklist).then((pred_output) => {
            console.log(pred_output[1][0].shape);
            let pred = pred_output[0];
            let img_id = get_imgID();
            this.update(pred_output[1], gt, img_id);
        });
    }

    attachTo(graphs) {
        graphs.addHTMLElement(this.html);
    }

    addHTMLElement(element) {
        this.html.appendChild(element);
    }


    update(tensors, gt, img_id) {
        console.log("update", this.updaters);
        for (let i = 0; i < this.updaters.length; i++) {
            this.updaters[i](tensors[i], gt, img_id);
        }
        return img_id;
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