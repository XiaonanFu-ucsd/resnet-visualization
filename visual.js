export class Layer_visual {
    layer_html = null;
    name = null;
    subnode_count = null;
    blocks = [];
    block_container_html = null;
    constructor(name, subnode_count) {
        this.name = name;
        this.subnode_count = subnode_count;
    }

    attachTo(parent) {
        // create a div
        this.layer_html = document.createElement("div");
        this.layer_html.className = "layer";
        this.layer_html.innerText = this.name + " " + this.subnode_count;

        this.block_container_html = document.createElement("div");
        this.block_container_html.style.lineHeight = "0";
        this.block_container_html.style.width = 32 * BLOCK_DEFAULT_SIZE + "px";
        this.layer_html.appendChild(this.block_container_html);

        parent.addHTMLElement(this.layer_html);
        this.addBlocks(this.subnode_count);
    }

    addHTMLElement(element) {
        this.layer_html.appendChild(element);
    }

    addBlocks(count) {
        for (let i = 0; i < count; i++) {
            let block = new Block_visual();
            this.blocks.push(block);
            this.block_container_html.appendChild(block.get_html());
        }
    }

    update(arrays, img_id) {
        for (let i = 0; i < arrays.length; i++) {
            this.blocks[i].addColor(img_id, arrays[i]);
        }
    }
}

const BLOCK_DEFAULT_SIZE = 12;

export class Block_visual {
    block_html = null;
    name = null;
    colors = {}
    constructor(name) {
        this.name = name;
        this.block_html = document.createElement("span");
        this.block_html.className = "block";
        this.block_html.style.backgroundColor = "rgba(100, 100, 100, 0.5)";
    }

    get_html() {
        return this.block_html;
    }

    clear() {}
    addColor(img_id, rgba) {
        this.colors[img_id] = rgba;
        let [r, g, b, a] = cal_RGBA(Object.values(this.colors));
        r = Math.round(r);
        g = Math.round(g);
        b = Math.round(b);
        a = Math.round(a * 100) / 100;
        //console.log(r, g, b, a);
        this.block_html.style.backgroundColor = "rgba(" + r + ", " + g + ", " + b + ", " + a + ")";
    }
}

function cal_RGBA(rgba_list) {
    if (rgba_list.length == 1) {
        return rgba_list[0];
    }
    let r = 0;
    let g = 0;
    let b = 0;
    let a = 0;
    for (let i = 0; i < rgba_list.length; i++) {
        r += rgba_list[i][0];
        g += rgba_list[i][1];
        b += rgba_list[i][2];
        a += rgba_list[i][3];
    }
    // let rgb_sum = r + g + b;
    // if (rgb_sum == r || rgb_sum == g || rgb_sum == b) {
    //     r = r / length;
    //     g = g / length;
    //     b = b / length;
    //     a = 1;
    //     return [r, g, b, a];
    // }
    let length = rgba_list.length;
    // if (r / 1.5 > g + b) {
    //     r = 255;
    //     g = g / length;
    //     b = b / length;
    // } else if (g / 1.5 > r + b) {
    //     r = r / length;
    //     g = 255;
    //     b = b / length;
    // } else if (b / 1.5 > r + g) {
    //     r = r / length;
    //     g = g / length;
    //     b = 255;
    // } else {
    //     r = r / length;
    //     g = g / length;
    //     b = b / length;
    // }
    r = r / length * 2;
    g = g / length * 2;
    b = b / length * 2;
    a = 1;
    return [r, g, b, a];
}

export class VerticalLayerContainer {
    container_html = null;
    constructor() {
        // create a div
        this.container_html = document.createElement("div");
        this.container_html.style.pedding = "10px 25px 25px 10px";
        this.container_html.style.margin = "16px";
    }

    addLayer(layer) {
        layer.attachTo(this);
    }

    addHTMLElement(element) {
        this.container_html.appendChild(element);
    }

    attachTo(parent) {
        parent.addHTMLElement(this.container_html);
    }
}

export class Add_placeholder {
    html = null;
    constructor() {
        this.html = document.createElement("div");
        this.html.className = "add_placeholder";
    }

    attachTo(parent) {
        parent.addHTMLElement(this.html);
    }
}