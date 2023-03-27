// async function load_CIFAR10Test() {
//     const dataset_path = "./dataset/test_1_7_9-tf.json"
//     if (x_tensor_cache != null && y_tensor_cache != null) {
//         ret_x = tf.clone(x_tensor_cache);
//         ret_y = tf.clone(y_tensor_cache);
//         return [ret_x, ret_y];
//     }
//     const raw = await fetch(dataset_path);
//     const raw_obj = await raw.json();
//     let x = raw_obj.x;
//     let y = raw_obj.y;
//     let x_tensor = tf.tensor(x);
//     let y_tensor = tf.tensor(y);
//     // normalize x
//     x_tensor = x_tensor.div(255);
//     x_tensor = x_tensor.sub(0.5);
//     x_tensor = x_tensor.mul(2);
//     x_tensor_cache = tf.clone(x_tensor);
//     y_tensor_cache = tf.clone(y_tensor);
//     return [x_tensor, y_tensor];
// }

var x_tensor_cache = null;
var y_tensor_cache = null;

export function get_one_batch(x, y, from_index) {
    let x_batch = tf.slice4d(x, [from_index, 0, 0, 0], [1, 32, 32, 3]);
    let y_batch = tf.slice(y, [from_index], [1]);
    return [x_batch, y_batch];
}

// export async function TEST_get_one() {
//     let [x, y] = await load_CIFAR10Test();
//     let ret = get_one_batch(x, y, 2000);
//     return ret;
// }

const file_paths = [
    './img/1/7.json',
    './img/9/5.json',
    './img/7/2.json',
]

const label_to_tensor = {}

export async function load() {
    for (let p of file_paths) {
        const raw = await fetch(p);
        const raw_obj = await raw.json();
        let x = raw_obj.x;
        let y = raw_obj.y;
        let x_tensor = tf.tensor(x);
        // normalize x
        x_tensor = x_tensor.div(255);
        x_tensor = x_tensor.sub(0.5);
        x_tensor = x_tensor.mul(2);
        for (let i = 0; i < y.length; i++) {
            let label = y[i];
            if (label_to_tensor[label] == null) {
                label_to_tensor[label] = [];
            }
            label_to_tensor[label].push(x_tensor.slice([i, 0, 0, 0], [1, 32, 32, 3]));
        }
    }
    console.log(label_to_tensor);
}

export function unique_label() {
    return Object.keys(label_to_tensor);
}

export function getImagesByLabel(label) {
    return label_to_tensor[label];
}

export function getImagesByLabelAndy(label) {
    let ret = [];
    const t = label_to_tensor[label]
    for (let i = 0; i < t.length; i++) {
        ret.push([t[i], label]);
    }
    return ret;
}