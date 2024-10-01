#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy
import vart
import xir_extra_ops


class SubgraphDescriptor:
    def __init__(self, name, attrs, children):
        self.name = name
        self.attrs = attrs
        self.children = children


def create_subgraphs_for_graph(g, sd):
    create_subgraphs(g.get_root_subgraph(), sd, 0)


def create_subgraphs(sg, sd, level):
    sg.set_name(sd.name)
    for (k, v) in sd.attrs.items():
        sg.set_attr(k, v)
    if not sd.children:
        return
    sg.create_child_subgraph()
    map = {}
    ops = sg.get_ops()
    for op in ops:
        sname = op.get_attr("subgraph_path")[level]
        value = map.get(sname, set())
        value.add(sg.get_graph().get_leaf_subgraph(op))
        map[sname] = value

    children = {}
    for (k, v) in map.items():
        children[k] = sg.merge_children(v)
        children[k].set_name(k)
    for child_sd in sd.children:
        create_subgraphs(children[child_sd.name], child_sd, level + 1)


def create_simple_cpu_graph(g):
    create_subgraphs_for_graph(
        g,
        SubgraphDescriptor(
            "root", {}, [SubgraphDescriptor("cpu1", {"device": "CPU"}, [])]
        ),
    )
    normal_setting_for_graph(g)


def normal_setting_for_graph(g):
    g.get_root_subgraph().set_attr(
        "runner",
        {
            "ref": "libvitis_ai_library-graph_runner.so.3",
            "sim": "libvitis_ai_library-graph_runner.so.3",
            "run": "libvitis_ai_library-graph_runner.so.3",
        },
    )
    for child in g.get_root_subgraph().get_children():
        if child.has_attr("device"):
            device = child.get_attr("device")
            if device == "CPU" or device == "USER":
                child.set_attr(
                    "runner",
                    {
                        "ref": "libvitis_ai_library-cpu_task.so.3",
                        "sim": "libvitis_ai_library-cpu_task.so.3",
                        "run": "libvitis_ai_library-cpu_task.so.3",
                    },
                )


def run_graph(g):
    user = os.environ.get("USER", "NOBODY")
    os.makedirs(os.path.join("/", "tmp", user), exist_ok=True)
    SAVE_MODEL = os.environ.get("SAVE_MODEL")
    if SAVE_MODEL:
        g.save_as_image(os.path.join("/", "tmp", user, "a.svg"), "svg")
        g.serialize(os.path.join("/", "tmp", user, "a.xmodel"))
    runner = vart.RunnerExt.create_runner(g.get_root_subgraph(), "run")
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()
    job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    runner.wait(job_id)
    ret = {}
    for tb in output_tensor_buffers:
        name = tb.get_tensor().name
        if name.startswith("compare"):
            r = numpy.asarray(tb)
            shape = r.shape
            for b in range(shape[0]):
                actual_md5sum = "".join([chr(item) for item in r[b, 0, :]])
                expected_md5sum = "".join([chr(item) for item in r[b, 1, :]])
                if expected_md5sum != actual_md5sum:
                    name = tb.get_tensor().get_attr("input_tensor_name")
                    ret[name] = ret.get(name, [])
                    ret[name].append((b, actual_md5sum, expected_md5sum))
                    # print("DEBUG " + str([r.dtype, r.shape, name,
                    # actual_md5sum, expected_md5sum, b]))
    return ret


def set_reference_input(g, tensor_name, filenames):
    tensor = find_tensor(g, tensor_name)
    op = tensor.producer
    xir_extra_ops.set_op_py_buffer_type_attr(
        op, "data", [read_from_cache(file) for file in filenames]
    )
    op.set_attr("md5sum", [file["md5sum"] for file in filenames])


def default_file_entries(tensor, batch):
    return [
        {
            "file": xir_extra_ops.remove_xfix(tensor.name) + ".bin",
            "md5sum": "0" * 32,
            "size": tensor.get_data_size(),
        }
        for i in range(batch)
    ]


# def workaround_bug_in_tf_maybe_remove_last_fixneuron(tensor):
#    op = tensor.producer
#    if op.get_type() == 'fix':
#        return op.get_input_op('input', 0).get_output_tensor()
#    return tensor


def add_check_point(g, dump_tensors_ref, batch):
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        if sg.has_attr("device") and sg.get_attr("device") == "USER":
            continue
        for tensor in sg.get_output_tensors():
            #            tensor = workaround_bug_in_tf_maybe_remove_last_fixneuron(tensor0)
            tensor_name = xir_extra_ops.remove_xfix(tensor.name)
            file_entries = dump_tensors_ref.get(
                tensor_name, default_file_entries(tensor, batch)
            )
            dump_tensors_ref[tensor_name] = file_entries
            add_check_point2(g, tensor_name, file_entries)


def add_check_point2(g, tensor_name, file_entries):
    user = os.environ["USER"]
    tensor = find_tensor(g, tensor_name)
    op = tensor.producer
    subgraph = g.get_leaf_subgraph(op)
    is_cpu = subgraph.has_attr("device") and subgraph.get_attr("device") == "CPU"
    attrs = {
        "md5sum": [file["md5sum"] for file in file_entries],
        "dump_directory": os.path.join("/", "tmp", user, "error_result"),
    }
    if is_cpu:
        compare_op = g.create_op(
            "compare_" + tensor_name,
            "compare",
            attrs=attrs,
            input_ops={"input": [op]},
            subgraph=subgraph,
        )
    else:
        compare_op = g.create_op(
            "compare_" + tensor_name, "compare", attrs=attrs, input_ops={"input": [op]}
        )
        g.get_leaf_subgraph(compare_op).set_attr("device", "CPU")
    xir_extra_ops.set_op_py_buffer_type_attr(
        compare_op,
        "baseline",
        [read_cache_md5_data(file["md5sum"]) for file in file_entries],
    )
    compare_op.get_output_tensor().set_attr("input_tensor_name", tensor_name)


def find_tensor(g, tensor_name):
    tensors = [
        tensor
        for tensor in g.get_tensors()
        if xir_extra_ops.remove_xfix(tensor.name) == tensor_name
    ]
    assert len(tensors) == 1, (
        "cannot find tensor: tensor_name="
        + str(tensor_name)
        + " len="
        + str(len(tensors))
        + " all tensors: "
        + "\n\t".join(
            [xir_extra_ops.remove_xfix(tensor.name) for tensor in g.get_tensors()]
        )
    )
    return tensors[0]


def read_from_cache(file):
    filename = file["file"]
    md5sum = file["md5sum"]
    size = file["size"]
    root = os.environ.get(
        "GOLDEN_CACHE", os.path.join("/", "scratch", "models", "cache", "golden")
    )
    if md5sum == "0" * 32:
        return memoryview(bytes(size))
    rel = os.path.join(root, md5sum[0:2], md5sum[2:])
    assert size == os.stat(rel).st_size
    with open(rel, "rb") as f:
        return memoryview(f.read())


def read_cache_md5_data(md5sum):
    root = os.environ.get(
        "GOLDEN_CACHE", os.path.join("/", "scratch", "models", "cache", "golden")
    )
    rel = os.path.join(root, md5sum[0:2], md5sum[2:])
    if not os.path.isfile(rel):
        return memoryview(bytes(0))
    size = os.stat(rel).st_size
    with open(rel, "rb") as f:
        return memoryview(f.read())
