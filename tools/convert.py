"""Convert SkyRemoval v1.0.6's fixed-shape ONNX graph to Core ML.

Experiment only. Supply separately obtained weights; see README for licensing.
Dependencies: numpy, onnx, coremltools. Does not require PyTorch or an ONNX runtime
in the macOS application. The converter rejects unrecognized operators.
"""
import argparse
import collections
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes


def convert(source, destination):
    model = onnx.shape_inference.infer_shapes(onnx.load(source), data_prop=True)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
    builder = NeuralNetworkBuilder(
        [(model.graph.input[0].name, datatypes.Array(*shapes[model.graph.input[0].name]))],
        [(model.graph.output[0].name, datatypes.Array(*shapes[model.graph.output[0].name]))],
        disable_rank5_shape_mapping=True, use_float_arraytype=True)
    loaded = set()
    retained = collections.Counter()
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == 'Shape' and all(shapes.get(node.input[0], [])):
            constants[node.output[0]] = np.array(shapes[node.input[0]], dtype=np.int64)
            continue
        if all(name in constants for name in node.input if name):
            result = ReferenceEvaluator(node).run(None, {name: constants[name] for name in node.input if name})
            constants.update(zip(node.output, result))
            continue
        inputs, out = list(node.input), node.output[0]
        name = f'{index}_{node.op_type}'
        # Convolution parameters and Resize scales are attributes in Core ML.
        data_inputs = inputs[:1] if node.op_type in ('Conv', 'Resize') else inputs
        for value in data_inputs:
            if value in constants and value not in loaded:
                array = constants[value].astype(np.float32)
                builder.add_load_constant_nd(f'constant_{value}', value, array, array.shape)
                loaded.add(value)
        if node.op_type == 'Conv':
            w = constants[inputs[1]]
            b = constants[inputs[2]] if len(inputs) > 2 else None
            pads = attrs.get('pads', [0]*4); stride = attrs.get('strides', [1,1])
            builder.add_convolution(name, w.shape[1], w.shape[0], w.shape[2], w.shape[3],
                stride[0], stride[1], 'valid', attrs.get('group',1), w.transpose(2,3,1,0), b,
                b is not None, input_name=inputs[0], output_name=out,
                dilation_factors=attrs.get('dilations',[1,1]), padding_top=pads[0],
                padding_left=pads[1], padding_bottom=pads[2], padding_right=pads[3])
        elif node.op_type in ('Relu','Sigmoid'):
            builder.add_activation(name, 'RELU' if node.op_type == 'Relu' else 'SIGMOID',inputs[0],out)
        elif node.op_type == 'Add':
            builder.add_elementwise(name, inputs, out, 'ADD')
        elif node.op_type == 'Concat':
            builder.add_concat_nd(name, inputs, out, attrs['axis'])
        elif node.op_type == 'Resize':
            assert attrs['mode'] == b'nearest' and attrs['coordinate_transformation_mode'] == b'asymmetric' and attrs['nearest_mode'] == b'floor'
            scales = constants[inputs[2]]
            assert list(scales[:2]) == [1,1] and all(int(v)==v for v in scales[2:])
            builder.add_upsample(name, int(scales[2]), int(scales[3]), inputs[0], out, mode='NN')
        elif node.op_type == 'MaxPool':
            assert attrs.get('ceil_mode',0) == 0
            k=attrs['kernel_shape']; stride=attrs['strides']; pads=attrs['pads']
            builder.add_pooling(name,k[0],k[1],stride[0],stride[1],'MAX','VALID',inputs[0],out,
                padding_top=pads[0],padding_left=pads[1],padding_bottom=pads[2],padding_right=pads[3])
        else:
            raise ValueError(f'Unsupported dynamic operator: {node.op_type}')
        retained[node.op_type] += 1
    builder.spec.description.metadata.shortDescription = 'SkyRemoval v1.0.6 conversion experiment; upstream model licensing applies.'
    builder.spec.description.metadata.userDefined['upstream'] = 'https://github.com/OpenDroneMap/SkyRemoval'
    ct.utils.save_spec(builder.spec, destination)
    print('Saved', destination, dict(retained))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('onnx'); parser.add_argument('coreml')
    args=parser.parse_args(); convert(args.onnx,args.coreml)
