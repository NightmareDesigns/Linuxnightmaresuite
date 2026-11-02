"""
💀 NIGHTMARE ONNX PARSER — Pure-Numpy ONNX Weight Extractor
============================================================
Reads ONNX model files and extracts weight tensors (initializers)
without any external dependency (no onnx, no onnxruntime, no protobuf).

ONNX is a protobuf wire-format file. This parser walks the protobuf
fields to extract TensorProto initializers (the model weights) and
converts them to numpy float32 arrays — the same interface as
safetensors_parser.py and gguf_parser.py.

Usage:
    from onnx_parser import ONNXParser
    parser = ONNXParser()
    tensors = parser.parse("model.onnx")
    # tensors: dict of {name: numpy_array(float32)}

Author: Nightmare Designs 💀
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


# ════════════════════════════════════════════════════════════════
# MINIMAL PROTOBUF WIRE FORMAT READER
# ════════════════════════════════════════════════════════════════

# Protobuf wire types
WT_VARINT = 0
WT_FIXED64 = 1
WT_LENGTH_DELIMITED = 2
WT_FIXED32 = 5


def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a base-128 varint. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint")
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos


def _read_field_tag(data: bytes, pos: int) -> Tuple[int, int, int]:
    """Read a protobuf field tag. Returns (field_number, wire_type, new_pos)."""
    tag, pos = _read_varint(data, pos)
    field_number = tag >> 3
    wire_type = tag & 0x07
    return field_number, wire_type, pos


def _read_length_delimited(data: bytes, pos: int) -> Tuple[bytes, int]:
    """Read a length-delimited field. Returns (bytes, new_pos)."""
    length, pos = _read_varint(data, pos)
    end = pos + length
    if end > len(data):
        raise ValueError("Truncated length-delimited field")
    return data[pos:end], end


def _read_fixed32(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a 32-bit fixed value. Returns (value, new_pos)."""
    if pos + 4 > len(data):
        raise ValueError("Truncated fixed32")
    return struct.unpack_from('<I', data, pos)[0], pos + 4


def _read_fixed64(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a 64-bit fixed value. Returns (value, new_pos)."""
    if pos + 8 > len(data):
        raise ValueError("Truncated fixed64")
    return struct.unpack_from('<Q', data, pos)[0], pos + 8


# ════════════════════════════════════════════════════════════════
# ONNX TENSOR DATA TYPES
# ════════════════════════════════════════════════════════════════

# ONNX TensorProto.DataType enum values
ONNX_FLOAT = 1
ONNX_UINT8 = 2
ONNX_INT8 = 3
ONNX_UINT16 = 4
ONNX_INT16 = 5
ONNX_INT32 = 6
ONNX_INT64 = 7
ONNX_BOOL = 9
ONNX_FLOAT16 = 10
ONNX_DOUBLE = 11
ONNX_UINT32 = 12
ONNX_UINT64 = 13
ONNX_BFLOAT16 = 14

# Numpy dtype mapping for each ONNX type
_NUMPY_DTYPE = {
    ONNX_FLOAT: np.float32,
    ONNX_UINT8: np.uint8,
    ONNX_INT8: np.int8,
    ONNX_UINT16: np.uint16,
    ONNX_INT16: np.int16,
    ONNX_INT32: np.int32,
    ONNX_INT64: np.int64,
    ONNX_BOOL: np.bool_,
    ONNX_FLOAT16: np.float16,
    ONNX_DOUBLE: np.float64,
    ONNX_UINT32: np.uint32,
    ONNX_UINT64: np.uint64,
}

# Itemsize for each type (for raw_data interpretation)
_ITEMSIZE = {
    ONNX_FLOAT: 4, ONNX_UINT8: 1, ONNX_INT8: 1, ONNX_UINT16: 2,
    ONNX_INT16: 2, ONNX_INT32: 4, ONNX_INT64: 8, ONNX_BOOL: 1,
    ONNX_FLOAT16: 2, ONNX_DOUBLE: 8, ONNX_UINT32: 4, ONNX_UINT64: 8,
    ONNX_BFLOAT16: 2,
}


# ════════════════════════════════════════════════════════════════
# ONNX TENSOR PROTO PARSER
# ════════════════════════════════════════════════════════════════

def _parse_tensor_proto(data: bytes) -> Tuple[str, np.ndarray]:
    """Parse a TensorProto message. Returns (name, numpy_array).

    TensorProto fields (from ONNX schema):
      1: dims (repeated int64, varint)
      2: data_type (int32, varint)
      3: float_data (repeated float, fixed32)
      4: int32_data (repeated int32, varint)
      5: string_data (repeated bytes, length-delimited) [deprecated]
      6: int64_data (repeated int64, varint)
      7: name (string, length-delimited) [field 8 in newer schema]
      8: name (string, length-delimited)
      9: raw_data (bytes, length-delimited)
      12: double_data (repeated double, fixed64)
      13: uint64_data (repeated uint64, varint)
    """
    dims = []
    data_type = ONNX_FLOAT
    name = ""
    raw_data = None
    float_data = []
    int32_data = []
    int64_data = []
    double_data = []
    uint64_data = []

    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = _read_field_tag(data, pos)

        if field_num == 1:  # dims
            if wire_type == WT_VARINT:
                val, pos = _read_varint(data, pos)
                dims.append(val)
            elif wire_type == WT_LENGTH_DELIMITED:  # packed repeated
                chunk, pos = _read_length_delimited(data, pos)
                cpos = 0
                while cpos < len(chunk):
                    val, cpos = _read_varint(chunk, cpos)
                    dims.append(val)
        elif field_num == 2:  # data_type
            data_type, pos = _read_varint(data, pos)
        elif field_num == 3:  # float_data
            if wire_type == WT_FIXED32:
                val, pos = _read_fixed32(data, pos)
                float_data.append(struct.unpack('<f', struct.pack('<I', val))[0])
            elif wire_type == WT_LENGTH_DELIMITED:  # packed
                chunk, pos = _read_length_delimited(data, pos)
                float_data = list(np.frombuffer(chunk, dtype='<f4'))
        elif field_num == 4:  # int32_data
            if wire_type == WT_VARINT:
                val, pos = _read_varint(data, pos)
                int32_data.append(val)
            elif wire_type == WT_LENGTH_DELIMITED:
                chunk, pos = _read_length_delimited(data, pos)
                int32_data = list(np.frombuffer(chunk, dtype='<i4'))
        elif field_num == 6:  # int64_data
            if wire_type == WT_VARINT:
                val, pos = _read_varint(data, pos)
                int64_data.append(val)
            elif wire_type == WT_LENGTH_DELIMITED:
                chunk, pos = _read_length_delimited(data, pos)
                int64_data = list(np.frombuffer(chunk, dtype='<i8'))
        elif field_num == 7 or field_num == 8:  # name
            if wire_type == WT_LENGTH_DELIMITED:
                name_bytes, pos = _read_length_delimited(data, pos)
                name = name_bytes.decode('utf-8', errors='replace')
        elif field_num == 9:  # raw_data
            raw_data, pos = _read_length_delimited(data, pos)
        elif field_num == 12:  # double_data
            if wire_type == WT_FIXED64:
                val, pos = _read_fixed64(data, pos)
                double_data.append(struct.unpack('<d', struct.pack('<Q', val))[0])
            elif wire_type == WT_LENGTH_DELIMITED:
                chunk, pos = _read_length_delimited(data, pos)
                double_data = list(np.frombuffer(chunk, dtype='<f8'))
        elif field_num == 13:  # uint64_data
            if wire_type == WT_VARINT:
                val, pos = _read_varint(data, pos)
                uint64_data.append(val)
            elif wire_type == WT_LENGTH_DELIMITED:
                chunk, pos = _read_length_delimited(data, pos)
                uint64_data = list(np.frombuffer(chunk, dtype='<u8'))
        else:
            # Skip unknown field
            pos = _skip_field(data, pos, wire_type)

    # Build the numpy array from whichever data source is present
    arr = _build_array(data_type, dims, raw_data, float_data,
                       int32_data, int64_data, double_data, uint64_data)
    return name, arr


def _skip_field(data: bytes, pos: int, wire_type: int) -> int:
    """Skip an unknown protobuf field."""
    if wire_type == WT_VARINT:
        _, pos = _read_varint(data, pos)
    elif wire_type == WT_FIXED64:
        pos += 8
    elif wire_type == WT_LENGTH_DELIMITED:
        length, pos = _read_varint(data, pos)
        pos += length
    elif wire_type == WT_FIXED32:
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")
    return pos


def _build_array(data_type, dims, raw_data, float_data,
                 int32_data, int64_data, double_data, uint64_data):
    """Build a numpy array from the parsed TensorProto fields."""
    shape = [int(d) for d in dims] if dims else []
    n_elems = 1
    for d in shape:
        n_elems *= d

    # Priority: raw_data > typed data arrays
    if raw_data is not None and len(raw_data) > 0:
        itemsize = _ITEMSIZE.get(data_type, 4)
        if data_type == ONNX_BFLOAT16:
            # BF16: upper 16 bits of float32, stored as 2 bytes little-endian
            raw = np.frombuffer(raw_data, dtype=np.uint16)
            # Shift left 16 to get float32
            f32 = np.frombuffer((raw.astype(np.uint32) << 16).astype('<u4').tobytes(),
                               dtype=np.float32)
            arr = f32.copy()
        elif data_type == ONNX_FLOAT16:
            arr = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32).copy()
        else:
            np_dtype = _NUMPY_DTYPE.get(data_type, np.float32)
            arr = np.frombuffer(raw_data, dtype=np_dtype).copy()
    elif float_data:
        arr = np.array(float_data, dtype=np.float32)
    elif int32_data:
        arr = np.array(int32_data, dtype=np.int32)
    elif int64_data:
        arr = np.array(int64_data, dtype=np.int64)
    elif double_data:
        arr = np.array(double_data, dtype=np.float64)
    elif uint64_data:
        arr = np.array(uint64_data, dtype=np.uint64)
    else:
        arr = np.zeros(n_elems, dtype=np.float32)

    # Reshape to the declared shape
    if shape:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            # Shape mismatch — flatten
            arr = arr.reshape(-1)

    # Convert to float32 for the suite's inference engine
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    return arr


# ════════════════════════════════════════════════════════════════
# ONNX MODEL PROTO PARSER
# ════════════════════════════════════════════════════════════════

class ONNXParser:
    """Parse an ONNX model file and extract weight tensors (initializers).

    ONNX ModelProto structure:
      1: ir_version (int64)
      3: producer_name (string)
      5: opset_import (repeated)
      7: graph (GraphProto) — the main subgraph

    GraphProto structure:
      1: node (repeated NodeProto)
      2: name (string)
      5: initializer (repeated TensorProto) — WEIGHT TENSORS
      10: input (repeated ValueInfoProto)
      11: output (repeated ValueInfoProto)
    """

    def __init__(self):
        self.tensors = {}
        self.metadata = {}
        self.name = ""

    def parse(self, path: str) -> Dict[str, np.ndarray]:
        """Parse an ONNX file. Returns dict of {tensor_name: numpy_array}."""
        path = Path(path)
        print(f"  [ONNX] Parsing: {path.name}")

        with open(path, 'rb') as f:
            data = f.read()

        # Verify ONNX magic (protobuf doesn't have a fixed magic, but ONNX
        # files often start with field 1 (ir_version) = varint tag 0x08)
        if len(data) < 2:
            raise ValueError("File too small to be ONNX")

        self._parse_model_proto(data)

        print(f"  [ONNX] Loaded {len(self.tensors)} weight tensors")
        if self.name:
            print(f"  [ONNX] Graph: {self.name}")
        return self.tensors

    def _parse_model_proto(self, data: bytes):
        """Walk the ModelProto to find the GraphProto (field 7)."""
        pos = 0
        while pos < len(data):
            field_num, wire_type, pos = _read_field_tag(data, pos)
            if field_num == 7 and wire_type == WT_LENGTH_DELIMITED:
                # GraphProto
                graph_data, pos = _read_length_delimited(data, pos)
                self._parse_graph_proto(graph_data)
            elif field_num == 3 and wire_type == WT_LENGTH_DELIMITED:
                # producer_name
                name_bytes, pos = _read_length_delimited(data, pos)
                self.metadata['producer'] = name_bytes.decode('utf-8', errors='replace')
            elif field_num == 1 and wire_type == WT_VARINT:
                # ir_version
                ir_ver, pos = _read_varint(data, pos)
                self.metadata['ir_version'] = ir_ver
            else:
                pos = _skip_field(data, pos, wire_type)

    def _parse_graph_proto(self, data: bytes):
        """Extract initializers (weight tensors) from a GraphProto."""
        pos = 0
        while pos < len(data):
            field_num, wire_type, pos = _read_field_tag(data, pos)
            if field_num == 5 and wire_type == WT_LENGTH_DELIMITED:
                # initializer (TensorProto)
                tensor_data, pos = _read_length_delimited(data, pos)
                try:
                    name, arr = _parse_tensor_proto(tensor_data)
                    if name:
                        self.tensors[name] = arr
                except Exception as e:
                    print(f"  [ONNX] Warning: failed to parse tensor: {e}")
            elif field_num == 2 and wire_type == WT_LENGTH_DELIMITED:
                # graph name
                name_bytes, pos = _read_length_delimited(data, pos)
                self.name = name_bytes.decode('utf-8', errors='replace')
            elif field_num == 1 and wire_type == WT_LENGTH_DELIMITED:
                # node (NodeProto) — skip, we only want weights
                _, pos = _read_length_delimited(data, pos)
            elif field_num == 10 and wire_type == WT_LENGTH_DELIMITED:
                # input — skip
                _, pos = _read_length_delimited(data, pos)
            elif field_num == 11 and wire_type == WT_LENGTH_DELIMITED:
                # output — skip
                _, pos = _read_length_delimited(data, pos)
            else:
                pos = _skip_field(data, pos, wire_type)

    def close(self):
        """No resources to free (file already closed)."""
        pass


def load_onnx(path: str) -> Dict[str, np.ndarray]:
    """Convenience function: load an ONNX file → dict of tensors."""
    parser = ONNXParser()
    return parser.parse(path)
